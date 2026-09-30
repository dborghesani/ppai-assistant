from crewai import BaseLLM
import re
import requests
import time
import os
import structlog
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import create_urllib3_context

# Disable OpenTelemetry spam
os.environ["OTEL_SDK_DISABLED"] = "true"

REASONING_TAGS = ["think", "thinking", "reasoning"]
_TAG_PAIRS = [(rf"<{t}>", rf"</{t}>") for t in REASONING_TAGS]
_TAG_PAIRS += [(rf"<\|{t}\|>", rf"<\|/{t}\|>") for t in REASONING_TAGS]
_CLOSING_RE = re.compile("|".join(close for _, close in _TAG_PAIRS), re.IGNORECASE)
_OPENING_RE = re.compile("|".join(open_ for open_, _ in _TAG_PAIRS), re.IGNORECASE)
_BLOCK_RES = [re.compile(f"{o}.*?{c}", re.DOTALL | re.IGNORECASE) for o, c in _TAG_PAIRS]


class InsecureHTTPAdapter(HTTPAdapter):
    """HTTPAdapter that skips certificate verification (equivalent to `curl -k`)."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = 0  # ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _to_text(value) -> str:
    """Flatten an OpenAI-style content field (str, None or list of parts) into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value)

    chunks = []
    for item in value:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, dict) and item.get("type") != "reasoning":
            text = item.get("text") or item.get("content") or ""
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(c for c in chunks if c).strip()


def _drop_reasoning(text: str) -> str:
    """Return the answer part of `text`, discarding inline <think>-style reasoning."""
    if not text:
        return ""
    # Models emit the answer after the thinking block, so prefer what follows the last closing tag.
    closings = list(_CLOSING_RE.finditer(text))
    if closings:
        return text[closings[-1].end():].strip()
    for pattern in _BLOCK_RES:
        text = pattern.sub("", text)
    # An opening tag that is never closed means the generation was truncated
    # mid-thinking: everything from the tag onward is reasoning, not answer.
    opening = _OPENING_RE.search(text)
    if opening:
        return text[:opening.start()].strip()
    return text.strip()


def _unwrap_code_fence(text: str) -> str:
    """Unwrap a fence that encloses the whole answer; CrewAI's ReAct parser needs bare text."""
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```") or len(stripped) < 6:
        return text
    inner = stripped[3:-3]
    if "```" in inner:  # several separate blocks, the fences are part of the content
        return text
    first_line, newline, rest = inner.partition("\n")
    # Drop the opening language tag only when it is one (i.e. a single bare word).
    if newline and (not first_line.strip() or first_line.strip().isalnum()):
        inner = rest
    return inner.strip()


def _normalize_messages(messages) -> list[dict]:
    """Accept a string or message list and return messages with a single leading system turn."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    system = [m["content"] for m in messages if m["role"] == "system"]
    others = [m for m in messages if m["role"] != "system"]
    if not system:
        return others
    return [{"role": "system", "content": "\n\n".join(system)}] + others


class StellantisVLLM(BaseLLM):

    _NO_THINK_NUDGE = (
        "IMPORTANT: keep any internal reasoning extremely short. "
        "Reply with the final answer only, in the requested format."
    )

    def __init__(self,
                 model: str,
                 max_tokens: int = 50000,
                 verify_ssl: bool = True,
                 timeout: int = 1200,
                 max_retries: int = 3,
                 temperature: float | None = None,
                 enable_thinking: bool | None = None):

        super().__init__(model=model)
        self.model_name = model
        self.endpoint = f"https://apps.services.calypso.intra.chrysler.com/{model}/v1/chat/completions"
        self.max_tokens = max_tokens
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        # None: let the server/template decide; False: ask vLLM to render the chat
        # template without the thinking block (Qwen3-style models honour this).
        self.enable_thinking = enable_thinking
        self.logger = structlog.get_logger(__name__)

        self.session = requests.Session()
        adapter_cls = HTTPAdapter if verify_ssl else InsecureHTTPAdapter
        adapter = adapter_cls(
            max_retries=Retry(
                total=max_retries,
                backoff_factor=2.0,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST", "GET"],
                raise_on_status=False,  # let the loop in call() decide
            ),
            pool_connections=10,
            pool_maxsize=10,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _post(self, payload: dict) -> dict:
        response = self.session.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Connection": "close",  # avoid stale keep-alive connections
            },
            json=payload,
            verify=self.verify_ssl,
            timeout=(20, self.timeout),
        )
        if response.status_code >= 400:
            self.logger.error(f"VLLM {response.status_code}: {response.text[:500]}")
        response.raise_for_status()
        return response.json()

    def _extract_answer(self, choice: dict) -> str:
        message = choice.get("message", {})
        content = _to_text(message.get("content"))
        reasoning = _to_text(message.get("reasoning_content") or message.get("reasoning"))

        # When reasoning has its own field, `content` is already the final answer.
        # Never fall back to the reasoning text itself: feeding thinking downstream
        # is worse than returning empty and letting call() retry without thinking.
        candidates = [
            content if reasoning else _drop_reasoning(content),
            _to_text(message.get("output_text")),
            _drop_reasoning(_to_text(choice.get("text"))),
        ]
        return _unwrap_code_fence(next((c for c in candidates if c), ""))

    def _base_payload(self, messages: list[dict]) -> dict:
        payload: dict = {"messages": messages, "max_tokens": self.max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return payload

    def _anti_runaway_payload(self, messages: list[dict]) -> dict:
        """Payload for retries after the model burned the token budget in thinking.

        Resending the same request tends to reproduce the same reasoning loop, so:
        disable the thinking block at the template level, nudge the model to answer
        directly, and penalize the repeated phrases that keep such loops alive.
        """
        nudged = [dict(m) for m in messages]
        if nudged and nudged[-1]["role"] == "user" and isinstance(nudged[-1].get("content"), str):
            nudged[-1]["content"] = f"{nudged[-1]['content']}\n\n{self._NO_THINK_NUDGE}"
        else:
            nudged.append({"role": "user", "content": self._NO_THINK_NUDGE})
        return {
            "messages": nudged,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if self.temperature is not None else 0.6,
            "presence_penalty": 0.5,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def call(self, messages, tools=None, **kwargs):
        normalized = _normalize_messages(messages)
        payload = self._base_payload(normalized)
        anti_runaway = False

        last_error: Exception = ValueError("VLLM request failed")
        for attempt in range(1, self.max_retries + 1):
            if attempt > 1:
                time.sleep(5 * 2 ** (attempt - 2))  # 5s, 10s, 20s, ...
            try:
                data = self._post(payload)
                choice = data["choices"][0]
                finish_reason = choice.get("finish_reason")
                answer = self._extract_answer(choice)
                if answer:
                    if finish_reason == "length":
                        self.logger.warning("VLLM hit max_tokens, the answer may be truncated")
                    return answer
                # Empty answer: either the model spent the whole token budget inside its
                # thinking block (finish_reason=length) or the answer ended up in the
                # reasoning field only. Either way, retry with thinking discouraged.
                usage = data.get("usage") or {}
                last_error = ValueError(f"VLLM returned empty content (finish_reason={finish_reason})")
                self.logger.warning(
                    f"Empty response (attempt {attempt}/{self.max_retries}, "
                    f"finish_reason={finish_reason}, completion_tokens={usage.get('completion_tokens')})"
                )
                if not anti_runaway:
                    self.logger.warning("Retrying with thinking disabled")
                    payload = self._anti_runaway_payload(normalized)
                    anti_runaway = True
            except requests.exceptions.HTTPError as e:
                # If the server rejects the anti-runaway extras (old vLLM), fall back
                # to the plain payload instead of failing the whole call.
                status = e.response.status_code if e.response is not None else None
                if anti_runaway and status is not None and 400 <= status < 500:
                    self.logger.warning(
                        f"Anti-runaway payload rejected with HTTP {status}, reverting to base payload"
                    )
                    payload = self._base_payload(normalized)
                    last_error = e
                    continue
                raise
            except requests.exceptions.Timeout as e:
                # A generation stuck in a thinking loop often surfaces as a read timeout:
                # apply the anti-runaway measures before the next attempt.
                last_error = e
                self.logger.warning(
                    f"VLLM request timed out (attempt {attempt}/{self.max_retries}), "
                    "possible runaway thinking"
                )
                if not anti_runaway:
                    payload = self._anti_runaway_payload(normalized)
                    anti_runaway = True
            except requests.RequestException as e:
                last_error = e
                self.logger.warning(
                    f"VLLM request failed (attempt {attempt}/{self.max_retries}): "
                    f"{type(e).__name__}: {e}"
                )

        raise last_error

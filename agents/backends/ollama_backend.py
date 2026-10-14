import asyncio
import re
import time
from typing import Any, AsyncGenerator, Callable

import structlog
from openai import AsyncOpenAI

from config import ConfigAssistant
from agents.backends.llm_logging import usage_log_fields


class OllamaBackend:
    def __init__(self, opt: ConfigAssistant, on_usage: Callable[[str, Any], None], *, client=None):
        self.opt = opt
        self.on_usage = on_usage
        self.logger = structlog.get_logger()
        self.client = client if client is not None else AsyncOpenAI(
            api_key="ollama", base_url=f"http://{opt.ollama_host}:{opt.ollama_port}/v1",
            timeout=opt.ollama_timeout,
        )

    async def complete(self, *, call_name: str, **request: Any) -> Any:
        started = time.perf_counter()
        usage = None
        text = ""
        status = "error"
        try:
            response = await self.client.chat.completions.create(**request)
            usage = getattr(response, "usage", None)
            if response.choices:
                text = response.choices[0].message.content or ""
            status = "completed"
            return response
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            self._log_result(call_name, request.get("model"), usage, text, started, status)

    async def stream(self, *, call_name: str, **request: Any) -> AsyncGenerator[Any, None]:
        started = time.perf_counter()
        usage = None
        stream = None
        ttft_ms = None
        text_parts = []
        status = "error"
        try:
            stream = await self.client.chat.completions.create(
                **{**request, "stream": True, "stream_options": {"include_usage": True}},
            )
            async for chunk in stream:
                usage = getattr(chunk, "usage", None) or usage
                if chunk.choices and chunk.choices[0].delta.content:
                    text_parts.append(chunk.choices[0].delta.content)
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started) * 1000, 2)
                        self.logger.debug("[LLM] Time to first token", llm_call=call_name,
                                          model=request.get("model"), ttft_ms=ttft_ms)
                yield chunk
            status = "completed"
        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        finally:
            try:
                if stream is not None:
                    await stream.close()
            finally:
                self._log_result(call_name, request.get("model"), usage, "".join(text_parts), started, status, ttft_ms)

    def _log_result(self, call_name: str, model: Any, usage: Any, text: str,
                    started: float, status: str, ttft_ms: float | None = None) -> None:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        self.on_usage(call_name, usage)
        self.logger.debug("[LLM] Response generation time", llm_call=call_name,
                          model=model, duration_ms=duration_ms)
        fields = usage_log_fields(usage, self.opt.context_window_size)
        if ttft_ms is not None:
            fields["ttft_ms"] = ttft_ms
        self.logger.info(
            "[LLM] Call finished", llm_call=call_name, model=model, status=status,
            response_text=re.sub(r"^\s*\[\[tone:[^\]]+\]\]\s*", "", text).strip(),
            duration_ms=duration_ms, **fields,
        )

    async def close(self) -> None:
        await self.client.close()
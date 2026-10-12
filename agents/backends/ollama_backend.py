import time
from typing import Any, AsyncGenerator, Callable

import structlog
from openai import AsyncOpenAI

from config import ConfigAssistant


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
        try:
            response = await self.client.chat.completions.create(**request)
            usage = getattr(response, "usage", None)
            return response
        finally:
            self.on_usage(call_name, usage)
            self.logger.info("[LLM] Response generation time", llm_call=call_name,
                             model=request.get("model"), duration_ms=round((time.perf_counter() - started) * 1000, 2))

    async def stream(self, *, call_name: str, **request: Any) -> AsyncGenerator[Any, None]:
        started = time.perf_counter()
        usage = None
        stream = None
        first_token_seen = False
        try:
            stream = await self.client.chat.completions.create(
                **{**request, "stream": True, "stream_options": {"include_usage": True}},
            )
            async for chunk in stream:
                usage = getattr(chunk, "usage", None) or usage
                if not first_token_seen and chunk.choices and chunk.choices[0].delta.content:
                    first_token_seen = True
                    self.logger.info("[LLM] Time to first token", llm_call=call_name,
                                     model=request.get("model"), ttft_ms=round((time.perf_counter() - started) * 1000, 2))
                yield chunk
        finally:
            try:
                if stream is not None:
                    await stream.close()
            finally:
                self.on_usage(call_name, usage)
                self.logger.info("[LLM] Response generation time", llm_call=call_name,
                                 model=request.get("model"), duration_ms=round((time.perf_counter() - started) * 1000, 2))

    async def close(self) -> None:
        await self.client.close()
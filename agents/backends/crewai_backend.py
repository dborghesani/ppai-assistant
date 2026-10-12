import asyncio
import time
from typing import Any, Callable, TypeVar

import structlog
from crewai import Crew, LLM
from pydantic import BaseModel

from config import ConfigAssistant

Output = TypeVar("Output", bound=BaseModel)


class CrewAIBackend:
    def __init__(self, opt: ConfigAssistant, on_usage: Callable[[str, Any], None], *, model: str | None = None):
        self.opt = opt
        self.model = (model or opt.ollama_model).removeprefix("ollama/")
        self.on_usage = on_usage
        self.logger = structlog.get_logger()
        self._lock = asyncio.Lock()
        self.llm = LLM(
            model=f"ollama/{self.model}",
            base_url=f"http://{opt.ollama_host}:{opt.ollama_port}",
            timeout=opt.ollama_timeout, max_tokens=opt.max_tokens,
            temperature=0.1, additional_params={"reasoning_effort": "none"},
        )

    def usage_snapshot(self) -> dict[str, int] | None:
        getter = getattr(self.llm, "get_token_usage_summary", None)
        if getter is None:
            return None
        usage = getter()
        keys = ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(usage, dict) and all(isinstance(usage.get(key), int) for key in keys):
            return {key: usage[key] for key in keys}
        return None

    async def run(self, crew: Crew, inputs: dict[str, Any], output: type[Output], name: str) -> Output:
        async with self._lock:
            started = time.perf_counter()
            before = self.usage_snapshot()
            try:
                result = await crew.kickoff_async(inputs=inputs)
                value = getattr(result, "pydantic", None)
                if not isinstance(value, output):
                    raise ValueError(f"{name} did not return {output.__name__}")
                return value
            finally:
                after = self.usage_snapshot()
                usage = None
                if before is not None and after is not None:
                    delta = {key: after[key] - before[key] for key in before}
                    if all(value >= 0 for value in delta.values()):
                        usage = delta
                self.on_usage(name, usage)
                self.logger.info("[LLM] Task execution time", task=name, model=self.model,
                                 duration_ms=round((time.perf_counter() - started) * 1000, 2))
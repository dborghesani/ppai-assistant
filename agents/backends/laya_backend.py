import asyncio
import time
from pathlib import Path
from typing import Any, Callable

import structlog

class LayaBackend:
    def __init__(self, laya_model: str | Path | None, on_usage: Callable[[str, Any], None], *, predictor=None):
        self.on_usage = on_usage
        self.logger = structlog.get_logger()
        self._lock = asyncio.Lock()
        if predictor is not None:
            self.predictor = predictor
            return
        import laya

        if laya_model:
            model_source = Path(laya_model).expanduser()
            if not model_source.is_dir():
                raise FileNotFoundError(f"Custom Laya model directory does not exist: {model_source}")
            self.predictor = laya.load(model_source, subfolder=None)
        else:
            self.predictor = laya.load("convaiinnovations/laya", subfolder="typed-decisions")

    async def predict(self, state: dict, questions: dict, *, call_name: str = "laya prediction") -> dict:
        async with self._lock:
            started = time.perf_counter()
            try:
                return await asyncio.to_thread(self.predictor.predict, state, questions)
            finally:
                self.on_usage(call_name, None)
                self.logger.info("[LAYA] Prediction time", task=call_name,
                                 duration_ms=round((time.perf_counter() - started) * 1000, 2))
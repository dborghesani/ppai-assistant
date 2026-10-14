"""Decision-model backend interface.

Both Laya and Kev expose the same contract: typed questions answered about one
state document in a single pass. `DecisionBackend` fixes that interface
(`execute_decision`) so tasks never need to know which decision model engine
is configured. The output is always a raw dict containing an `answers` object
mapping question ids to their answer payloads.
"""
from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import structlog


class DecisionBackend(ABC):
    def __init__(self, on_usage: Callable[[str, Any], None]):
        self.on_usage = on_usage
        self.logger = structlog.get_logger()
        self.last_duration_ms: float | None = None

    async def warmup(self) -> None:
        await self.execute_decision(
            {"user_input": "Hello", "conversation_history": [], "pending_message_count": 0},
            {
                "requires_action_classification": {
                    "type": "choice",
                    "instructions": "Does this greeting request an action?",
                    "criteria": {"required": "An action is requested.", "not_required": "A greeting only."},
                },
                "response_source": {
                    "type": "choice",
                    "instructions": "Choose the information source for this greeting.",
                    "criteria": {"general": "General conversation.", "vehicle_state": "Live telemetry.",
                                 "vehicle_manual": "Vehicle instructions."},
                },
            },
            call_name="decision warmup",
        )

    @abstractmethod
    async def execute_decision(self, state: dict, questions: dict, *, call_name: str = "decision") -> dict:
        """Answer typed questions about a state document.

        Args:
            state: The state document the model reads (dict serialized as the state).
            questions: Mapping of question id -> typed question
                (`{"type": "choice", "instructions": str, "criteria": {...}}`).
            call_name: Label used for usage logging.
        Returns:
            Raw provider response containing an `answers` dict.
        """


class KevBackend(DecisionBackend):
    """Decision backend backed by a Kev checkpoint from the Hugging Face Hub.

    No additional server is needed: the checkpoint (`jaredpalmer/kev-0.8b`,
    `jaredpalmer/kev-4b`, ...) is loaded during backend initialization and scored
    in-process with the `kev` library — a LoRA adapter plus a pointer head on a
    Qwen3.5 base model. Loading follows `kev.serve`:
    `Checkpoint(repo).load(device, LoadOptions())` gives tokenizer and model,
    `SystemOneRequest` validates the request, `to_record`/`admit` prepare the
    encoded record and `model.probs(enc)` returns one probability distribution
    per question. No text is generated.

    The local fork lives in agents/backends/kev; its provenance and current
    compatibility limitations are documented in FORK.md in that directory.
    """

    def __init__(self, opt, on_usage: Callable[[str, Any], None], *, model: str | None = None):
        super().__init__(on_usage)
        self.model = model or opt.kev_model or "jaredpalmer/kev-0.8b"
        self._lock = asyncio.Lock()
        import torch
        from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
        from kev.checkpoint import Checkpoint, LoadOptions
        from transformers.utils import logging as transformers_logging

        device = "cuda" if torch.cuda.is_available() else "cpu"
        hub_progress_disabled = are_progress_bars_disabled()
        transformers_progress_enabled = transformers_logging.is_progress_bar_enabled()
        self.logger.debug("[KEV] Loading model", model=self.model, device=device)
        transformers_logging.disable_progress_bar()
        disable_progress_bars()
        try:
            checkpoint = Checkpoint(self.model)
            self._tokenizer, self._model = checkpoint.load(device, LoadOptions())
            self._temperature = getattr(checkpoint.meta, "temperature", 1.0)
            self._model.head.temperature = self._temperature
        finally:
            if transformers_progress_enabled:
                transformers_logging.enable_progress_bar()
            if hub_progress_disabled:
                disable_progress_bars()
            else:
                enable_progress_bars()
        self.logger.debug("[KEV] Model loaded", model=self.model, device=device)

    async def execute_decision(self, state: dict, questions: dict, *, call_name: str = "decision") -> dict:
        async with self._lock:
            started = time.perf_counter()
            try:
                return await asyncio.to_thread(self._decide_sync, state, questions)
            except Exception:
                self.logger.exception("[KEV] Decision failed", task=call_name, model=self.model)
                raise
            finally:
                self.last_duration_ms = round((time.perf_counter() - started) * 1000, 2)
                self.on_usage(call_name, None)
                self.logger.info(
                    "[KEV] Decision time",
                    task=call_name,
                    model=self.model,
                    duration_ms=self.last_duration_ms,
                )

    def _decide_sync(self, state: dict, questions: dict) -> dict:
        from kev.api import SystemOneRequest, to_answers, to_record
        from kev.model import admit

        request = SystemOneRequest.model_validate({
            "state": state if isinstance(state, str) else json.dumps(state, ensure_ascii=True),
            "model": "kev-latest",
            "questions": questions,
        })
        record, meta = to_record(request)
        encodings = admit(self._model, self._tokenizer, record)
        probabilities = self._model.probs(encodings)
        return {"answers": to_answers(probabilities, meta)}


class LayaBackend(DecisionBackend):
    """Decision backend backed by a Laya typed-decisions model."""

    def __init__(self, laya_model: str | Path | None, on_usage: Callable[[str, Any], None], *, predictor=None):
        super().__init__(on_usage)
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

    async def execute_decision(self, state: dict, questions: dict, *, call_name: str = "decision") -> dict:
        async with self._lock:
            started = time.perf_counter()
            try:
                return await asyncio.to_thread(self.predictor.predict, state, questions)
            finally:
                self.last_duration_ms = round((time.perf_counter() - started) * 1000, 2)
                self.on_usage(call_name, None)
                self.logger.info("[LAYA] Prediction time", task=call_name,
                                 duration_ms=self.last_duration_ms)

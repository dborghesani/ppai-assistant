import asyncio
import queue
import time

import numpy as np
import structlog

try:
    import torch
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
except ImportError:
    torch = None  # type: ignore[assignment]
    CheckpointInfo = None  # type: ignore[assignment]
    TTSModel = None  # type: ignore[assignment]
    DEFAULT_DSM_TTS_REPO = ""

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]

logger = structlog.get_logger()

# Smaller (0.75B, English-only) DSM TTS model; use "" for the library default (1.6B, en_fr)
DEFAULT_TTS_REPO = "kyutai/tts-0.75b-en-public"
DEFAULT_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"


class TTSManager:
    """Text-to-speech using Kyutai TTS (Delayed Streams Modeling, PyTorch backend)."""

    def __init__(
        self,
        enabled: bool = True,
        hf_repo: str = DEFAULT_TTS_REPO,
        voice: str = DEFAULT_VOICE,
        device: str = "cuda",
        n_q: int = 16,
        cfg_coef: float = 3.0,
    ):
        self.enabled = enabled and sd is not None and TTSModel is not None
        self._model = None
        self._prefix = None
        self._condition_attributes = None
        self._cfg_coef = cfg_coef

        if enabled and (sd is None or TTSModel is None):
            logger.warning("sounddevice/moshi not installed, TTS disabled")
            return
        if not self.enabled:
            return

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for TTS (will be slow)")
            device = "cpu"

        try:
            logger.info("Loading Kyutai TTS model", hf_repo=hf_repo or DEFAULT_DSM_TTS_REPO)
            checkpoint_info = CheckpointInfo.from_hf_repo(hf_repo or DEFAULT_DSM_TTS_REPO)
            self._model = TTSModel.from_checkpoint_info(
                checkpoint_info, n_q=n_q, temp=0.6, cfg_coef=cfg_coef, device=device
            )
            voice_path = self._model.get_voice_path(voice)
            if self._model.multi_speaker:
                # CFG-conditioned voice embedding (requires a precomputed .safetensors file)
                self._prefix = None
                self._condition_attributes = self._model.make_condition_attributes(
                    [voice_path], cfg_coef=self._cfg_coef
                )
            else:
                # Single-speaker model: clone the voice via an audio prefix instead
                self._prefix = self._model.get_prefix(voice_path)
                self._condition_attributes = None
        except Exception as e:
            logger.warning("Failed to load Kyutai TTS model, TTS disabled", error=str(e))
            self.enabled = False

    async def speak(self, text: str) -> None:
        if not self.enabled or not text:
            return
        await asyncio.to_thread(self._speak_blocking, text)

    def _speak_blocking(self, text: str) -> None:
        try:
            entries = self._model.prepare_script([text], padding_between=1)
            attributes = [self._condition_attributes] if self._condition_attributes else []
            prefixes = [self._prefix] if self._prefix is not None else None
            # Frames corresponding to the voice-cloning prefix must be skipped, they
            # reproduce the reference audio, not the requested text.
            skip_frames = self._prefix.shape[-1] if self._prefix is not None else 0

            pcms: queue.Queue = queue.Queue()
            frame_count = 0

            def _on_frame(frame):
                nonlocal frame_count
                if (frame[:, 1:] != -1).all():
                    frame_count += 1
                    if frame_count <= skip_frames:
                        return
                    pcm = self._model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcms.put_nowait(np.clip(pcm[0, 0], -1, 1))

            def _audio_callback(outdata, _frames, _time_info, _status):
                try:
                    outdata[:, 0] = pcms.get(block=False)
                except queue.Empty:
                    outdata[:] = 0

            with sd.OutputStream(
                samplerate=self._model.mimi.sample_rate,
                blocksize=1920,
                channels=1,
                callback=_audio_callback,
            ):
                with self._model.mimi.streaming(1):
                    self._model.generate([entries], attributes, prefixes=prefixes, on_frame=_on_frame)
                while pcms.qsize() > 0:
                    time.sleep(0.1)
        except Exception as e:
            logger.warning("TTS playback failed", error=str(e))

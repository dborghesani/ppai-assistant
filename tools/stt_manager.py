import itertools
import math

import numpy as np
import structlog

try:
    import torch
    import moshi.models
except ImportError:
    torch = None  # type: ignore[assignment]
    moshi = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]

logger = structlog.get_logger()

DEFAULT_STT_REPO = "kyutai/stt-1b-en_fr"


class STTManager:
    """Push-to-talk speech-to-text using Kyutai STT (Delayed Streams Modeling, PyTorch backend)."""

    def __init__(
        self,
        enabled: bool = True,
        hf_repo: str = DEFAULT_STT_REPO,
        device: str = "cuda",
    ):
        self.enabled = enabled and sd is not None and moshi is not None
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._info = None
        self._mimi = None
        self._tokenizer = None
        self._lm = None
        self._device = device
        self._sample_rate = 24000

        if enabled and (sd is None or moshi is None):
            logger.warning("sounddevice/moshi not installed, STT disabled")
            return
        if not self.enabled:
            return

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for STT (will be slow)")
            device = "cpu"
            self._device = device

        try:
            self._info = moshi.models.loaders.CheckpointInfo.from_hf_repo(hf_repo)
            self._mimi = self._info.get_mimi(device=device)
            self._sample_rate = self._mimi.sample_rate
            self._tokenizer = self._info.get_text_tokenizer()
            self._lm = self._info.get_moshi(device=device, dtype=torch.bfloat16)
        except Exception as e:
            logger.warning("Failed to load Kyutai STT model, STT disabled", error=str(e))
            self.enabled = False

    def start_recording(self) -> None:
        if not self.enabled or self._stream is not None:
            return
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            logger.debug("Audio input status", status=str(status))
        self._frames.append(indata.copy())

    def stop_and_transcribe(self) -> str:
        """Blocking: stops the stream and runs the model. Call via asyncio.to_thread."""
        if not self.enabled or self._stream is None:
            return ""

        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._frames:
            return ""

        audio = np.concatenate(self._frames, axis=0).flatten()
        self._frames = []

        try:
            return self._run_model(audio)
        except Exception as e:
            logger.warning("Transcription failed", error=str(e))
            return ""

    def _run_model(self, audio: np.ndarray) -> str:
        lm_gen = moshi.models.LMGen(self._lm, temp=0, temp_text=0.0)
        frame_size = self._mimi.frame_size

        audio_t = torch.from_numpy(audio).to(self._device)[None, None]
        if audio_t.shape[-1] % frame_size != 0:
            to_pad = frame_size - audio_t.shape[-1] % frame_size
            audio_t = torch.nn.functional.pad(audio_t, (0, to_pad))

        stt_config = self._info.stt_config
        prefix_seconds = stt_config.get("audio_silence_prefix_seconds", 1.0)
        delay_seconds = stt_config.get("audio_delay_seconds", 5.0)
        n_prefix_chunks = math.ceil(prefix_seconds * self._mimi.frame_rate)
        n_suffix_chunks = math.ceil(delay_seconds * self._mimi.frame_rate)
        silence_chunk = torch.zeros(
            (1, 1, frame_size), dtype=torch.float32, device=self._device
        )

        chunks = itertools.chain(
            itertools.repeat(silence_chunk, n_prefix_chunks),
            torch.split(audio_t, frame_size, dim=-1),
            itertools.repeat(silence_chunk, n_suffix_chunks),
        )

        pieces: list[str] = []
        with torch.no_grad(), self._mimi.streaming(1), lm_gen.streaming(1):
            for chunk in chunks:
                if chunk.shape[-1] != frame_size:
                    continue
                audio_tokens = self._mimi.encode(chunk)
                text_tokens = lm_gen.step(audio_tokens)
                if text_tokens is None:
                    continue
                token = text_tokens[0, 0, 0].cpu().item()
                if token not in (0, 3):
                    pieces.append(self._tokenizer.id_to_piece(token).replace("▁", " "))

        return "".join(pieces).strip()

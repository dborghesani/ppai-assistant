import asyncio
import queue
import threading
import time

import numpy as np
import structlog
from voice import moshi_compat  # noqa: F401  # must run before importing Moshi
from voice.gpu_lock import GPU_LOCK

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
        self.hf_repo = hf_repo
        self.enabled = enabled and sd is not None and TTSModel is not None
        self._model = None
        self._prefix = None
        self._condition_attributes = None
        self._cfg_coef = cfg_coef
        self._stop_flag = threading.Event()
        self._playback_stream = None
        self.is_speaking = False
        # Rolling buffer of recently-played audio, used by STTManager to tell
        # apart mic-captured TTS echo from a genuine user barge-in.
        self._playback_ring = np.zeros(0, dtype=np.float32)
        self._playback_ring_lock = threading.Lock()
        self._playback_ring_max_samples = 0

        if enabled and (sd is None or TTSModel is None):
            logger.warning("sounddevice/moshi not installed, TTS disabled")
            return
        if not self.enabled:
            return

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA not available, falling back to CPU for TTS (will be slow)"
            )
            device = "cpu"

        try:
            logger.info(
                "Loading Kyutai TTS model", hf_repo=hf_repo
            )
            checkpoint_info = CheckpointInfo.from_hf_repo(
                hf_repo
            )
            self._model = TTSModel.from_checkpoint_info(
                checkpoint_info, n_q=n_q, temp=0.6, device=device
            )
            voice_path = self._model.get_voice_path(voice)
            # Not every checkpoint was trained with CFG distillation (e.g. the 0.75B
            # public model); passing cfg_coef to a model that doesn't support it raises.
            supports_cfg = bool(self._model.valid_cfg_conditionings)
            effective_cfg_coef = self._cfg_coef if supports_cfg else None
            if not supports_cfg and self._cfg_coef != 1.0:
                logger.info(
                    "TTS checkpoint has no CFG distillation support, ignoring cfg_coef",
                    requested_cfg_coef=self._cfg_coef,
                )

            if self._model.multi_speaker:
                # CFG-conditioned voice embedding (requires a precomputed .safetensors file)
                self._prefix = None
                self._condition_attributes = self._model.make_condition_attributes(
                    [voice_path], cfg_coef=effective_cfg_coef
                )
            else:
                # Single-speaker model: clone the voice via an audio prefix instead
                self._prefix = self._model.get_prefix(voice_path)
                self._condition_attributes = (
                    self._model.make_condition_attributes(
                        [], cfg_coef=effective_cfg_coef
                    )
                    if supports_cfg
                    else None
                )
            logger.info("Kyutai TTS model loaded successfully")
        except Exception as e:
            logger.warning(
                "Failed to load Kyutai TTS model, TTS disabled", error=str(e)
            )
            self.enabled = False
            return

        self._playback_ring_max_samples = int(self._model.mimi.sample_rate * 1.5)
        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy generation (no audio playback) so CUDA kernel compilation/
        autotuning happens at startup instead of during the first spoken response."""
        try:
            t0 = time.time()
            entries = self._model.prepare_script(["Hello."], padding_between=1)
            attributes = (
                [self._condition_attributes]
                if self._condition_attributes is not None
                else []
            )
            prefixes = [self._prefix] if self._prefix is not None else None
            with GPU_LOCK, self._model.mimi.streaming(1):
                self._model.generate(
                    [entries],
                    attributes,
                    prefixes=prefixes,
                    on_frame=lambda frame: None,
                )
            logger.info(f"TTS warmup done in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning("TTS warmup failed", error=str(e))

    async def speak(self, text: str) -> None:
        if not self.enabled or not text:
            return
        await asyncio.to_thread(self._speak_blocking, text)

    def stop(self) -> None:
        """Immediately halt any ongoing playback (barge-in on user speech)."""
        self._stop_flag.set()
        if self._playback_stream is not None:
            try:
                self._playback_stream.abort()
            except Exception:
                pass

    def _push_playback_reference(self, pcm: np.ndarray) -> None:
        with self._playback_ring_lock:
            self._playback_ring = np.concatenate([self._playback_ring, pcm])
            max_len = self._playback_ring_max_samples
            if max_len and self._playback_ring.shape[0] > max_len:
                self._playback_ring = self._playback_ring[-max_len:]

    def get_recent_playback(self, n_samples: int) -> np.ndarray:
        """Last `n_samples` of audio actually sent to the speakers, zero-padded
        on the left if not enough history is available yet."""
        with self._playback_ring_lock:
            ring = self._playback_ring
            if ring.shape[0] >= n_samples:
                return ring[-n_samples:].copy()
            pad = n_samples - ring.shape[0]
            return np.concatenate([np.zeros(pad, dtype=np.float32), ring.copy()])

    def _speak_blocking(self, text: str) -> None:
        self._stop_flag.clear()
        self.is_speaking = True
        try:
            entries = self._model.prepare_script([text], padding_between=1)
            attributes = (
                [self._condition_attributes]
                if self._condition_attributes is not None
                else []
            )
            prefixes = [self._prefix] if self._prefix is not None else None
            # Frames corresponding to the voice-cloning prefix must be skipped, they
            # reproduce the reference audio, not the requested text.
            skip_frames = self._prefix.shape[-1] if self._prefix is not None else 0

            pcms: queue.Queue = queue.Queue()
            frame_count = 0

            def _on_frame(frame):
                nonlocal frame_count
                if self._stop_flag.is_set():
                    return
                if (frame[:, 1:] != -1).all():
                    frame_count += 1
                    if frame_count <= skip_frames:
                        return
                    pcm = self._model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcm = np.clip(pcm[0, 0], -1, 1)
                    pcms.put_nowait(pcm)
                    self._push_playback_reference(pcm)

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
            ) as stream:
                self._playback_stream = stream
                with GPU_LOCK, self._model.mimi.streaming(1):
                    self._model.generate(
                        [entries], attributes, prefixes=prefixes, on_frame=_on_frame
                    )
                while pcms.qsize() > 0 and not self._stop_flag.is_set():
                    time.sleep(0.1)
        except Exception as e:
            logger.warning("TTS playback failed", error=str(e))
        finally:
            self._playback_stream = None
            self.is_speaking = False

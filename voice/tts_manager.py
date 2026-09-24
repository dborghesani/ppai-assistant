import asyncio
import queue
import threading
import time
from pathlib import Path

import numpy as np
import structlog
from agents.agents_dataclasses import VoiceType
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

# Project-local custom voice references. These are not guaranteed to be part of the
# base Kyutai voice pack, so we keep a verified fallback and only use custom files
# when they exist on disk.
CUSTOM_VOICES_DIR = Path(__file__).resolve().parent / "custom"
DEFAULT_VOICES_BY_TYPE: dict[VoiceType, str] = {
    VoiceType.CALM: DEFAULT_VOICE,
    VoiceType.ENTHUSIASTIC: DEFAULT_VOICE,
    VoiceType.SERIOUS: DEFAULT_VOICE,
    VoiceType.EMPATIC: DEFAULT_VOICE,
    VoiceType.WHISPER: DEFAULT_VOICE,
}


class TTSManager:
    """Text-to-speech using Kyutai TTS (Delayed Streams Modeling, PyTorch backend)."""

    def __init__(
        self,
        enabled: bool = True,
        hf_repo: str = DEFAULT_TTS_REPO,
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
        self._speech_lock = asyncio.Lock()
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
            self.set_voice(VoiceType.CALM)
            logger.info("Kyutai TTS model loaded successfully")
        except Exception as e:
            logger.warning(
                "Failed to load Kyutai TTS model, TTS disabled", error=str(e)
            )
            self.enabled = False
            return

        self._playback_ring_max_samples = int(self._model.mimi.sample_rate * 1.5)
        self._warmup()

    @staticmethod
    def resolve_voice_name(voice: str | VoiceType | None) -> str:
        if isinstance(voice, VoiceType):
            resolved = DEFAULT_VOICES_BY_TYPE.get(voice)
            if resolved and Path(resolved).exists():
                return resolved
            return DEFAULT_VOICE

        if isinstance(voice, str):
            normalized = voice.strip().lower()
            for voice_type, default_voice in DEFAULT_VOICES_BY_TYPE.items():
                if normalized == voice_type.value.lower():
                    if Path(default_voice).exists():
                        return default_voice
                    return DEFAULT_VOICE

            if Path(voice).exists():
                return voice
            return DEFAULT_VOICE

        return DEFAULT_VOICE

    def set_voice(self, voice: str | VoiceType | None) -> bool:
        """Switch the active TTS voice at runtime for multi-speaker models."""
        if not self.enabled or self._model is None:
            return False

        try:
            resolved_voice = self.resolve_voice_name(voice) or DEFAULT_VOICE
            voice_path = self._model.get_voice_path(resolved_voice)
            supports_cfg = bool(self._model.valid_cfg_conditionings)
            effective_cfg_coef = self._cfg_coef if supports_cfg else None

            if not supports_cfg and self._cfg_coef != 1.0:
                logger.info(
                    "TTS checkpoint has no CFG distillation support, ignoring cfg_coef",
                    requested_cfg_coef=self._cfg_coef,
                )

            if self._model.multi_speaker:
                self._prefix = None
                self._condition_attributes = self._model.make_condition_attributes(
                    [voice_path], cfg_coef=effective_cfg_coef
                )
            else:
                self._prefix = self._model.get_prefix(voice_path)
                self._condition_attributes = (
                    self._model.make_condition_attributes(
                        [], cfg_coef=effective_cfg_coef
                    )
                    if supports_cfg
                    else None
                )

            self._current_voice = resolved_voice
            logger.info("TTS voice updated", voice=resolved_voice)
            return True
        except Exception as e:
            logger.warning(
                "Failed to switch TTS voice, falling back to default voice",
                requested_voice=voice,
                error=str(e),
            )
            if self._current_voice != DEFAULT_VOICE:
                self._current_voice = DEFAULT_VOICE
                return self.set_voice(DEFAULT_VOICE)
            return False

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
        async with self._speech_lock:
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
            playback_buffer = np.zeros(0, dtype=np.float32)
            generation_done = threading.Event()
            playback_done = threading.Event()
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

            def _audio_callback(outdata, _frames, _time_info, _status):
                nonlocal playback_buffer

                # A Mimi frame isn't guaranteed to have the same length as
                # sounddevice's requested block. Keep surplus samples and
                # combine short frames instead of producing gaps or shape
                # errors in the PortAudio callback.
                while playback_buffer.shape[0] < len(outdata):
                    try:
                        pcm = pcms.get(block=False)
                    except queue.Empty:
                        break
                    if pcm.size:
                        if playback_buffer.size:
                            playback_buffer = np.concatenate((playback_buffer, pcm))
                        else:
                            playback_buffer = pcm

                outdata.fill(0)
                count = min(len(outdata), playback_buffer.shape[0])
                if count:
                    outdata[:count, 0] = playback_buffer[:count]
                    playback_buffer = playback_buffer[count:]
                    self._push_playback_reference(outdata[:count, 0])

                if (
                    generation_done.is_set()
                    and pcms.empty()
                    and playback_buffer.size == 0
                ):
                    playback_done.set()

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
                generation_done.set()
                while not playback_done.is_set() and not self._stop_flag.is_set():
                    time.sleep(0.1)
        except Exception as e:
            logger.warning("TTS playback failed", error=str(e))
        finally:
            self._playback_stream = None
            self.is_speaking = False

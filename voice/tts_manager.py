import asyncio
import re
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import structlog
from data.agents_dataclasses import ToneType
from voice import moshi_compat  # noqa: F401  # must run before importing Moshi
from voice.gpu_lock import GPU_LOCK
from voice.hf_cache import (
    checkpoint_info_cache_first as _checkpoint_info_cache_first,
    download_hf_file_cache_first as _download_hf_file_cache_first,
    resolve_hf_reference as _resolve_hf_reference,
)

try:
    import torch
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
except ImportError:
    torch = None  # type: ignore[assignment]
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
DEFAULT_VOICES_BY_TYPE: dict[ToneType, str] = {
    ToneType.CALM: DEFAULT_VOICE,
    ToneType.ENTHUSIASTIC: DEFAULT_VOICE,
    ToneType.SERIOUS: DEFAULT_VOICE,
    ToneType.EMPATHETIC: DEFAULT_VOICE,
    ToneType.DISCREET: DEFAULT_VOICE,
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
        self._finish_word = threading.Event()
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
            load_started = time.perf_counter()
            checkpoint_info = _checkpoint_info_cache_first(hf_repo)
            self._model = TTSModel.from_checkpoint_info(
                checkpoint_info, n_q=n_q, temp=0.6, device=device
            )
            self.set_voice(ToneType.CALM)
            logger.info(
                "Kyutai TTS model loaded successfully",
                device=device,
                elapsed_seconds=round(time.perf_counter() - load_started, 2),
            )
        except Exception as e:
            logger.warning(
                "Failed to load Kyutai TTS model, TTS disabled", error=str(e)
            )
            self.enabled = False
            return

        self._playback_ring_max_samples = int(self._model.mimi.sample_rate * 1.5)
        self._warmup()

    @staticmethod
    def resolve_voice_name(voice: str | ToneType | None) -> str:
        if isinstance(voice, ToneType):
            resolved = DEFAULT_VOICES_BY_TYPE.get(voice)
            if resolved and Path(resolved).exists():
                return resolved
            return DEFAULT_VOICE

        if isinstance(voice, str):
            normalized = voice.strip().lower()
            for tone_type, default_voice in DEFAULT_VOICES_BY_TYPE.items():
                if normalized == tone_type.value.lower():
                    if Path(default_voice).exists():
                        return default_voice
                    return DEFAULT_VOICE

            if Path(voice).exists():
                return voice
            return DEFAULT_VOICE

        return DEFAULT_VOICE

    def set_voice(self, voice: str | ToneType | None) -> bool:
        """Switch the active TTS voice at runtime for multi-speaker models."""
        if not self.enabled or self._model is None:
            return False

        try:
            resolved_voice = self.resolve_voice_name(voice) or DEFAULT_VOICE
            voice_repo = getattr(self._model, "voice_repo", None)
            voice_suffix = getattr(self._model, "voice_suffix", None)
            if Path(resolved_voice).is_file():
                voice_path = Path(resolved_voice)
            elif isinstance(voice_repo, str) and isinstance(voice_suffix, str):
                voice_path = _resolve_hf_reference(
                    voice_repo, f"{resolved_voice}{voice_suffix}"
                )
            else:
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
            logger.debug(f"TTS warmup done in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning("TTS warmup failed", error=str(e))

    async def speak(self, text: str, on_word: Callable[[str], None] | None = None) -> None:
        """Speak `text`, optionally reporting the spoken text prefix every time a
        word finishes playing (word boundaries estimated proportionally to the
        generated audio; accurate because the sentence is fully generated before
        playback starts)."""
        if not self.enabled or not text:
            return
        async with self._speech_lock:
            await asyncio.to_thread(self._speak_blocking, text, on_word)

    def stop(self, finish_word: bool = False) -> None:
        """Halt any ongoing playback. With `finish_word=True` the word currently
        being spoken is allowed to complete before playback stops (a natural
        interruption instead of a mid-word cutoff)."""
        self._stop_flag.set()
        if finish_word:
            self._finish_word.set()
            return
        self._finish_word.clear()
        if self._playback_stream is not None:
            try:
                self._playback_stream.abort()
            except Exception:
                pass

    @staticmethod
    def _word_boundaries(text: str) -> tuple[list[str], list[int], list[int]]:
        """Split `text` into words with their character start/end offsets."""
        words, starts, ends = [], [], []
        for match in re.finditer(r"\S+", text):
            words.append(match.group())
            starts.append(match.start())
            ends.append(match.end())
        return words, starts, ends

    @staticmethod
    def _word_end_samples(text: str, total_samples: int) -> list[int]:
        """Estimated sample offset at which each word finishes, distributed
        proportionally to character position across the generated audio."""
        if not text or total_samples <= 0:
            return []
        _, _, ends = TTSManager._word_boundaries(text)
        char_total = len(text.rstrip())
        return [max(1, int(round(total_samples * min(end, char_total) / char_total))) for end in ends]

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

    def _speak_blocking(self, text: str, on_word: Callable[[str], None] | None = None) -> None:
        self._stop_flag.clear()
        self._finish_word.clear()
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

            # Generate the full sentence first: playback is sequential anyway
            # (one sentence at a time under the speech lock) and knowing the
            # whole audio up-front gives exact word timing for on_word events
            # and finish-word stops.
            pcms: list[np.ndarray] = []
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
                    pcms.append(np.clip(pcm[0, 0], -1, 1))

            with GPU_LOCK, self._model.mimi.streaming(1):
                self._model.generate(
                    [entries], attributes, prefixes=prefixes, on_frame=_on_frame
                )
            if self._stop_flag.is_set() or not pcms:
                return
            pcm = pcms[0] if len(pcms) == 1 else np.concatenate(pcms)
            # Keep the GPU lock through playback: releasing it lets other CUDA
            # work interleave with mimi's compiled (cudagraph) state and kills
            # subsequent generations with "Offset increment outside graph
            # capture encountered unexpectedly".
            with GPU_LOCK:
                self._play_audio(pcm, text, on_word)
        except Exception as e:
            logger.warning("TTS playback failed", error=str(e))
        finally:
            self._playback_stream = None
            self.is_speaking = False

    def _play_audio(self, pcm: np.ndarray, text: str, on_word: Callable[[str], None] | None) -> None:
        sample_rate = self._model.mimi.sample_rate
        block = 1920
        total = pcm.shape[0]
        word_end_samples = self._word_end_samples(text, total) if on_word is not None else []
        _, starts, ends = self._word_boundaries(text) if word_end_samples else ([], [], [])
        next_word = 0
        finish_at = -1  # sample index at which a finish-word stop must cut
        played = 0

        with sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32"
        ) as stream:
            self._playback_stream = stream
            for start in range(0, total, block):
                if self._stop_flag.is_set():
                    if not self._finish_word.is_set():
                        return
                    if finish_at < 0:
                        # Finish the word currently being spoken: cut at the
                        # first word boundary ahead of the playback position.
                        finish_at = next(
                            (end for end in word_end_samples if end > played),
                            played,
                        )
                    if played >= finish_at:
                        return
                samples = np.clip(pcm[start:start + block], -1, 1)
                try:
                    stream.write(samples.reshape(-1, 1))
                except Exception:
                    return
                played = min(start + block, total)
                self._push_playback_reference(samples)
                while next_word < len(word_end_samples) and played >= word_end_samples[next_word]:
                    if on_word is not None and self.is_speaking:
                        on_word(text[: ends[next_word]].strip())
                    next_word += 1
            while next_word < len(word_end_samples):
                if on_word is not None and self.is_speaking:
                    on_word(text[: ends[next_word]].strip())
                next_word += 1

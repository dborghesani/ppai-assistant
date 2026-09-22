import itertools
import math
import queue
import threading
import time
from typing import Callable

import numpy as np
import structlog
from voice import moshi_compat  # noqa: F401  # must run before importing Moshi
from voice.gpu_lock import GPU_LOCK

try:
    import moshi.models
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
    moshi = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]

logger = structlog.get_logger()

DEFAULT_STT_REPO = "kyutai/stt-1b-en_fr"


def _max_normalized_correlation(mic: np.ndarray, reference: np.ndarray) -> float:
    """Best-lag normalized cross-correlation of `mic` against `reference`.

    Returns a value close to 1.0 when `mic` closely matches some contiguous
    segment of `reference` (i.e. it looks like echo of that reference audio),
    and close to 0.0 when it doesn't (i.e. it looks like unrelated new audio).
    `reference` must be at least as long as `mic`.
    """
    if mic.size == 0 or reference.size < mic.size:
        return 0.0
    mic = mic.astype(np.float64) - mic.mean()
    reference = reference.astype(np.float64) - reference.mean()
    mic_norm = np.sqrt(np.sum(mic**2))
    if mic_norm < 1e-6:
        return 0.0

    corr = np.correlate(reference, mic, mode="valid")
    if corr.size == 0:
        return 0.0

    window = mic.size
    cumsum = np.cumsum(np.insert(reference**2, 0, 0.0))
    local_energy = cumsum[window:] - cumsum[:-window]
    local_norm = np.sqrt(np.clip(local_energy, 0.0, None)) + 1e-8

    normalized = np.abs(corr) / (mic_norm * local_norm)
    return float(np.max(normalized))


class STTManager:
    """Push-to-talk speech-to-text using Kyutai STT (Delayed Streams Modeling, PyTorch backend)."""

    def __init__(
        self,
        enabled: bool = True,
        hf_repo: str = DEFAULT_STT_REPO,
        device: str = "cuda",
    ):
        self.hf_repo = hf_repo
        self.enabled = enabled and sd is not None and moshi is not None
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._info = None
        self._mimi = None
        self._tokenizer = None
        self._lm = None
        self._device = device
        self._sample_rate = 24000
        self._conversation_thread: threading.Thread | None = None
        self._conversation_stop = threading.Event()
        self._conversation_stream = None

        if enabled and (sd is None or moshi is None):
            logger.warning("sounddevice/moshi not installed, STT disabled")
            return
        if not self.enabled:
            return

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA not available, falling back to CPU for STT (will be slow)"
            )
            device = "cpu"
            self._device = device

        try:
            logger.info(f"Loading Kyutai STT model", hf_repo=hf_repo)
            self._info = moshi.models.loaders.CheckpointInfo.from_hf_repo(hf_repo)
            self._mimi = self._info.get_mimi(device=device)
            self._sample_rate = self._mimi.sample_rate
            self._tokenizer = self._info.get_text_tokenizer()
            self._lm = self._info.get_moshi(device=device, dtype=torch.bfloat16)
        except Exception as e:
            logger.warning(
                "Failed to load Kyutai STT model, STT disabled", error=str(e)
            )
            self.enabled = False
            return

        logger.info("Kyutai STT model loaded successfully")

        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy inference so CUDA kernel compilation/autotuning happens at
        startup instead of during the user's first push-to-talk request."""
        try:
            t0 = time.time()
            self._run_model(np.zeros(self._sample_rate, dtype=np.float32))
            logger.info(f"STT warmup done in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning("STT warmup failed", error=str(e))

    def start_recording(self) -> None:
        if not self.enabled or self._stream is not None:
            return
        self._frames = []
        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
            self._stream = None
            self._frames = []
            raise

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
        with GPU_LOCK, torch.no_grad(), self._mimi.streaming(1), lm_gen.streaming(1):
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

    def start_conversation(
        self,
        on_utterance: Callable[[str], None],
        on_speech_start: Callable[[], None] | None = None,
        on_session_timeout: Callable[[], None] | None = None,
        vad_head_index: int = 2,
        vad_threshold: float = 0.5,
        session_silence_timeout: float = 10.0,
        pause_seconds: float = 1.0,
        is_tts_speaking: Callable[[], bool] | None = None,
        mute_mic_during_tts: bool = True,
        barge_in_energy_threshold: float = 0.035,
        barge_in_min_frames: int = 4,
        get_tts_reference: Callable[[int], "np.ndarray"] | None = None,
        echo_correlation_threshold: float = 0.6,
    ) -> bool:
        """Start a continuous, always-listening session with turn-taking.

        Returns True if the microphone stream and session thread were started
        successfully, False otherwise (callers should not assume success just
        because no exception was raised).

        Uses the model's semantic-VAD extra heads when the checkpoint provides them;
        otherwise falls back to counting consecutive silence tokens emitted by the
        model itself (most public STT checkpoints, e.g. stt-1b-en_fr, have no VAD heads).

        Without acoustic echo cancellation, leaving the mic active while the
        assistant speaks can make it transcribe its own voice as new user input,
        causing a runaway feedback loop. When `mute_mic_during_tts` is True and
        `is_tts_speaking()` returns True, incoming audio is not transcribed. A
        raw-energy check (`barge_in_energy_threshold`/`barge_in_min_frames`) stays
        active during that window so the user can still interrupt the assistant.
        If `get_tts_reference` is provided, loud audio is additionally compared
        (via cross-correlation) against the audio actually being played; a strong
        match is treated as TTS echo and ignored instead of triggering barge-in.

        Callbacks are invoked from a background thread; callers must marshal them
        back onto their own event loop/thread as needed.
        """
        if not self.enabled:
            logger.warning("start_conversation called but STTManager is disabled")
            return False
        if (
            self._conversation_thread is not None
            and self._conversation_thread.is_alive()
        ):
            logger.warning(
                "start_conversation called while a session is already running"
            )
            return False

        self._conversation_stop.clear()
        audio_queue: queue.Queue = queue.Queue()

        def _on_audio(indata, frames, time_info, status):
            if status:
                logger.debug("Audio input status", status=str(status))
            audio_queue.put(indata.copy())

        try:
            self._conversation_stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._mimi.frame_size,
                callback=_on_audio,
            )
            self._conversation_stream.start()
        except Exception as e:
            logger.warning(
                "Failed to open microphone for conversation mode", error=str(e)
            )
            self._conversation_stream = None
            return False

        logger.info("Conversation mode: microphone stream started")
        self._conversation_thread = threading.Thread(
            target=self._conversation_loop,
            args=(
                audio_queue,
                on_utterance,
                on_speech_start,
                on_session_timeout,
                vad_head_index,
                vad_threshold,
                session_silence_timeout,
                pause_seconds,
                is_tts_speaking,
                mute_mic_during_tts,
                barge_in_energy_threshold,
                barge_in_min_frames,
                get_tts_reference,
                echo_correlation_threshold,
            ),
            daemon=True,
        )
        self._conversation_thread.start()
        return True

    def stop_conversation(self) -> None:
        self._conversation_stop.set()
        if self._conversation_thread is not None:
            self._conversation_thread.join(timeout=2.0)
            self._conversation_thread = None
        if self._conversation_stream is not None:
            try:
                self._conversation_stream.stop()
                self._conversation_stream.close()
            except Exception:
                pass
            self._conversation_stream = None

    def close(self) -> None:
        """Release every audio input stream without starting transcription."""
        self.stop_conversation()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._frames = []

    def _conversation_loop(
        self,
        audio_queue: "queue.Queue",
        on_utterance: Callable[[str], None],
        on_speech_start: Callable[[], None] | None,
        on_session_timeout: Callable[[], None] | None,
        vad_head_index: int,
        vad_threshold: float,
        session_silence_timeout: float,
        pause_seconds: float,
        is_tts_speaking: Callable[[], bool] | None,
        mute_mic_during_tts: bool,
        barge_in_energy_threshold: float,
        barge_in_min_frames: int,
        get_tts_reference: Callable[[int], "np.ndarray"] | None,
        echo_correlation_threshold: float,
    ) -> None:
        frame_size = self._mimi.frame_size
        idle_frame_threshold = max(1, round(pause_seconds * self._mimi.frame_rate))
        pieces: list[str] = []
        speaking = False
        idle_frames = 0
        last_activity = time.time()
        utterance_start_time: float | None = None
        # While muted (assistant speaking), track raw audio energy instead of
        # transcribing, so the user can still barge in without echo triggering a
        # false transcription/feedback loop.
        barge_in_energy_frames = 0
        barge_in_triggered = False
        mic_echo_window = np.zeros(0, dtype=np.float32)
        mic_echo_window_max = int(0.24 * self._sample_rate)
        echo_reference_len = mic_echo_window_max + int(0.5 * self._sample_rate)
        # Buffer incoming audio and slice it ourselves: the input device is not
        # guaranteed to deliver callbacks of exactly `frame_size` samples.
        pending = np.zeros((0,), dtype=np.float32)

        logger.info("Conversation loop started")
        try:
            lm_gen = moshi.models.LMGen(self._lm, temp=0, temp_text=0.0)
            has_vad_heads = len(self._lm.extra_heads) > 0
            if not has_vad_heads:
                logger.warning(
                    "Checkpoint has no semantic-VAD heads, using silence-token fallback",
                    pause_seconds=pause_seconds,
                )
            with torch.no_grad(), self._mimi.streaming(1), lm_gen.streaming(1):
                while not self._conversation_stop.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        if time.time() - last_activity > session_silence_timeout:
                            if on_session_timeout is not None:
                                on_session_timeout()
                            return
                        continue

                    pending = np.concatenate([pending, chunk[:, 0]])

                    # Without echo cancellation, the mic can pick up the assistant's
                    # own speaker output while it talks. Discard audio in that window
                    # instead of feeding it to the model, to avoid a self-feedback loop.
                    if (
                        mute_mic_during_tts
                        and is_tts_speaking is not None
                        and is_tts_speaking()
                    ):
                        mic_echo_window = np.concatenate(
                            [mic_echo_window, chunk[:, 0]]
                        )[-mic_echo_window_max:]
                        if (
                            not barge_in_triggered
                            and mic_echo_window.shape[0] >= mic_echo_window_max
                        ):
                            rms = float(
                                np.sqrt(
                                    np.mean(
                                        np.square(mic_echo_window, dtype=np.float64)
                                    )
                                )
                            )
                            if rms > barge_in_energy_threshold:
                                is_echo = False
                                correlation = None
                                if get_tts_reference is not None:
                                    reference = get_tts_reference(echo_reference_len)
                                    correlation = _max_normalized_correlation(
                                        mic_echo_window, reference
                                    )
                                    is_echo = correlation > echo_correlation_threshold
                                if is_echo:
                                    logger.debug(
                                        "Barge-in candidate suppressed as TTS echo",
                                        rms=round(rms, 4),
                                        correlation=round(correlation, 3)
                                        if correlation is not None
                                        else None,
                                    )
                                    barge_in_energy_frames = 0
                                else:
                                    barge_in_energy_frames += 1
                                    if barge_in_energy_frames >= barge_in_min_frames:
                                        barge_in_triggered = True
                                        logger.info(
                                            "Barge-in triggered by raw audio energy",
                                            rms=round(rms, 4),
                                            threshold=barge_in_energy_threshold,
                                            frames=barge_in_energy_frames,
                                            correlation=correlation,
                                        )
                                        if on_speech_start is not None:
                                            on_speech_start()
                            else:
                                barge_in_energy_frames = 0
                        pending = np.zeros((0,), dtype=np.float32)
                        pieces = []
                        speaking = False
                        idle_frames = 0
                        utterance_start_time = None
                        continue

                    # No longer muted: reset the barge-in gate for the next mute window.
                    barge_in_energy_frames = 0
                    barge_in_triggered = False
                    mic_echo_window = np.zeros(0, dtype=np.float32)

                    while pending.shape[0] >= frame_size:
                        frame, pending = pending[:frame_size], pending[frame_size:]

                        chunk_t = torch.from_numpy(frame).to(self._device)[None, None]
                        with GPU_LOCK:
                            audio_tokens = self._mimi.encode(chunk_t)
                            step_out = lm_gen.step_with_extra_heads(audio_tokens)
                        if step_out is None:
                            continue
                        text_tokens, vad_heads = step_out

                        token = text_tokens[0, 0, 0].cpu().item()
                        if token not in (0, 3):
                            piece = self._tokenizer.id_to_piece(token).replace("▁", " ")
                            pieces.append(piece)
                            idle_frames = 0
                            last_activity = time.time()
                            if not speaking:
                                utterance_start_time = time.time()
                                if on_speech_start is not None:
                                    on_speech_start()
                            speaking = True
                        elif speaking:
                            idle_frames += 1

                        end_of_turn = False
                        if vad_heads:
                            pause_probability = (
                                vad_heads[vad_head_index][0, 0, 0].cpu().item()
                            )
                            end_of_turn = pause_probability > vad_threshold
                        elif speaking and idle_frames >= idle_frame_threshold:
                            end_of_turn = True

                        if end_of_turn and pieces:
                            tail_latency = time.time() - last_activity
                            speech_duration = (
                                last_activity - utterance_start_time
                                if utterance_start_time
                                else 0.0
                            )
                            utterance = "".join(pieces).strip()
                            pieces = []
                            speaking = False
                            idle_frames = 0
                            utterance_start_time = None
                            last_activity = time.time()
                            if utterance:
                                logger.info(
                                    f">>> STT utterance ready: {speech_duration:.2f}s speech, "
                                    f"{tail_latency:.2f}s tail latency",
                                    text=utterance,
                                )
                                on_utterance(utterance)
        except Exception as e:
            logger.warning("Conversation loop stopped due to error", error=str(e))
        finally:
            logger.info("Conversation loop exiting")
            self._conversation_thread = None
            if self._conversation_stream is not None:
                try:
                    self._conversation_stream.stop()
                    self._conversation_stream.close()
                except Exception:
                    pass
                self._conversation_stream = None

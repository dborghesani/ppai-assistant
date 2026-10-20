import queue
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from voice.gpu_lock import GPU_LOCK
from voice.stt_manager import STTManager, _max_normalized_correlation


def run_conversation(monkeypatch, predictions, *, echo=False, heads=True,
                     reference=True, stop_playback=True, mute=True, audio=None,
                     pause_seconds=0.16):
    manager = STTManager.__new__(STTManager)
    manager._device = "cpu"
    manager._sample_rate = 100
    manager._conversation_stop = threading.Event()
    manager._conversation_stream = None
    manager._info = SimpleNamespace(stt_config={"audio_delay_seconds": 0.16})
    manager._lm = SimpleNamespace(extra_heads=[None] * 4 if heads else [])
    manager._tokenizer = SimpleNamespace(id_to_piece=lambda token: {
        101: "Stop", 102: " please", 103: " now", 104: "?",
        105: "Hello, can you hear", 106: " me?",
    }[token])
    encoded = []

    def streaming(batch):
        assert GPU_LOCK.locked()
        return nullcontext()

    def encode(chunk):
        assert GPU_LOCK.locked()
        encoded.append(chunk.clone())
        return chunk

    manager._mimi = SimpleNamespace(frame_size=8, frame_rate=12.5,
                                    encode=encode, streaming=streaming)
    outputs = iter(predictions)

    def step(chunk):
        assert GPU_LOCK.locked()
        token, pause = next(outputs)
        probability = torch.tensor([[[pause, 1 - pause, 0, 0, 0, 0]]])
        return torch.tensor([[[token]]]), [probability] * 4 if heads else []

    generator = SimpleNamespace(step_with_extra_heads=step, streaming=streaming)
    monkeypatch.setattr("voice.stt_manager.moshi.models.LMGen", lambda *args, **kwargs: generator)
    monkeypatch.setattr(
        "voice.stt_manager._max_normalized_correlation",
        lambda *args: float(echo(len(encoded)) if callable(echo) else echo),
    )
    frames = iter([np.ones((8, 1), dtype=np.float32) * 0.1 if audio is None else audio
                   for _ in predictions])

    class AudioQueue:
        def get(self, timeout):
            try:
                return next(frames)
            except StopIteration:
                manager._conversation_stop.set()
                raise queue.Empty

    tts_active = [True]
    interruptions = []
    utterances = []

    def interrupt():
        interruptions.append(tts_active[0])
        if stop_playback:
            tts_active[0] = False

    logger = Mock()
    monkeypatch.setattr("voice.stt_manager.logger", logger)
    manager._conversation_loop(
        AudioQueue(), utterances.append, interrupt, None, 2, 0.5, 100, pause_seconds,
        lambda: tts_active[0], mute, 0.035, 4,
        (lambda count: np.zeros(count, dtype=np.float32)) if reference else None,
        0.6, 0.4,
    )
    if heads:
        logger.warning.assert_not_called()
    else:
        logger.warning.assert_called_once_with(
            "Checkpoint has no semantic-VAD heads, using silence-token fallback",
            pause_seconds=pause_seconds,
        )
    return interruptions, utterances, encoded


def test_semantic_vad_interrupts_during_tts_and_keeps_transcription(monkeypatch):
    predictions = [(3, 0.9)] * 12 + [(3, 0.1)] * 4
    predictions += [(101, 0.1), (102, 0.1), (3, 0.9), (3, 0.9), (103, 0.9)]
    predictions += [(3, 0.9)] * 3
    interruptions, utterances, encoded = run_conversation(monkeypatch, predictions)
    assert interruptions == [True]
    assert utterances == ["Stop please now"]
    assert len(encoded) == len(predictions)


def test_turn_waits_for_last_text_tokens_and_punctuation(monkeypatch):
    predictions = [(3, 0.9)] * 2 + [(101, 0.1), (102, 0.9)]
    predictions += [(3, 0.9), (103, 0.9), (104, 0.9)] + [(3, 0.9)] * 3
    interruptions, utterances, _ = run_conversation(monkeypatch, predictions)
    assert interruptions == [True]
    assert utterances == ["Stop please now?"]


def test_pause_inside_request_does_not_split_last_word(monkeypatch):
    predictions = [(3, 0.9)] * 2 + [(105, 0.1)]
    predictions += [(3, 0.9)] * 8 + [(106, 0.1)] + [(3, 0.9)] * 14
    interruptions, utterances, _ = run_conversation(
        monkeypatch, predictions, pause_seconds=1.0,
    )
    assert interruptions == [True]
    assert utterances == ["Hello, can you hear me?"]


def test_distinct_requests_are_not_merged_after_confirmed_silence(monkeypatch):
    predictions = [(3, 0.9)] * 2 + [(101, 0.1)] + [(3, 0.9)] * 14
    predictions += [(105, 0.1), (106, 0.1)] + [(3, 0.9)] * 14
    _, utterances, _ = run_conversation(monkeypatch, predictions, pause_seconds=1.0)
    assert utterances == ["Stop", "Hello, can you hear me?"]


def test_punctuation_alone_does_not_start_speech_or_submit_request(monkeypatch):
    predictions = [(3, 0.9)] * 2 + [(104, 0.9)] + [(3, 0.9)] * 5
    interruptions, utterances, _ = run_conversation(monkeypatch, predictions)
    assert interruptions == []
    assert utterances == []


def test_recognized_word_interrupts_before_vad_warmup(monkeypatch):
    predictions = [(3, 0.9)] * 2 + [(101, 0.1), (102, 0.1)] + [(3, 0.9)] * 3
    interruptions, utterances, _ = run_conversation(monkeypatch, predictions)
    assert interruptions == [True]
    assert utterances == ["Stop please"]


@pytest.mark.parametrize("pause", [0.1, 0.9])
def test_echo_does_not_interrupt_even_with_spurious_model_output(monkeypatch, pause):
    predictions = [(101, pause)] * 20
    interruptions, utterances, encoded = run_conversation(monkeypatch, predictions, echo=True)
    assert interruptions == []
    assert utterances == []
    assert all(not frame.any() for frame in encoded)


def test_loud_noise_without_model_speech_does_not_interrupt(monkeypatch):
    interruptions, utterances, _ = run_conversation(monkeypatch, [(3, 0.9)] * 20)
    assert interruptions == []
    assert utterances == []


def test_echo_tail_is_filtered_after_interruption(monkeypatch):
    predictions = [(3, 0.9)] * 2 + [(101, 0.1)] + [(3, 0.9)] * 3 + [(102, 0.9)] * 3
    interruptions, utterances, encoded = run_conversation(
        monkeypatch, predictions, echo=lambda frame: frame >= 3,
    )
    assert interruptions == [True]
    assert utterances == ["Stop"]
    assert all(not frame.any() for frame in encoded[3:])


def test_initial_unstable_vad_and_short_dip_do_not_interrupt(monkeypatch):
    predictions = [(3, 0.1)] * 12 + [(3, 0.9)] + [(3, 0.1)] * 3 + [(3, 0.9)] * 4
    interruptions, utterances, _ = run_conversation(monkeypatch, predictions)
    assert interruptions == []
    assert utterances == []


def test_vad_interrupts_only_once_while_playback_is_stopping(monkeypatch):
    predictions = [(3, 0.9)] * 12 + [(3, 0.1)] * 12
    interruptions, _, _ = run_conversation(monkeypatch, predictions, stop_playback=False)
    assert interruptions == [True]


def test_headphones_allow_model_barge_in_without_playback_reference(monkeypatch):
    predictions = [(3, 0.9)] * 2 + [(101, 0.1)] + [(3, 0.9)] * 3
    interruptions, utterances, _ = run_conversation(monkeypatch, predictions, reference=False, mute=False)
    assert interruptions == [True]
    assert utterances == ["Stop"]


@pytest.mark.parametrize("heads,reference", [(False, True), (True, False)])
def test_missing_vad_or_echo_reference_retains_muted_energy_path(monkeypatch, heads, reference):
    predictions = [(3, 0.9)] * 12
    interruptions, _, encoded = run_conversation(
        monkeypatch, predictions, heads=heads, reference=reference, stop_playback=False,
    )
    assert interruptions == [True]
    assert encoded == []


def test_structured_config_exposes_barge_in_vad_threshold():
    from omegaconf import OmegaConf

    from config import ConfigAssistant

    config = OmegaConf.structured(ConfigAssistant)
    assert config.conversation_barge_in_vad_threshold == 0.4
    updated = OmegaConf.merge(config, {"conversation_barge_in_vad_threshold": 0.3})
    assert updated.conversation_barge_in_vad_threshold == 0.3


def test_echo_correlation_distinguishes_delayed_echo_from_other_speech():
    random = np.random.default_rng(42)
    reference = random.normal(size=1000).astype(np.float32)
    echo = reference[200:440] * 0.3
    other = random.normal(size=240).astype(np.float32)
    assert _max_normalized_correlation(echo, reference) > 0.99
    assert _max_normalized_correlation(other, reference) < 0.6
from __future__ import annotations

from pathlib import Path

import pytest

from data.agents_dataclasses import ToneType
from voice.hf_cache import download_hf_file_cache_first
from voice.tts_manager import TTSManager


class FakeModel:
    def __init__(self, multi_speaker: bool):
        self.multi_speaker = multi_speaker
        self.valid_cfg_conditionings = ["cfg"]
        self.calls = []

    def get_voice_path(self, voice: str) -> str:
        self.calls.append(("get_voice_path", voice))
        return f"voice::{voice}"

    def make_condition_attributes(self, voices, cfg_coef=None):
        self.calls.append(("make_condition_attributes", list(voices), cfg_coef))
        return {"voices": list(voices), "cfg_coef": cfg_coef}

    def get_prefix(self, voice_path: str):
        self.calls.append(("get_prefix", voice_path))
        return f"prefix::{voice_path}"


def test_cancelled_speech_waits_for_audio_worker_before_next_sentence(monkeypatch):
    import asyncio
    import threading

    async def check():
        manager = TTSManager.__new__(TTSManager)
        manager.enabled = True
        manager._speech_lock = asyncio.Lock()
        manager._stop_flag = threading.Event()
        manager._finish_word = threading.Event()
        manager._playback_stream = None
        started = asyncio.Event()
        release_worker = threading.Event()
        calls = []
        loop = asyncio.get_running_loop()

        def blocking_speak(text, on_word):
            calls.append(text)
            if text == "First":
                loop.call_soon_threadsafe(started.set)
                if not release_worker.wait(timeout=2):
                    raise RuntimeError("audio worker was not released")

        monkeypatch.setattr(manager, "_speak_blocking", blocking_speak)
        first = asyncio.create_task(manager.speak("First"))
        second = None
        try:
            await asyncio.wait_for(started.wait(), 1)
            first.cancel()
            second = asyncio.create_task(manager.speak("Second"))
            await asyncio.wait({first, second}, timeout=0.05)
            assert manager._stop_flag.is_set()
            assert not first.done()
            assert calls == ["First"]
            release_worker.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            await second
            assert calls == ["First", "Second"]
        finally:
            release_worker.set()
            await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)

    asyncio.run(check())


def test_stop_does_not_abort_stream_from_another_thread():
    import threading
    from unittest.mock import Mock

    manager = TTSManager.__new__(TTSManager)
    manager._stop_flag = threading.Event()
    manager._finish_word = threading.Event()
    manager._playback_stream = Mock()
    manager.stop()
    assert manager._stop_flag.is_set()
    manager._playback_stream.abort.assert_not_called()


@pytest.mark.parametrize("stopping", [False, True])
def test_audio_write_errors_are_not_silenced_unless_stopping(monkeypatch, stopping):
    import threading
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np

    manager = TTSManager.__new__(TTSManager)
    manager._model = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))
    manager._stop_flag = threading.Event()
    manager._finish_word = threading.Event()
    stream = MagicMock()
    stream.__enter__.return_value = stream

    def fail_write(samples):
        if stopping:
            manager._stop_flag.set()
        raise RuntimeError("ALSA playback failed")

    stream.write.side_effect = fail_write
    monkeypatch.setattr("voice.tts_manager.sd.OutputStream", lambda **kwargs: stream)
    pcm = np.zeros(1920, dtype=np.float32)

    if stopping:
        manager._play_audio(pcm, "Hello.", None)
    else:
        with pytest.raises(RuntimeError, match="ALSA playback failed"):
            manager._play_audio(pcm, "Hello.", None)


def test_playback_owner_aborts_after_write_when_stop_requested(monkeypatch):
    import threading
    from types import SimpleNamespace
    from unittest.mock import MagicMock, Mock

    import numpy as np

    manager = TTSManager.__new__(TTSManager)
    manager._model = SimpleNamespace(mimi=SimpleNamespace(sample_rate=24000))
    manager._stop_flag = threading.Event()
    manager._finish_word = threading.Event()
    stream = MagicMock()
    stream.__enter__.return_value = stream
    owner = threading.get_ident()
    abort_threads = []
    stream.abort.side_effect = lambda: abort_threads.append(threading.get_ident())

    def write(samples):
        interrupt = threading.Thread(target=manager.stop)
        interrupt.start()
        interrupt.join(timeout=1)
        assert not interrupt.is_alive()
        stream.abort.assert_not_called()
        return False

    stream.write.side_effect = write
    monkeypatch.setattr("voice.tts_manager.sd.OutputStream", lambda **kwargs: stream)
    on_word = Mock()
    manager._play_audio(np.zeros(3840, dtype=np.float32), "Hello there.", on_word)
    assert abort_threads == [owner]
    stream.write.assert_called_once()
    on_word.assert_not_called()


def test_tts_releases_gpu_lock_during_playback(monkeypatch):
    import threading
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np
    from voice.gpu_lock import GPU_LOCK

    manager = TTSManager.__new__(TTSManager)
    manager._stop_flag = threading.Event()
    manager._finish_word = threading.Event()
    manager._prefix = None
    manager._condition_attributes = None
    frame = Mock()
    frame.__getitem__ = Mock(return_value=frame)
    frame.__ne__ = Mock(return_value=SimpleNamespace(all=lambda: True))
    pcm = np.zeros((1, 1, 1920), dtype=np.float32)

    def generate(*args, on_frame, **kwargs):
        assert GPU_LOCK.locked()
        on_frame(frame)

    manager._model = SimpleNamespace(
        prepare_script=lambda *args, **kwargs: [],
        generate=generate,
        mimi=SimpleNamespace(streaming=lambda batch: nullcontext(),
                             decode=lambda frame: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: pcm))),
    )
    played = []

    def play(*args):
        acquired = GPU_LOCK.acquire(blocking=False)
        assert acquired
        GPU_LOCK.release()
        played.append(True)

    monkeypatch.setattr(manager, "_play_audio", play)
    manager._speak_blocking("Hello.")
    assert played == [True]


def test_set_voice_multi_speaker_updates_condition_attributes():
    manager = TTSManager.__new__(TTSManager)
    manager.enabled = True
    manager._model = FakeModel(multi_speaker=True)
    manager._condition_attributes = None
    manager._prefix = None
    manager._cfg_coef = 3.0
    manager._current_voice = "default"

    manager.set_voice("calm")

    assert manager._current_voice == "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    assert manager._prefix is None
    assert manager._condition_attributes == {
        "voices": ["voice::expresso/ex03-ex01_happy_001_channel1_334s.wav"],
        "cfg_coef": 3.0,
    }


def test_set_voice_single_speaker_updates_prefix():
    manager = TTSManager.__new__(TTSManager)
    manager.enabled = True
    manager._model = FakeModel(multi_speaker=False)
    manager._condition_attributes = None
    manager._prefix = None
    manager._cfg_coef = 3.0
    manager._current_voice = "default"

    manager.set_voice("energetic")

    assert manager._current_voice == "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    assert manager._prefix == "prefix::voice::expresso/ex03-ex01_happy_001_channel1_334s.wav"
    assert manager._condition_attributes == {"voices": [], "cfg_coef": 3.0}


def test_resolve_voice_name_maps_voice_type_to_default_voice():
    assert TTSManager.resolve_voice_name(ToneType.CALM) == "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    assert TTSManager.resolve_voice_name("energetic") == "expresso/ex03-ex01_happy_001_channel1_334s.wav"


def test_hf_file_download_uses_cache_without_network(monkeypatch, tmp_path):
    cached_file = tmp_path / "cached.safetensors"
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, revision=None: str(cached_file),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit must not call hf_hub_download")
        ),
    )

    assert download_hf_file_cache_first("kyutai/model", "weights.safetensors") == cached_file


def test_hf_file_download_contacts_server_only_after_cache_miss(monkeypatch, tmp_path):
    downloaded_file = tmp_path / "downloaded.safetensors"
    calls = []

    def fake_download(repo_id, filename, **kwargs):
        calls.append(kwargs)
        return str(downloaded_file)

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, revision=None: None,
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    assert download_hf_file_cache_first("kyutai/model", "weights.safetensors") == downloaded_file
    assert len(calls) == 1
    assert "local_files_only" not in calls[0]


def test_set_voice_uses_cached_voice_embedding_without_hub_download(
    monkeypatch, tmp_path
):
    cached_voice = tmp_path / "voice.safetensors"
    model = FakeModel(multi_speaker=True)
    model.voice_repo = "kyutai/tts-voices"
    model.voice_suffix = ".model.safetensors"
    manager = TTSManager.__new__(TTSManager)
    manager.enabled = True
    manager._model = model
    manager._condition_attributes = None
    manager._prefix = None
    manager._cfg_coef = 3.0
    manager._current_voice = "default"

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, revision=None: str(cached_voice),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached voice must not call hf_hub_download")
        ),
    )

    assert manager.set_voice(ToneType.CALM)
    assert model.calls == [
        ("make_condition_attributes", [cached_voice], 3.0)
    ]


@pytest.mark.parametrize("reference", ["weights.safetensors", "hf://kyutai/model/weights.safetensors"])
def test_moshi_hub_loader_preserves_revision(monkeypatch, tmp_path, reference):
    from inspect import signature

    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    downloaded_file = tmp_path / "weights.safetensors"
    calls = []

    def fake_download(repo_id, filename, *, revision=None):
        signature(hf_hub_download).bind(repo_id, filename, revision=revision)
        calls.append((repo_id, filename, revision))
        return str(downloaded_file)

    monkeypatch.setattr(loaders, "hf_hub_download", fake_download)

    assert loaders.hf_get(reference, "kyutai/model", revision="test-revision") == downloaded_file
    assert calls == [("kyutai/model", "weights.safetensors", "test-revision")]


def test_moshi_checkpoint_loader_handles_hub_missing_config(monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError
    from moshi.models import loaders

    def missing_config(*args, **kwargs):
        raise EntryNotFoundError("config.json is missing")

    monkeypatch.setattr(loaders, "hf_hub_download", missing_config)
    with pytest.warns(UserWarning, match="contains no config.json"):
        checkpoint = loaders.CheckpointInfo.from_hf_repo(
            "kyutai/legacy-model",
            moshi_weights=Path("moshi.safetensors"),
            mimi_weights=Path("mimi.safetensors"),
            tokenizer=Path("tokenizer.model"),
        )

    assert checkpoint.moshi_weights == Path("moshi.safetensors")
    assert checkpoint.mimi_weights == Path("mimi.safetensors")
    assert checkpoint.tokenizer == Path("tokenizer.model")
    assert checkpoint.lm_config is None

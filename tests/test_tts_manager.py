from __future__ import annotations

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

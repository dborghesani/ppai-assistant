from __future__ import annotations

from agents.agents_dataclasses import VoiceType
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
    assert TTSManager.resolve_voice_name(VoiceType.CALM) == "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    assert TTSManager.resolve_voice_name("energetic") == "expresso/ex03-ex01_happy_001_channel1_334s.wav"

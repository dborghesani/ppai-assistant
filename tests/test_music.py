from data.assistant_dataclasses import DriverEmotionState, DriverPreferences
from dataclasses import fields
import asyncio
from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace
import json

from config import ConfigAssistant
from managers.music_manager import MusicManager, MusicSearchPlan, jamendo_url
from managers.knowledge_manager import KnowledgeManager


def test_emotion_persistence_requires_duration_and_resets_when_emotion_drops():
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = Mock()
    for field in fields(DriverEmotionState):
        if field.name == "neutral":
            assert not field.metadata["persistence_levels"]
            continue
        key = f"DriverEmotionState.{field.name}"
        fact = f"{field.name} level is medium."
        assert manager.add_persistence_knowledge(key, 0.5, fact, field.metadata, 10.0) == (fact, False)
        assert manager.add_persistence_knowledge(key, 0.5, fact, field.metadata, 39.9) == (fact, False)
        persisted, notify = manager.add_persistence_knowledge(key, 0.6, fact, field.metadata, 40.0)
        assert "persisted for a while" in persisted and notify
        assert not manager.add_persistence_knowledge(key, 0.5, fact, field.metadata, 45.0)[1]
        assert manager.add_persistence_knowledge(key, 0.1, fact, field.metadata, 46.0) == (fact, False)
        assert key not in manager.persistence_since
        assert manager.add_persistence_knowledge(key, 0.5, fact, field.metadata, 50.0) == (fact, False)


def test_emotion_persistence_is_published_without_new_measurements():
    async def check():
        manager = KnowledgeManager.__new__(KnowledgeManager)
        manager.logger = Mock()
        manager.database_manager = SimpleNamespace()
        manager.event_queue = asyncio.Queue()
        manager.knowledge_event_queue = asyncio.Queue()
        manager.opt = SimpleNamespace(use_laya=True)
        manager.context = {}
        manager._last_evaluated = {}
        manager._raw_values = {}
        manager._last_speed_status = None
        manager.persistence_since = {}
        manager.persistence_notified = set()
        manager._is_first_batch = False
        clock = iter((10.0, 39.9, 40.1, 41.0))
        manager.monotonic_clock = lambda: next(clock)
        manager.on_context_updated = Mock()
        manager.on_knowledge_changed = Mock()
        key = "DriverEmotionState.happy"
        await manager.process_pending({("DriverEmotionState", "happy"): 0.5})
        assert "persisted" not in manager.context[key]
        assert manager._due_persistence_updates(39.9) == {}
        await manager.process_pending({("DriverPreferences", "preferred_music"): "rock"})
        due = manager._due_persistence_updates(40.1)
        assert due == {("DriverEmotionState", "happy"): 0.5}
        await manager.process_pending(due)
        assert "persisted for a while" in manager.context[key]
        assert key in manager.persistence_notified
        await manager.process_pending({("DriverEmotionState", "happy"): 0.1})
        assert "persisted" not in manager.context[key]
        assert key not in manager.persistence_since
    asyncio.run(check())


def test_music_preferences_include_styles_and_emotion_associations():
    preference = DriverPreferences().preferred_music
    assert len(preference) <= 500
    assert all(name in preference for name in ["1970s rock", "Radiohead", "Zero 7", "classical"])
    assert "When happy" in preference
    assert "When sad" in preference
    assert "When tense" in preference


def test_music_plan_receives_all_emotions_preferences_and_history():
    async def check():
        history = [{"role": "user", "content": "I prefer gentle jazz today."}]
        create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "should_propose": False, "title": "", "queries": [], "tags": [], "permission_question": ""
        })))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        manager = MusicManager(ConfigAssistant(), client, Mock(), history_provider=lambda: history)
        emotions = {"sad": 0.7, "happy": 0.5, "neutral": 0.1}
        plan = await manager.plan("When reflective I enjoy jazz.", emotions, "", {"traffic": "Heavy"})
        assert not plan.should_propose
        payload = json.loads(create.call_args.kwargs["messages"][1]["content"])
        assert payload["emotions"] == emotions
        assert payload["history"] == history
        assert payload["context"] == {"traffic": "Heavy"}
        assert payload["preferences"] == "When reflective I enjoy jazz."
        assert payload["request_kind"] == "automatic_proposal"
        schema = create.call_args.kwargs["response_format"]["json_schema"]["schema"]
        assert set(schema["required"]) == {"title", "queries", "tags", "should_propose", "permission_question"}
    asyncio.run(check())


def test_live_playlist_and_fallback():
    async def check():
        manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        plan = MusicSearchPlan(title="Mellow", queries=["downtempo"], tags=["chillout"])
        track = {"id": "1", "name": "Track", "audio": "https://prod-1.storage.jamendo.com/track/1"}
        manager.get = AsyncMock(side_effect=[[{"id": "42", "name": "Live playlist"}], [{"tracks": [track]}]])
        result = await manager.search(plan)
        assert result["kind"] == "playlist"
        assert manager.get.call_args_list[0].kwargs["namesearch"] == "downtempo"
        manager.get = AsyncMock(side_effect=[[], [track]])
        result = await manager.search(plan)
        assert result["kind"] == "mix"
        assert manager.get.call_args.kwargs["fuzzytags"] == "chillout"
    asyncio.run(check())


def test_music_failures_do_not_start_playback_and_controls_emit_events(monkeypatch):
    async def check():
        manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        manager.find = AsyncMock(side_effect=RuntimeError("Service unavailable"))
        assert await manager.request("rock", autoplay=True) is None
        assert manager.current["status"] == "error"
        for command in ["pause", "resume", "next", "stop"]:
            manager.control(command)
            manager.on_update.assert_called_with({"status": "control", "command": command})
        monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)
        unconfigured = MusicManager(ConfigAssistant(), None, Mock())
        assert await unconfigured.request("rock", autoplay=True) is None
        assert unconfigured.current["status"] == "not_configured"
    asyncio.run(check())


def test_consent_is_single_use_and_stop_clears_it():
    manager = MusicManager(ConfigAssistant(), None, Mock())
    manager.pending = {"tracks": [], "title": "Mellow"}
    assert manager.consent(True)
    assert manager.current["autoplay"] is True
    assert not manager.consent(True)
    manager.pending = {"tracks": []}
    assert manager.consent(False)
    assert manager.current["status"] == "idle"
    manager.pending = {"tracks": []}
    manager.control("stop")
    assert manager.pending is None
    assert jamendo_url("javascript:alert(1)") == ""


def test_consent_cannot_bypass_a_newer_assistant_question():
    history = [{"role": "assistant", "content": "Would you like music?"}]
    manager = MusicManager(ConfigAssistant(), None, Mock(), history_provider=lambda: history)
    manager.pending = {"title": "Mellow", "tracks": []}
    manager.permission_prompt = "Would you like music?"
    history.append({"role": "assistant", "content": "Should I join your meeting?"})
    assert not manager.consent(True)
    assert manager.pending is None
    assert manager.current["status"] == "idle"


def test_emotional_proposal_waits_for_answer():
    async def check():
        manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        manager.plan = AsyncMock(return_value=MusicSearchPlan(
            title="Mellow", queries=["mellow rock"], tags=["rock"],
            permission_question="Would you enjoy some mellow rock right now?",
        ))
        manager.search = AsyncMock(return_value={"title": "Mellow", "tracks": []})
        ask = AsyncMock(return_value=True)
        state = {"DriverEmotionState": {"happy": 0.5}, "DriverPreferences": {"preferred_music": "rock"}}
        manager.observe(state, ask)
        await manager._task
        assert manager.current["status"] == "proposal"
        assert manager.current["autoplay"] is False
        assert manager.current["permission_question"] == "Would you enjoy some mellow rock right now?"
        assert manager.pending is not None
        ask.assert_awaited_once_with("Would you enjoy some mellow rock right now?")
        manager.observe(state, ask)
        assert manager.plan.await_count == 1
        assert manager.consent(True)
        assert manager.current["autoplay"] is True
        assert not hasattr(MusicManager, "select_emotion")
    asyncio.run(check())


def test_llm_can_decline_music_without_search_or_permission():
    async def check():
        manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        manager.plan = AsyncMock(return_value=MusicSearchPlan(should_propose=False))
        manager.search = AsyncMock()
        ask = AsyncMock()
        state = {"DriverEmotionState": {"happy": 0.9, "sad": 0.6},
                 "DriverPreferences": {"preferred_music": "jazz"}, "EnvironmentState": {"traffic": "Heavy"}}
        manager.observe(state, ask)
        await manager._task
        manager.plan.assert_awaited_once_with("jazz", {"happy": 0.9, "sad": 0.6}, "", state | {
            "DriverPhysicalState": {}, "DetectedObjects": {}
        })
        manager.search.assert_not_awaited()
        ask.assert_not_awaited()
        assert manager.current["status"] == "idle"
    asyncio.run(check())


def test_persistence_update_is_not_blocked_by_transient_silence_cooldown():
    async def check():
        manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        manager.plan = AsyncMock(side_effect=[MusicSearchPlan(should_propose=False), MusicSearchPlan(
            title="Mellow rock", queries=["mellow rock"], tags=["rock"],
            permission_question="Would you like some mellow rock?",
        )])
        manager.search = AsyncMock(return_value={"title": "Mellow rock", "tracks": []})
        ask = AsyncMock(return_value=True)
        state = {"DriverEmotionState": {"happy": "Happiness level is medium."},
             "DriverPreferences": {"preferred_music": "rock"}}
        manager.observe(state, ask)
        await manager._task
        assert manager.current["status"] == "idle"
        assert manager._last_attempt == float("-inf")
        state["DriverEmotionState"]["happy"] += " This condition has persisted for a while."
        manager.observe(state, ask)
        await manager._task
        assert manager.plan.await_count == 2
        assert manager.current["status"] == "proposal"
        assert not manager.current["autoplay"]
        ask.assert_awaited_once()
        manager.observe({**state, "DriverEmotionState": {"sad": "Sadness level is medium."}}, ask)
        assert manager.plan.await_count == 2
    asyncio.run(check())
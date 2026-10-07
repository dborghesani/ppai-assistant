from data.assistant_dataclasses import DriverEmotionState, DriverPreferences
from dataclasses import fields
import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock
import json

import pytest

from config import ConfigAssistant
from data.agents_dataclasses import (
    ActionType,
    InterventionType,
    SkillType,
    SuggestionType,
    ToneType,
    UrgencyType,
)
from data.events import EventName
from agents.automotive_agent import AutomotiveAgent
from agents.llm_backend import LLMBackend, NotificationDecision
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
            "title": "Gentle jazz", "queries": ["gentle jazz"], "tags": ["jazz"]
        })))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        manager = MusicManager(ConfigAssistant(), client, Mock())
        emotions = {"sad": 0.7, "happy": 0.5, "neutral": 0.1}
        plan = await manager.plan(
            "When reflective I enjoy jazz.", emotions, "", {"traffic": "Heavy"}, history=history
        )
        assert plan.title == "Gentle jazz"
        payload = json.loads(create.call_args.kwargs["messages"][1]["content"])
        assert payload["emotions"] == emotions
        assert payload["history"] == history
        assert payload["context"] == {"traffic": "Heavy"}
        assert payload["preferences"] == "When reflective I enjoy jazz."
        schema = create.call_args.kwargs["response_format"]["json_schema"]["schema"]
        assert set(schema["required"]) == {"title", "queries", "tags"}
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


def test_manager_plays_a_selection_and_stop_interrupts_it():
    manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
    selection = {"tracks": [{"id": "1"}], "title": "Mellow"}
    assert manager.play(selection)
    assert manager.current["autoplay"] is True
    manager.control("stop")
    assert manager.current["status"] == "idle"
    assert jamendo_url("javascript:alert(1)") == ""


def test_agent_rejects_music_confirmation_after_a_newer_assistant_question():
    history = [{"role": "assistant", "content": "Would you like music?"}]
    manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
    manager.play = Mock(return_value=True)
    agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
    agent.pending_music_selection = {"title": "Mellow", "tracks": [{"id": "1"}]}
    agent.pending_music_prompt = "Would you like music?"
    agent.pending_confirmation = SimpleNamespace(action=ActionType.EVALUATE_MUSIC_PROPOSAL)
    agent.conversation_history = history
    agent.action_manager = SimpleNamespace(music_manager=manager)
    agent.on_music_update = Mock()
    history.append({"role": "assistant", "content": "Should I join your meeting?"})
    assert not agent.confirm_pending_music_selection(True)
    assert agent.pending_music_selection is None
    assert manager.play.call_count == 0
    assert agent.pending_confirmation is None


def test_music_proposal_reply_acknowledges_selection_without_asking_again():
    instructions, reference = LLMBackend.music_proposal_reply_context(
        True, {"title": "Mellow piano"}
    )
    assert "without asking again" in instructions
    assert '"queued_playlist_title": "Mellow piano"' in reference


def test_music_proposal_reply_asks_for_selection_confirmation_after_first_yes():
    instructions, reference = LLMBackend.music_proposal_reply_context(
        True, {"title": "Mellow Nights"}, asking_selection_confirmation=True,
    )
    assert "second, separate confirmation" in instructions
    assert "I found the playlist <title>." in instructions
    assert "Do not praise the choice" in instructions
    assert '"queued_playlist_title": "Mellow Nights"' in reference


def test_music_proposal_requires_two_confirmations_to_start_selected_tracks():
    async def check():
        history = [{"role": "assistant", "content": "Would you like a mood-matching playlist?"}]
        manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        selection = {"kind": "playlist", "title": "Mellow Nights", "tracks": [{"id": "1"}]}
        manager.current = {"status": "ready", "autoplay": False, **selection}
        manager.request = AsyncMock()

        agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
        agent.pending_confirmation = SimpleNamespace(action=ActionType.PROPOSE_MUSIC)
        agent.interpret_confirmation = AsyncMock(side_effect=[True, True])
        agent.voice_response_task = None
        agent.cancel_voice_response = Mock()
        agent.knowledge_context = Mock(return_value=[])
        agent.music_preferences_provider = Mock(return_value="rock")
        agent.emotion_state = Mock(return_value={"sad": "Sadness level is medium."})
        agent.conversation_history = history
        agent.pending_music_selection = selection
        agent.pending_music_prompt = None
        agent.on_music_update = Mock()
        agent.action_manager = SimpleNamespace(music_manager=manager)
        agent.llm_backend = SimpleNamespace(
            music_proposal_reply_context=LLMBackend.music_proposal_reply_context,
        )
        async def generate_reply(_event, *, response_context):
            if response_context[1].find('"asking_selection_confirmation": true') >= 0:
                reply = 'I found "Mellow Nights". Does this choice sound good? Shall I start it?'
            else:
                reply = "I will start Mellow Nights."
            history.append({"role": "assistant", "content": reply})
            return reply
        agent.stream_user_response = AsyncMock(side_effect=generate_reply)

        await agent.handle_pending_confirmation("Yes, find me some music.")

        assert agent.pending_confirmation.action is ActionType.EVALUATE_MUSIC_PROPOSAL
        assert agent.pending_music_selection == selection
        assert manager.current["status"] == "ready"
        assert manager.current["autoplay"] is False
        agent.on_music_update.assert_called_with({
            "status": "proposal", "autoplay": False,
            **selection,
        })
        manager.request.assert_not_awaited()

        await agent.handle_pending_confirmation("Yes, start it.")

        assert manager.current["status"] == "ready"
        assert manager.current["autoplay"] is True
        assert manager.current["title"] == "Mellow Nights"
        assert agent.pending_music_selection is None
        assert agent.pending_music_prompt is None
        manager.request.assert_not_awaited()
        assert agent.pending_confirmation is None
        agent.stream_user_response.assert_awaited_once()
        assert '"queued_playlist_title": "Mellow Nights"' in (
            agent.stream_user_response.call_args.kwargs["response_context"][1]
        )

    asyncio.run(check())


def test_unclassified_music_confirmation_is_treated_as_decline_in_both_stages():
    async def check():
        manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        manager.request = AsyncMock()
        manager.play = Mock(return_value=True)
        agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
        agent.pending_confirmation = SimpleNamespace(action=ActionType.PROPOSE_MUSIC)
        agent.interpret_confirmation = AsyncMock(side_effect=[None, None])
        agent.voice_response_task = None
        agent.cancel_voice_response = Mock()
        agent.knowledge_context = Mock(return_value=[])
        agent.conversation_history = [{"role": "assistant", "content": "Would you like music?"}]
        agent.pending_music_selection = {"title": "Mellow Nights", "tracks": [{"id": "1"}]}
        agent.pending_music_prompt = None
        agent.on_music_update = Mock()
        agent.action_manager = SimpleNamespace(music_manager=manager)
        agent.llm_backend = SimpleNamespace(
            music_proposal_reply_context=LLMBackend.music_proposal_reply_context,
        )
        agent.stream_user_response = AsyncMock()

        await agent.handle_pending_confirmation("Maybe.")
        assert agent.pending_confirmation is None
        assert agent.pending_music_selection is None
        manager.request.assert_not_awaited()
        assert "removed from the" in agent.stream_user_response.call_args.kwargs[
            "response_context"
        ][0]
        assert '"confirmed": false' in agent.stream_user_response.call_args.kwargs["response_context"][1]

        agent.pending_confirmation = SimpleNamespace(action=ActionType.EVALUATE_MUSIC_PROPOSAL)
        agent.pending_music_selection = {"title": "Mellow Nights", "tracks": [{"id": "1"}]}
        agent.pending_music_prompt = "Should I start Mellow Nights?"
        agent.conversation_history.append({"role": "assistant", "content": agent.pending_music_prompt})
        await agent.handle_pending_confirmation("Maybe.")
        assert agent.pending_confirmation is None
        assert agent.pending_music_selection is None
        manager.play.assert_not_called()
        assert "removed from the" in agent.stream_user_response.call_args.kwargs[
            "response_context"
        ][0]
        assert '"confirmed": false' in agent.stream_user_response.call_args.kwargs["response_context"][1]

    asyncio.run(check())


def test_propose_music_decision_waits_for_confirmation_in_the_agent():
    async def check():
        agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
        agent.should_offer_music_proposal = Mock(return_value=True)
        agent.speak = AsyncMock()
        agent.pending_confirmation = None
        selection = {"kind": "playlist", "title": "Calm piano", "tracks": [{"id": "1"}]}
        manager = MusicManager(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        manager.request = AsyncMock(return_value=selection)
        agent.action_manager = SimpleNamespace(handle_decision=Mock(), music_manager=manager)
        agent.music_preferences_provider = Mock(return_value="calm classical")
        agent.emotion_state = Mock(return_value={"angry": "persisted"})
        agent.conversation_history = []
        agent.pending_music_selection = None
        agent.pending_music_prompt = None
        agent.on_music_update = Mock()
        agent.recent_notifications = []
        agent.duplicate_suppression_enabled = False

        decision = NotificationDecision(
            urgency=UrgencyType.LOW,
            tone=ToneType.CALM,
            intervention_type=InterventionType.ACT,
            skill=SkillType.WELLBEING,
            action=ActionType.PROPOSE_MUSIC,
            suggestion_type=SuggestionType.NONE,
            reason="Anger has persisted for a while; a calming playlist may help.",
            spoken_message="You've seemed tense for a while. Would you like some calming music?",
        )
        event = SimpleNamespace(
            event_name=EventName.KNOWLEDGE_UPDATED,
            event_value={"DriverEmotionState.angry": 0.9},
        )
        await agent.handle_notification_decision(event, decision)
        assert agent.pending_confirmation.action is ActionType.PROPOSE_MUSIC
        assert agent.pending_music_selection == selection
        manager.request.assert_awaited_once_with(
            "calm classical", emotion={"angry": "persisted"},
            request="Prepare a mood-matching selection for confirmation", autoplay=False,
            history=[],
        )
        agent.on_music_update.assert_called_once_with({
            "status": "proposal", "autoplay": False, **selection,
        })
        agent.action_manager.handle_decision.assert_not_called()
        agent.speak.assert_awaited_once_with(decision.spoken_message, tone=decision.tone)
    asyncio.run(check())


def test_ask_for_music_decision_requires_spoken_permission_question():
    payload = dict(
        urgency=UrgencyType.LOW.value, tone=ToneType.CALM.value,
        intervention_type=InterventionType.ACT.value, skill=SkillType.WELLBEING.value,
        action=ActionType.PROPOSE_MUSIC.value, suggestion_type=SuggestionType.NONE.value,
        reason="Anger persisted for a while.", spoken_message=None,
    )
    with pytest.raises(ValueError, match="propose_music requires"):
        NotificationDecision.model_validate(payload)
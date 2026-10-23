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
from agents.tasks.decision import NotificationDecision
from agents.tasks.mood_music import MusicSplit, music_proposal_reply_context
from agents.tasks.routing import TaskType
from agents.tasks.mood_music import MoodMusicTask, MusicSearchPlan, jamendo_url


def _playback_task(opt, client, on_update):
    """A bare MoodMusicTask exposing only the playback search pipeline."""
    task = cast(Any, MoodMusicTask.__new__(MoodMusicTask))
    task.opt = opt
    task.voice_llm = client
    task.notify_music_update = on_update
    task.backend = SimpleNamespace(logger=Mock())
    task.client_id = opt.jamendo_client_id or ""
    task._playback_generation = 0
    status = "idle" if task.client_id else "not_configured"
    task.current = {"status": status if opt.music_enabled else "disabled"}
    return task
from managers.knowledge_manager import KnowledgeManager


def test_emotion_persistence_requires_duration_and_resets_when_emotion_drops():
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = Mock()
    for field in fields(DriverEmotionState):
        if not field.metadata.get("knowledge", False):
            continue
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
        task = _playback_task(ConfigAssistant(), None, Mock())
        task.backend = SimpleNamespace(run=AsyncMock(return_value=MusicSearchPlan(
            title="Gentle jazz", queries=["gentle jazz"], tags=["jazz"]
        )), logger=Mock())
        task.plan_crew = Mock()
        emotions = {"sad": 0.7, "happy": 0.5, "neutral": 0.1}
        plan = await task.plan_search(
            "When reflective I enjoy jazz.", emotions, "", {"traffic": "Heavy"}, history=history
        )
        assert plan.title == "Gentle jazz"
        inputs = task.backend.run.call_args.args[1]
        assert inputs["emotions"] == json.dumps(emotions)
        assert json.loads(inputs["history"]) == history
        assert json.loads(inputs["context"]) == {"traffic": "Heavy"}
        assert inputs["preferences"] == "When reflective I enjoy jazz."
        assert task.backend.run.call_args.args[2] is MusicSearchPlan
    asyncio.run(check())


def test_live_playlist_and_fallback():
    async def check():
        task = _playback_task(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        plan = MusicSearchPlan(title="Mellow", queries=["downtempo"], tags=["chillout"])
        track = {"id": "1", "name": "Track", "audio": "https://prod-1.storage.jamendo.com/track/1"}
        task.get = AsyncMock(side_effect=[[{"id": "42", "name": "Live playlist"}], [{"tracks": [track]}]])
        result = await task.search(plan)
        assert result["kind"] == "playlist"
        assert task.get.call_args_list[0].kwargs["namesearch"] == "downtempo"
        assert task.get.call_args_list[1].kwargs["audioformat"] == "ogg"
        task.get = AsyncMock(side_effect=[[], [track]])
        result = await task.search(plan)
        assert result["kind"] == "mix"
        assert task.get.call_args.kwargs["fuzzytags"] == "chillout"
        assert task.get.call_args.kwargs["audioformat"] == "ogg"
    asyncio.run(check())


def test_music_failures_do_not_start_playback_and_controls_emit_events(monkeypatch):
    async def check():
        task = _playback_task(ConfigAssistant(jamendo_client_id="test"), None, Mock())
        task.find = AsyncMock(side_effect=RuntimeError("Service unavailable"))
        assert await task.request_music("rock", autoplay=True) is None
        assert task.current["status"] == "error"
        for command in ["pause", "resume", "next", "stop"]:
            task.control_playback(command)
            task.notify_music_update.assert_called_with({"status": "control", "command": command})
        monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)
        unconfigured = _playback_task(ConfigAssistant(), None, Mock())
        assert await unconfigured.request_music("rock", autoplay=True) is None
        assert unconfigured.current["status"] == "not_configured"
    asyncio.run(check())


def test_task_plays_a_selection_and_stop_interrupts_it():
    task = _playback_task(ConfigAssistant(jamendo_client_id="test"), None, Mock())
    selection = {"tracks": [{"id": "1"}], "title": "Mellow"}
    assert task.start_playback(selection)
    assert task.current["autoplay"] is True
    task.control_playback("stop")
    assert task.current["status"] == "idle"
    assert jamendo_url("javascript:alert(1)") == ""


def _music_task(*, selection=None, prompt=None, stage="selection", awaiting=True):
    task = cast(Any, MoodMusicTask.__new__(MoodMusicTask))
    task.awaiting_reply = awaiting
    task.stage = stage
    task.pending_selection = selection
    task.pending_prompt = prompt
    task.speak = None
    task.respond = None
    task.notify_music_update = Mock()
    task.preferences_provider = Mock(return_value="rock")
    task.emotion_provider = Mock(return_value={})
    task.opt = ConfigAssistant(jamendo_client_id="test")
    task.client_id = "test"
    task._playback_generation = 0
    task.current = {"status": "idle"}
    task.backend = SimpleNamespace(logger=Mock())
    return task


def _stub_interpret(task, confirmed):
    task.interpret_reply = AsyncMock(return_value=MusicSplit(confirmed=confirmed))


def test_agent_rejects_music_confirmation_after_a_newer_assistant_question():
    history = [{"role": "assistant", "content": "Should I join your meeting?"}]
    task = _music_task(selection={"title": "Mellow", "tracks": [{"id": "1"}]},
                       prompt="Would you like music?")
    task.start_playback = Mock(return_value=True)
    _stub_interpret(task, True)

    asyncio.run(task.handle_user_input("Yes, start it.", history))

    assert task.pending_selection is None
    assert task.stage is None
    assert task.start_playback.call_count == 0


def test_music_proposal_reply_acknowledges_selection_without_asking_again():
    instructions, reference = music_proposal_reply_context(
        True, {"title": "Mellow piano"}
    )
    assert "without asking again" in instructions
    assert '"queued_playlist_title": "Mellow piano"' in reference


def test_music_proposal_reply_asks_for_selection_confirmation_after_first_yes():
    instructions, reference = music_proposal_reply_context(
        True, {"title": "Mellow Nights"}, asking_selection_confirmation=True,
    )
    assert "second, separate confirmation" in instructions
    assert "I found the playlist <title>." in instructions
    assert "Do not praise the choice" in instructions
    assert '"queued_playlist_title": "Mellow Nights"' in reference


def test_music_proposal_with_no_selection_does_not_claim_a_playlist_was_found():
    async def check():
        task = _music_task(stage="offered", awaiting=True)
        task.request_music = AsyncMock(return_value=None)
        task.interpret_reply = AsyncMock(return_value=MusicSplit(confirmed=True))
        task.speak = AsyncMock()
        task.respond = AsyncMock()

        await task.handle_user_input("Yes, find me some music.", [])

        task.speak.assert_awaited_once_with("I couldn't find a suitable playlist right now.")
        task.respond.assert_not_awaited()
        assert task.stage is None
        assert task.awaiting_reply is False

    asyncio.run(check())


def test_unclassified_music_confirmation_is_treated_as_decline_in_both_stages():
    async def check():
        task = _music_task(stage="offered", awaiting=True)
        task.request_music = AsyncMock()
        task.start_playback = Mock(return_value=True)
        replies = []

        async def deliver(reply_context, short_history, user_input):
            replies.append(reply_context)
            return "reply"

        task.interpret_reply = AsyncMock(return_value=MusicSplit(confirmed=None))
        task._deliver_reply = deliver

        await task.handle_user_input("Maybe.", [])
        assert task.stage is None
        assert task.awaiting_reply is False
        task.request_music.assert_not_awaited()
        assert "removed from the" in replies[-1][0]
        assert '"confirmed": false' in replies[-1][1]

        task.stage = "selection"
        task.awaiting_reply = True
        task.pending_selection = {"title": "Mellow Nights", "tracks": [{"id": "1"}]}
        task.pending_prompt = "Should I start Mellow Nights?"
        await task.handle_user_input("Maybe.", [])
        assert task.stage is None
        assert task.pending_selection is None
        task.start_playback.assert_not_called()
        assert "removed from the" in replies[-1][0]
        assert '"confirmed": false' in replies[-1][1]

    asyncio.run(check())


def test_music_selection_can_be_changed_and_requires_a_new_confirmation():
    async def check():
        task = _music_task(selection={"title": "Mellow Nights", "tracks": [{"id": "1"}]},
                           prompt="Should I start Mellow Nights?")
        task.interpret_reply = AsyncMock(return_value=MusicSplit(confirmed=False, change_requested=True))
        alternate = {"title": "Upbeat Jazz", "tracks": [{"id": "2"}]}
        task.request_music = AsyncMock(return_value=alternate)
        task._deliver_reply = AsyncMock(return_value="Would you like me to start Upbeat Jazz?")

        await task.handle_user_input("Could you find something more upbeat, maybe jazz?", [
            {"role": "assistant", "content": "Should I start Mellow Nights?"},
        ])

        task.request_music.assert_awaited_once_with(
            "rock", emotion={}, request="Could you find something more upbeat, maybe jazz?",
            autoplay=False, history=[{"role": "assistant", "content": "Should I start Mellow Nights?"}],
        )
        assert task.pending_selection is alternate
        assert task.stage == "selection"
        assert task.awaiting_reply is True
        assert task.pending_prompt == "Would you like me to start Upbeat Jazz?"
        task.start_playback = Mock()
        task.interpret_reply = AsyncMock(return_value=MusicSplit(confirmed=True))
        await task.handle_user_input("Yes", [{"role": "assistant", "content": task.pending_prompt}])
        task.start_playback.assert_called_once_with(alternate)

    asyncio.run(check())


def test_accepted_music_selection_uses_conversation_reply_with_title_constraints():
    async def check():
        task = _music_task(selection={"title": "Jazz at Dawn", "tracks": [{"id": "2"}]},
                           prompt="Would you like me to start Jazz at Dawn?")
        task.interpret_reply = AsyncMock(return_value=MusicSplit(confirmed=True))
        task.start_playback = Mock(return_value=True)
        task.respond = AsyncMock()
        history = [{"role": "assistant", "content": task.pending_prompt}]

        await task.handle_user_input("yes please", history)

        task.start_playback.assert_called_once()
        task.respond.assert_awaited_once()
        user_input, reply_context = task.respond.await_args.args
        assert user_input == "yes please"
        instructions, reference = reply_context
        assert "using that exact title" in instructions
        assert "Do not mention or infer emotions" in instructions
        assert '"queued_playlist_title": "Jazz at Dawn"' in reference
        assert task.stage is None
        assert task.awaiting_reply is False

    asyncio.run(check())


def test_music_offer_can_be_refined_before_search():
    async def check():
        task = _music_task(stage="offered", selection=None)
        task.interpret_reply = AsyncMock(return_value=MusicSplit(confirmed=False, change_requested=True))
        task.request_music = AsyncMock(return_value={"title": "Jazz", "tracks": [{"id": "2"}]})
        task._deliver_reply = AsyncMock(return_value="Would you like me to start Jazz?")

        await task.handle_user_input("Yes, but make it jazz.", [])

        assert task.request_music.call_args.kwargs["request"] == "Yes, but make it jazz."
        assert task.stage == "selection"
        assert task.awaiting_reply is True

    asyncio.run(check())


def test_propose_music_decision_waits_for_confirmation_in_the_agent():
    async def check():
        agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
        agent.speak = AsyncMock()
        agent.action_manager = SimpleNamespace(handle_decision=Mock())
        agent.knowledge_facts_provider = Mock(return_value={})
        agent.tasks = {TaskType.MOOD_MUSIC: _music_task(stage=None, awaiting=False)}
        agent.music_task.compose_suggestion = AsyncMock(return_value="Shall I put on some music?")
        agent.music_task.notify_music_update = Mock()
        agent.music_task.request_music = AsyncMock()
        agent.music_task._last_selection = None
        agent.music_task._last_attempt = float("-inf")
        agent.music_task.emotion_provider = Mock(return_value={"angry": "persisted"})
        agent.music_task.preferences_provider = Mock(return_value="calm classical")
        agent.opt = SimpleNamespace(music_cooldown_seconds=300.0)

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
        assert agent.music_task.stage == "offered"
        assert agent.music_task.awaiting_reply is True
        assert agent.music_task.pending_selection is None
        agent.music_task.request_music.assert_not_awaited()
        agent.music_task.notify_music_update.assert_not_called()
        agent.action_manager.handle_decision.assert_not_called()
        agent.speak.assert_awaited_once_with("Shall I put on some music?", tone=decision.tone)
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
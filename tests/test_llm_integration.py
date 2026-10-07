"""Opt-in integration tests for the production reasoning prompts and local Ollama model."""

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

from data.agents_dataclasses import (
    ActionType,
    SkillType,
    SuggestionType,
    ToneType,
    UrgencyType,
)
from agents.automotive_agent import AutomotiveAgent
from config import ConfigAssistant
from data.events import CarEvent, EventName, IncomingMessage


pytestmark = pytest.mark.skipif(
    os.environ.get("PPAI_RUN_LLM_INTEGRATION") != "1",
    reason="Set PPAI_RUN_LLM_INTEGRATION=1 to run against the configured Ollama model.",
)


def make_live_agent() -> AutomotiveAgent:
    return AutomotiveAgent(
        ConfigAssistant(
            ollama_model=os.environ.get(
                "PPAI_OLLAMA_TEST_MODEL", "qwen3.5:4b"
            ),
            context_window_size=16384,
            tts_enabled=False,
            stt_enabled=False,
            mqtt_enabled=False,
            is_socket_enabled=False,
        )
    )


@pytest.mark.parametrize(
    ("level", "expected_action", "heating_on", "ac_on"),
    [
        ("too high", ActionType.ENABLE_AIR_CONDITIONING, False, False),
        ("high", ActionType.ENABLE_AIR_CONDITIONING, False, False),
        ("too low", ActionType.ENABLE_HEATING, False, False),
        ("low", ActionType.ENABLE_HEATING, False, False),
        ("optimal", ActionType.NONE, False, False),
        ("too high", ActionType.NONE, False, True),
        ("too low", ActionType.NONE, True, False),
    ],
    ids=["too-high", "high", "too-low", "low", "optimal", "ac-already-on", "heating-already-on"],
)
def test_relative_cabin_temperature_with_real_reasoner(
    level, expected_action, heating_on, ac_on
):
    async def run() -> None:
        agent = make_live_agent()
        actions: list[ActionType] = []
        spoken: list[str] = []
        agent.action_manager.on_action = lambda action, _parameters: actions.append(action)
        agent.on_response = spoken.append
        event = CarEvent(
            SkillType.WELLBEING,
            EventName.KNOWLEDGE_UPDATED,
            {"VehicleState.internal_temperature": level},
            [
                f"Changed just now: Cabin temperature is {level}.",
                "The driver's preferred cabin temperature is 22 C.",
                f"The air conditioning is {'on' if ac_on else 'off'}.",
                f"Cabin heating is {'on' if heating_on else 'off'}.",
                "Attention level is very high.",
                "No fatigue detected.",
                "The vehicle is stationary.",
            ],
        )
        try:
            await agent.process_event_llm(event)
            if expected_action is ActionType.NONE:
                assert actions == []
                assert spoken == []
            else:
                assert actions == [expected_action]
                assert spoken
        finally:
            await agent.voice_llm.close()

    asyncio.run(run())


def test_late_meeting_change_asks_permission_without_user_request():
    async def run():
        agent = make_live_agent()
        spoken = []
        agent.on_response = spoken.append
        event = CarEvent(SkillType.CONVERSATION, EventName.KNOWLEDGE_UPDATED,
                         {"DriverAgenda.late_for_meeting": True},
                         ["Changed just now: The driver is running late for an upcoming meeting."])
        try:
            await agent.process_event_llm(event)
            assert agent.action_manager.awaiting_confirmation
            assert agent.action_manager._meeting_task is None
            assert spoken
        finally:
            agent.action_manager.close()
            await agent.voice_llm.close()

    asyncio.run(run())


def test_high_fatigue_with_real_reasoner_suggests_break_not_music():
    async def run() -> None:
        agent = make_live_agent()
        decisions = []
        agent.handle_notification_decision = AsyncMock(
            side_effect=lambda _event, decision: decisions.append(decision)
        )
        event = CarEvent(
            SkillType.WELLBEING,
            EventName.KNOWLEDGE_UPDATED,
            {"DriverPhysicalState.fatigue_level": 0.9},
            [
                "Changed just now: Fatigue level is very high.",
                "The vehicle is moving.",
                "No emotion change is reported.",
            ],
        )
        try:
            await agent.process_event_llm(event)
            assert len(decisions) == 1
            assert decisions[0].action is not ActionType.PROPOSE_MUSIC
            assert decisions[0].suggestion_type is SuggestionType.TAKE_BREAK
        finally:
            await agent.voice_llm.close()

    asyncio.run(run())


def test_music_manager_selects_from_context_without_owning_permission_decisions():
    async def run():
        agent = make_live_agent()
        manager = agent.action_manager.music_manager
        emotions = {"happy": "Happiness level is medium.", "sad": "No sadness detected."}
        context = {"DriverPhysicalState": {"attention_level": 1.0, "fatigue_level": 0.0, "activity": "Idle"},
                   "EnvironmentState": {"traffic": "Light"},
                   "DriverEmotionState": emotions}
        preferences = "When happy I like warm, upbeat 1970s-style rock."
        try:
            transient = await manager.plan(preferences, emotions, "", context)
            assert transient.queries and transient.tags
            assert not hasattr(transient, "should_propose")
            context["DriverEmotionState"]["happy"] += " This condition has persisted for a while."
            sustained = await manager.plan(preferences, emotions, "", context)
            assert sustained.queries and sustained.tags
            context["DriverEmotionState"]["happy"] = "Happiness level is medium."
            explicit = await manager.plan(preferences, emotions, "Play some 1970s-style rock.", context)
            assert explicit.queries and explicit.tags
        finally:
            agent.action_manager.close()
            await agent.voice_llm.close()

    asyncio.run(run())


def test_message_privacy_sequence_with_same_backend():
    async def run():
        agent = make_live_agent()
        backend = agent.llm_backend
        text = (
            "Hey there! Did you hear about Giulia and Marco's disastrous cooking attempt? "
            "Apparently, Chiara accidentally stole Marco's secret sauce recipe while he was "
            "distracted by his car app, leading to a real culinary disaster at my place. "
            "Want to know how Sara got involved in the chaos?"
        )
        try:
            for privacy_on in (True, False, True):
                facts = [f"Privacy mode is {'on' if privacy_on else 'off'}.",
                         "One person is detected inside the vehicle.",
                         "Attention level is very high.", "No fatigue detected.", "Current traffic is light."]
                decision = await backend.evaluate_event({
                    "event_name": EventName.INCOMING_MESSAGE_RECEIVED,
                    "driver_name": agent.driver_name,
                    "event_details": str({"sender": "Luca", "text": text}),
                    "changed_facts": f"New incoming message from Luca: {text}",
                    "context": "\n".join(facts), "user_input": "",
                    "skill_instructions": agent.skill_manager.get_skill(SkillType.CONVERSATION),
                    "vehicle_action_guidance": agent.skill_manager.vehicle_action_guidance,
                    **backend.decision_options,
                    "silent_decision_guidance": backend.silent_decision_guidance,
                })
                expected = (
                    ActionType.ASK_PERMISSION_TO_TALK
                    if privacy_on
                    else ActionType.READ_PENDING_MESSAGES
                )
                assert decision.action is expected, decision.reason
                if privacy_on:
                    assert decision.spoken_message is None
        finally:
            agent.action_manager.close()
            await agent.voice_llm.close()

    asyncio.run(run())


def test_multiple_occupants_make_message_delivery_privacy_sensitive():
    async def run():
        agent = make_live_agent()
        text = "Please call me when you can."
        try:
            decision = await agent.llm_backend.evaluate_event({
                "event_name": EventName.INCOMING_MESSAGE_RECEIVED,
                "driver_name": agent.driver_name,
                "event_details": str({"sender": "Luca", "text": text}),
                "changed_facts": f"New incoming message from Luca: {text}",
                "context": "\n".join([
                    "Privacy mode is off.",
                    "Multiple people are detected inside the vehicle.",
                    "Attention level is very high.",
                    "No fatigue detected.",
                    "Traffic is light.",
                ]),
                "user_input": "",
                "skill_instructions": agent.skill_manager.get_skill(SkillType.CONVERSATION),
                "vehicle_action_guidance": agent.skill_manager.vehicle_action_guidance,
                **agent.llm_backend.decision_options,
                "silent_decision_guidance": agent.llm_backend.silent_decision_guidance,
            })
            assert decision.action is ActionType.ASK_PERMISSION_TO_TALK
            assert decision.spoken_message is None
        finally:
            agent.action_manager.close()
            await agent.voice_llm.close()

    asyncio.run(run())


def test_low_fatigue_with_pending_messages_reads_pending_messages():
    async def run() -> None:
        agent = make_live_agent()
        manager = agent.action_manager.message_manager
        manager.pending_messages.append(
            IncomingMessage(sender="Luca", text="Please call me when you can.")
        )
        manager.summarize_and_speak = AsyncMock()
        event = CarEvent(
            SkillType.WELLBEING,
            EventName.KNOWLEDGE_UPDATED,
            {"DriverPhysicalState.fatigue_level": 0.1},
            [
                "Changed just now: Fatigue level is low.",
                "Privacy mode is off.",
            ],
        )

        try:
            await agent.process_event_llm(event)
            assert manager.read_task is not None
            assert manager.pending_count == 1
            await manager.read_task
            assert manager.pending_count == 0
            manager.summarize_and_speak.assert_awaited_once()
        finally:
            agent.action_manager.close()
            await agent.voice_llm.close()

    asyncio.run(run())


def test_fatigue_postpones_incoming_message_then_reads_it_after_recovery():
    async def run() -> None:
        agent = make_live_agent()
        manager = agent.action_manager.message_manager
        manager.summarize_and_speak = AsyncMock()
        message = IncomingMessage(sender="Mara", text="Please call me when you can.")
        manager.pending_messages.append(message)
        incoming_event = CarEvent(
            SkillType.CONVERSATION,
            EventName.INCOMING_MESSAGE_RECEIVED,
            {"sender": message.sender, "text": message.text},
            [
                f"Changed just now: New incoming message from {message.sender}: {message.text}",
                "Privacy mode is off.",
                "Fatigue level is high.",
                "Attention level is high.",
                "Traffic is light.",
            ],
        )
        recovery_event = CarEvent(
            SkillType.WELLBEING,
            EventName.KNOWLEDGE_UPDATED,
            {"DriverPhysicalState.fatigue_level": 0.1},
            [
                "Changed just now: Fatigue level is low.",
                "Privacy mode is off.",
                "Fatigue level is low.",
            ],
        )

        try:
            await agent.process_event_llm(incoming_event)
            assert manager.pending_count == 1

            await agent.process_event_llm(recovery_event)
            assert manager.read_task is not None
            assert manager.pending_count == 1
            await manager.read_task
            assert manager.pending_count == 0
            manager.summarize_and_speak.assert_awaited_once()
        finally:
            agent.action_manager.close()
            await agent.voice_llm.close()

    asyncio.run(run())


def test_low_attention_while_moving_selects_adas_profile_with_real_reasoner():
    async def run() -> None:
        agent = make_live_agent()
        actions: list[ActionType] = []
        agent.action_manager.on_action = lambda action, _parameters: actions.append(
            action
        )
        event = CarEvent(
            SkillType.WELLBEING,
            EventName.KNOWLEDGE_UPDATED,
            {"DriverPhysicalState.attention_level": 0.2},
            [
                "Changed just now: Attention level is low. This condition has persisted for a while.",
                "The vehicle is moving.",
            ],
        )

        try:
            await agent.process_event_llm(event)
            assert ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE in actions
        finally:
            await agent.voice_llm.close()

    asyncio.run(run())


@pytest.mark.parametrize("privacy_on", [False, True], ids=["privacy-off-asks", "privacy-on-asks"])
def test_urgent_message_delivery_depends_on_privacy(privacy_on):
    async def run() -> None:
        agent = make_live_agent()
        actions: list[ActionType] = []
        classifications: list[tuple[ToneType, UrgencyType]] = []
        spoken: list[str] = []
        text = (
            "My car has broken down on the highway shoulder with traffic passing close. "
            "Please call me ASAP."
        )
        manager = agent.action_manager.message_manager
        message = IncomingMessage(sender="Mara", text=text)
        manager.pending_messages.append(message)
        manager.summarize_and_speak = AsyncMock()
        agent.action_manager.on_action = lambda action, _parameters: actions.append(
            action
        )
        agent.on_incoming_message_classified = lambda tone, urgency: classifications.append(
            (tone, urgency)
        )
        agent.on_response = spoken.append
        event = CarEvent(
            SkillType.CONVERSATION,
            EventName.INCOMING_MESSAGE_RECEIVED,
            {"sender": "Mara", "text": text},
            [
                f"Changed just now: New incoming message from Mara: {text}",
                f"Privacy mode is {'on' if privacy_on else 'off'}.",
                "One person is detected inside the vehicle.",
                "Driver attention is very high.",
                "Fatigue is low.",
                "Traffic is light.",
            ],
        )

        try:
            await agent.process_event_llm(event)
            assert classifications
            tone, urgency = classifications[-1]
            assert tone is ToneType.SERIOUS
            assert urgency in {UrgencyType.HIGH, UrgencyType.CRITICAL}
            if privacy_on:
                assert agent.assistant_status.value == "ask_permission_to_talk"
                assert spoken == []
                assert manager.pending_count == 1
            else:
                assert agent.assistant_status.value == "idle"
                assert manager.read_task is not None
                assert manager.pending_count == 1
                await manager.read_task
                assert manager.pending_count == 0
                manager.summarize_and_speak.assert_awaited_once()
        finally:
            await agent.voice_llm.close()

    asyncio.run(run())


def test_fatigued_pending_message_reminder_is_postponed_by_real_reasoner():
    async def run() -> None:
        agent = make_live_agent()
        manager = agent.action_manager.message_manager
        manager.pending_messages.append(
            IncomingMessage(sender="Mara", text="Please call me later.")
        )
        event = CarEvent(
            SkillType.CONVERSATION,
            EventName.PENDING_MESSAGES_REMINDER,
            {"pending_count": 1},
            [
                "There is 1 incoming message pending delivery.",
                "Privacy mode is off.",
                "Fatigue level is high.",
            ],
        )

        try:
            await agent.process_event_llm(event)
            assert manager.pending_count == 1
        finally:
            await agent.voice_llm.close()

    asyncio.run(run())
"""Opt-in integration tests for the production reasoning prompts and local Ollama model."""

import asyncio
import os

import pytest

from data.agents_dataclasses import ActionType, SkillType, ToneType, UrgencyType
from agents.automotive_agent import AutomotiveAgent
from config import ConfigAssistant
from data.events import CarEvent, EventName


pytestmark = pytest.mark.skipif(
    os.environ.get("PPAI_RUN_LLM_INTEGRATION") != "1",
    reason="Set PPAI_RUN_LLM_INTEGRATION=1 to run against the configured Ollama model.",
)


def make_live_agent() -> AutomotiveAgent:
    return AutomotiveAgent(
        ConfigAssistant(
            ollama_model=os.environ.get(
                "PPAI_OLLAMA_TEST_MODEL", "ollama/qwen3.5:4b-ctx16384"
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


def test_music_model_requires_emotional_persistence_unless_explicitly_requested():
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
            assert not transient.should_propose
            context["DriverEmotionState"]["happy"] += " This condition has persisted for a while."
            sustained = await manager.plan(preferences, emotions, "", context)
            assert sustained.should_propose
            assert sustained.permission_question and sustained.queries
            context["DriverEmotionState"]["happy"] = "Happiness level is medium."
            explicit = await manager.plan(preferences, emotions, "Play some 1970s-style rock.", context)
            assert explicit.should_propose and explicit.queries
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
                expected = ActionType.ASK_PERMISSION_TO_TALK if privacy_on else ActionType.ANNOUNCE_INCOMING_MESSAGE
                assert decision.action is expected, decision.reason
                if privacy_on:
                    assert decision.spoken_message is None
                else:
                    assert decision.spoken_message
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


@pytest.mark.parametrize("privacy_on", [False, True], ids=["privacy-off-announces", "privacy-on-asks"])
def test_urgent_message_delivery_depends_on_privacy(privacy_on):
    async def run() -> None:
        agent = make_live_agent()
        actions: list[ActionType] = []
        classifications: list[tuple[ToneType, UrgencyType]] = []
        spoken: list[str] = []
        agent.action_manager.on_action = lambda action, _parameters: actions.append(
            action
        )
        agent.on_incoming_message_classified = lambda tone, urgency: classifications.append(
            (tone, urgency)
        )
        agent.on_response = spoken.append
        text = (
            "My car has broken down on the highway shoulder with traffic passing close. "
            "Please call me ASAP."
        )
        event = CarEvent(
            SkillType.CONVERSATION,
            EventName.INCOMING_MESSAGE_RECEIVED,
            {"sender": "Mara", "text": text},
            [
                f"Changed just now: New incoming message from Mara: {text}",
                f"Privacy mode is {'on' if privacy_on else 'off'}.",
                "Driver attention is very high.",
                "Fatigue is low.",
                "Traffic is light.",
            ],
        )

        try:
            await agent.process_event_llm(event)
            assert agent.assistant_status.value == ("ask_permission_to_talk" if privacy_on else "idle")
            assert classifications
            tone, urgency = classifications[-1]
            assert tone is ToneType.SERIOUS
            assert urgency in {UrgencyType.HIGH, UrgencyType.CRITICAL}
            if privacy_on:
                assert spoken == []
            else:
                assert spoken
                assert "Mara" in " ".join(spoken)
                assert " says" not in " ".join(spoken).lower()
        finally:
            await agent.voice_llm.close()

    asyncio.run(run())
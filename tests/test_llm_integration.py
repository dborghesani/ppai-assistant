"""Opt-in integration tests for the production reasoning prompts and local Ollama model."""

import asyncio
import os

import pytest

from agents.agents_dataclasses import ActionType, SkillType, ToneType, UrgencyType
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
            await agent.llm_agent.process_event(event)
            if expected_action is ActionType.NONE:
                assert actions == []
                assert spoken == []
            else:
                assert actions == [expected_action]
                assert spoken
        finally:
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
            await agent.llm_agent.process_event(event)
            assert ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE in actions
        finally:
            await agent.voice_llm.close()

    asyncio.run(run())


def test_private_urgent_message_is_classified_and_asks_without_speaking():
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
                "Privacy mode is on.",
                "Driver attention is very high.",
                "Fatigue is low.",
                "Traffic is light.",
            ],
        )

        try:
            await agent.llm_agent.process_event(event)
            assert ActionType.ASK_PERMISSION_TO_TALK in actions
            assert classifications
            tone, urgency = classifications[-1]
            assert tone is ToneType.SERIOUS
            assert urgency in {UrgencyType.HIGH, UrgencyType.CRITICAL}
            assert spoken == []
        finally:
            await agent.voice_llm.close()

    asyncio.run(run())
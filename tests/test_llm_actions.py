from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
import structlog

from agents.agents_dataclasses import (
    ActionType,
    InterventionType,
    SkillType,
    SuggestionType,
    ToneType,
    UrgencyType,
)
from agents.llm_agent import LLMAgent, NotificationDecision
from data.events import CarEvent


class FakeCrew:
    def __init__(self, decision: NotificationDecision) -> None:
        self.decision = decision
        self.inputs = None

    async def kickoff_async(self, inputs):
        self.inputs = inputs
        return SimpleNamespace(raw="structured decision", pydantic=self.decision)


def make_llm_agent(decision: NotificationDecision):
    agent = SimpleNamespace(
        maybe_handle_meeting_confirmation=AsyncMock(return_value=False),
        cancel_voice_response=Mock(),
        action_manager=SimpleNamespace(handle_decision=Mock()),
        speak=AsyncMock(),
        stream_user_response=AsyncMock(),
        _voice_response_task=None,
        skill_manager=SimpleNamespace(
            get_skill=Mock(return_value="conversation rules"),
            vehicle_action_guidance="vehicle action reference",
        ),
    )
    llm_agent = LLMAgent.__new__(LLMAgent)
    llm_agent.logger = structlog.get_logger()
    llm_agent.agent = cast(Any, agent)
    llm_agent.crew = cast(Any, FakeCrew(decision))
    llm_agent.direct_action_crew = cast(Any, FakeCrew(decision))
    llm_agent.llm = cast(Any, SimpleNamespace(temperature=0.7))
    llm_agent.decision_options = {
        "skill_options": "driving, wellbeing",
        "tone_options": "calm, discreet",
        "action_options": "ask_attend_meeting: attend the meeting",
    }
    llm_agent.direct_action_options = "increase_temperature: increase temperature"
    llm_agent.direct_vehicle_actions = (ActionType.INCREASE_TEMPERATURE, ActionType.OPEN_WINDOWS)
    llm_agent.silent_decision_guidance = "silent guidance"
    llm_agent.recent_notifications = []
    return llm_agent, agent


def make_decision(action: ActionType) -> NotificationDecision:
    is_action = action is not ActionType.NONE
    return NotificationDecision(
        urgency=UrgencyType.LOW if is_action else UrgencyType.NONE,
        tone=ToneType.CALM,
        intervention_type=InterventionType.ACT if is_action else InterventionType.NONE,
        skill=SkillType.DRIVING if is_action else SkillType.NONE,
        action=action,
        suggestion_type=SuggestionType.NONE,
        reason="The driver explicitly requested this action." if is_action else "No supported action requested.",
        spoken_message="I received your request." if is_action else None,
    )


def test_direct_vehicle_command_dispatches_structured_action():
    command = "Could you open the windows?"
    llm_agent, agent = make_llm_agent(make_decision(ActionType.OPEN_WINDOWS))
    event = CarEvent(SkillType.CONVERSATION, "user_input", command, [], command)

    import asyncio

    asyncio.run(llm_agent.process_event(event))

    agent.action_manager.handle_decision.assert_called_once_with(
        ActionType.OPEN_WINDOWS, {}
    )
    agent.speak.assert_awaited_once_with("I received your request.", tone=ToneType.CALM)
    assert llm_agent.direct_action_crew.inputs["user_input"] == command
    assert llm_agent.direct_action_crew.inputs["vehicle_action_guidance"] == "vehicle action reference"
    assert "context" not in llm_agent.direct_action_crew.inputs


def test_direct_vehicle_action_is_dispatched_with_unrelated_meeting_context():
    command = "increase the temperature, please"
    llm_agent, agent = make_llm_agent(make_decision(ActionType.INCREASE_TEMPERATURE))
    event = CarEvent(
        SkillType.CONVERSATION,
        "user_input",
        command,
        ["The driver is running late for an upcoming meeting."],
        command,
    )

    import asyncio

    asyncio.run(llm_agent.process_event(event))

    agent.action_manager.handle_decision.assert_called_once_with(
        ActionType.INCREASE_TEMPERATURE, {}
    )
    agent.speak.assert_awaited_once_with("I received your request.", tone=ToneType.CALM)
    assert "context" not in llm_agent.direct_action_crew.inputs


def test_non_command_user_input_falls_back_to_conversation():
    llm_agent, agent = make_llm_agent(make_decision(ActionType.NONE))
    event = CarEvent(
        SkillType.CONVERSATION,
        "user_input",
        "What is the weather like?",
        [],
        "What is the weather like?",
    )

    import asyncio

    async def run_event():
        await llm_agent.process_event(event)
        await asyncio.sleep(0)

    asyncio.run(run_event())

    agent.action_manager.handle_decision.assert_not_called()
    agent.stream_user_response.assert_awaited_once_with(event)


def test_direct_command_fallback_logs_decision_and_reason():
    command = "Can you increase the temperature?"
    llm_agent, agent = make_llm_agent(make_decision(ActionType.NONE))
    llm_agent.logger = Mock()
    event = CarEvent(SkillType.CONVERSATION, "user_input", command, [], command)

    import asyncio

    async def run_event():
        await llm_agent.process_event(event)
        await asyncio.sleep(0)

    asyncio.run(run_event())

    agent.action_manager.handle_decision.assert_not_called()
    agent.stream_user_response.assert_awaited_once_with(event)
    llm_agent.logger.warning.assert_called_once_with(
        "Direct user request was not classified as a vehicle action; falling back to conversation",
        user_input=command,
        fallback_reason="model selected action=none",
        intervention_type="none",
        action="none",
        skill="none",
        suggestion_type="none",
        decision_reason="No supported action requested.",
        spoken_message=None,
    )


def test_knowledge_update_uses_readable_context_without_raw_event_json():
    llm_agent, _ = make_llm_agent(make_decision(ActionType.NONE))
    event = CarEvent(
        SkillType.DRIVING,
        "knowledge_updated",
        {"DetectedObjects.dangerous_objects_around": True},
        [
            "Changed just now: A dangerous object is detected around the vehicle.",
            "The vehicle doors are unlocked.",
        ],
    )

    import asyncio

    asyncio.run(llm_agent.process_event(event))

    assert llm_agent.crew.inputs["context"] == (
        "- The vehicle doors are unlocked."
    )
    assert llm_agent.crew.inputs["changed_facts"] == (
        "- A dangerous object is detected around the vehicle."
    )
    assert "changed_knowledge" not in llm_agent.crew.inputs
    assert "meeting_decision_guidance" not in llm_agent.crew.inputs
    assert "ask_attend_meeting" in llm_agent.crew.inputs["action_options"]


def test_meeting_action_is_available_without_event_specific_gate():
    llm_agent, _ = make_llm_agent(make_decision(ActionType.NONE))
    event = CarEvent(
        SkillType.CONVERSATION,
        "knowledge_updated",
        {"DriverAgenda.late_for_meeting": True},
        ["The driver is running late for an upcoming meeting."],
    )

    import asyncio

    asyncio.run(llm_agent.process_event(event))

    assert "meeting_decision_guidance" not in llm_agent.crew.inputs
    assert "ask_attend_meeting" in llm_agent.crew.inputs["action_options"]

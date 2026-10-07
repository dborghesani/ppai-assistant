from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
import structlog

from data.agents_dataclasses import (
    ActionType,
    InterventionType,
    SkillType,
    SuggestionType,
    ToneType,
    UrgencyType,
)
from agents.llm_backend import LLMBackend, NotificationDecision, ResponseSource
from agents.automotive_agent import AutomotiveAgent
from managers.message_manager import MessageManager
from data.events import CarEvent, EventName


class FakeCrew:
    def __init__(self, decision: NotificationDecision) -> None:
        self.decision = decision
        self.inputs = None

    async def kickoff_async(self, inputs):
        self.inputs = inputs
        return SimpleNamespace(raw="structured decision", pydantic=self.decision)


def make_llm_agent(decision: NotificationDecision):
    agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
    agent.__dict__.update(vars(SimpleNamespace(
        maybe_handle_meeting_confirmation=AsyncMock(return_value=False),
        log_llm_usage=Mock(),
        cancel_voice_response=Mock(),
        notify_incoming_message_classification=Mock(),
        remember_pending_message_classification=Mock(),
        most_urgent_pending_message_classification=Mock(return_value=None),
        action_manager=SimpleNamespace(
            handle_decision=Mock(),
            registered_actions=frozenset(),
            registered_action_options="",
            registered_action_guidance="",
            message_manager=SimpleNamespace(pending_count=0, simulation_stopped=Mock(return_value=False),
                                            simulator=SimpleNamespace(active=True)),
        ),
        pending_friend_message_count=0,
        speak=AsyncMock(),
        stream_user_response=AsyncMock(),
        voice_response_task=None,
        skill_manager=SimpleNamespace(
            get_skill=Mock(return_value="conversation rules"),
            vehicle_action_guidance="vehicle action reference",
        ),
    )))
    agent.logger = structlog.get_logger()
    agent.conversation_history = []
    agent.recent_notifications = []
    agent.duplicate_suppression_enabled = False
    agent.action_manager.message_manager.simulation_stopped = lambda event: MessageManager.simulation_stopped(agent.action_manager.message_manager, event)
    async def prepare_response(event, source):
        await agent.stream_user_response(event)

    agent.prepare_user_response = AsyncMock(side_effect=prepare_response)
    llm_agent = LLMBackend.__new__(LLMBackend)
    llm_agent.logger = structlog.get_logger()
    llm_agent.on_usage = agent.log_llm_usage
    llm_agent.crew = cast(Any, FakeCrew(decision))
    llm_agent.message_delivery_crew = llm_agent.crew
    llm_agent.direct_action_crew = cast(Any, FakeCrew(decision))
    llm_agent.llm = cast(Any, SimpleNamespace(temperature=0.7))
    llm_agent.decision_options = {
        "skill_options": "driving, wellbeing",
        "tone_options": "calm, discreet",
        "action_options": "ask_attend_meeting: attend the meeting",
    }
    llm_agent.direct_action_options = "enable_heating: enable cabin heating"
    llm_agent.direct_vehicle_actions = (ActionType.ENABLE_HEATING, ActionType.OPEN_WINDOWS)
    llm_agent.silent_decision_guidance = "silent guidance"
    agent.llm_backend = llm_agent
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


@pytest.mark.parametrize("source", list(ResponseSource))
def test_llm_selected_source_is_forwarded_to_response_preparation(source):
    decision = make_decision(ActionType.NONE)
    decision.response_source = source
    backend, agent = make_llm_agent(decision)
    agent.conversation_history = [{"role": "assistant", "content": "We were discussing cruise control."}]
    event = CarEvent(SkillType.CONVERSATION, "user_input", "How does it work?", [], "How does it work?")

    async def check():
        await agent.process_event_llm(event)
        await agent.voice_response_task

    import asyncio
    asyncio.run(check())
    agent.prepare_user_response.assert_awaited_once_with(event, source)
    assert "cruise control" in backend.direct_action_crew.inputs["conversation_history"]
    agent.action_manager.handle_decision.assert_not_called()


def test_direct_vehicle_command_dispatches_structured_action():
    command = "Could you open the windows?"
    llm_agent, agent = make_llm_agent(make_decision(ActionType.OPEN_WINDOWS))
    event = CarEvent(SkillType.CONVERSATION, "user_input", command, [], command)

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    agent.action_manager.handle_decision.assert_called_once_with(
        ActionType.OPEN_WINDOWS, {}
    )
    agent.speak.assert_awaited_once_with("I received your request.", tone=ToneType.CALM)
    assert llm_agent.direct_action_crew.inputs["user_input"] == command
    assert llm_agent.direct_action_crew.inputs["vehicle_action_guidance"] == "vehicle action reference"
    assert "context" not in llm_agent.direct_action_crew.inputs


def test_direct_vehicle_action_is_dispatched_with_unrelated_meeting_context():
    command = "increase the temperature, please"
    llm_agent, agent = make_llm_agent(make_decision(ActionType.ENABLE_HEATING))
    event = CarEvent(
        SkillType.CONVERSATION,
        "user_input",
        command,
        ["The driver is running late for an upcoming meeting."],
        command,
    )

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    agent.action_manager.handle_decision.assert_called_once_with(
        ActionType.ENABLE_HEATING, {}
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
        await agent.process_event_llm(event)
        await asyncio.sleep(0)

    asyncio.run(run_event())
    agent.action_manager.handle_decision.assert_not_called()
    agent.stream_user_response.assert_awaited_once_with(event)


@pytest.mark.parametrize(
    "action",
    [ActionType.ANNOUNCE_INCOMING_MESSAGE, ActionType.ASK_PERMISSION_TO_TALK],
)
def test_friend_message_event_dispatches_reasoner_selected_action(action):
    decision = NotificationDecision(
        urgency=UrgencyType.LOW,
        tone=ToneType.DISCREET,
        intervention_type=InterventionType.ACT,
        skill=SkillType.CONVERSATION,
        action=action,
        suggestion_type=SuggestionType.NONE,
        reason="The message event requires a privacy-aware response.",
        spoken_message=(
            "Luca says that he heard a harmless rumor."
            if action is ActionType.ANNOUNCE_INCOMING_MESSAGE
            else None
        ),
    )
    llm_agent, agent = make_llm_agent(decision)
    event = CarEvent(
        SkillType.CONVERSATION,
        "incoming_message_received",
        {"sender": "Luca", "text": "A harmless rumor"},
        ["Changed just now: New incoming message from Luca: A harmless rumor", "Privacy mode is on."],
    )

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    if action is ActionType.ANNOUNCE_INCOMING_MESSAGE:
        agent.speak.assert_awaited_once_with(
            "Luca says that he heard a harmless rumor.", tone=ToneType.DISCREET
        )
    else:
        agent.speak.assert_not_awaited()
    agent.action_manager.handle_decision.assert_called_once_with(action, {})
    agent.notify_incoming_message_classification.assert_called_once_with(
        decision.tone, decision.urgency
    )
    assert llm_agent.crew.inputs["event_name"] == "incoming_message_received"
    assert "A harmless rumor" in llm_agent.crew.inputs["event_details"]
    assert llm_agent.crew.inputs["driver_name"] == "David"


def test_permission_decision_cannot_include_private_spoken_content():
    payload = make_decision(ActionType.NONE).model_dump()
    payload.update(action=ActionType.ASK_PERMISSION_TO_TALK, spoken_message="Luca asks you to call now.")
    with pytest.raises(ValueError, match="spoken_message=null"):
        NotificationDecision.model_validate(payload)


def test_incoming_notification_classification_reaches_ui_and_speech():
    decision = NotificationDecision(
        urgency=UrgencyType.HIGH,
        tone=ToneType.SERIOUS,
        intervention_type=InterventionType.ACT,
        skill=SkillType.CONVERSATION,
        action=ActionType.ANNOUNCE_INCOMING_MESSAGE,
        suggestion_type=SuggestionType.NONE,
        reason="The driver can hear the message.",
        spoken_message="Luca says Giulia needs help on the highway ASAP.",
    )
    llm_agent, agent = make_llm_agent(decision)
    event = CarEvent(
        SkillType.CONVERSATION,
        "incoming_message_received",
        {
            "sender": "Luca",
            "text": "Giulia is down on Highway 402 and needs help ASAP.",
        },
        ["Privacy mode is off."],
    )

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    assert decision.tone is ToneType.SERIOUS
    assert decision.urgency is UrgencyType.HIGH
    agent.speak.assert_awaited_once_with(
        "Luca says Giulia needs help on the highway ASAP.", tone=ToneType.SERIOUS
    )
    agent.notify_incoming_message_classification.assert_called_once_with(
        ToneType.SERIOUS, UrgencyType.HIGH
    )


def test_pending_reminder_reuses_classification_of_waiting_message():
    decision = NotificationDecision(
        urgency=UrgencyType.LOW,
        tone=ToneType.DISCREET,
        intervention_type=InterventionType.ACT,
        skill=SkillType.CONVERSATION,
        action=ActionType.ASK_PERMISSION_TO_TALK,
        suggestion_type=SuggestionType.NONE,
        reason="A pending message requires permission.",
        spoken_message=None,
    )
    llm_agent, agent = make_llm_agent(decision)
    agent.action_manager.message_manager.pending_count = 1
    agent.most_urgent_pending_message_classification.return_value = (
        ToneType.SERIOUS,
        UrgencyType.HIGH,
    )
    event = CarEvent(
        SkillType.CONVERSATION,
        EventName.PENDING_MESSAGES_REMINDER,
        {"pending_count": 1},
        ["Privacy mode is on."],
    )

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    assert decision.tone is ToneType.SERIOUS
    assert decision.urgency is UrgencyType.HIGH
    agent.notify_incoming_message_classification.assert_called_once_with(
        ToneType.SERIOUS, UrgencyType.HIGH
    )
    agent.action_manager.handle_decision.assert_called_once_with(
        ActionType.ASK_PERMISSION_TO_TALK, {}
    )
    agent.speak.assert_not_awaited()


def test_stopping_friend_simulation_during_classification_keeps_message_pending():
    decision = NotificationDecision(
        urgency=UrgencyType.LOW,
        tone=ToneType.DISCREET,
        intervention_type=InterventionType.ACT,
        skill=SkillType.CONVERSATION,
        action=ActionType.ANNOUNCE_INCOMING_MESSAGE,
        suggestion_type=SuggestionType.NONE,
        reason="The message can be announced.",
        spoken_message="Luca says that Giulia named her plant Roberto.",
    )
    llm_agent, agent = make_llm_agent(decision)
    agent.action_manager.message_manager.pending_count = 1
    agent.action_manager.message_manager.simulator = SimpleNamespace(active=True)

    async def kickoff_and_stop(inputs):
        agent.message_simulator.active = False
        return SimpleNamespace(raw="structured decision", pydantic=decision)

    llm_agent.crew = SimpleNamespace(
        kickoff_async=AsyncMock(side_effect=kickoff_and_stop)
    )
    llm_agent.message_delivery_crew = llm_agent.crew
    event = CarEvent(
        SkillType.CONVERSATION,
        "incoming_message_received",
        {"sender": "Luca", "text": "Giulia named her plant Roberto."},
        [],
    )

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    assert agent.pending_message_count == 1
    agent.action_manager.handle_decision.assert_not_called()
    agent.speak.assert_not_awaited()


def test_notification_with_explicit_urgency_is_not_hidden_by_reason_wording():
    decision = NotificationDecision(
        urgency=UrgencyType.HIGH,
        tone=ToneType.SERIOUS,
        intervention_type=InterventionType.SUGGEST,
        skill=SkillType.DRIVING,
        action=ActionType.NONE,
        suggestion_type=SuggestionType.REDUCE_SPEED,
        reason="No immediate safety concern is known, but speed exceeds the limit.",
        spoken_message="Speed exceeds the limit; please reduce it.",
    )
    llm_agent, agent = make_llm_agent(decision)
    event = CarEvent(
        SkillType.DRIVING,
        EventName.KNOWLEDGE_UPDATED,
        ["current_speed", "speed_limit"],
        ["Changed just now: Speed exceeds the posted limit."],
    )

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    agent.speak.assert_awaited_once_with(
        "Speed exceeds the limit; please reduce it.", tone=ToneType.SERIOUS
    )


def test_explicit_pending_message_request_dispatches_read_action():
    llm_agent, agent = make_llm_agent(make_decision(ActionType.READ_PENDING_MESSAGES))
    agent.action_manager.message_manager.pending_count = 2
    event = CarEvent(
        SkillType.CONVERSATION,
        "user_input",
        "Tell me the pending messages",
        [],
        "Tell me the pending messages",
    )

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    agent.action_manager.handle_decision.assert_called_once_with(
        ActionType.READ_PENDING_MESSAGES, {}
    )
    agent.speak.assert_not_awaited()
    assert llm_agent.direct_action_crew.inputs["pending_message_count"] == 2
    assert "read_pending_messages" in llm_agent.direct_action_crew.inputs["action_options"]


def test_duplicate_suppression_is_opt_in_and_defaults_off():
    decision = NotificationDecision(
        urgency=UrgencyType.MEDIUM,
        tone=ToneType.CALM,
        intervention_type=InterventionType.SUGGEST,
        skill=SkillType.DRIVING,
        action=ActionType.NONE,
        suggestion_type=SuggestionType.CALM_DRIVING,
        reason="The current driving context supports a brief suggestion.",
        spoken_message="Please continue driving smoothly.",
    )
    llm_agent, agent = make_llm_agent(decision)
    agent.is_duplicate_or_cooling_down = Mock(
        return_value=(True, "duplicate")
    )
    event = CarEvent(
        SkillType.DRIVING,
        "knowledge_updated",
        {"VehicleMotion.speed": 70},
        ["Changed just now: Current speed is 70 km/h."],
    )

    import asyncio

    assert agent.duplicate_suppression_enabled is False
    asyncio.run(agent.process_event_llm(event))
    agent.speak.assert_awaited_once_with(
        "Please continue driving smoothly.", tone=ToneType.CALM
    )
    agent.is_duplicate_or_cooling_down.assert_not_called()

    agent.speak.reset_mock()
    agent.duplicate_suppression_enabled = True
    asyncio.run(agent.process_event_llm(event))

    agent.speak.assert_not_awaited()
    agent.is_duplicate_or_cooling_down.assert_called_once()


def test_direct_command_fallback_logs_decision_and_reason():
    command = "Can you increase the temperature?"
    llm_agent, agent = make_llm_agent(make_decision(ActionType.NONE))
    agent.logger = Mock()
    event = CarEvent(SkillType.CONVERSATION, "user_input", command, [], command)

    import asyncio

    async def run_event():
        await agent.process_event_llm(event)
        await asyncio.sleep(0)

    asyncio.run(run_event())

    agent.action_manager.handle_decision.assert_not_called()
    agent.stream_user_response.assert_awaited_once_with(event)
    agent.logger.warning.assert_called_once_with(
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
    llm_agent, agent = make_llm_agent(make_decision(ActionType.NONE))
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

    asyncio.run(agent.process_event_llm(event))

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
    llm_agent, agent = make_llm_agent(make_decision(ActionType.NONE))
    event = CarEvent(
        SkillType.CONVERSATION,
        "knowledge_updated",
        {"DriverAgenda.late_for_meeting": True},
        ["The driver is running late for an upcoming meeting."],
    )

    import asyncio

    asyncio.run(agent.process_event_llm(event))

    assert "meeting_decision_guidance" not in llm_agent.crew.inputs
    assert "ask_attend_meeting" in llm_agent.crew.inputs["action_options"]

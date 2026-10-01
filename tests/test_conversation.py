from __future__ import annotations

import asyncio
from pathlib import Path
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

	async def kickoff_async(self, *, inputs):
		self.inputs = inputs
		return SimpleNamespace(raw="structured decision", pydantic=self.decision)


def make_llm_agent(decision: NotificationDecision):
	direct_actions = tuple(
		action
		for action in ActionType
		if action
		not in {
			ActionType.NONE,
			ActionType.FIND_REST_AREA,
			ActionType.ASK_ATTEND_MEETING,
		}
	)
	vehicle_action_guidance = (
		Path(__file__).resolve().parents[1] / "skills" / "vehicle_actions.md"
	).read_text(encoding="utf-8")
	agent = SimpleNamespace(
		maybe_handle_meeting_confirmation=AsyncMock(return_value=False),
		cancel_voice_response=Mock(),
		action_manager=SimpleNamespace(handle_decision=Mock()),
		speak=AsyncMock(),
		stream_user_response=AsyncMock(),
		_voice_response_task=None,
		skill_manager=SimpleNamespace(
			get_skill=Mock(return_value="conversation instructions"),
			vehicle_action_guidance=vehicle_action_guidance,
		),
	)
	llm_agent = LLMAgent.__new__(LLMAgent)
	llm_agent.logger = structlog.get_logger()
	llm_agent.agent = cast(Any, agent)
	llm_agent.direct_action_crew = cast(Any, FakeCrew(decision))
	llm_agent.llm = cast(Any, SimpleNamespace(temperature=0.7))
	llm_agent.decision_options = {
		"skill_options": "driving, wellbeing",
		"tone_options": "calm, discreet",
	}
	llm_agent.direct_vehicle_actions = direct_actions
	llm_agent.direct_action_options = "\n".join(
		f"- {action.value}: {action.description}" for action in direct_actions
	)
	return llm_agent, agent


VEHICLE_COMMANDS = [
	("Could you raise the cabin temperature?", ActionType.INCREASE_TEMPERATURE),
	("Could you lower the cabin temperature?", ActionType.DECREASE_TEMPERATURE),
	("Could you increase the fan speed?", ActionType.INCREASE_FAN_SPEED),
	("Could you decrease the fan speed?", ActionType.DECREASE_FAN_SPEED),
	("Could you open the windows?", ActionType.OPEN_WINDOWS),
	("Could you close the windows?", ActionType.CLOSE_WINDOWS),
	("Could you open the sunroof?", ActionType.OPEN_SUNROOF),
	("Could you close the sunroof?", ActionType.CLOSE_SUNROOF),
	("Please turn on the air conditioning.", ActionType.ENABLE_AIR_CONDITIONING),
	("Please turn off the air conditioning.", ActionType.DISABLE_AIR_CONDITIONING),
	("Please enable air recirculation.", ActionType.ENABLE_AIR_RECIRCULATION),
	("Please disable air recirculation.", ActionType.DISABLE_AIR_RECIRCULATION),
	("Please turn on the seat heating.", ActionType.ENABLE_SEAT_HEATING),
	("Please turn off the seat heating.", ActionType.DISABLE_SEAT_HEATING),
	("Please lock the doors.", ActionType.LOCK_DOORS),
	("Please unlock the doors.", ActionType.UNLOCK_DOORS),
	("Please start the radio.", ActionType.START_RADIO),
	("Please stop the radio.", ActionType.STOP_RADIO),
	("Please start navigation.", ActionType.START_NAVIGATION),
	("Please stop navigation.", ActionType.STOP_NAVIGATION),
	(
		"Please enable adaptive cruise control.",
		ActionType.ENABLE_ADAPTIVE_CRUISE_CONTROL,
	),
	(
		"Please disable adaptive cruise control.",
		ActionType.DISABLE_ADAPTIVE_CRUISE_CONTROL,
	),
	("Please enable lane keep assist.", ActionType.ENABLE_LANE_KEEP_ASSIST),
	("Please disable lane keep assist.", ActionType.DISABLE_LANE_KEEP_ASSIST),
	("Please enable blind spot monitoring.", ActionType.ENABLE_BLIND_SPOT_MONITOR),
	("Please disable blind spot monitoring.", ActionType.DISABLE_BLIND_SPOT_MONITOR),
	("Please turn on the sidelights.", ActionType.ENABLE_SIDELIGHTS),
	("Please turn off the sidelights.", ActionType.DISABLE_SIDELIGHTS),
	(
		"Please turn on the low-beam headlights.",
		ActionType.ENABLE_LOW_BEAM_HEADLIGHTS,
	),
	(
		"Please turn off the low-beam headlights.",
		ActionType.DISABLE_LOW_BEAM_HEADLIGHTS,
	),
	(
		"Please turn on the high-beam headlights.",
		ActionType.ENABLE_HIGH_BEAM_HEADLIGHTS,
	),
	(
		"Please turn off the high-beam headlights.",
		ActionType.DISABLE_HIGH_BEAM_HEADLIGHTS,
	),
	("Please turn on the fog lights.", ActionType.ENABLE_FOG_LIGHTS),
	("Please turn off the fog lights.", ActionType.DISABLE_FOG_LIGHTS),
	("Reduce the target speed.", ActionType.REDUCE_TARGET_SPEED),
	("Increase the target speed.", ActionType.INCREASE_TARGET_SPEED),
	(
		"Increase the following distance.",
		ActionType.INCREASE_FOLLOWING_DISTANCE,
	),
	(
		"Decrease the following distance.",
		ActionType.DECREASE_FOLLOWING_DISTANCE,
	),
	("Increase the audio volume.", ActionType.INCREASE_AUDIO_VOLUME),
	("Decrease the audio volume.", ActionType.DECREASE_AUDIO_VOLUME),
	(
		"Apply the restrictive ADAS safety profile.",
		ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE,
	),
]

def test_vehicle_command_matrix_covers_all_direct_actions() -> None:
	excluded_actions = {
		ActionType.NONE,
		ActionType.FIND_REST_AREA,
		ActionType.ASK_ATTEND_MEETING,
	}

	assert {action for _, action in VEHICLE_COMMANDS} == set(ActionType) - excluded_actions


@pytest.mark.parametrize(("command", "expected_action"), VEHICLE_COMMANDS)
def test_explicit_vehicle_command_dispatches_matching_action(
	command: str, expected_action: ActionType
) -> None:
	decision = NotificationDecision(
		urgency=UrgencyType.LOW,
		tone=ToneType.CALM,
		intervention_type=InterventionType.ACT,
		skill=SkillType.DRIVING,
		action=expected_action,
		suggestion_type=SuggestionType.NONE,
		reason="The driver explicitly requested this vehicle action.",
		spoken_message="I received your request.",
	)
	llm_agent, agent = make_llm_agent(decision)
	event = CarEvent(SkillType.CONVERSATION, "user_input", command, [], command)

	asyncio.run(llm_agent.process_event(event))

	assert llm_agent.direct_action_crew.inputs["user_input"] == command
	assert expected_action.value in llm_agent.direct_action_crew.inputs["action_options"]
	assert expected_action.value in llm_agent.direct_action_crew.inputs[
		"vehicle_action_guidance"
	]
	agent.action_manager.handle_decision.assert_called_once_with(expected_action, {})
	agent.speak.assert_awaited_once_with(
		"I received your request.", tone=ToneType.CALM
	)


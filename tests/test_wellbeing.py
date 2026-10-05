from __future__ import annotations

import asyncio
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
from agents.llm_backend import LLMBackend, NotificationDecision
from agents.automotive_agent import AutomotiveAgent
from data.events import CarEvent, EventName


class FakeCrew:
	def __init__(self, decision: NotificationDecision) -> None:
		self.decision = decision
		self.inputs = None

	async def kickoff_async(self, *, inputs):
		self.inputs = inputs
		return SimpleNamespace(raw="structured decision", pydantic=self.decision)


def make_llm_agent(decision: NotificationDecision):
	agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
	agent.__dict__.update(vars(SimpleNamespace(
		maybe_handle_meeting_confirmation=AsyncMock(return_value=False),
		log_llm_usage=Mock(),
		cancel_voice_response=Mock(),
		action_manager=SimpleNamespace(handle_decision=Mock(), message_manager=SimpleNamespace(simulation_stopped=Mock(return_value=False))),
		speak=AsyncMock(),
		stream_user_response=AsyncMock(),
		voice_response_task=None,
		skill_manager=SimpleNamespace(
			get_skill=Mock(return_value="driving instructions"),
			vehicle_action_guidance=(
				"increase or decrease cabin temperature or fan speed by one step"
			),
		),
	)))
	agent.logger = structlog.get_logger()
	agent.recent_notifications = []
	agent.duplicate_suppression_enabled = False
	llm_agent = LLMBackend.__new__(LLMBackend)
	llm_agent.logger = structlog.get_logger()
	llm_agent.on_usage = agent.log_llm_usage
	llm_agent.crew = cast(Any, FakeCrew(decision))
	llm_agent.llm = cast(Any, SimpleNamespace(temperature=0.7))
	llm_agent.decision_options = {
		"skill_options": "driving, wellbeing",
		"tone_options": "calm, discreet",
	}
	llm_agent.silent_decision_guidance = "silent guidance"
	agent.llm_backend = llm_agent
	return llm_agent, agent


def test_dangerous_object_update_warns_driver_and_locks_unlocked_doors() -> None:
	warning = "A dangerous object has been detected nearby. I have locked the doors."
	decision = NotificationDecision(
		urgency=UrgencyType.HIGH,
		tone=ToneType.SERIOUS,
		intervention_type=InterventionType.ACT,
		skill=SkillType.DRIVING,
		action=ActionType.LOCK_DOORS,
		suggestion_type=SuggestionType.SECURE_VEHICLE,
		reason="A dangerous object is detected and the vehicle doors are unlocked.",
		spoken_message=warning,
	)
	llm_agent, agent = make_llm_agent(decision)
	event = CarEvent(
		SkillType.DRIVING,
		EventName.KNOWLEDGE_UPDATED,
		{"DetectedObjects.dangerous_objects_around": True},
		[
			"Changed just now: A dangerous object is detected around the vehicle.",
			"The vehicle doors are unlocked.",
		],
	)

	asyncio.run(agent.process_event_llm(event))

	assert llm_agent.crew.inputs["changed_facts"] == (
		"- A dangerous object is detected around the vehicle."
	)
	assert llm_agent.crew.inputs["context"] == "- The vehicle doors are unlocked."
	agent.speak.assert_awaited_once_with(warning, tone=ToneType.SERIOUS)
	agent.action_manager.handle_decision.assert_called_once_with(
		ActionType.LOCK_DOORS, {}
	)


@pytest.mark.parametrize(
	("measure", "value", "changed_fact", "supporting_context"),
	[
		(
			"DriverPhysicalState.fatigue_level",
			0.7,
			"Fatigue level is high.",
			"It is night.",
		),
	],
)
def test_high_fatigue_or_low_attention_suggests_a_break(
	measure: str, value: float, changed_fact: str, supporting_context: str
) -> None:
	suggestion = "Please take a break at the next safe opportunity."
	decision = NotificationDecision(
		urgency=UrgencyType.HIGH,
		tone=ToneType.CALM,
		intervention_type=InterventionType.SUGGEST,
		skill=SkillType.WELLBEING,
		action=ActionType.NONE,
		suggestion_type=SuggestionType.TAKE_BREAK,
		reason="The reported fatigue or low attention is increased by the conditions.",
		spoken_message=suggestion,
	)
	llm_agent, agent = make_llm_agent(decision)
	agent.skill_manager.get_skill.return_value = "wellbeing instructions"
	event = CarEvent(
		SkillType.WELLBEING,
		EventName.KNOWLEDGE_UPDATED,
		{measure: value},
		[f"Changed just now: {changed_fact}", supporting_context],
	)

	asyncio.run(agent.process_event_llm(event))

	assert llm_agent.crew.inputs["skill_instructions"] == "wellbeing instructions"
	assert llm_agent.crew.inputs["changed_facts"] == f"- {changed_fact}"
	assert llm_agent.crew.inputs["context"] == f"- {supporting_context}"
	agent.speak.assert_awaited_once_with(suggestion, tone=ToneType.CALM)
	agent.action_manager.handle_decision.assert_not_called()


@pytest.mark.parametrize(
	("measure", "value", "fact", "persisted", "vehicle_moving", "expected_action"),
	[
		(
			"DriverPhysicalState.attention_level",
			0.2,
			"Attention level is low.",
			True,
			True,
			ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE,
		),
		(
			"DriverPhysicalState.attention_level",
			0.2,
			"Attention level is low.",
			False,
			True,
			ActionType.NONE,
		),
		(
			"DriverPhysicalState.attention_level",
			0.2,
			"Attention level is low.",
			True,
			False,
			ActionType.NONE,
		),
		(
			"DriverPhysicalState.fatigue_level",
			0.7,
			"Fatigue level is high.",
			True,
			True,
			ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE,
		),
	],
)
def test_adas_profile_requires_persistent_attention_or_fatigue_while_moving(
	measure: str,
	value: float,
	fact: str,
	persisted: bool,
	vehicle_moving: bool,
	expected_action: ActionType,
) -> None:
	changed_fact = (
		f"{fact} This condition has persisted for at least one minute."
		if persisted
		else fact
	)
	warning = "Your attention or fatigue needs attention. Please take a break when safe."
	decision = NotificationDecision(
		urgency=UrgencyType.HIGH,
		tone=ToneType.SERIOUS,
		intervention_type=(
			InterventionType.ACT
			if vehicle_moving
			else InterventionType.SUGGEST
		),
		skill=SkillType.WELLBEING,
		action=expected_action,
		suggestion_type=(
			SuggestionType.NONE
			if vehicle_moving
			else SuggestionType.TAKE_BREAK
		),
		reason="The reported wellbeing condition requires a response.",
		spoken_message=warning,
	)
	llm_agent, agent = make_llm_agent(decision)
	agent.skill_manager.get_skill.return_value = "wellbeing instructions"
	event = CarEvent(
		SkillType.WELLBEING,
		EventName.KNOWLEDGE_UPDATED,
		{measure: value},
		[
			f"Changed just now: {changed_fact}",
			"The vehicle is moving." if vehicle_moving else "The vehicle is stationary.",
		],
	)

	asyncio.run(agent.process_event_llm(event))

	agent.speak.assert_awaited_once_with(warning, tone=ToneType.SERIOUS)
	if expected_action is ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE:
		agent.action_manager.handle_decision.assert_called_once_with(
			ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE, {}
		)
	else:
		agent.action_manager.handle_decision.assert_not_called()


@pytest.mark.parametrize(
	("activity", "changed_fact", "warning", "expected_action"),
	[
		(
			"Talking",
			"The driver is talking, which may distract from driving.",
			"Talking can distract you from driving. Please keep your attention on the road.",
			ActionType.NONE,
		),
		(
			"On the phone",
			"The driver is using a phone, which seriously distracts from driving.",
			"Using your phone is dangerous while driving. Please focus on the road.",
			ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE,
		),
		(
			"Eating",
			"The driver is eating, which distracts from driving.",
			"Eating can distract you from driving. Please keep your attention on the road.",
			ActionType.NONE,
		),
		(
			"Sleeping",
			"The driver appears to be asleep and unable to drive safely.",
			"You appear to be asleep. Please stop as soon as it is safe.",
			ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE,
		),
	],
)
def test_dangerous_driver_activity_warns_and_applies_supported_action(
	activity: str,
	changed_fact: str,
	warning: str,
	expected_action: ActionType,
) -> None:
	decision = NotificationDecision(
		urgency=UrgencyType.HIGH,
		tone=ToneType.SERIOUS,
		intervention_type=(
			InterventionType.ACT
			if expected_action is not ActionType.NONE
			else InterventionType.SUGGEST
		),
		skill=SkillType.WELLBEING,
		action=expected_action,
		suggestion_type=(
			SuggestionType.NONE
			if expected_action is not ActionType.NONE
			else SuggestionType.REDUCE_DISTRACTION
		),
		reason="The detected activity compromises safe driving.",
		spoken_message=warning,
	)
	llm_agent, agent = make_llm_agent(decision)
	agent.skill_manager.get_skill.return_value = "wellbeing instructions"
	event = CarEvent(
		SkillType.WELLBEING,
		EventName.KNOWLEDGE_UPDATED,
		{"DriverPhysicalState.activity": activity},
		[f"Changed just now: {changed_fact}", "The vehicle is moving."],
	)

	asyncio.run(agent.process_event_llm(event))

	assert llm_agent.crew.inputs["changed_facts"] == f"- {changed_fact}"
	assert llm_agent.crew.inputs["context"] == "- The vehicle is moving."
	agent.speak.assert_awaited_once_with(warning, tone=ToneType.SERIOUS)
	if expected_action is ActionType.NONE:
		agent.action_manager.handle_decision.assert_not_called()
	else:
		agent.action_manager.handle_decision.assert_called_once_with(
			expected_action, {}
		)


def test_children_out_of_place_triggers_high_priority_safety_warning() -> None:
	warning = (
		"Children are out of place near the vehicle. Please ensure they are safely "
		"supervised and clear of the vehicle before moving."
	)
	decision = NotificationDecision(
		urgency=UrgencyType.HIGH,
		tone=ToneType.CALM,
		intervention_type=InterventionType.SUGGEST,
		skill=SkillType.WELLBEING,
		action=ActionType.NONE,
		suggestion_type=SuggestionType.SECURE_VEHICLE,
		reason="Children are explicitly reported out of place near the vehicle.",
		spoken_message=warning,
	)
	llm_agent, agent = make_llm_agent(decision)
	agent.skill_manager.get_skill.return_value = "wellbeing instructions"
	event = CarEvent(
		SkillType.WELLBEING,
		EventName.KNOWLEDGE_UPDATED,
		{"DriverPhysicalState.activity": "Children out of place"},
		[
			"Changed just now: Children are out of place near the vehicle.",
			"The vehicle is stationary.",
		],
	)

	asyncio.run(agent.process_event_llm(event))

	assert llm_agent.crew.inputs["changed_facts"] == (
		"- Children are out of place near the vehicle."
	)
	assert llm_agent.crew.inputs["context"] == "- The vehicle is stationary."
	agent.speak.assert_awaited_once_with(warning, tone=ToneType.CALM)
	agent.action_manager.handle_decision.assert_not_called()

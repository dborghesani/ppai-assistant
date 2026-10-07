from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

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


def test_aggressive_driving_style_warns_driver() -> None:
	warning = "Your driving style appears very aggressive. Please drive calmly and smoothly."
	decision = NotificationDecision(
		urgency=UrgencyType.HIGH,
		tone=ToneType.SERIOUS,
		intervention_type=InterventionType.SUGGEST,
		skill=SkillType.DRIVING,
		action=ActionType.NONE,
		suggestion_type=SuggestionType.CALM_DRIVING,
		reason="The reported driving tension is very high.",
		spoken_message=warning,
	)
	driving_instructions = (
		Path(__file__).resolve().parents[1] / "skills" / "driving.md"
	).read_text(encoding="utf-8")
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
			get_skill=Mock(return_value=driving_instructions),
			vehicle_action_guidance="vehicle action reference",
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
		"urgency_options": "none, low, medium, high, critical",
		"tone_options": "calm, serious",
		"intervention_options": "none, suggest, act",
		"skill_options": "driving, wellbeing",
		"action_options": "none, apply_restrictive_adas_profile",
		"suggestion_options": "none, calm_driving",
	}
	llm_agent.silent_decision_guidance = "silent guidance"
	agent.llm_backend = llm_agent
	event = CarEvent(
		SkillType.DRIVING,
		EventName.KNOWLEDGE_UPDATED,
		{"DriverDrivingStyle.driving_tension": 0.9},
		[
			"Changed just now: The driving style shows a very high level of tension."
		],
	)

	asyncio.run(agent.process_event_llm(event))

	crew = cast(Any, llm_agent.crew)
	assert "excessively aggressive" in crew.inputs["skill_instructions"]
	assert "Treat reported driving tension as a driving-style signal" in crew.inputs["skill_instructions"]
	assert crew.inputs["changed_facts"] == (
		"- The driving style shows a very high level of tension."
	)
	assert decision.suggestion_type is SuggestionType.CALM_DRIVING
	agent.speak.assert_awaited_once_with(warning, tone=ToneType.SERIOUS)
	agent.action_manager.handle_decision.assert_not_called()

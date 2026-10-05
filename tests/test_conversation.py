from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
import structlog

from agents.automotive_agent import AutomotiveAgent
from data.agents_dataclasses import (
	ActionType,
	AssistantStatus,
	InterventionType,
	SkillType,
	SuggestionType,
	ToneType,
	UrgencyType,
)
from agents.llm_backend import LLMBackend, NotificationDecision
from managers.action_manager import ActionManager
from managers.music_manager import MUSIC_ACTIONS
from config import ConfigAssistant
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
			*MUSIC_ACTIONS,
		}
	)
	vehicle_action_guidance = (
		Path(__file__).resolve().parents[1] / "skills" / "vehicle_actions.md"
	).read_text(encoding="utf-8")
	agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
	agent.__dict__.update(vars(SimpleNamespace(
		maybe_handle_meeting_confirmation=AsyncMock(return_value=False),
		log_llm_usage=Mock(),
		cancel_voice_response=Mock(),
		action_manager=SimpleNamespace(handle_decision=Mock()),
		speak=AsyncMock(),
		stream_user_response=AsyncMock(),
		voice_response_task=None,
		skill_manager=SimpleNamespace(
			get_skill=Mock(return_value="conversation instructions"),
			vehicle_action_guidance=vehicle_action_guidance,
		),
	)))
	agent.opt = ConfigAssistant()
	agent.voice_llm = SimpleNamespace()
	agent.conversation_history = []
	agent.knowledge_context_provider = lambda: []
	agent.logger = structlog.get_logger()
	agent.recent_notifications = []
	agent.duplicate_suppression_enabled = False
	async def prepare_response(event, source):
		await agent.stream_user_response(event)

	agent.prepare_user_response = AsyncMock(side_effect=prepare_response)
	agent.action_manager = cast(Any, ActionManager(cast(Any, agent)))
	agent.action_manager.handle_decision = Mock()
	llm_agent = LLMBackend.__new__(LLMBackend)
	llm_agent.logger = structlog.get_logger()
	llm_agent.on_usage = agent.log_llm_usage
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
	agent.llm_backend = llm_agent
	return llm_agent, agent


def test_llm_classification_returns_decision_without_applying_it() -> None:
	async def check():
		decision = NotificationDecision(
			urgency=UrgencyType.LOW, tone=ToneType.CALM,
			intervention_type=InterventionType.ACT, skill=SkillType.DRIVING,
			action=ActionType.OPEN_WINDOWS, suggestion_type=SuggestionType.NONE, reason="Explicit request",
		)
		llm_agent, agent = make_llm_agent(decision)
		assert await llm_agent.classify_request({"user_input": "Open the windows"}) is decision
		agent.speak.assert_not_awaited()
		agent.action_manager.handle_decision.assert_not_called()
		agent.stream_user_response.assert_not_awaited()
		assert llm_agent.llm.temperature == 0.7
		assert not hasattr(llm_agent, "agent")
		assert not hasattr(llm_agent, "process_event")
	asyncio.run(check())


def test_backend_usage_is_per_call_not_cumulative():
	async def check():
		decision = NotificationDecision(urgency=UrgencyType.NONE, tone=ToneType.CALM,
			intervention_type=InterventionType.NONE, skill=SkillType.NONE,
			action=ActionType.NONE, suggestion_type=SuggestionType.NONE, reason="No action")
		backend, agent = make_llm_agent(decision)
		counters = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
		backend.llm.get_token_usage_summary = lambda: counters
		async def kickoff(*, inputs):
			counters["prompt_tokens"] += 200
			counters["completion_tokens"] += 20
			counters["total_tokens"] += 220
			return SimpleNamespace(pydantic=decision, token_usage=SimpleNamespace(**counters))
		backend.direct_action_crew.kickoff_async = kickoff
		backend.crew = backend.direct_action_crew
		await backend.classify_request({})
		await backend.evaluate_event({"event_name": "knowledge_updated"})
		assert agent.log_llm_usage.call_count == 2
		for call in agent.log_llm_usage.call_args_list:
			assert call.args[1] == {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}
		assert counters["total_tokens"] == 1540
	asyncio.run(check())


def test_orchestrator_applies_backend_decision() -> None:
	async def check():
		decision = NotificationDecision(urgency=UrgencyType.LOW, tone=ToneType.CALM,
			intervention_type=InterventionType.ACT, skill=SkillType.DRIVING,
			action=ActionType.OPEN_WINDOWS, suggestion_type=SuggestionType.NONE, reason="Explicit request")
		agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
		agent.logger = Mock()
		agent.action_manager = SimpleNamespace(message_manager=SimpleNamespace(pending_count=0),
			registered_action_options="", registered_action_guidance="", registered_actions=frozenset(), handle_decision=Mock())
		agent.skill_manager = SimpleNamespace(vehicle_action_guidance="Vehicle actions")
		agent.conversation_history = []
		agent.speak = AsyncMock()
		agent.llm_backend = SimpleNamespace(classify_request=AsyncMock(return_value=decision),
			direct_action_options="open_windows", direct_vehicle_actions=(ActionType.OPEN_WINDOWS,),
			decision_options={"skill_options": "driving", "tone_options": "calm"})
		await agent._process_direct_user_input(CarEvent(SkillType.CONVERSATION, "user_input", "Open windows", [], "Open windows"))
		agent.action_manager.handle_decision.assert_called_once_with(ActionType.OPEN_WINDOWS, {})
		agent.speak.assert_awaited_once_with("I received the action request.", tone=ToneType.CALM)
	asyncio.run(check())


def test_log_llm_usage_reports_context_window_fill() -> None:
	agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
	agent.opt = ConfigAssistant(context_window_size=4096)
	agent.logger = Mock()
	agent.log_llm_usage(
		"test call",
		SimpleNamespace(prompt_tokens=1000, completion_tokens=250, total_tokens=1250),
	)

	agent.logger.info.assert_called_once_with(
		"LLM token usage",
		llm_call="test call",
		available=True,
		prompt_tokens=1000,
		completion_tokens=250,
		total_tokens=1250,
		context_tokens=1250,
		context_window_tokens=4096,
		context_window_percent=30.5,
	)


def test_car_event_dataclass_preserves_context_copy_and_timestamp() -> None:
	context = ["vehicle is stopped"]
	event = CarEvent(SkillType.CONVERSATION, "user_input", "Hello", context, "Hello")

	context.append("external mutation")

	assert event.context == ["vehicle is stopped"]
	assert event.timestamp is not None
	assert event.user_input == "Hello"


VEHICLE_COMMANDS = [
	("Could you raise the cabin temperature?", ActionType.ENABLE_HEATING),
	("Could you lower the cabin temperature?", ActionType.ENABLE_AIR_CONDITIONING),
	("Please turn on the cabin heating.", ActionType.ENABLE_HEATING),
	("Please turn off the cabin heating.", ActionType.DISABLE_HEATING),
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
	("Please enable privacy mode.", ActionType.ENABLE_PRIVACY_MODE),
	("Please disable privacy mode.", ActionType.DISABLE_PRIVACY_MODE),
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
		ActionType.ANNOUNCE_INCOMING_MESSAGE,
		ActionType.ASK_PERMISSION_TO_TALK,
		ActionType.READ_PENDING_MESSAGES,
		*MUSIC_ACTIONS,
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

	asyncio.run(agent.process_event_llm(event))

	assert llm_agent.direct_action_crew.inputs["user_input"] == command
	assert expected_action.value in llm_agent.direct_action_crew.inputs["action_options"]
	assert expected_action.value in llm_agent.direct_action_crew.inputs[
		"vehicle_action_guidance"
	]
	agent.action_manager.handle_decision.assert_called_once_with(expected_action, {})
	agent.speak.assert_awaited_once_with(
		"I received your request.", tone=ToneType.CALM
	)


def make_meeting_agent(confirmed: bool):
	agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
	agent.action_manager = ActionManager(agent)
	agent.action_manager.logger = Mock()
	agent.conversation_history = []
	agent.interpret_confirmation = AsyncMock(return_value=confirmed)
	agent._assistant_status = AssistantStatus.IDLE
	agent.set_assistant_status = Mock()
	agent.speak = AsyncMock()
	async def generated_reply(event, *, response_context):
		await agent.speak("A contextual response generated by the LLM.")
	agent.stream_user_response = AsyncMock(side_effect=generated_reply)
	agent.knowledge_context_provider = lambda: ["The driver is late for a meeting."]
	agent.llm_backend = SimpleNamespace(llm=object(), meeting_reply_context=LLMBackend.meeting_reply_context)
	agent.opt = SimpleNamespace(crewai_verbose=False)
	return agent


def test_invalid_confirmation_keeps_meeting_pending():
	async def check():
		agent = make_meeting_agent(confirmed=False)
		agent.interpret_confirmation = AsyncMock(return_value=None)
		agent.action_manager.awaiting_confirmation = True
		assert await agent.maybe_handle_meeting_confirmation("yes, please")
		assert agent.action_manager.awaiting_confirmation
		assert agent.action_manager._meeting_task is None
		agent.stream_user_response.assert_awaited_once()
		assert '"confirmed": null' in agent.stream_user_response.call_args.kwargs["response_context"][1]
	asyncio.run(check())


@pytest.mark.parametrize("content, expected", [
	('{"confirmed": true}', True), ('{"confirmed": false}', False),
	('', None), ('{}', None), ('{"confirmed": "false"}', None),
])
def test_confirmation_backend_requires_boolean_output(content, expected):
	async def check():
		backend = LLMBackend.__new__(LLMBackend)
		backend.opt = ConfigAssistant()
		backend.on_usage = Mock()
		backend.logger = Mock()
		create = AsyncMock(return_value=SimpleNamespace(usage=None, choices=[
			SimpleNamespace(message=SimpleNamespace(content=content))
		]))
		backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
		result = await backend.interpret_confirmation("yes, please", [{"role": "assistant", "content": "Should I attend the meeting?"}])
		assert result is expected
		assert create.call_args.kwargs["reasoning_effort"] == "none"
		assert create.call_args.kwargs["max_tokens"] == 64
		assert create.call_args.kwargs["response_format"]["json_schema"]["schema"]["required"] == ["confirmed"]
	asyncio.run(check())


def test_meeting_permission_yes_starts_meeting_and_speaks_summary() -> None:
	async def run_flow() -> None:
		agent = make_meeting_agent(confirmed=True)
		meeting_crew = SimpleNamespace(kickoff=Mock())
		driver_summary_task = SimpleNamespace(
			output=SimpleNamespace(raw="We agreed on a project timeline.")
		)
		with patch(
			"managers.action_manager.build_crew",
			return_value=(meeting_crew, driver_summary_task),
		) as build_crew_mock:
			agent.action_manager.handle_decision(ActionType.ASK_ATTEND_MEETING)
			assert agent.action_manager.awaiting_confirmation is True

			handled = await agent.maybe_handle_meeting_confirmation("Yes, please.")
			assert handled is True
			assert agent.action_manager._meeting_task is not None
			await agent.action_manager._meeting_task

		build_crew_mock.assert_called_once()
		meeting_crew.kickoff.assert_called_once_with()
		assert agent.action_manager.awaiting_confirmation is False
		agent.set_assistant_status.assert_any_call(
			AssistantStatus.BACKGROUND_TASK_RUNNING
		)
		assert [call.args[0] for call in agent.speak.await_args_list] == [
			"A contextual response generated by the LLM.",
			"We agreed on a project timeline.",
		]
		assert '"confirmed": true' in agent.stream_user_response.call_args.kwargs["response_context"][1]
		assert agent.stream_user_response.call_args.args[0].context == ["The driver is late for a meeting."]

	asyncio.run(run_flow())


@pytest.mark.parametrize("action", [
	ActionType.PLAY_MUSIC, ActionType.PAUSE_MUSIC, ActionType.RESUME_MUSIC,
	ActionType.NEXT_MUSIC, ActionType.STOP_MUSIC,
])
def test_music_commands_use_music_handler_not_vehicle_controls(action) -> None:
	async def check():
		decision = NotificationDecision(
			urgency=UrgencyType.LOW, tone=ToneType.CALM,
			intervention_type=InterventionType.ACT, skill=SkillType.CONVERSATION,
			action=action, suggestion_type=SuggestionType.NONE, reason="Explicit music request",
		)
		llm_agent, agent = make_llm_agent(decision)
		agent.handle_music_action = AsyncMock()
		agent.action_manager._music_manager = cast(Any, SimpleNamespace(execute=agent.handle_music_action))
		event = CarEvent(SkillType.CONVERSATION, "user_input", "Music command", [], "Music command")
		await agent._process_direct_user_input(event)
		if agent.voice_response_task is not None:
			await agent.voice_response_task
		agent.handle_music_action.assert_awaited_once_with(action, event.user_input, agent=agent)
		agent.action_manager.handle_decision.assert_not_called()
		assert "play_music" in llm_agent.direct_action_crew.inputs["action_options"]
	asyncio.run(check())


def test_registered_action_dispatch_is_generic_and_receives_conversation_history() -> None:
	async def check():
		decision = NotificationDecision(
			urgency=UrgencyType.LOW, tone=ToneType.CALM,
			intervention_type=InterventionType.ACT, skill=SkillType.CONVERSATION,
			action=ActionType.START_RADIO, suggestion_type=SuggestionType.NONE, reason="Contextual request",
		)
		llm_agent, agent = make_llm_agent(decision)
		agent.conversation_history = [{"role": "assistant", "content": "Would you like the radio?"}]
		handler = AsyncMock()
		agent.action_manager.register_actions([ActionType.START_RADIO], handler, "Use the conversation for radio replies.")
		event = CarEvent(SkillType.CONVERSATION, "user_input", "Yes please", [], "Yes please")
		await agent._process_direct_user_input(event)
		await agent.voice_response_task
		handler.assert_awaited_once_with(ActionType.START_RADIO, "Yes please")
		inputs = llm_agent.direct_action_crew.inputs
		assert "Would you like the radio?" in inputs["conversation_history"]
		assert "Use the conversation for radio replies." in inputs["registered_action_guidance"]
		assert "music_permission_prompt" not in inputs
	asyncio.run(check())


def test_meeting_permission_no_declines_without_starting_meeting() -> None:
	async def run_flow() -> None:
		agent = make_meeting_agent(confirmed=False)
		with patch("managers.action_manager.build_crew") as build_crew_mock:
			agent.action_manager.handle_decision(ActionType.ASK_ATTEND_MEETING)
			assert agent.action_manager.awaiting_confirmation is True

			handled = await agent.maybe_handle_meeting_confirmation("No, thanks.")

		assert handled is True
		assert agent.action_manager.awaiting_confirmation is False
		assert agent.action_manager._meeting_task is None
		build_crew_mock.assert_not_called()
		agent.speak.assert_awaited_once_with(
			"A contextual response generated by the LLM."
		)
		assert '"confirmed": false' in agent.stream_user_response.call_args.kwargs["response_context"][1]

	asyncio.run(run_flow())


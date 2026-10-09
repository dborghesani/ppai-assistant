from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
import structlog
import httpx
from crewai import LLM, Agent, Crew, Task
from openai import OpenAI

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
from agents.llm_backend import (
	ConfirmationReply, LLMBackend, NotificationDecision, PreClassification, ResponseSource,
	_before_timed_llm_call, _after_timed_llm_call, _crew_llm_timings,
)
from agents.tasks.pre_classify import (
	PreClassificationInput,
	PreClassificationUpdate,
	RequestPreClassifier,
)
from crewai.hooks.llm_hooks import LLMCallHookContext
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


def test_crewai_logs_full_response_generation_time():
	async def check():
		backend = LLMBackend.__new__(LLMBackend)
		backend.opt = ConfigAssistant(ollama_model="test-model")
		backend.logger = Mock()
		result = object()
		crew = SimpleNamespace(kickoff_async=AsyncMock(return_value=result))
		with patch("agents.llm_backend.time.perf_counter", side_effect=[10.0, 10.25]):
			assert await backend._kickoff_timed(crew, {"user_input": "Hi"}, "test call") is result
		crew.kickoff_async.assert_awaited_once_with(inputs={"user_input": "Hi"})
		backend.logger.info.assert_called_once_with(
			"[LLM] Response generation time", llm_call="test call",
			model="test-model", duration_ms=250.0,
			llm_calls=0, completed_llm_calls=0, llm_duration_ms=0.0,
			crew_overhead_ms=None, success=True,
		)
	asyncio.run(check())


def test_pre_classify_request_uses_short_history_and_returns_route():
	async def check():
		classifier = RequestPreClassifier.__new__(RequestPreClassifier)
		classifier.streaming = True
		classifier.model = "fast-model"
		classifier.on_usage = Mock()
		chunks = [
			'{"requires_action_classification":true,',
			'"response_source":"vehicle_manual",',
			'"wait_message":"Checking how to activate cruise ',
			'control for your vehicle."}',
		]
		events = []
		class Stream:
			def __init__(self):
				self.chunks = iter(chunks)

			def __aiter__(self):
				return self

			async def __anext__(self):
				try:
					text = next(self.chunks)
				except StopIteration:
					raise StopAsyncIteration
				events.append("stream_chunk")
				return SimpleNamespace(
					choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None,
				)

			async def close(self):
				pass

		stream = Stream()
		classifier.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
			create=AsyncMock(return_value=stream),
		)))
		result = await classifier.execute(
			PreClassificationInput(
				"How fast am I going?",
				[{"role": "user", "content": str(index)} for index in range(8)],
				0,
			),
			on_update=lambda update: events.append(update),
		)

		assert result.requires_action_classification is True
		assert result.wait_message == "Checking how to activate cruise control for your vehicle."
		assert events == [
			"stream_chunk",
			PreClassificationUpdate(requires_action_classification=True),
			"stream_chunk",
			PreClassificationUpdate(response_source=ResponseSource.VEHICLE_MANUAL),
			"stream_chunk",
			"stream_chunk",
			PreClassificationUpdate(
				wait_message_chunk="Checking how to activate cruise control for your vehicle."
			),
		]
		call = classifier.client.chat.completions.create.call_args.kwargs
		assert call["model"] == "fast-model"
		assert call["stream"] is True
		json_schema = call["response_format"]["json_schema"]
		assert json_schema["strict"] is True
		assert json_schema["schema"]["required"] == list(
			json_schema["schema"]["properties"]
		)
		assert json_schema["schema"]["additionalProperties"] is False
		assert len(json.loads(call["messages"][1]["content"])["conversation_history"]) == 6
		classifier.on_usage.assert_called_once_with("direct request pre-classification", None)
	asyncio.run(check())


def test_pre_classify_crew_mode_emits_updates_after_final_result():
	async def check():
		classifier = RequestPreClassifier.__new__(RequestPreClassifier)
		classifier.streaming = False
		classifier.model = "fast-model"
		classifier.crew_llm = SimpleNamespace(
			temperature=0.7,
			get_token_usage_summary=Mock(return_value=None),
		)
		classifier.on_usage = Mock()
		classification = PreClassification(
			requires_action_classification=True,
			wait_message="Un momento.",
		)
		crew = SimpleNamespace(kickoff_async=AsyncMock(
			return_value=SimpleNamespace(pydantic=classification),
		))
		classifier.crew = crew
		updates = []

		result = await classifier.execute(
			PreClassificationInput("Open the windows", [], 0),
			on_update=updates.append,
		)

		assert result is classification
		assert updates == [
			PreClassificationUpdate(requires_action_classification=True),
			PreClassificationUpdate(wait_message_chunk="Un momento."),
		]
		crew.kickoff_async.assert_awaited_once()
		classifier.on_usage.assert_called_once_with("direct request pre-classification", None)
	asyncio.run(check())


def test_pre_classify_conversation_route_does_not_emit_wait_message():
	async def check():
		classifier = RequestPreClassifier.__new__(RequestPreClassifier)
		classifier.streaming = True
		classifier.model = "fast-model"
		classifier.on_usage = Mock()
		chunks = [
			'{"requires_action_classification":false,',
			'"wait_message":"","response_source":"general"}',
		]
		class Stream:
			def __init__(self):
				self.chunks = iter(chunks)

			def __aiter__(self):
				return self

			async def __anext__(self):
				try:
					text = next(self.chunks)
				except StopIteration:
					raise StopAsyncIteration
				return SimpleNamespace(
					choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None,
				)

			async def close(self):
				pass

		classifier.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
			create=AsyncMock(return_value=Stream()),
		)))
		updates = []

		result = await classifier.execute(
			PreClassificationInput("How are you?", [], 0), on_update=updates.append,
		)

		assert result.requires_action_classification is False
		assert updates == [
			PreClassificationUpdate(requires_action_classification=False),
			PreClassificationUpdate(response_source=ResponseSource.GENERAL),
		]
	asyncio.run(check())


def test_pre_classification_rejects_question_as_wait_message():
	with pytest.raises(ValueError, match="brief non-question holding phrase"):
		PreClassification(
			requires_action_classification=True,
			wait_message="Which make and model do you drive?",
		)


def test_pre_classification_accepts_natural_nine_word_wait_message():
	classification = PreClassification(
		requires_action_classification=True,
		wait_message="Checking how to activate cruise control for your vehicle.",
	)
	assert classification.wait_message == (
		"Checking how to activate cruise control for your vehicle."
	)


def test_pre_classify_wait_message_is_spoken_verbatim():
	async def check():
		agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
		agent.on_response = Mock()
		agent.tts_manager = cast(Any, SimpleNamespace(speak=AsyncMock()))
		message = "Un momento, controllo."

		await agent._speak_wait_message(message)

		agent.on_response.assert_called_once_with(message + "\n")
		agent.tts_manager.speak.assert_awaited_once_with(message)
	asyncio.run(check())


def test_wait_message_chunks_update_one_bubble_and_finalize_once():
	async def check():
		agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
		agent.on_response = Mock()
		agent.on_response_update = Mock()
		tts_manager = SimpleNamespace(speak=AsyncMock())
		agent.tts_manager = cast(Any, tts_manager)
		chunks = asyncio.Queue()
		chunks.put_nowait("Sure, I")
		chunks.put_nowait("can help with that.")
		chunks.put_nowait(None)

		await agent._speak_wait_message_chunks(chunks)

		assert [entry.args[0] for entry in agent.on_response_update.call_args_list] == [
			"Sure, I",
			"Sure, I can help with that.",
		]
		agent.on_response.assert_called_once_with("Sure, I can help with that.\n")
		assert [entry.args[0] for entry in tts_manager.speak.await_args_list] == [
			"Sure, I", "can help with that.",
		]
	asyncio.run(check())


def test_clear_conversation_removes_history_and_pending_confirmation():
	agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
	agent.conversation_history = [
		{"role": "user", "content": "Open the windows."},
		{"role": "assistant", "content": "Would you like me to?"},
	]
	agent.pending_confirmation = object()
	agent.cancel_voice_response = Mock()
	agent.clear_pending_music_selection = Mock()

	agent.clear_conversation()

	assert agent.conversation_history == []
	assert agent.pending_confirmation is None
	agent.cancel_voice_response.assert_called_once_with()
	agent.clear_pending_music_selection.assert_called_once_with()


def test_pre_classify_failure_routes_to_full_classifier():
	async def check():
		classifier = RequestPreClassifier.__new__(RequestPreClassifier)
		classifier.streaming = True
		classifier.model = "fast-model"
		classifier.on_usage = Mock()
		classifier.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
			create=AsyncMock(side_effect=RuntimeError("offline")),
		)))
		updates = []

		result = await classifier.execute(
			PreClassificationInput("Set the temperature", [], 0),
			on_update=updates.append,
		)

		assert result.requires_action_classification is True
		assert result.response_source is ResponseSource.GENERAL
		assert updates == [
			PreClassificationUpdate(requires_action_classification=True),
		]
	asyncio.run(check())


def test_pre_classify_empty_json_routes_to_full_classifier():
	async def check():
		classifier = RequestPreClassifier.__new__(RequestPreClassifier)
		classifier.streaming = True
		classifier.model = "fast-model"
		classifier.on_usage = Mock()
		class Stream:
			def __init__(self):
				self.chunks = iter(["{}"])

			def __aiter__(self):
				return self

			async def __anext__(self):
				try:
					text = next(self.chunks)
				except StopIteration:
					raise StopAsyncIteration
				return SimpleNamespace(
					choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None,
				)

			async def close(self):
				pass

		classifier.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
			create=AsyncMock(return_value=Stream()),
		)))
		updates = []

		result = await classifier.execute(
			PreClassificationInput("Hello", [], 0), on_update=updates.append,
		)

		assert result.requires_action_classification is True
		assert updates == [PreClassificationUpdate(requires_action_classification=True)]
	asyncio.run(check())


def test_invalid_wait_phrase_preserves_vehicle_manual_route():
	async def check():
		classifier = RequestPreClassifier.__new__(RequestPreClassifier)
		classifier.streaming = True
		classifier.model = "fast-model"
		classifier.on_usage = Mock()
		chunks = [
			'{"requires_action_classification":true,',
			'"response_source":"vehicle_manual",',
			'"wait_message":"Checking how to activate cruise control for your vehicle."}',
		]
		class Stream:
			def __init__(self):
				self.chunks = iter(chunks)

			def __aiter__(self):
				return self

			async def __anext__(self):
				try:
					text = next(self.chunks)
				except StopIteration:
					raise StopAsyncIteration
				return SimpleNamespace(
					choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None,
				)

			async def close(self):
				pass

		classifier.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
			create=AsyncMock(return_value=Stream()),
		)))
		updates = []

		result = await classifier.execute(
			PreClassificationInput("How do I activate cruise control?", [], 0),
			on_update=updates.append,
		)

		assert result.requires_action_classification is True
		assert result.response_source is ResponseSource.VEHICLE_MANUAL
		assert result.wait_message == ""
		assert updates == [
			PreClassificationUpdate(requires_action_classification=True),
			PreClassificationUpdate(response_source=ResponseSource.VEHICLE_MANUAL),
		]
	asyncio.run(check())


def test_crewai_counts_individual_calls_and_residual_time():
	async def check():
		backend = LLMBackend.__new__(LLMBackend)
		backend.opt = ConfigAssistant(ollama_model="test-model")
		backend.logger = Mock()
		context = LLMCallHookContext(llm=SimpleNamespace(model="test-model"), response="Done")
		result = object()

		def generate():
			for _ in range(2):
				_before_timed_llm_call(context)
				_after_timed_llm_call(context)
			return result

		async def kickoff(*, inputs):
			return await asyncio.to_thread(generate)

		crew = SimpleNamespace(kickoff_async=kickoff)
		with patch("agents.llm_backend.time.perf_counter", side_effect=[10.0, 10.1, 10.3, 10.4, 10.7, 11.0]):
			assert await backend._kickoff_timed(crew, {}, "notification decision") is result
		assert _crew_llm_timings.get() is None
		calls = backend.logger.info.call_args_list
		assert [call.kwargs["duration_ms"] for call in calls] == [200.0, 300.0, 1000.0]
		assert [call.kwargs["call_index"] for call in calls[:-1]] == [1, 2]
		assert calls[-1].kwargs["llm_calls"] == 2
		assert calls[-1].kwargs["completed_llm_calls"] == 2
		assert calls[-1].kwargs["llm_duration_ms"] == 500.0
		assert calls[-1].kwargs["crew_overhead_ms"] == 500.0
	asyncio.run(check())


def test_crewai_timing_preserves_errors_and_reports_unfinished_calls():
	async def check():
		backend = LLMBackend.__new__(LLMBackend)
		backend.opt = ConfigAssistant()
		backend.logger = Mock()
		context = LLMCallHookContext(llm=SimpleNamespace(model="test-model"))

		async def kickoff(*, inputs):
			_before_timed_llm_call(context)
			raise RuntimeError("Model failed")

		with pytest.raises(RuntimeError, match="Model failed"):
			await backend._kickoff_timed(SimpleNamespace(kickoff_async=kickoff), {}, "test call")
		assert _crew_llm_timings.get() is None
		fields = backend.logger.info.call_args.kwargs
		assert fields["llm_calls"] == 1
		assert fields["completed_llm_calls"] == 0
		assert fields["crew_overhead_ms"] is None
		assert fields["success"] is False
	asyncio.run(check())


def test_crewai_timing_observes_real_executor_without_duplicate_calls():
	async def check():
		def respond(request):
			payload = json.loads(request.content)
			assert not payload.get("stream", False)
			return httpx.Response(200, json={
				"id": "test", "object": "chat.completion", "created": 1, "model": "test-model",
				"choices": [{"index": 0, "message": {
					"role": "assistant", "content": "Thought: Ready.\nFinal Answer: Done",
				}, "finish_reason": "stop"}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
			})

		backend = LLMBackend.__new__(LLMBackend)
		backend.opt = ConfigAssistant(ollama_model="test-model")
		backend.logger = Mock()
		llm = LLM(model="ollama/test-model")
		with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
			llm._client = client
			agent = Agent(role="Test agent", goal="Reply", backstory="Test", llm=llm, verbose=False)
			task = Task(description="Reply Done", expected_output="Done", agent=agent)
			crew = Crew(agents=[agent], tasks=[task], verbose=False, tracing=False)
			result = await backend._kickoff_timed(crew, {}, "test call")
		assert result.raw == "Done"
		calls = backend.logger.info.call_args_list
		assert len(calls) == 2
		assert calls[0].args[0] == "[LLM] CrewAI call completed"
		assert calls[0].kwargs["call_index"] == 1
		assert calls[-1].kwargs["llm_calls"] == 1
		assert calls[-1].kwargs["completed_llm_calls"] == 1
		assert calls[-1].kwargs["crew_overhead_ms"] is not None
	asyncio.run(check())


def test_conversation_logs_time_to_first_nonempty_token_once():
	async def check():
		class Stream:
			async def __aiter__(self):
				for text in [None, "", "[[tone:empathetic]]", "Hello", " world"]:
					yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

			async def close(self):
				pass

		agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
		agent.opt = ConfigAssistant(ollama_model="test-model")
		agent.skill_manager = SimpleNamespace(get_skill=Mock(return_value="Conversation"))
		agent.classify_conversation_tone = AsyncMock(return_value=ToneType.CALM)
		agent.conversation_history = []
		agent.voice_llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
			create=AsyncMock(return_value=Stream()),
		)))
		agent.set_assistant_status = Mock()
		agent.on_speaking_tone_changed = None
		tone_updates = []
		agent.on_speaking_tone_changed = tone_updates.append
		agent.on_response_update = None
		agent.on_response = None
		agent.tts_manager = None
		agent.log_llm_usage = Mock()
		logger = Mock()
		with patch("agents.automotive_agent.logger", logger), patch(
			"agents.automotive_agent.time.perf_counter", side_effect=[10.0, 10.125]
		):
			response = await agent.stream_user_response(
				CarEvent(SkillType.CONVERSATION, "user_input", "Hi", [], "Hi"),
			)
		assert response == "Hello world"
		assert tone_updates == [ToneType.EMPATHETIC.value, None]
		agent.classify_conversation_tone.assert_not_awaited()
		prompt = agent.voice_llm.chat.completions.create.call_args.kwargs["messages"][0]["content"]
		assert "[[tone:<value>]]" in prompt
		ttft_calls = [call for call in logger.info.call_args_list if call.args[0] == "[LLM] Time to first token"]
		assert len(ttft_calls) == 1
		assert ttft_calls[0].kwargs == {
			"llm_call": "conversation response", "model": "test-model", "ttft_ms": 125.0,
		}
	asyncio.run(check())


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
	agent.pending_confirmation = None
	agent.voice_llm = SimpleNamespace()
	agent.conversation_history = []
	agent.knowledge_facts_provider = lambda: {}
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
		await agent.process_direct_user_input(CarEvent(SkillType.CONVERSATION, "user_input", "Open windows", [], "Open windows"))
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
		"[LLM] Token usage",
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
		ActionType.ASK_PERMISSION_TO_TALK,
		ActionType.POSTPONE_NOTIFICATION_DELIVERY,
		ActionType.READ_PENDING_MESSAGES,
		ActionType.PROPOSE_MUSIC,
		ActionType.EVALUATE_MUSIC_PROPOSAL,
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
	agent.knowledge_facts_provider = lambda: {"MeetingState.scheduled": "The driver is late for a meeting."}
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
def test_confirmation_crew_output_is_validated(content, expected):
	async def check():
		backend = LLMBackend.__new__(LLMBackend)
		backend.logger = Mock()
		if content == '{"confirmed": true}':
			backend.meeting_confirmation_crew = SimpleNamespace(
				kickoff_async=AsyncMock(return_value=SimpleNamespace(pydantic=ConfirmationReply(confirmed=True))))
		elif content == '{"confirmed": false}':
			backend.meeting_confirmation_crew = SimpleNamespace(
				kickoff_async=AsyncMock(return_value=SimpleNamespace(pydantic=ConfirmationReply(confirmed=False))))
		else:
			backend.meeting_confirmation_crew = SimpleNamespace(
				kickoff_async=AsyncMock(side_effect=ValueError("unparseable crew output")))
		result = await backend.interpret_confirmation("yes, please", [{"role": "assistant", "content": "Should I attend the meeting?"}])
		assert result is expected
		if expected is not None:
			backend.meeting_confirmation_crew.kickoff_async.assert_awaited_once_with(inputs={
				"conversation_history": json.dumps([{"role": "assistant", "content": "Should I attend the meeting?"}]),
				"user_input": "yes, please",
			})
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
		agent.execute_music_action = AsyncMock()
		for music_action in MUSIC_ACTIONS:
			agent.action_manager._registered_handlers[music_action] = agent.execute_music_action
		event = CarEvent(SkillType.CONVERSATION, "user_input", "Music command", [], "Music command")
		await agent.process_direct_user_input(event)
		if agent.voice_response_task is not None:
			await agent.voice_response_task
		agent.execute_music_action.assert_awaited_once_with(action, event.user_input)
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
		await agent.process_direct_user_input(event)
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


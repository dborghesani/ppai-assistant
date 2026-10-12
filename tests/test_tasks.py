import pytest
from pydantic import ValidationError

from agents.tasks.routing import TaskRoute, TaskType


def test_telemetry_route_deduplicates_tasks_in_priority_order():
    route = TaskRoute(tasks=[TaskType.WELLBEING, TaskType.DRIVING, TaskType.WELLBEING])
    assert route.tasks == [TaskType.WELLBEING, TaskType.DRIVING]


@pytest.mark.parametrize("task", [
    TaskType.MESSAGES, TaskType.MEETING, TaskType.CONVERSATION,
    TaskType.VEHICLE_MANUAL, TaskType.VEHICLE_COMMAND,
])
def test_telemetry_route_rejects_manual_and_direct_scenarios(task):
    with pytest.raises(ValidationError):
        TaskRoute(tasks=[task])


def test_telemetry_route_can_be_empty():
    assert TaskRoute(tasks=[]).tasks == []


@pytest.mark.parametrize("selected", [[], [TaskType.WELLBEING], [TaskType.WELLBEING, TaskType.MOOD_MUSIC]])
def test_laya_router_preserves_multiple_task_routes(selected):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.laya_backend import LayaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from config import ConfigAssistant

    answers = {task_type.value: {"choice": "route" if task_type in selected else "skip"}
               for task_type in TelemetryRouter.TASK_CRITERIA}
    backend = Mock(spec=LayaBackend)
    backend.predict = AsyncMock(return_value={"answers": answers})
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)
    changed = {"DriverPhysicalState.fatigue_level": "Fatigue level is high.",
               "DriverEmotionState.angry": "Anger level is high."}
    route = asyncio.run(router.execute(RoutingInput(changed, [{"role": "user", "content": "Old conversation"}])))
    assert route.tasks == selected
    state, questions = backend.predict.await_args.args
    assert state == {"changed_facts": changed}
    assert set(questions) == {task_type.value for task_type in TelemetryRouter.TASK_CRITERIA}
    assert all(question["type"] == "choice" for question in questions.values())
    assert all(set(question["criteria"]) == {"route", "skip"} for question in questions.values())
    assert "messages" not in questions and "meeting" not in questions


@pytest.mark.parametrize("result", [None, {}, {"answers": {}}, {"answers": {
    "wellbeing": {"choice": "invented"},
}}])
def test_laya_router_rejects_invalid_or_incomplete_answers(result):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.laya_backend import LayaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from config import ConfigAssistant

    backend = Mock(spec=LayaBackend)
    backend.predict = AsyncMock(return_value=result)
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)
    with pytest.raises(ValueError):
        asyncio.run(router.execute(RoutingInput({"VehicleMotion.speed": "Speed is high."})))


def test_laya_router_skips_inference_for_empty_changes():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.laya_backend import LayaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from config import ConfigAssistant

    backend = Mock(spec=LayaBackend)
    backend.predict = AsyncMock()
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)
    assert asyncio.run(router.execute(RoutingInput({}))).tasks == []
    backend.predict.assert_not_awaited()


@pytest.mark.parametrize("confirmed", [True, False, None])
def test_meeting_reply_context_is_owned_by_meeting_task(confirmed):
    import json
    from agents.tasks.meeting import meeting_reply_context

    instructions, reference = meeting_reply_context(confirmed)
    assert "Never invent meeting details or completed actions." in instructions
    assert json.loads(reference.removeprefix("Meeting confirmation result: ")) == {"confirmed": confirmed}


def test_crewai_backend_executes_supplied_crew_and_rejects_wrong_output():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from pydantic import BaseModel
    from agents.backends.crewai_backend import CrewAIBackend
    from config import ConfigAssistant

    class Result(BaseModel):
        value: str

    usage = Mock()
    backend = CrewAIBackend(ConfigAssistant(), usage)
    crew = SimpleNamespace(kickoff_async=AsyncMock(return_value=SimpleNamespace(pydantic=Result(value="ok"))))
    assert asyncio.run(backend.run(crew, {"input": "data"}, Result, "test")).value == "ok"
    crew.kickoff_async.assert_awaited_once_with(inputs={"input": "data"})
    crew.kickoff_async.return_value = SimpleNamespace(pydantic=None)
    with pytest.raises(ValueError, match="did not return"):
        asyncio.run(backend.run(crew, {}, Result, "test"))
    assert usage.call_count == 2


def test_ollama_backend_closes_stream_and_reports_usage_on_early_exit():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.ollama_backend import OllamaBackend
    from config import ConfigAssistant

    class Stream:
        close = AsyncMock()

        async def __aiter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))], usage=None)

    stream = Stream()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))))
    usage = Mock()
    backend = OllamaBackend(ConfigAssistant(), usage, client=client)

    async def check():
        response = backend.stream(call_name="test", model="test", messages=[])
        await anext(response)
        await response.aclose()

    asyncio.run(check())
    stream.close.assert_awaited_once()
    usage.assert_called_once_with("test", None)


@pytest.mark.parametrize("streaming", [True, False])
def test_preclassifier_uses_injected_backend(streaming):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.crewai_backend import CrewAIBackend
    from agents.backends.ollama_backend import OllamaBackend
    from agents.tasks.pre_classify import PreClassification, PreClassificationInput, RequestPreClassifier
    from config import ConfigAssistant

    usage = Mock()
    opt = ConfigAssistant()
    if streaming:
        class Stream:
            close = AsyncMock()

            async def __aiter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=
                    '{"requires_action_classification":false,"response_source":"general","wait_message":""}'))], usage=None)

        stream = Stream()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))))
        backend = OllamaBackend(opt, usage, client=client)
        task = RequestPreClassifier(opt, usage, backend=backend)
    else:
        backend = CrewAIBackend(opt, usage)
        task = RequestPreClassifier(opt, usage, streaming=False, backend=backend)
        task.crew = SimpleNamespace(kickoff_async=AsyncMock(return_value=SimpleNamespace(
            pydantic=PreClassification(requires_action_classification=False),
        )))
    result = asyncio.run(task.execute(PreClassificationInput("Hello", [], 0)))
    assert result.requires_action_classification is False
    usage.assert_called_once()


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("source", ["general", "vehicle_state", "vehicle_manual"])
def test_preclassifier_with_laya_preserves_contract_and_updates(required, source):
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from agents.backends.laya_backend import LayaBackend
    from agents.tasks.pre_classify import (
        RequestPreClassifier, PreClassificationInput, PreClassificationUpdate, ResponseSource,
    )
    from config import ConfigAssistant

    backend = Mock(spec=LayaBackend)
    backend.predict = AsyncMock(return_value={"answers": {
        "requires_action_classification": {"choice": "required" if required else "not_required"},
        "response_source": {"choice": source},
    }})
    task = RequestPreClassifier(ConfigAssistant(), Mock(), backend=backend)
    history = [{"role": "user", "content": str(index)} for index in range(8)]
    updates = []
    result = asyncio.run(task.execute(PreClassificationInput("Current turn", history, 2), on_update=updates.append))
    assert result.requires_action_classification is required
    assert result.response_source is ResponseSource(source)
    assert result.wait_message == ""
    assert updates == [
        PreClassificationUpdate(requires_action_classification=required),
        PreClassificationUpdate(response_source=ResponseSource(source)),
    ] + ([PreClassificationUpdate(wait_message_chunk="")] if required else [])
    state, questions = backend.predict.await_args.args
    assert state == {"user_input": "Current turn", "conversation_history": history[-6:], "pending_message_count": 2}
    assert set(questions) == {"requires_action_classification", "response_source"}
    assert set(questions["response_source"]["criteria"]) == {"general", "vehicle_state", "vehicle_manual"}


@pytest.mark.parametrize("result", [None, {}, {"answers": {}}, {"answers": {
    "requires_action_classification": {"choice": "not_required"},
    "response_source": {"choice": "invented"},
}}, {"answers": {
    "requires_action_classification": {"choice": "invented"},
    "response_source": {"choice": "general"},
}}])
def test_preclassifier_with_laya_invalid_answers_fall_back_before_updates(result):
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from agents.backends.laya_backend import LayaBackend
    from agents.tasks.pre_classify import RequestPreClassifier, PreClassificationInput
    from config import ConfigAssistant

    backend = Mock(spec=LayaBackend)
    backend.predict = AsyncMock(return_value=result)
    task = RequestPreClassifier(ConfigAssistant(), Mock(), backend=backend)
    updates = []
    output = asyncio.run(task.execute(PreClassificationInput("Ambiguous turn", [], 0), on_update=updates.append))
    assert output.requires_action_classification is True
    assert output.response_source.value == "general"
    assert updates[0].requires_action_classification is True


@pytest.mark.parametrize("routing_backend", ["llm", "laya"])
def test_agent_reuses_laya_backend_when_models_match(routing_backend):
    from unittest.mock import AsyncMock, Mock, patch
    from agents.backends.laya_backend import LayaBackend
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    opt = ConfigAssistant(routing_backend=routing_backend, pre_classify_backend="laya")
    backend = Mock(spec=LayaBackend)
    backend.predict = AsyncMock()
    with patch("agents.automotive_agent.LayaBackend", return_value=backend) as load:
        agent = AutomotiveAgent(opt)
    assert agent.request_pre_classifier.backend is backend
    assert all(task.backend is agent.crewai_backend for task in agent.tasks.values())
    assert load.call_count == 1
    if routing_backend == "laya":
        assert agent.telemetry_router.backend is backend
        assert agent.telemetry_router.backend is agent.request_pre_classifier.backend
    else:
        assert agent.telemetry_router.backend is agent.ollama_backend


def test_agent_rejects_unknown_preclassification_backend():
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    with pytest.raises(ValueError, match="pre_classify_backend"):
        AutomotiveAgent(ConfigAssistant(pre_classify_backend="unknown"))


def test_agent_keeps_distinct_laya_checkpoints_separate():
    from unittest.mock import AsyncMock, Mock, patch
    from agents.backends.laya_backend import LayaBackend
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    opt = ConfigAssistant(routing_backend="laya", pre_classify_backend="laya",
                          routing_model_laya="router-model", pre_classify_model_laya="request-model")
    router_backend, request_backend = Mock(spec=LayaBackend), Mock(spec=LayaBackend)
    router_backend.predict = AsyncMock()
    request_backend.predict = AsyncMock()
    with patch("agents.automotive_agent.LayaBackend", side_effect=[router_backend, request_backend]) as load:
        agent = AutomotiveAgent(opt)
    assert agent.telemetry_router.backend is router_backend
    assert agent.request_pre_classifier.backend is request_backend
    assert [call.args[0] for call in load.call_args_list] == ["router-model", "request-model"]


def test_laya_preclassification_uses_agent_holding_phrase_fallback():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.laya_backend import LayaBackend
    from agents.tasks.pre_classify import RequestPreClassifier
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.action_manager.registered_action_options = ""
        agent.action_manager.registered_action_guidance = ""
        agent.command_task = SimpleNamespace(
            direct_action_options="vehicle options", vehicle_action_guidance="vehicle reference",
            decision_options={"skill_options": "skills", "tone_options": "tones"},
            execute=AsyncMock(return_value=agent.tasks[TaskType.WELLBEING].execute.return_value),
        )
        backend = Mock(spec=LayaBackend)
        backend.predict = AsyncMock(return_value={"answers": {
            "requires_action_classification": {"choice": "required"},
            "response_source": {"choice": "general"},
        }})
        agent.request_pre_classifier = RequestPreClassifier(agent.opt, Mock(), backend=backend)
        wait_messages = []

        async def speak_wait(chunks):
            while (chunk := await chunks.get()) is not None:
                if chunk.strip():
                    wait_messages.append(chunk)

        async def prepare_response(event, source, *, response_gate):
            await response_gate.wait()

        agent._speak_wait_message_chunks = AsyncMock(side_effect=speak_wait)
        agent.prepare_user_response = AsyncMock(side_effect=prepare_response)
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Open the windows", [], "Open the windows")
        preparation = await agent.preclassify_direct_input(event, 0)
        await preparation.classifier_task
        await preparation.wait_task
        assert wait_messages == ["Hm."]
        assert preparation.pre_classification.wait_message == ""
        preparation.response_gate.set()
        await preparation.response_task

    asyncio.run(check())


@pytest.mark.parametrize("requires_action", [False, True])
def test_agent_owns_preclassification_orchestration(requires_action):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from agents.tasks.pre_classify import PreClassification, PreClassificationUpdate, ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = [{"role": "user", "content": "Previous turn"}]
        agent.action_manager.registered_action_options = "registered options"
        agent.action_manager.registered_action_guidance = "registered guidance"
        classifier_started, response_started = asyncio.Event(), asyncio.Event()
        events, wait_messages = [], []

        async def classify(inputs):
            events.append("classifier_started")
            classifier_started.set()
            return agent.tasks[TaskType.WELLBEING].execute.return_value

        agent.command_task = SimpleNamespace(
            direct_action_options="vehicle options", vehicle_action_guidance="vehicle reference",
            decision_options={"skill_options": "skills", "tone_options": "tones"},
            execute=AsyncMock(side_effect=classify),
        )

        async def prepare_response(event, source, *, response_gate):
            events.append("response_started")
            response_started.set()
            await response_gate.wait()

        async def speak_wait(chunks):
            while (chunk := await chunks.get()) is not None:
                wait_messages.append(chunk)

        async def preclassify(task_input, *, on_update):
            assert task_input.user_input == "Current turn"
            assert task_input.pending_message_count == 2
            assert task_input.history == agent.conversation_history
            on_update(PreClassificationUpdate(requires_action_classification=requires_action))
            if requires_action:
                await asyncio.wait_for(classifier_started.wait(), 1)
            on_update(PreClassificationUpdate(response_source=ResponseSource.VEHICLE_MANUAL))
            await asyncio.wait_for(response_started.wait(), 1)
            if requires_action:
                on_update(PreClassificationUpdate(wait_message_chunk="Checking."))
            events.append("preclassification_finished")
            return PreClassification(requires_action_classification=requires_action,
                                     response_source=ResponseSource.VEHICLE_MANUAL,
                                     wait_message="Checking." if requires_action else "")

        agent.prepare_user_response = AsyncMock(side_effect=prepare_response)
        agent._speak_wait_message_chunks = AsyncMock(side_effect=speak_wait)
        agent.request_pre_classifier = SimpleNamespace(execute=AsyncMock(side_effect=preclassify))
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Current turn", [], "Current turn")
        preparation = await agent.preclassify_direct_input(event, 2)
        assert not preparation.response_gate.is_set()
        assert not preparation.response_task.done()
        assert events[-1] == "preclassification_finished"
        if requires_action:
            await preparation.classifier_task
            await preparation.wait_task
            assert events.index("classifier_started") < events.index("preclassification_finished")
            assert wait_messages == ["Checking."]
            assert "read_pending_messages" in agent.command_task.execute.await_args.args[0]["action_options"]
        else:
            assert preparation.classifier_task is None
            assert preparation.wait_task is None
            agent.command_task.execute.assert_not_awaited()
            agent._speak_wait_message_chunks.assert_not_awaited()
        preparation.response_gate.set()
        await preparation.response_task

    asyncio.run(check())


def test_laya_backend_only_predicts_with_supplied_state_and_questions():
    import asyncio
    from unittest.mock import Mock
    from agents.backends.laya_backend import LayaBackend
    from config import ConfigAssistant

    predictor = Mock()
    predictor.predict.return_value = {"answers": {"task": {"choice": "wellbeing"}}}
    usage = Mock()
    backend = LayaBackend(None, usage, predictor=predictor)
    state, questions = {"fatigue": "high"}, {"task": {"type": "choice"}}
    assert asyncio.run(backend.predict(state, questions)) == predictor.predict.return_value
    predictor.predict.assert_called_once_with(state, questions)
    assert not hasattr(backend, "agent")
    assert not hasattr(backend, "process_event")
    usage.assert_called_once_with("laya prediction", None)


@pytest.mark.parametrize("module_name,class_name", [
    ("wellbeing", "WellbeingTask"), ("driving", "DrivingTask"),
    ("comfort", "ComfortTask"), ("mood_music", "MoodMusicTask"),
    ("messages", "MessagesTask"), ("meeting", "MeetingTask"),
])
def test_specialist_task_executes_without_backend_and_validates_capabilities(module_name, class_name):
    import asyncio
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.tasks.decision import DecisionInput, NotificationDecision
    from config import ConfigAssistant
    from data.agents_dataclasses import ActionType, InterventionType, SuggestionType, ToneType, UrgencyType

    task_class = getattr(importlib.import_module(f"agents.tasks.{module_name}"), class_name)
    task = task_class(ConfigAssistant(), Mock())
    decision = NotificationDecision(
        urgency=UrgencyType.NONE, tone=ToneType.DISCREET,
        intervention_type=InterventionType.NONE, action=ActionType.NONE,
        suggestion_type=SuggestionType.NONE, reason="Routine.",
    )
    task.crew = SimpleNamespace(kickoff_async=AsyncMock(return_value=SimpleNamespace(pydantic=decision)))
    inputs = DecisionInput("knowledge_updated", "David", 2, "event data", "changed data", "supporting data")
    assert asyncio.run(task.execute(inputs)) is decision
    crew_inputs = task.crew.kickoff_async.await_args.kwargs["inputs"]
    assert crew_inputs["event_name"] == "knowledge_updated"
    assert crew_inputs["driver_name"] == "David"
    assert crew_inputs["pending_message_count"] == 2
    assert crew_inputs["event_details"] == "event data"
    assert crew_inputs["changed_facts"] == "changed data"
    assert crew_inputs["context"] == "supporting data"
    assert crew_inputs["task_instructions"] == task.spec.instructions
    assert not hasattr(task, "run_decision")
    decision.action = ActionType.OPEN_WINDOWS
    with pytest.raises(ValueError, match="outside its contract"):
        asyncio.run(task.execute(inputs))


def test_wellbeing_owns_activities_but_does_not_read_emotions_or_preferences():
    from agents.tasks.wellbeing import TASK

    facts = {
        "DriverPhysicalState.activity": "The driver is using a phone.",
        "DriverPhysicalState.fatigue_level": "Fatigue level is high.",
        "DetectedObjects.children_inside": "One child is inside.",
        "VehicleMotion.speed": "Vehicle speed is optimal.",
        "DriverEmotionState.angry": "Anger level is high.",
        "DriverPreferences.preferred_music": "Rock music.",
    }
    assert set(TASK.select_facts(facts)) == {
        "DriverPhysicalState.activity", "DriverPhysicalState.fatigue_level",
        "DetectedObjects.children_inside", "VehicleMotion.speed",
    }


def test_mood_task_cannot_apply_adas_or_read_fatigue():
    from agents.tasks.mood_music import TASK
    from data.agents_dataclasses import ActionType

    assert ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE not in TASK.actions
    assert TASK.select_facts({"DriverPhysicalState.fatigue_level": "Fatigue is high."}) == {}


def test_agent_builds_specialist_tasks_without_connecting_to_ollama():
    from agents.automotive_agent import DECISION_TASKS, AutomotiveAgent
    from config import ConfigAssistant

    agent = AutomotiveAgent(ConfigAssistant())
    assert set(agent.tasks) == set(DECISION_TASKS)
    assert agent.telemetry_router.model == "qwen2.5:3b-instruct"
    assert "Vehicle action reference" in agent.command_task.crew.tasks[0].description
    assert agent.command_task.backend is agent.crewai_backend
    assert not hasattr(agent, "process_event_llm")
    assert not hasattr(agent, "process_event_laya")


def test_agent_selects_laya_router_without_changing_specialist_backends():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock, patch
    from agents.backends.laya_backend import LayaBackend
    from agents.automotive_agent import AutomotiveAgent
    from agents.tasks.routing import TelemetryRouter
    from config import ConfigAssistant

    opt = ConfigAssistant(routing_backend="laya", routing_model_laya="router-model",
                          laya_custom_model="old-decision-model")
    backend = Mock(spec=LayaBackend)
    backend.predict = AsyncMock()
    with patch("agents.automotive_agent.LayaBackend", return_value=backend) as build_backend:
        agent = AutomotiveAgent(opt)
    assert isinstance(agent.telemetry_router, TelemetryRouter)
    assert agent.telemetry_router.backend is backend
    assert all(task.backend is agent.crewai_backend for task in agent.tasks.values())
    assert build_backend.call_args.args[0] == "router-model"
    assert opt.laya_custom_model == "old-decision-model"


@pytest.mark.parametrize("model_kind", ["default", "path", "string"])
def test_laya_backend_loads_model_without_assistant_configuration(tmp_path, model_kind):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from agents.backends.laya_backend import LayaBackend

    predictor = Mock()
    load = Mock(return_value=predictor)
    model = None if model_kind == "default" else tmp_path if model_kind == "path" else str(tmp_path)
    with patch.dict("sys.modules", {"laya": SimpleNamespace(load=load)}):
        backend = LayaBackend(model, Mock())
    assert backend.predictor is predictor
    if model_kind == "default":
        load.assert_called_once_with("convaiinnovations/laya", subfolder="typed-decisions")
    else:
        load.assert_called_once_with(tmp_path, subfolder=None)


def test_laya_backend_rejects_missing_local_model(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from agents.backends.laya_backend import LayaBackend

    load = Mock()
    with patch.dict("sys.modules", {"laya": SimpleNamespace(load=load)}):
        with pytest.raises(FileNotFoundError, match="Custom Laya model directory"):
            LayaBackend(tmp_path / "missing", Mock())
    load.assert_not_called()


def test_agent_rejects_unknown_routing_backend():
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    with pytest.raises(ValueError, match="routing_backend"):
        AutomotiveAgent(ConfigAssistant(routing_backend="unknown"))


def test_routing_configuration_can_be_selected_through_omegaconf():
    from omegaconf import OmegaConf
    from config import ConfigAssistant

    configuration = OmegaConf.merge(
        OmegaConf.structured(ConfigAssistant),
        OmegaConf.from_dotlist(["routing_backend=laya", "routing_model_laya=/tmp/router-checkpoint",
                       "pre_classify_model_laya=/tmp/request-checkpoint", "routing_model=router-llm"]),
    )
    opt = OmegaConf.to_object(configuration)
    assert opt.routing_backend == "laya"
    assert opt.routing_model_laya == "/tmp/router-checkpoint"
    assert opt.pre_classify_model_laya == "/tmp/request-checkpoint"
    assert opt.routing_model == "router-llm"
    assert not hasattr(opt, "use_laya")


@pytest.mark.parametrize("routing_model,expected", [
    ("router-model", "router-model"), ("ollama/router-model", "router-model"), (None, "main-model"),
])
def test_telemetry_router_uses_its_own_llm_model(routing_model, expected):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.ollama_backend import OllamaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from agents.tasks.pre_classify import RequestPreClassifier
    from config import ConfigAssistant

    opt = ConfigAssistant(ollama_model="main-model", pre_classify_model="request-model", routing_model=routing_model)
    backend = Mock(spec=OllamaBackend)
    backend.complete = AsyncMock(return_value=SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content='{"tasks": []}')),
    ]))
    router = TelemetryRouter(opt, Mock(), backend=backend)
    preclassifier = RequestPreClassifier(opt, Mock(), backend=backend)
    assert asyncio.run(router.execute(RoutingInput({"VehicleMotion.speed": "Speed is high."}))).tasks == []
    assert backend.complete.await_args.kwargs["model"] == expected
    assert preclassifier.model == "request-model"


def make_routed_agent(task_types):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.automotive_agent import AutomotiveAgent
    from agents.tasks.decision import NotificationDecision
    from data.agents_dataclasses import ActionType, InterventionType, SuggestionType, ToneType, UrgencyType

    agent = AutomotiveAgent.__new__(AutomotiveAgent)
    agent.logger = Mock()
    agent.pending_confirmation = None
    agent.knowledge_facts_provider = lambda: {}
    agent.action_manager = SimpleNamespace(message_manager=SimpleNamespace(
        pending_count=0, simulation_stopped=Mock(return_value=False),
    ))
    agent.telemetry_router = SimpleNamespace(execute=AsyncMock(return_value=TaskRoute(tasks=task_types)))
    agent.tasks = {task_type: SimpleNamespace(execute=AsyncMock(return_value=NotificationDecision(
            urgency=UrgencyType.NONE, tone=ToneType.DISCREET,
            intervention_type=InterventionType.NONE, action=ActionType.NONE,
            suggestion_type=SuggestionType.NONE, reason="Routine facts.",
        ))) for task_type in [TaskType.WELLBEING, TaskType.MEETING, TaskType.MESSAGES]}
    agent.handle_notification_decision = AsyncMock()
    return agent


@pytest.mark.parametrize("user_input", ["Hello", ""])
def test_process_event_dispatches_by_input_without_global_backend_mode(user_input):
    import asyncio
    from unittest.mock import AsyncMock
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    agent = make_routed_agent([])
    agent.process_direct_user_input = AsyncMock()
    agent.process_car_event = AsyncMock()
    event = CarEvent(SkillType.NONE, EventName.USER_INPUT if user_input else EventName.KNOWLEDGE_UPDATED,
                     {}, [], user_input)
    asyncio.run(agent.process_event(event))
    selected = agent.process_direct_user_input if user_input else agent.process_car_event
    other = agent.process_car_event if user_input else agent.process_direct_user_input
    selected.assert_awaited_once_with(event)
    other.assert_not_awaited()


def test_process_event_skips_empty_pending_message_reminders():
    import asyncio
    from unittest.mock import AsyncMock
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    agent = make_routed_agent([])
    agent.process_car_event = AsyncMock()
    event = CarEvent(SkillType.NONE, EventName.PENDING_MESSAGES_REMINDER, {}, [])
    asyncio.run(agent.process_event(event))
    agent.process_car_event.assert_not_awaited()


def test_automotive_agent_dispatches_with_only_task_context():
    import asyncio
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    agent = make_routed_agent([TaskType.WELLBEING])
    event = CarEvent(SkillType.NONE, EventName.KNOWLEDGE_UPDATED,
        {"DriverPhysicalState.activity": "On the phone"}, [], facts={
            "DriverPhysicalState.activity": "The driver is using a phone.",
            "VehicleMotion.speed": "The vehicle is moving.",
            "DriverEmotionState.angry": "Anger level is high.",
            "DriverPreferences.preferred_music": "Rock music.",
        })
    asyncio.run(agent.process_car_event(event))
    inputs = agent.tasks[TaskType.WELLBEING].execute.await_args.args[0]
    assert inputs.changed_facts == "- The driver is using a phone."
    assert inputs.context == "- The vehicle is moving."
    assert "Rock" not in str(inputs)


def test_manual_meeting_scenario_bypasses_router():
    import asyncio
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    agent = make_routed_agent([])
    event = CarEvent(SkillType.NONE, EventName.KNOWLEDGE_UPDATED,
        {"DriverAgenda.late_for_meeting": True}, [], facts={
            "DriverAgenda.late_for_meeting": "The driver is late for an upcoming meeting.",
        })
    asyncio.run(agent.process_car_event(event))
    agent.telemetry_router.execute.assert_not_awaited()
    agent.tasks[TaskType.MEETING].execute.assert_awaited_once()


def test_conversation_task_streams_without_agent_and_hides_tone_metadata():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock, patch
    from agents.tasks.conversation import ConversationDelivery, ConversationInput, ConversationTask
    from config import ConfigAssistant
    from data.agents_dataclasses import ToneType

    class Stream:
        closed = False

        def __aiter__(self):
            return self.chunks()

        async def chunks(self):
            for text in ["", "[[to", "ne:empathetic]]", "Hello. ", "Take care."]:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)

        async def close(self):
            self.closed = True

    stream = Stream()
    create = AsyncMock(return_value=stream)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    speak, on_text, on_tone = AsyncMock(), Mock(), Mock()
    logger = Mock()
    with patch("structlog.get_logger", return_value=logger):
        task = ConversationTask(ConfigAssistant(), Mock(), client=client)
    result = asyncio.run(task.execute(
        ConversationInput("Hello", [], []), ConversationDelivery(speak, on_text, on_tone, Mock()),
    ))
    assert result.text == "Hello. Take care."
    assert result.tone is ToneType.EMPATHETIC
    assert speak.await_count == 2
    assert "[[tone:" not in str(speak.await_args_list)
    assert stream.closed
    create.assert_awaited_once()
    ttft_calls = [call for call in logger.info.call_args_list if call.args[0] == "[LLM] Time to first token"]
    assert len(ttft_calls) == 1
    assert ttft_calls[0].kwargs["llm_call"] == "conversation response"


def test_knowledge_publishes_one_unscoped_snapshot_with_motion():
    import asyncio
    from managers.knowledge_manager import KnowledgeManager
    from data.agents_dataclasses import SkillType

    async def check():
        manager = KnowledgeManager.__new__(KnowledgeManager)
        manager.context = {
            "DriverPhysicalState.fatigue_level": "Fatigue level is high.",
            "DriverEmotionState.angry": "Anger level is high.",
        }
        manager._raw_values = {"VehicleMotion.speed": 30.0}
        manager.knowledge_event_queue = asyncio.Queue()
        changes = {"DriverPhysicalState.fatigue_level": 0.7, "DriverEmotionState.angry": 0.7}
        await manager._notify_agent(changes)
        event = manager.knowledge_event_queue.get_nowait()
        assert event.event_value == changes
        assert event.skill is SkillType.NONE
        assert event.facts["VehicleMotion.moving"] == "The vehicle is moving."
        assert manager.knowledge_event_queue.empty()
        manager.context.clear()
        assert "DriverEmotionState.angry" in event.facts

    asyncio.run(check())
import pytest
from pydantic import ValidationError

from agents.tasks.routing import TaskRoute, TaskType


def test_llm_preclassification_disables_reasoning_for_short_json_budget():
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from agents.tasks.pre_classify import PreClassificationInput, RequestPreClassifier, ResponseSource
    from config import ConfigAssistant

    requests = []

    async def stream(**request):
        requests.append(request)
        content = json.dumps({
            "requires_action_classification": False,
            "response_source": "general",
            "wait_message": "",
        }) if request.get("reasoning_effort") == "none" else ""
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    classifier = RequestPreClassifier(
        ConfigAssistant(pre_classify_backend="llm"), Mock(),
        backend=SimpleNamespace(stream=stream),
    )
    result = asyncio.run(classifier.execute(PreClassificationInput(
        user_input="Hello", history=[], pending_message_count=0,
    )))

    assert not result.requires_action_classification
    assert result.response_source is ResponseSource.GENERAL
    assert requests[0]["reasoning_effort"] == "none"
    assert requests[0]["max_tokens"] == 128


def test_telemetry_route_deduplicates_tasks_in_priority_order():
    route = TaskRoute(tasks=[TaskType.WELLBEING, TaskType.DRIVING, TaskType.WELLBEING])
    assert route.tasks == [TaskType.WELLBEING, TaskType.DRIVING]


@pytest.mark.parametrize("task", [
    TaskType.CONVERSATION, TaskType.VEHICLE_MANUAL, TaskType.VEHICLE_COMMAND,
])
def test_telemetry_route_rejects_manual_scenarios(task):
    with pytest.raises(ValidationError):
        TaskRoute(tasks=[task])


@pytest.mark.parametrize("task", [TaskType.MESSAGES, TaskType.MEETING])
def test_telemetry_route_accepts_direct_scenarios(task):
    assert TaskRoute(tasks=[task]).tasks == [task]


@pytest.mark.parametrize("status,expected", [
    ("none", "No meeting phase is active."),
    ("prepare", "The meeting is being prepared and has not started yet."),
    ("started", "The meeting has started and is in progress."),
    ("finished", "The meeting has finished."),
])
def test_meeting_phase_produces_distinct_knowledge_fact(status, expected):
    from dataclasses import fields
    from data.assistant_dataclasses import DriverAgenda
    from managers.knowledge_manager import KnowledgeManager

    assert DriverAgenda().meeting_status == "none"
    phase = next(item for item in fields(DriverAgenda) if item.name == "meeting_status")
    assert phase.metadata["knowledge"]
    assert phase.metadata["value_kind"] == "category"
    assert KnowledgeManager._CATEGORY_KNOWLEDGE_FORMATTERS[("DriverAgenda", "meeting_status")](status) == expected


@pytest.mark.parametrize("timing,expected", [
    ("none", "No meeting is scheduled."),
    ("on_time", "The driver is on time for the meeting."),
    ("late", "The driver is late for the meeting."),
    ("arrived", "The driver has arrived at the meeting."),
])
def test_meeting_timing_produces_three_distinct_facts(timing, expected):
    from data.assistant_dataclasses import DriverAgenda
    from managers.knowledge_manager import KnowledgeManager

    assert DriverAgenda().meeting_timing == "none"
    formatter = KnowledgeManager._CATEGORY_KNOWLEDGE_FORMATTERS[("DriverAgenda", "meeting_timing")]
    assert formatter(timing) == expected


def test_meeting_preparation_and_simulation_share_planned_agenda():
    from unittest.mock import Mock
    from agents.backends.crewai_backend import CrewAIBackend
    from config import ConfigAssistant
    from sim.meeting_sim import DEFAULT_TOPIC, PLANNED_AGENDA, build_crew, build_preparation_crew, planned_agenda

    backend = CrewAIBackend(ConfigAssistant(), Mock())
    preparation, briefing = build_preparation_crew(backend.llm, DEFAULT_TOPIC, False)
    meeting, summary = build_crew(backend.llm, DEFAULT_TOPIC, False)
    agenda = planned_agenda(DEFAULT_TOPIC)

    assert agenda in briefing.description
    assert agenda in meeting.tasks[0].description
    assert preparation.tasks == [briefing]
    assert len(preparation.agents) == 1
    assert summary is meeting.tasks[-1]
    assert "45 minutes" in agenda
    assert len(meeting.tasks) == 9
    safety_review, architecture_response, close_meeting = meeting.tasks[5:8]
    assert safety_review.agent.role.startswith("Elena Novak")
    assert safety_review.context == meeting.tasks[1:3]
    assert architecture_response.agent.role.startswith("Marco Bellini")
    assert architecture_response.context == [meeting.tasks[2], safety_review]
    assert close_meeting.context == meeting.tasks[:7]
    for owner, minutes, subject in PLANNED_AGENDA:
        assert f"{owner}, {minutes} minutes" in agenda
        assert subject in agenda
    assert "meeting has not started" in briefing.description
    assert "not decisions already made" in agenda
    assert "Driver's Meeting Briefing Assistant" == preparation.agents[0].role
    assert "Address the driver directly as 'you' and 'your meeting'" in briefing.description
    assert "Do not offer to " in briefing.description
    assert "assume they are late" in briefing.description


def test_meeting_permission_prompt_uses_lateness_without_prepare_prerequisite():
    from agents.tasks.meeting import TASK

    assert TASK.context_fields == ("DriverAgenda.meeting_status", "DriverAgenda.meeting_timing")
    assert "Prepare is ONLY an optional briefing" in TASK.instructions
    assert "timing is absent/unknown" in TASK.instructions
    assert 'Prepared meeting + "The driver is on time for the meeting." => silent.' in TASK.instructions
    assert '"No meeting phase is active." => ask_attend_meeting.' in TASK.instructions
    assert "timing-only decision table" in TASK.instructions


@pytest.mark.parametrize("request_briefing", [False, True])
def test_meeting_late_authorization_is_independent_of_optional_prepare(request_briefing):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from managers.action_manager import ActionManager

    async def check():
        manager = ActionManager(SimpleNamespace(set_assistant_status=Mock()))
        manager.attend_meeting = AsyncMock()
        manager.prepare_meeting = AsyncMock()
        manager.set_meeting_timing("late")
        manager.confirm_attend_meeting()
        assert manager._meeting_authorized
        manager.attend_meeting.assert_not_called()
        if request_briefing:
            manager.set_meeting_status("prepare")
            await manager._meeting_preparation_task
            assert manager._meeting_authorized
        manager.set_meeting_status("started")
        await manager._meeting_task
        manager.attend_meeting.assert_awaited_once()
        assert manager.prepare_meeting.await_count == (1 if request_briefing else 0)

    asyncio.run(check())


@pytest.mark.parametrize("confirmed,state", [(True, "accepted"), (False, "declined"), (None, "pending")])
def test_meeting_preserves_permission_state_for_started_decision(confirmed, state):
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from agents.tasks.meeting import MeetingTask
    from agents.tasks.decision import DecisionInput, NotificationDecision
    from config import ConfigAssistant
    from data.agents_dataclasses import ActionType, InterventionType, SuggestionType, ToneType, UrgencyType

    async def check():
        task = MeetingTask(ConfigAssistant(), Mock())
        task.interpret_reply = AsyncMock(return_value=confirmed)
        task._deliver_reply = AsyncMock()
        assert await task.on_asked()
        await task.handle_user_input("Yes" if confirmed else "No", [])
        assert task.permission_state == state
        assert task.awaiting_reply == (confirmed is None)
        assert not await task.on_asked()
        task.backend.run = AsyncMock(return_value=NotificationDecision(
            action=ActionType.NONE, urgency=UrgencyType.NONE, tone=ToneType.CALM,
            intervention_type=InterventionType.NONE, suggestion_type=SuggestionType.NONE,
            reason="Only the meeting phase changed.",
        ))
        await task.execute(DecisionInput(
            driver_name="David",
            changed_facts="The meeting has started and is in progress.",
            context="The driver is late for the meeting.",
        ))
        inputs = task.backend.run.await_args.args[1]
        assert inputs == {
            "driver_name": "David",
            "changed_facts": "The meeting has started and is in progress.",
            "context": "The driver is late for the meeting.",
            "meeting_permission_state": state,
        }
        prompt = task.crew.tasks[0]
        prompt.interpolate_inputs_and_add_conversation_history(inputs)
        assert "Current meeting_permission_state: " + state in prompt.description
        assert "only in supporting context" in prompt.description
        assert "<SUPPORTING_MEETING_FACTS>\nThe driver is late" in prompt.description
        task.reset()
        assert task.permission_state == "not_asked"
        assert await task.on_asked()

    asyncio.run(check())


@pytest.mark.parametrize(
    "changed_facts,should_call_backend",
    [
        ("The meeting is being prepared and has not started yet.", False),
        (
            "The meeting is being prepared and has not started yet.\n"
            "The driver is late for the meeting.",
            True,
        ),
    ],
)
def test_meeting_prepare_skips_decision_call_unless_lateness_changed(changed_facts, should_call_backend):
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from agents.tasks.meeting import MeetingTask
    from agents.tasks.decision import DecisionInput, NotificationDecision
    from config import ConfigAssistant
    from data.agents_dataclasses import ActionType, InterventionType, SuggestionType, ToneType, UrgencyType

    async def check():
        task = MeetingTask(ConfigAssistant(), Mock())
        task.backend.run = AsyncMock(return_value=NotificationDecision(
            urgency=UrgencyType.NONE,
            tone=ToneType.CALM,
            intervention_type=InterventionType.NONE,
            action=ActionType.NONE,
            suggestion_type=SuggestionType.NONE,
            reason="No new lateness.",
        ))
        decision = await task.execute(DecisionInput(
            driver_name="David", changed_facts=changed_facts, context="No additional facts.",
        ))

        assert task.backend.run.await_count == int(should_call_backend)
        if not should_call_backend:
            assert decision.action is ActionType.NONE
            assert decision.urgency is UrgencyType.NONE
            assert decision.spoken_message is None

    asyncio.run(check())


@pytest.mark.parametrize("choice", ["route", "skip"])
@pytest.mark.parametrize("fact", [
    "The driver has arrived at the meeting.",
    "The driver is late for the meeting.",
    "The driver is on time for the meeting.",
    "No meeting is scheduled.",
    "The meeting is being prepared and has not started yet.",
    "The meeting has started and is in progress.",
    "The meeting has finished.",
])
def test_meeting_router_classifies_domain_without_forcing_action(fact, choice):
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from agents.backends.decision_backend import KevBackend
    from agents.tasks.routing import RoutingInput, TelemetryRouter
    from config import ConfigAssistant

    backend = Mock(spec=KevBackend)
    backend.execute_decision = AsyncMock(return_value={"answers": {
        task.value: {"choice": choice if task is TaskType.MEETING else "skip"}
        for task in TelemetryRouter.TASK_CRITERIA
    }})
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)
    route = asyncio.run(router.execute(RoutingInput({"agenda_fact": fact})))

    assert route.tasks == ([TaskType.MEETING] if choice == "route" else [])
    state, questions = backend.execute_decision.await_args.args
    assert state == {"changed_facts": [fact]}
    assert set(questions) == {task.value for task in TelemetryRouter.TASK_CRITERIA}
    assert "no meeting is scheduled" in questions["meeting"]["instructions"]
    assert "preparation and arrival" in questions["meeting"]["criteria"]["route"]


def test_meeting_arrival_snapshot_excludes_later_contributions():
    from types import SimpleNamespace
    from sim.meeting_sim import MeetingInterrupted, MeetingProgress

    progress = MeetingProgress()
    progress.record(SimpleNamespace(
        agent="Priya", description="Product discussion", raw="Only parallel parking was proposed.",
    ))
    transcript = progress.stop()
    with pytest.raises(MeetingInterrupted):
        progress.record(SimpleNamespace(
            agent="Marco", description="Architecture discussion", raw="Later sensor proposal.",
        ))
    assert transcript == [{
        "speaker": "Priya", "task": "Product discussion", "text": "Only parallel parking was proposed.",
    }]
    assert progress.stop() == transcript


def test_meeting_arrival_interrupts_pause_between_contributions():
    from threading import Event, Thread
    from types import SimpleNamespace
    from unittest.mock import patch
    from sim.meeting_sim import MeetingInterrupted, MeetingProgress

    progress = MeetingProgress(pause_seconds=60, expected_contributions=2)
    pause_started = Event()
    callback_finished = Event()
    errors = []
    wait_between_turns = progress._wait_between_turns

    def wait_for_meeting_pace():
        pause_started.set()
        wait_between_turns()

    def record_contribution():
        try:
            progress.record(SimpleNamespace(agent="Priya", description="Product", raw="Proposal."))
        except MeetingInterrupted as error:
            errors.append(error)
        finally:
            callback_finished.set()

    with patch.object(progress, "_wait_between_turns", side_effect=wait_for_meeting_pace):
        thread = Thread(target=record_contribution)
        thread.start()
        assert pause_started.wait(1)
        assert len(progress.stop()) == 1
        assert callback_finished.wait(1)
        thread.join()

    assert len(errors) == 1


def test_meeting_arrival_callback_stops_actual_crewai_sequence():
    from unittest.mock import Mock, patch
    from agents.backends.crewai_backend import CrewAIBackend
    from config import ConfigAssistant
    from sim.meeting_sim import DEFAULT_TOPIC, MeetingInterrupted, MeetingProgress, build_crew

    backend = CrewAIBackend(ConfigAssistant(), Mock())
    crew, _ = build_crew(backend.llm, DEFAULT_TOPIC, False)
    progress = MeetingProgress()
    crew.task_callback = progress.record
    calls = []

    def execute(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            return "The chair introduced the planned agenda. No decisions were made."
        progress.stop()
        return "A contribution completed after the driver arrived."

    with patch("crewai.Agent.execute_task", side_effect=execute):
        with pytest.raises(MeetingInterrupted):
            crew.kickoff()
    assert len(calls) == 2
    assert [entry["text"] for entry in progress.stop()] == [
        "The chair introduced the planned agenda. No decisions were made.",
    ]


@pytest.mark.parametrize("driver_name", [None, "David", "Alex"])
def test_meeting_arrival_summary_has_no_planned_or_future_discussion(driver_name):
    import json
    from unittest.mock import Mock
    from agents.backends.crewai_backend import CrewAIBackend
    from config import ConfigAssistant
    from sim.meeting_sim import build_arrival_summary_crew

    backend = CrewAIBackend(ConfigAssistant(), Mock())
    crew, summary = build_arrival_summary_crew(backend.llm, [], False, driver_name=driver_name)
    assert crew.tasks == [summary]
    assert "Completed contributions at the moment of arrival:\n[]" in summary.description
    assert "no completed discussion" in summary.description
    assert "An opening agenda is only a plan" in summary.description
    assert "Do not say the whole meeting concluded" in summary.description
    assert crew.agents[0].role == "Driver's Meeting Handover Assistant"
    assert json.dumps({"driver_name": driver_name}) in summary.description
    assert "Speak directly to the driver as 'you'" in summary.description
    assert "Never infer the driver's name from the transcript" in summary.description
    assert "Speaking plan (planned turns, not completed discussion)" in summary.description
    assert "name who has already spoken" in summary.description
    assert "does not mean his closing review has happened" in summary.description
    assert "Do not assert who is currently speaking" in summary.description


def test_meeting_arrival_summary_includes_shared_speaking_order_separate_from_transcript():
    import json
    from unittest.mock import Mock
    from agents.backends.crewai_backend import CrewAIBackend
    from config import ConfigAssistant
    from sim.meeting_sim import PLANNED_AGENDA, build_arrival_summary_crew

    backend = CrewAIBackend(ConfigAssistant(), Mock())
    transcript = [
        {"speaker": "Jordan Reyes, Engineering Meeting Manager", "task": "Open meeting", "text": "Here is the planned agenda."},
        {"speaker": "Priya Shah, Product Manager", "task": "Product scope", "text": "Parallel parking is proposed for the MVP."},
    ]
    _, summary = build_arrival_summary_crew(backend.llm, transcript, False, driver_name="David")

    plan_text = summary.description.split("Speaking plan (planned turns, not completed discussion):\n", 1)[1]
    plan_text, transcript_text = plan_text.split("\n\nCompleted contributions at the moment of arrival:\n", 1)
    plan = json.loads(plan_text)
    assert plan["opening_chair"] == "Jordan"
    assert plan["agenda"] == [{"speaker": owner, "topic": subject} for owner, _, subject in PLANNED_AGENDA]
    assert json.loads(transcript_text) == transcript
    assert "who still has an agenda contribution" in summary.description


@pytest.mark.parametrize("completed_first", [True, False])
@pytest.mark.parametrize("arrival_timing", ["none", "on_time", "arrived"])
def test_meeting_arrival_stops_following_turns_and_speaks_only_partial_summary(arrival_timing, completed_first):
    import asyncio
    from threading import Event
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock, patch
    from managers.action_manager import ActionManager

    async def check():
        agent = SimpleNamespace(
            set_assistant_status=Mock(), speak=AsyncMock(), llm=Mock(),
            driver_name="Alex",
            opt=SimpleNamespace(crewai_verbose=False),
        )
        manager = ActionManager(agent)
        manager.on_meeting_completed = Mock()
        manager.set_meeting_timing("late")
        first_done = asyncio.Event()
        continue_turn = Event()
        loop = asyncio.get_running_loop()
        meeting_crew = Mock()
        final_summary = SimpleNamespace(output=SimpleNamespace(raw="Full meeting summary must not be spoken."))
        handover_crew = Mock()
        handover = SimpleNamespace(output=SimpleNamespace(raw=(
            "Priya proposed parallel parking. No decisions were taken yet." if completed_first
            else "You have arrived before any discussion was completed."
        )))
        later_turn = Mock()

        def discuss():
            if completed_first:
                meeting_crew.task_callback(SimpleNamespace(
                    agent="Priya", description="Product discussion", raw="Parallel parking was proposed.",
                ))
            loop.call_soon_threadsafe(first_done.set)
            assert continue_turn.wait(timeout=2)
            meeting_crew.task_callback(SimpleNamespace(
                agent="Marco", description="Architecture discussion", raw="Later sensor proposal.",
            ))
            later_turn()

        meeting_crew.kickoff.side_effect = discuss
        with (
            patch("managers.action_manager.build_crew", return_value=(meeting_crew, final_summary)),
            patch("managers.action_manager.build_arrival_summary_crew", return_value=(handover_crew, handover)) as summarize,
        ):
            manager.set_meeting_status("started")
            manager.confirm_attend_meeting()
            running = manager._meeting_task
            try:
                await asyncio.wait_for(first_done.wait(), timeout=2)
                manager.set_meeting_timing(arrival_timing)
            finally:
                continue_turn.set()
            await asyncio.wait_for(running, timeout=2)
            later_turn.assert_not_called()
            expected = [{
                "speaker": "Priya", "task": "Product discussion", "text": "Parallel parking was proposed.",
            }] if completed_first else []
            assert summarize.call_args.args[1] == expected
            assert summarize.call_args.kwargs["driver_name"] == "Alex"
            handover_crew.kickoff.assert_called_once()
            agent.speak.assert_awaited_once_with(handover.output.raw)
            manager.on_meeting_completed.assert_called_once()
            assert not manager._meeting_authorized

    asyncio.run(check())


def test_meeting_started_launches_simulation_only_after_permission():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from managers.action_manager import ActionManager

    async def check():
        manager = ActionManager(SimpleNamespace(set_assistant_status=Mock()))
        manager.attend_meeting = AsyncMock()
        manager.prepare_meeting = AsyncMock()
        manager.set_meeting_status("prepare")
        manager.confirm_attend_meeting()
        manager.attend_meeting.assert_not_called()
        manager.set_meeting_status("started")
        await manager._meeting_task
        manager.attend_meeting.assert_awaited_once()
        manager.set_meeting_status("started")
        manager.attend_meeting.assert_awaited_once()
        manager.set_meeting_status("finished")
        assert not manager._meeting_authorized
        manager.set_meeting_status("prepare")
        manager.set_meeting_status("started")
        assert manager.attend_meeting.await_count == 1
        manager.confirm_attend_meeting()
        await manager._meeting_task
        assert manager.attend_meeting.await_count == 2

    asyncio.run(check())


def test_meeting_prepare_delivers_planned_agenda_without_starting_simulation():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock, patch
    from managers.action_manager import ActionManager

    async def check():
        briefing = "The meeting will cover parking scope, sensors, safety and driver feedback."
        agent = SimpleNamespace(
            set_assistant_status=Mock(), speak=AsyncMock(), llm=Mock(),
            opt=SimpleNamespace(crewai_verbose=False),
        )
        manager = ActionManager(agent)
        preparation = Mock()
        briefing_task = SimpleNamespace(output=SimpleNamespace(raw=briefing))
        manager.on_meeting_completed = Mock()
        with (
            patch("managers.action_manager.build_preparation_crew", return_value=(preparation, briefing_task)),
            patch("managers.action_manager.build_crew") as simulate,
        ):
            manager.set_meeting_status("prepare")
            await manager._meeting_preparation_task
            preparation.kickoff.assert_called_once()
            agent.speak.assert_awaited_once_with(briefing)
            simulate.assert_not_called()
            manager.on_meeting_completed.assert_not_called()
            manager.set_meeting_status("prepare")
            assert preparation.kickoff.call_count == 1

    asyncio.run(check())


def test_telemetry_route_can_be_empty():
    assert TaskRoute(tasks=[]).tasks == []


@pytest.mark.parametrize("selected", [[], [TaskType.WELLBEING], [TaskType.WELLBEING, TaskType.MOOD_MUSIC]])
def test_laya_router_preserves_multiple_task_routes(selected):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.decision_backend import LayaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from config import ConfigAssistant

    answers = {task_type.value: {"choice": "route" if task_type in selected else "skip"}
               for task_type in TelemetryRouter.TASK_CRITERIA}
    backend = Mock(spec=LayaBackend)
    backend.execute_decision = AsyncMock(return_value={"answers": answers})
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)
    changed = {"DriverPhysicalState.fatigue_level": "Fatigue level is high.",
               "DriverEmotionState.angry": "Anger level is high."}
    route = asyncio.run(router.execute(RoutingInput(changed)))
    assert route.tasks == selected
    state, questions = backend.execute_decision.await_args.args
    assert state == {"changed_facts": list(changed.values())}
    assert set(questions) == {task_type.value for task_type in TelemetryRouter.TASK_CRITERIA}
    assert all(question["type"] == "choice" for question in questions.values())
    assert all(set(question["criteria"]) == {"route", "skip"} for question in questions.values())
    assert "conversation" not in questions and "vehicle_command" not in questions


@pytest.mark.parametrize("message_route", ["route", "skip"])
@pytest.mark.parametrize("key,fact", [
    ("UnreadMessages.count", "There are 3 unread messages."),
    ("UnreadMessages.count", "There are no unread messages."),
    ("DriverPhysicalState.attention_level", "Attention level is medium."),
    ("DriverPhysicalState.attention_level", "Attention level is high."),
    ("DriverPhysicalState.attention_level", "Attention level is very high."),
])
def test_decision_router_treats_message_delivery_changes_as_domain_facts(key, fact, message_route):
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from agents.backends.decision_backend import KevBackend
    from agents.tasks.routing import RoutingInput, TelemetryRouter
    from config import ConfigAssistant

    backend = Mock(spec=KevBackend)
    answers = {
        task.value: {"choice": message_route if task is TaskType.MESSAGES else "skip"}
        for task in TelemetryRouter.TASK_CRITERIA
    }
    backend.execute_decision = AsyncMock(return_value={"answers": answers})
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)

    route = asyncio.run(router.execute(RoutingInput({key: fact})))

    expected = [TaskType.MESSAGES] if message_route == "route" else []
    assert route.tasks == expected
    state, questions = backend.execute_decision.await_args.args
    assert state == {"changed_facts": [fact]}
    assert set(questions) == {task.value for task in TelemetryRouter.TASK_CRITERIA}
    question = questions["messages"]
    assert "Attention level, unread messages" in question["instructions"]
    assert "privacy and people inside" in question["instructions"]
    assert question["criteria"] == {
        "route": "The facts describe attention, unread messages, privacy or people inside.",
        "skip": "The facts describe something unrelated to message delivery.",
    }


def test_message_prompt_explicitly_allows_delivery_after_occupancy_returns_to_one():
    from agents.tasks.messages import TASK as MESSAGE_TASK

    instructions = MESSAGE_TASK.instructions
    assert "0 or 1 permits delivery; 2 or more blocks it." in instructions
    assert "occupancy returning from 2 or more\npeople to 0 or 1" in instructions
    assert "a republished pending count can trigger\nthis recovery decision" in instructions


@pytest.mark.parametrize("choice", ["route", "skip"])
@pytest.mark.parametrize("fact", [
    "Anger level is high.", "Disgust level is high.", "Fear level is high.",
    "Happiness level is high.", "Sadness level is high.", "Surprise level is high.",
    "The driver seems emotionally neutral.", "No anger detected.",
])
def test_decision_router_leaves_emotion_routing_to_model(fact, choice):
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from agents.backends.decision_backend import KevBackend
    from agents.tasks.routing import RoutingInput, TelemetryRouter
    from config import ConfigAssistant

    backend = Mock(spec=KevBackend)
    backend.execute_decision = AsyncMock(return_value={"answers": {
        task.value: {"choice": choice if task is TaskType.MOOD_MUSIC else "skip"}
        for task in TelemetryRouter.TASK_CRITERIA
    }})
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)

    result = asyncio.run(router.execute(RoutingInput({"emotion": fact})))

    assert result.tasks == ([TaskType.MOOD_MUSIC] if choice == "route" else [])
    state, questions = backend.execute_decision.await_args.args
    assert state == {"changed_facts": [fact]}
    assert set(questions) == {task.value for task in TelemetryRouter.TASK_CRITERIA}
    assert questions["mood_music"]["criteria"] == {
        "route": "The changed facts describe the driver's emotions, including their absence.",
        "skip": "The changed facts describe only non-emotional information such as driving tension, fatigue or attention.",
    }


@pytest.mark.parametrize("result", [None, {}, {"answers": {}}, {"answers": {
    "wellbeing": {"choice": "invented"},
}}])
def test_laya_router_rejects_invalid_or_incomplete_answers(result):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.decision_backend import LayaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from config import ConfigAssistant

    backend = Mock(spec=LayaBackend)
    backend.execute_decision = AsyncMock(return_value=result)
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)
    with pytest.raises(ValueError):
        asyncio.run(router.execute(RoutingInput({"VehicleMotion.speed": "Speed is high."})))


def test_laya_router_skips_inference_for_empty_changes():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.decision_backend import LayaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from config import ConfigAssistant

    backend = Mock(spec=LayaBackend)
    backend.execute_decision = AsyncMock()
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)
    assert asyncio.run(router.execute(RoutingInput({}))).tasks == []
    backend.execute_decision.assert_not_awaited()


def test_ollama_router_recovers_route_from_reasoning_field():
    """Reasoning models can reply with an empty content field: the route must be
    recovered from the reasoning text instead of failing JSON validation."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.ollama_backend import OllamaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from config import ConfigAssistant

    backend = Mock(spec=OllamaBackend)
    backend.complete = AsyncMock(return_value=SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(
            content=None,
            reasoning_content='The facts mention fatigue and attention. {"tasks": ["wellbeing"]}',
        )),
    ]))
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)

    route = asyncio.run(router.execute(RoutingInput({"DriverPhysicalState.fatigue_level": "Fatigue level is high."})))

    assert route.tasks == [TaskType.WELLBEING]
    assert backend.complete.await_count == 1


def test_ollama_router_retries_on_empty_response_then_degrades():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.ollama_backend import OllamaBackend
    from agents.tasks.routing import TelemetryRouter, RoutingInput
    from config import ConfigAssistant

    backend = Mock(spec=OllamaBackend)
    empty = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    backend.complete = AsyncMock(return_value=empty)
    router = TelemetryRouter(ConfigAssistant(), Mock(), backend=backend)

    route = asyncio.run(router.execute(RoutingInput({"VehicleMotion.speed": "Speed is high."})))

    assert route.tasks == []
    assert backend.complete.await_count == 2


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
    backend.logger = Mock()
    crew = SimpleNamespace(kickoff_async=AsyncMock(return_value=SimpleNamespace(pydantic=Result(value="ok"))))
    assert asyncio.run(backend.run(crew, {"input": "data"}, Result, "test")).value == "ok"
    backend.logger.info.assert_called_once()
    assert backend.logger.info.call_args.kwargs["response_text"] == '{"value":"ok"}'
    assert backend.logger.info.call_args.kwargs["status"] == "completed"
    assert "ttft_ms" not in backend.logger.info.call_args.kwargs
    crew.kickoff_async.assert_awaited_once_with(inputs={"input": "data"})
    crew.kickoff_async.return_value = SimpleNamespace(pydantic=None)
    with pytest.raises(ValueError, match="did not return"):
        asyncio.run(backend.run(crew, {}, Result, "test"))
    assert usage.call_count == 2
    assert backend.logger.info.call_args.kwargs["status"] == "error"


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
    backend.logger = Mock()

    async def check():
        response = backend.stream(call_name="test", model="test", messages=[])
        await anext(response)
        await response.aclose()

    asyncio.run(check())
    stream.close.assert_awaited_once()
    usage.assert_called_once_with("test", None)
    backend.logger.info.assert_called_once()
    assert backend.logger.info.call_args.kwargs["status"] == "cancelled"
    assert backend.logger.info.call_args.kwargs["response_text"] == "Hi"


@pytest.mark.parametrize("streaming", [False, True])
def test_llm_call_logs_one_info_summary_with_output_timing_and_usage(streaming):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.automotive_agent import AutomotiveAgent
    from agents.backends.ollama_backend import OllamaBackend
    from config import ConfigAssistant

    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    logger = Mock()
    agent = AutomotiveAgent.__new__(AutomotiveAgent)
    agent.opt = ConfigAssistant()
    agent.logger = logger

    class Stream:
        close = AsyncMock()

        async def __aiter__(self):
            for text in ("[[tone:calm]]", "Hello."):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)
            yield SimpleNamespace(choices=[], usage=usage)

    response = Stream() if streaming else SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello."))], usage=usage,
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
    backend = OllamaBackend(agent.opt, agent.log_llm_usage, client=client)
    backend.logger = logger

    async def check():
        if streaming:
            async for _ in backend.stream(call_name="test", model="test-model", messages=[]):
                pass
        else:
            await backend.complete(call_name="test", model="test-model", messages=[])

    asyncio.run(check())
    logger.info.assert_called_once()
    assert logger.info.call_args.args == ("[LLM] Call finished",)
    fields = logger.info.call_args.kwargs
    assert fields["llm_call"] == "test"
    assert fields["model"] == "test-model"
    assert fields["response_text"] == "Hello."
    assert fields["status"] == "completed"
    assert fields["available"] is True
    assert fields["prompt_tokens"] == 10
    assert fields["completion_tokens"] == 4
    assert fields["context_tokens"] == fields["total_tokens"] == 14
    assert fields["context_window_tokens"] == 4096
    assert fields["duration_ms"] >= 0
    assert ("ttft_ms" in fields) == streaming
    if streaming:
        assert fields["duration_ms"] >= fields["ttft_ms"] >= 0
        response.close.assert_awaited_once()
    debug_events = [call.args[0] for call in logger.debug.call_args_list]
    assert "[LLM] Token usage" in debug_events
    assert "[LLM] Response generation time" in debug_events
    assert ("[LLM] Time to first token" in debug_events) == streaming


def test_empty_llm_stream_summary_has_no_invented_ttft():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.ollama_backend import OllamaBackend
    from config import ConfigAssistant

    class Stream:
        close = AsyncMock()

        async def __aiter__(self):
            yield SimpleNamespace(choices=[], usage=None)

    stream = Stream()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))))
    backend = OllamaBackend(ConfigAssistant(), Mock(), client=client)
    backend.logger = Mock()

    async def check():
        async for _ in backend.stream(call_name="empty", model="test", messages=[]):
            pass

    asyncio.run(check())
    backend.logger.info.assert_called_once()
    fields = backend.logger.info.call_args.kwargs
    assert "ttft_ms" not in fields
    assert fields["response_text"] == ""
    assert fields["available"] is False


def test_failed_llm_call_logs_error_summary_and_preserves_exception():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.ollama_backend import OllamaBackend
    from config import ConfigAssistant

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=RuntimeError("backend unavailable")),
    )))
    backend = OllamaBackend(ConfigAssistant(), Mock(), client=client)
    backend.logger = Mock()
    with pytest.raises(RuntimeError, match="backend unavailable"):
        asyncio.run(backend.complete(call_name="failed", model="test", messages=[]))
    backend.logger.info.assert_called_once()
    assert backend.logger.info.call_args.kwargs["status"] == "error"


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
    from agents.backends.decision_backend import LayaBackend
    from agents.tasks.pre_classify import (
        RequestPreClassifier, PreClassificationInput, PreClassificationUpdate, ResponseSource,
    )
    from config import ConfigAssistant

    backend = Mock(spec=LayaBackend)
    backend.execute_decision = AsyncMock(return_value={"answers": {
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
    state, questions = backend.execute_decision.await_args.args
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
    from agents.backends.decision_backend import LayaBackend
    from agents.tasks.pre_classify import RequestPreClassifier, PreClassificationInput
    from config import ConfigAssistant

    backend = Mock(spec=LayaBackend)
    backend.execute_decision = AsyncMock(return_value=result)
    task = RequestPreClassifier(ConfigAssistant(), Mock(), backend=backend)
    updates = []
    output = asyncio.run(task.execute(PreClassificationInput("Ambiguous turn", [], 0), on_update=updates.append))
    assert output.requires_action_classification is True
    assert output.response_source.value == "general"
    assert updates[0].requires_action_classification is True


@pytest.mark.parametrize("routing_backend", ["llm", "decision"])
def test_agent_reuses_laya_backend_when_models_match(routing_backend):
    from unittest.mock import AsyncMock, Mock, patch
    from agents.backends.decision_backend import LayaBackend
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    opt = ConfigAssistant(routing_backend=routing_backend, pre_classify_backend="decision")
    backend = Mock(spec=LayaBackend)
    backend.execute_decision = AsyncMock()
    with patch("agents.automotive_agent.LayaBackend", return_value=backend) as load:
        agent = AutomotiveAgent(opt)
    assert agent.request_pre_classifier.backend is backend
    assert all(task.backend is agent.crewai_backend for task in agent.tasks.values())
    assert load.call_count == 1
    if routing_backend == "decision":
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
    from agents.backends.decision_backend import LayaBackend
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    opt = ConfigAssistant(routing_backend="decision", pre_classify_backend="decision",
                          routing_model="router-model", pre_classify_model="request-model")
    router_backend, request_backend = Mock(spec=LayaBackend), Mock(spec=LayaBackend)
    router_backend.execute_decision = AsyncMock()
    request_backend.execute_decision = AsyncMock()
    with patch("agents.automotive_agent.LayaBackend", side_effect=[router_backend, request_backend]) as load:
        agent = AutomotiveAgent(opt)
    assert agent.telemetry_router.backend is router_backend
    assert agent.request_pre_classifier.backend is request_backend
    # The laya engine uses the shared laya_model checkpoint; both backends are
    # still distinct instances.
    assert [call.args[0] for call in load.call_args_list] == [None, None]


@pytest.mark.parametrize("message,valid", [
    ("Controllo il manuale.", True),
    ("I'm currently checking your vehicle's manual for the cruise control activation steps.", True),
    ("A" * 120, True),
    (" ".join(["ok"] * 20), True),
    ("", False),
    ("I'm checking the vehicle manual for the cruise control instructions so I can give you the right steps for your particular vehicle.", True),
    (" ".join(["ok"] * 21), True),
    ("Va bene?", False),
    ("First\nSecond", False),
])
def test_wait_message_task_generates_and_validates_short_acknowledgement(message, valid):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.tasks.conversation import WaitMessageTask
    from config import ConfigAssistant

    class Stream:
        aclose = AsyncMock()

        async def __aiter__(self):
            for text in (None, message[:len(message) // 2], message[len(message) // 2:]):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
            yield SimpleNamespace(choices=[], usage=None)

    stream = Stream()
    backend = SimpleNamespace(stream=Mock(return_value=stream))
    task = WaitMessageTask(ConfigAssistant(), backend)
    request = task.execute("Come si attiva il cruise control?", [], purpose="vehicle manual lookup")
    if valid:
        assert asyncio.run(request) == message
    else:
        with pytest.raises(ValueError, match="Invalid wait message"):
            asyncio.run(request)
    assert backend.stream.call_args.kwargs["call_name"] == "wait message"
    assert backend.stream.call_args.kwargs["reasoning_effort"] == "none"
    prompt = backend.stream.call_args.kwargs["messages"][0]["content"]
    assert "characters" not in prompt
    assert "4-10" not in prompt
    stream.aclose.assert_awaited_once()


def make_wait_message_backend(message="Controllo la richiesta."):
    from types import SimpleNamespace
    from unittest.mock import Mock

    async def chunks(**kwargs):
        for text in (message[:len(message) // 2], message[len(message) // 2:]):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
        yield SimpleNamespace(choices=[], usage=None)

    return SimpleNamespace(stream=Mock(side_effect=chunks))


def test_streaming_wait_message_logs_ttft_in_final_summary():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.ollama_backend import OllamaBackend
    from agents.tasks.conversation import WaitMessageTask
    from config import ConfigAssistant

    class Stream:
        close = AsyncMock()

        async def __aiter__(self):
            yield SimpleNamespace(choices=[], usage=None)
            for text in ("Controllo ", "il manuale."):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)
            yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14))

    stream = Stream()
    create = AsyncMock(return_value=stream)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    opt = ConfigAssistant()
    usage = Mock()
    backend = OllamaBackend(opt, usage, client=client)
    backend.logger = Mock()
    task = WaitMessageTask(opt, backend)
    assert asyncio.run(task.execute("Cruise control?", [], purpose="manual lookup")) == "Controllo il manuale."
    assert create.await_args.kwargs["stream"] is True
    assert create.await_args.kwargs["stream_options"] == {"include_usage": True}
    backend.logger.info.assert_called_once()
    summary = backend.logger.info.call_args.kwargs
    assert summary["llm_call"] == "wait message"
    assert summary["response_text"] == "Controllo il manuale."
    assert summary["status"] == "completed"
    assert summary["duration_ms"] >= summary["ttft_ms"] >= 0
    assert summary["total_tokens"] == 14
    stream.close.assert_awaited_once()
    usage.assert_called_once()


@pytest.mark.parametrize("delay", [0, 0.1, 0.75])
def test_wait_message_uses_configured_delay(delay):
    import asyncio
    from unittest.mock import AsyncMock, patch
    from omegaconf import OmegaConf
    from agents.tasks.pre_classify import ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = OmegaConf.to_object(OmegaConf.merge(
            OmegaConf.structured(ConfigAssistant),
            OmegaConf.from_dotlist([f"wait_message_delay_seconds={delay}"]),
        ))
        ready = asyncio.create_task(asyncio.Event().wait())
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Cruise control?", [], "Cruise control?")
        try:
            with patch("agents.automotive_agent.asyncio.wait", new_callable=AsyncMock,
                       return_value=({ready}, set())) as wait:
                await agent.generate_wait_message(event, ResponseSource.VEHICLE_MANUAL, ready)
            wait.assert_awaited_once_with({ready}, timeout=delay)
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    asyncio.run(check())


def test_agent_rejects_negative_wait_message_delay():
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    with pytest.raises(ValueError, match="wait_message_delay_seconds"):
        AutomotiveAgent(ConfigAssistant(wait_message_delay_seconds=-0.1))


def test_slow_manual_lookup_speaks_llm_acknowledgement_before_final_response():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.tasks.pre_classify import ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent._assistant_flags = 0
        agent.opt.wait_message_delay_seconds = 0
        lookup_released, lookup_done = asyncio.Event(), asyncio.Event()
        speech_started, speech_released = asyncio.Event(), asyncio.Event()
        agent.ollama_backend = make_wait_message_backend("Controllo il manuale.")

        async def lookup(*args):
            await lookup_released.wait()
            lookup_done.set()
            return "manual instructions", "manual passages"

        async def speak(message):
            speech_started.set()
            lookup_released.set()
            await speech_released.wait()

        agent.response_context_providers = {ResponseSource.VEHICLE_MANUAL: lookup}
        agent.tts_manager = SimpleNamespace(speak=AsyncMock(side_effect=speak))
        agent.on_response, agent.on_response_update = Mock(), Mock()
        agent.stream_user_response = AsyncMock()
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Cruise control?", [], "Cruise control?")
        response = asyncio.create_task(agent.prepare_user_response(event, ResponseSource.VEHICLE_MANUAL))
        try:
            await asyncio.wait_for(speech_started.wait(), 1)
            await asyncio.wait_for(lookup_done.wait(), 1)
            agent.stream_user_response.assert_not_awaited()
            speech_released.set()
            await asyncio.wait_for(response, 1)
        finally:
            response.cancel()
            await asyncio.gather(response, return_exceptions=True)
        agent.tts_manager.speak.assert_awaited_once_with("Controllo il manuale.")
        agent.on_response.assert_called_once_with("Controllo il manuale.\n")
        assert agent.conversation_history == []
        assert agent.stream_user_response.await_args.kwargs["response_context"] == (
            "manual instructions", "manual passages",
        )

    asyncio.run(check())


@pytest.mark.parametrize("outcome", ["ready", "cancel", "error"])
def test_manual_lookup_cleans_up_pending_wait_generation(outcome):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.tasks.pre_classify import ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.opt.wait_message_delay_seconds = 0
        lookup_released, lookup_cancelled = asyncio.Event(), asyncio.Event()
        generation_started, generation_cancelled = asyncio.Event(), asyncio.Event()
        claimed = asyncio.Event()

        async def generate(**kwargs):
            generation_started.set()
            try:
                yield SimpleNamespace(choices=[])
                await asyncio.Future()
            finally:
                generation_cancelled.set()

        async def lookup(*args):
            try:
                await lookup_released.wait()
                if outcome == "error":
                    raise RuntimeError("lookup failed")
                return "", ""
            finally:
                lookup_cancelled.set()

        agent.ollama_backend = SimpleNamespace(stream=Mock(side_effect=generate))
        agent.response_context_providers = {ResponseSource.VEHICLE_MANUAL: lookup}
        agent.speak_wait_message_chunks = AsyncMock()
        agent.stream_user_response = AsyncMock()
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Cruise control?", [], "Cruise control?")
        response = asyncio.create_task(agent.prepare_user_response(
            event, ResponseSource.VEHICLE_MANUAL, wait_message_claimed=claimed,
        ))
        try:
            await asyncio.wait_for(generation_started.wait(), 1)
            if outcome == "cancel":
                response.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await response
            else:
                lookup_released.set()
                if outcome == "error":
                    with pytest.raises(RuntimeError, match="lookup failed"):
                        await asyncio.wait_for(response, 1)
                else:
                    await asyncio.wait_for(response, 1)
            assert generation_cancelled.is_set()
            assert lookup_cancelled.is_set()
            assert not claimed.is_set()
            agent.speak_wait_message_chunks.assert_not_awaited()
            assert agent.stream_user_response.await_count == (1 if outcome == "ready" else 0)
        finally:
            response.cancel()
            await asyncio.gather(response, return_exceptions=True)

    asyncio.run(check())


def test_unspoken_wait_does_not_suppress_later_manual_acknowledgement():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.tasks.pre_classify import ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.opt.wait_message_delay_seconds = 0
        generation_started, generation_closed = asyncio.Event(), asyncio.Event()
        first_ready, second_ready = asyncio.Event(), asyncio.Event()
        claimed = asyncio.Event()

        async def interrupted_stream():
            try:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Controllo"))])
                generation_started.set()
                await asyncio.Future()
            finally:
                generation_closed.set()

        agent.ollama_backend = SimpleNamespace(stream=Mock(side_effect=[
            interrupted_stream(), make_wait_message_backend("Consulto il manuale.").stream(),
        ]))
        agent.speak_wait_message_chunks = AsyncMock()
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Cruise control?", [], "Cruise control?")
        first_work = asyncio.create_task(first_ready.wait())
        second_work = asyncio.create_task(second_ready.wait())
        first_wait = asyncio.create_task(agent.generate_wait_message(
            event, ResponseSource.GENERAL, first_work, wait_message_claimed=claimed,
        ))
        try:
            await asyncio.wait_for(generation_started.wait(), 1)
            assert claimed.is_set()
            first_ready.set()
            await asyncio.wait_for(first_wait, 1)
            assert generation_closed.is_set()
            assert not claimed.is_set()
            agent.speak_wait_message_chunks.assert_not_awaited()
            await asyncio.wait_for(agent.generate_wait_message(
                event, ResponseSource.VEHICLE_MANUAL, second_work, wait_message_claimed=claimed,
            ), 1)
            assert claimed.is_set()
            assert agent.ollama_backend.stream.call_count == 2
            agent.speak_wait_message_chunks.assert_awaited_once()
        finally:
            for task in (first_wait, first_work, second_work):
                task.cancel()
            await asyncio.gather(first_wait, first_work, second_work, return_exceptions=True)

    asyncio.run(check())


def test_action_and_manual_wait_share_one_generation_per_turn():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from agents.tasks.pre_classify import ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.opt.wait_message_delay_seconds = 0
        agent.ollama_backend = make_wait_message_backend()
        agent.speak_wait_message_chunks = AsyncMock()
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Cruise control?", [], "Cruise control?")
        ready = asyncio.create_task(asyncio.Event().wait())
        claimed = asyncio.Event()
        gate = asyncio.Event()
        try:
            await agent.generate_wait_message(event, ResponseSource.GENERAL, ready, wait_message_claimed=claimed)
            gate.set()
            await agent.generate_wait_message(
                event, ResponseSource.VEHICLE_MANUAL, ready,
                response_gate=gate, wait_message_claimed=claimed,
            )
            agent.ollama_backend.stream.assert_called_once()
            agent.speak_wait_message_chunks.assert_awaited_once()
            await agent.generate_wait_message(
                event, ResponseSource.VEHICLE_MANUAL, ready, wait_message_claimed=asyncio.Event(),
            )
            assert agent.ollama_backend.stream.call_count == 2
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    asyncio.run(check())


@pytest.mark.parametrize("mode", ["informational", "classified", "failed"])
def test_manual_response_does_not_require_a_separate_wait_task(mode):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from agents.automotive_agent import DirectInputPreparation
    from agents.tasks.pre_classify import PreClassification, ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        decision = agent.tasks[TaskType.WELLBEING].execute.return_value.model_copy(
            update={"response_source": ResponseSource.VEHICLE_MANUAL},
        )

        async def classify():
            if mode == "failed":
                raise RuntimeError("classification failed")
            return decision

        classifier = asyncio.create_task(classify()) if mode != "informational" else None
        preparation = DirectInputPreparation(
            pre_classification=PreClassification(
                requires_action_classification=mode != "informational",
                response_source=ResponseSource.VEHICLE_MANUAL,
            ), classifier_task=classifier, wait_task=None, response_task=None, response_gate=asyncio.Event(),
        )
        agent.command_task = SimpleNamespace()
        agent.response_context_providers = {ResponseSource.VEHICLE_MANUAL: AsyncMock(return_value=("", "manual"))}
        agent.generate_wait_message = AsyncMock()
        agent.stream_user_response = AsyncMock()
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Cruise control?", [], "Cruise control?")
        preparation.response_task = asyncio.create_task(agent.prepare_user_response(
            event, ResponseSource.VEHICLE_MANUAL, response_gate=preparation.response_gate,
            wait_message_claimed=preparation.wait_message_claimed,
        ))
        agent.preclassify_direct_input = AsyncMock(return_value=preparation)
        try:
            if mode == "failed":
                with pytest.raises(RuntimeError, match="classification failed"):
                    await agent.handle_direct_user_input(event)
                assert preparation.response_task.done()
            else:
                await agent.handle_direct_user_input(event)
                await asyncio.wait_for(preparation.response_task, 1)
                assert preparation.response_gate.is_set()
                agent.stream_user_response.assert_awaited_once()
                agent.generate_wait_message.assert_awaited_once()
        finally:
            tasks = [task for task in (classifier, preparation.response_task) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(check())


def test_wait_message_generation_failure_is_nonfatal():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.tasks.pre_classify import ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.opt.wait_message_delay_seconds = 0
        agent.ollama_backend = SimpleNamespace(stream=Mock(side_effect=RuntimeError("LLM unavailable")))
        agent.speak_wait_message_chunks = AsyncMock()
        claimed = asyncio.Event()
        ready = asyncio.create_task(asyncio.Event().wait())
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Open windows", [], "Open windows")
        try:
            await asyncio.wait_for(agent.generate_wait_message(
                event, ResponseSource.GENERAL, ready, wait_message_claimed=claimed,
            ), 1)
            assert not ready.done()
            assert not claimed.is_set()
            agent.speak_wait_message_chunks.assert_not_awaited()
            agent.logger.warning.assert_called_once()
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)

    asyncio.run(check())


@pytest.mark.parametrize("slow_classifier", [False, True])
def test_decision_preclassification_uses_llm_wait_only_for_slow_classifier(slow_classifier):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.backends.decision_backend import LayaBackend
    from agents.tasks.pre_classify import RequestPreClassifier
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.opt.wait_message_delay_seconds = 0
        classifier_released = asyncio.Event()
        agent.ollama_backend = make_wait_message_backend()
        agent.action_manager.registered_action_options = ""
        agent.action_manager.registered_action_guidance = ""

        async def classify(inputs):
            if slow_classifier:
                await classifier_released.wait()
            return agent.tasks[TaskType.WELLBEING].execute.return_value

        agent.command_task = SimpleNamespace(
            direct_action_options="vehicle options", vehicle_action_guidance="vehicle reference",
            decision_options={"skill_options": "skills", "tone_options": "tones"},
            execute=AsyncMock(side_effect=classify),
        )
        backend = Mock(spec=LayaBackend)
        backend.execute_decision = AsyncMock(return_value={"answers": {
            "requires_action_classification": {"choice": "required"},
            "response_source": {"choice": "general"},
        }})
        agent.request_pre_classifier = RequestPreClassifier(agent.opt, Mock(), backend=backend)
        wait_messages = []

        async def speak_wait(chunks):
            while (chunk := await chunks.get()) is not None:
                if chunk.strip():
                    wait_messages.append(chunk)
                    classifier_released.set()

        async def prepare_response(event, source, *, response_gate, wait_message_claimed):
            await response_gate.wait()

        agent.speak_wait_message_chunks = AsyncMock(side_effect=speak_wait)
        agent.prepare_user_response = AsyncMock(side_effect=prepare_response)
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Open the windows", [], "Open the windows")
        preparation = await agent.preclassify_direct_input(event, 0)
        await asyncio.wait_for(preparation.classifier_task, 1)
        await asyncio.wait_for(preparation.wait_task, 1)
        assert wait_messages == (["Controllo la richiesta."] if slow_classifier else [])
        assert agent.ollama_backend.stream.call_count == (1 if slow_classifier else 0)
        assert preparation.pre_classification.wait_message == ""
        preparation.response_gate.set()
        await preparation.response_task

    asyncio.run(check())


def test_speculative_manual_wait_message_does_not_speak_before_route_confirmation():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from agents.tasks.pre_classify import ResponseSource
    from config import ConfigAssistant
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    async def check():
        agent = make_routed_agent([])
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.opt.wait_message_delay_seconds = 0
        agent.ollama_backend = make_wait_message_backend()
        agent.speak_wait_message_chunks = AsyncMock()
        lookup_released = asyncio.Event()
        ready = asyncio.create_task(lookup_released.wait())
        response_gate = asyncio.Event()
        event = CarEvent(SkillType.NONE, EventName.USER_INPUT, "Cruise control?", [], "Cruise control?")
        wait_task = asyncio.create_task(agent.generate_wait_message(
            event, ResponseSource.VEHICLE_MANUAL, ready, response_gate=response_gate,
        ))
        try:
            await asyncio.sleep(0)
            agent.ollama_backend.stream.assert_not_called()
            lookup_released.set()
            await asyncio.wait_for(wait_task, 1)
            agent.ollama_backend.stream.assert_not_called()
            agent.speak_wait_message_chunks.assert_not_awaited()
        finally:
            for task in (ready, wait_task):
                task.cancel()
            await asyncio.gather(ready, wait_task, return_exceptions=True)

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

        async def prepare_response(event, source, *, response_gate, wait_message_claimed):
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
        agent.speak_wait_message_chunks = AsyncMock(side_effect=speak_wait)
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
            agent.speak_wait_message_chunks.assert_not_awaited()
        preparation.response_gate.set()
        await preparation.response_task

    asyncio.run(check())


@pytest.mark.parametrize("cuda_available", [False, True])
def test_kev_backend_loads_during_init_and_reuses_model(cuda_available):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from agents.backends.decision_backend import KevBackend
    from config import ConfigAssistant
    from voice.gpu_lock import GPU_LOCK

    tokenizer, model, encodings = Mock(), Mock(), Mock()

    def infer(encoded):
        assert GPU_LOCK.locked()
        assert encoded is encodings
        return [[0.9, 0.1]]

    def encode(*args):
        assert GPU_LOCK.locked()
        return encodings

    model.probs.side_effect = infer
    checkpoint = Mock(meta=SimpleNamespace(temperature=2.0))
    checkpoint.load.return_value = (tokenizer, model)
    with (
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch("kev.checkpoint.Checkpoint", return_value=checkpoint) as load_checkpoint,
        patch("kev.model.admit", side_effect=encode) as admit,
    ):
        usage = Mock()
        backend = KevBackend(ConfigAssistant(), usage)
        load_checkpoint.assert_called_once_with("jaredpalmer/kev-0.8b")
        checkpoint.load.assert_called_once()
        assert checkpoint.load.call_args.args[0] == ("cuda" if cuda_available else "cpu")
        assert model.head.temperature == 2.0
        model.probs.assert_not_called()

        async def check():
            for _ in range(2):
                result = await backend.execute_decision(
                    {"fatigue": "high"},
                    {"task": {"type": "choice", "criteria": {"route": None, "skip": None}}},
                )
                assert result["answers"]["task"]["choice"] == "route"

        asyncio.run(check())
        load_checkpoint.assert_called_once()
        checkpoint.load.assert_called_once()
        assert model.probs.call_count == 2
        assert admit.call_args.args[:2] == (model, tokenizer)
        assert usage.call_count == 2
        assert not GPU_LOCK.locked()


def test_kev_backend_waits_for_voice_gpu_work():
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event
    from unittest.mock import Mock, patch
    from agents.backends.decision_backend import KevBackend
    from voice.gpu_lock import GPU_LOCK

    backend = KevBackend.__new__(KevBackend)
    backend._tokenizer = Mock()
    backend._model = Mock()
    backend._model.probs.return_value = [[0.9, 0.1]]
    started = Event()

    def decide():
        started.set()
        return backend._decide_sync({}, {
            "task": {"type": "choice", "criteria": {"route": None, "skip": None}},
        })

    with patch("kev.model.admit", return_value=Mock()) as admit:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with GPU_LOCK:
                result = executor.submit(decide)
                assert started.wait(timeout=1)
                with pytest.raises(TimeoutError):
                    result.result(timeout=0.05)
                admit.assert_not_called()
                backend._model.probs.assert_not_called()
            assert result.result(timeout=2)["answers"]["task"]["choice"] == "route"
    assert not GPU_LOCK.locked()


@pytest.mark.parametrize("failure_stage", ["encode", "infer"])
def test_kev_backend_releases_shared_gpu_lock_after_failure(failure_stage):
    from unittest.mock import Mock, patch
    from agents.backends.decision_backend import KevBackend
    from voice.gpu_lock import GPU_LOCK

    backend = KevBackend.__new__(KevBackend)
    backend._tokenizer = Mock()
    backend._model = Mock()

    def encode(*args):
        assert GPU_LOCK.locked()
        if failure_stage == "encode":
            raise RuntimeError("encoding failed")
        return Mock()

    def infer(*args):
        assert GPU_LOCK.locked()
        raise RuntimeError("inference failed")

    backend._model.probs.side_effect = infer
    with patch("kev.model.admit", side_effect=encode):
        with pytest.raises(RuntimeError, match="failed"):
            backend._decide_sync({}, {
                "task": {"type": "choice", "criteria": {"route": None, "skip": None}},
            })
    assert not GPU_LOCK.locked()


@pytest.mark.parametrize("hub_disabled,transformers_enabled", [
    (False, True), (True, True), (False, False), (True, False),
])
@pytest.mark.parametrize("load_fails", [False, True])
def test_kev_loading_hides_progress_and_restores_previous_state(hub_disabled, transformers_enabled, load_fails):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
    from transformers.utils import logging as transformers_logging
    from agents.backends.decision_backend import KevBackend
    from config import ConfigAssistant

    original_hub_disabled = are_progress_bars_disabled()
    original_transformers_enabled = transformers_logging.is_progress_bar_enabled()
    checkpoint = Mock(meta=SimpleNamespace(temperature=1.0))
    logger = Mock()

    def load(*args):
        assert are_progress_bars_disabled()
        assert not transformers_logging.is_progress_bar_enabled()
        if load_fails:
            raise RuntimeError("checkpoint load failed")
        return Mock(), Mock()

    checkpoint.load.side_effect = load
    try:
        if transformers_enabled:
            transformers_logging.enable_progress_bar()
        else:
            transformers_logging.disable_progress_bar()
        if hub_disabled:
            disable_progress_bars()
        else:
            enable_progress_bars()
        with (
            patch("kev.checkpoint.Checkpoint", return_value=checkpoint),
            patch("agents.backends.decision_backend.structlog.get_logger", return_value=logger),
        ):
            if load_fails:
                with pytest.raises(RuntimeError, match="checkpoint load failed"):
                    KevBackend(ConfigAssistant(), Mock())
            else:
                KevBackend(ConfigAssistant(), Mock())
        assert are_progress_bars_disabled() == hub_disabled
        assert transformers_logging.is_progress_bar_enabled() == transformers_enabled
        assert logger.debug.call_count == (1 if load_fails else 2)
    finally:
        if original_transformers_enabled:
            transformers_logging.enable_progress_bar()
        else:
            transformers_logging.disable_progress_bar()
        if original_hub_disabled:
            disable_progress_bars()
        else:
            enable_progress_bars()


def test_kev_backend_logs_traceback_and_propagates_inference_failure():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from agents.backends.decision_backend import KevBackend
    from config import ConfigAssistant

    checkpoint = Mock(meta=SimpleNamespace(temperature=1.0))
    checkpoint.load.return_value = (Mock(), Mock())
    usage = Mock()
    with patch("kev.checkpoint.Checkpoint", return_value=checkpoint):
        backend = KevBackend(ConfigAssistant(), usage)
    backend.logger = Mock()
    failure = RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")
    with patch.object(backend, "_decide_sync", side_effect=failure):
        with pytest.raises(RuntimeError) as raised:
            asyncio.run(backend.execute_decision({}, {}, call_name="test decision"))
    assert raised.value is failure
    backend.logger.exception.assert_called_once_with(
        "[KEV] Decision failed", task="test decision", model=backend.model,
    )
    usage.assert_called_once_with("test decision", None)


@pytest.mark.parametrize("routing_backend", ["llm", "decision"])
def test_agent_initializes_kev_before_first_request(routing_backend):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    checkpoint = Mock(meta=SimpleNamespace(temperature=1.0))
    checkpoint.load.return_value = (Mock(), Mock())
    with patch("kev.checkpoint.Checkpoint", return_value=checkpoint) as load_checkpoint:
        agent = AutomotiveAgent(ConfigAssistant(
            routing_backend=routing_backend, pre_classify_backend="decision",
            decision_model="kev",
        ))
    load_checkpoint.assert_called_once_with("jaredpalmer/kev-0.8b")
    checkpoint.load.assert_called_once()
    if routing_backend == "decision":
        assert agent.telemetry_router.backend is agent.request_pre_classifier.backend


def test_agent_does_not_load_kev_when_decision_backend_is_unused():
    from unittest.mock import patch
    from agents.automotive_agent import AutomotiveAgent
    from config import ConfigAssistant

    with patch("kev.checkpoint.Checkpoint") as load_checkpoint:
        AutomotiveAgent(ConfigAssistant(decision_model="kev"))
    load_checkpoint.assert_not_called()


def test_laya_backend_only_predicts_with_supplied_state_and_questions():
    import asyncio
    from unittest.mock import Mock
    from agents.backends.decision_backend import LayaBackend
    from config import ConfigAssistant

    predictor = Mock()
    predictor.predict.return_value = {"answers": {"task": {"choice": "wellbeing"}}}
    usage = Mock()
    backend = LayaBackend(None, usage, predictor=predictor)
    state, questions = {"fatigue": "high"}, {"task": {"type": "choice"}}
    assert asyncio.run(backend.execute_decision(state, questions)) == predictor.predict.return_value
    predictor.predict.assert_called_once_with(state, questions)
    assert not hasattr(backend, "agent")
    assert not hasattr(backend, "process_event")
    usage.assert_called_once_with("decision", None)


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
    inputs = DecisionInput(
        changed_facts="changed data", context="supporting data", driver_name="David",
        pending_message_count=2, event_details="event data",
    )
    assert asyncio.run(task.execute(inputs)) is decision
    crew_inputs = task.crew.kickoff_async.await_args.kwargs["inputs"]
    assert crew_inputs["changed_facts"] == "changed data"
    assert crew_inputs["context"] == "supporting data"
    if module_name == "meeting":
        assert set(crew_inputs) == {"driver_name", "changed_facts", "context", "meeting_permission_state"}
        assert crew_inputs["driver_name"] == "David"
        assert crew_inputs["meeting_permission_state"] == "not_asked"
    elif module_name == "messages":
        assert set(crew_inputs) == {
            "pending_message_count", "event_details", "changed_facts", "context",
            "task_instructions", "action_options", "suggestion_options", "urgency_options",
            "tone_options", "intervention_options",
        }
        assert crew_inputs["pending_message_count"] == 2
        assert crew_inputs["event_details"] == "event data"
    else:
        assert set(crew_inputs) == {
            "changed_facts", "context", "task_instructions", "action_options",
            "suggestion_options", "urgency_options", "tone_options", "intervention_options",
        }
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


def test_wellbeing_can_enable_sentry_when_changed_occupancy_reaches_zero():
    from agents.tasks.wellbeing import TASK
    from data.agents_dataclasses import ActionType

    assert ActionType.ENABLE_SENTRY_MODE in TASK.actions
    assert "the cabin is empty (the detected occupant count is" in TASK.instructions
    assert "Supporting facts show sentry mode is off" in TASK.instructions
    assert "an isolated sentry-mode change" in TASK.instructions


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
    # routing_model defaults to None: the router falls back to the main model.
    assert agent.telemetry_router.model == "qwen3.5:4b"
    assert "Vehicle action reference" in agent.command_task.crew.tasks[0].description
    assert agent.command_task.backend is agent.crewai_backend
    assert not hasattr(agent, "process_event_llm")
    assert not hasattr(agent, "process_event_laya")


def test_agent_selects_laya_router_without_changing_specialist_backends():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock, patch
    from agents.backends.decision_backend import LayaBackend
    from agents.automotive_agent import AutomotiveAgent
    from agents.tasks.routing import TelemetryRouter
    from config import ConfigAssistant

    opt = ConfigAssistant(routing_backend="decision", routing_model="router-model",
                          laya_model="router-checkpoint")
    backend = Mock(spec=LayaBackend)
    backend.execute_decision = AsyncMock()
    with patch("agents.automotive_agent.LayaBackend", return_value=backend) as build_backend:
        agent = AutomotiveAgent(opt)
    assert isinstance(agent.telemetry_router, TelemetryRouter)
    assert agent.telemetry_router.backend is backend
    assert all(task.backend is agent.crewai_backend for task in agent.tasks.values())
    assert build_backend.call_args.args[0] == "router-checkpoint"


@pytest.mark.parametrize("model_kind", ["default", "path", "string"])
def test_laya_backend_loads_model_without_assistant_configuration(tmp_path, model_kind):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from agents.backends.decision_backend import LayaBackend

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
    from agents.backends.decision_backend import LayaBackend

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
        OmegaConf.from_dotlist(["routing_backend=decision", "routing_model=/tmp/router-checkpoint",
                       "pre_classify_model=/tmp/request-checkpoint", "routing_model=router-llm"]),
    )
    opt = OmegaConf.to_object(configuration)
    assert opt.routing_backend == "decision"
    assert opt.pre_classify_model == "/tmp/request-checkpoint"
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
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from agents.automotive_agent import AutomotiveAgent
    from agents.tasks.decision import NotificationDecision
    from data.agents_dataclasses import ActionType, InterventionType, SuggestionType, ToneType, UrgencyType

    agent = AutomotiveAgent.__new__(AutomotiveAgent)
    agent.logger = Mock()
    agent.knowledge_facts_provider = lambda: {}
    agent.event_queue = asyncio.Queue()
    agent.action_manager = SimpleNamespace(
        registered_actions=frozenset(), registered_action_options="",
        registered_action_guidance="",
    )
    agent.telemetry_router = SimpleNamespace(execute=AsyncMock(return_value=TaskRoute(tasks=task_types)))
    agent.tasks = {task_type: SimpleNamespace(
        pending_count=0,
        execute=AsyncMock(return_value=NotificationDecision(
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


def test_automotive_agent_uses_structured_context_instead_of_prefix_parsing():
    import asyncio
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    changed_fact = "The driver is using a phone."
    supporting_fact = "Changed just now: This phrase is part of an unchanged fact."
    agent = make_routed_agent([TaskType.WELLBEING])
    event = CarEvent(
        SkillType.NONE,
        EventName.KNOWLEDGE_UPDATED,
        {"DriverPhysicalState.activity": "On the phone"},
        [f"Changed just now: {changed_fact}", supporting_fact],
        changed_facts=[changed_fact],
        supporting_facts=[supporting_fact],
    )

    asyncio.run(agent.process_car_event(event))

    inputs = agent.tasks[TaskType.WELLBEING].execute.await_args.args[0]
    assert inputs.changed_facts == f"- {changed_fact}"
    assert inputs.context == f"- {supporting_fact}"


def test_late_meeting_scenario_routes_through_router_prompt():
    import asyncio
    from agents.tasks import routing
    from data.events import CarEvent, EventName
    from data.agents_dataclasses import SkillType

    # Meeting routing is engine-driven: the prompt encodes the activation rule
    # in natural language, without internal fact identifiers.
    assert "late, on time, arrived" in routing.ROUTING_PROMPT
    assert "Driver arrival" in routing.ROUTING_PROMPT
    assert "meeting:" in routing.ROUTING_PROMPT
    assert "unread-message" in routing.ROUTING_PROMPT
    assert '"DriverAgenda.late_for_meeting"' not in routing.ROUTING_PROMPT
    assert '"UnreadMessages.count"' not in routing.ROUTING_PROMPT

    # When the router (the engine's answer) selects meeting, the task runs.
    agent = make_routed_agent([TaskType.MEETING])
    event = CarEvent(SkillType.NONE, EventName.KNOWLEDGE_UPDATED,
        {"DriverAgenda.meeting_timing": "late"}, [], facts={
            "DriverAgenda.meeting_timing": "The driver is late for the meeting.",
            "DriverAgenda.meeting_status": "The meeting is being prepared and has not started yet.",
        })
    asyncio.run(agent.process_car_event(event))
    agent.tasks[TaskType.MEETING].execute.assert_awaited_once()
    inputs = agent.tasks[TaskType.MEETING].execute.await_args.args[0]
    assert "late for the meeting" in inputs.changed_facts
    assert "being prepared" in inputs.context


def test_blocked_messages_use_unified_action_and_most_urgent_tone():
    import asyncio
    from unittest.mock import AsyncMock, Mock
    from agents.automotive_agent import AutomotiveAgent
    from agents.tasks.decision import NotificationDecision
    from data.agents_dataclasses import ActionType, InterventionType, SkillType, SuggestionType, ToneType, UrgencyType
    from data.events import CarEvent, EventName

    agent = make_routed_agent([TaskType.MESSAGES])
    agent.handle_notification_decision = AutomotiveAgent.handle_notification_decision.__get__(agent)
    agent.action_manager.handle_decision = Mock()
    agent.messages_task.most_urgent_classification = Mock(
        return_value=(ToneType.SERIOUS, UrgencyType.CRITICAL),
    )

    async def handle_message_decision(event, decision, *, on_classified):
        on_classified(decision.tone, decision.urgency)

    agent.messages_task.handle_decision = AsyncMock(side_effect=handle_message_decision)
    agent.on_incoming_message_classified = Mock()
    decision = NotificationDecision(
        urgency=UrgencyType.LOW,
        tone=ToneType.CALM,
        intervention_type=InterventionType.ACT,
        action=ActionType.BLOCK_PENDING_MESSAGES,
        suggestion_type=SuggestionType.NONE,
        reason="Driver attention is low.",
    )

    asyncio.run(agent.handle_notification_decision(
        CarEvent(SkillType.NONE, EventName.KNOWLEDGE_UPDATED, {}, []),
        decision,
        task_type=TaskType.MESSAGES,
    ))

    agent.messages_task.handle_decision.assert_awaited_once()
    agent.on_incoming_message_classified.assert_called_once_with(
        ToneType.SERIOUS, UrgencyType.CRITICAL,
    )
    agent.action_manager.handle_decision.assert_called_once_with(
        ActionType.BLOCK_PENDING_MESSAGES, {},
    )


@pytest.mark.parametrize("outcome", ["complete", "cancel_generation", "cancel_sentence", "cancel_playback", "error"])
def test_conversation_cleans_up_child_tasks(outcome):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock

    from agents.tasks.conversation import ConversationDelivery, ConversationInput, ConversationTask
    from config import ConfigAssistant

    async def check():
        baseline = set(asyncio.all_tasks())
        generation_waiting = asyncio.Event()
        release_generation = asyncio.Event()
        speech_started = asyncio.Event()
        release_speech = asyncio.Event()
        stream_closed = asyncio.Event()
        if outcome not in {"cancel_generation", "cancel_sentence"}:
            release_generation.set()
        if outcome == "complete":
            release_speech.set()

        async def stream(**kwargs):
            try:
                if outcome == "error":
                    raise RuntimeError("generation failed")
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                    content="[[tone:calm]]" if outcome == "cancel_sentence" else "[[tone:calm]]Hello there."
                ))])
                generation_waiting.set()
                await release_generation.wait()
            finally:
                stream_closed.set()

        async def speak(sentence, on_word):
            speech_started.set()
            await release_speech.wait()

        task = ConversationTask(ConfigAssistant(), Mock(), backend=SimpleNamespace(stream=stream))
        response = asyncio.create_task(task.execute(
            ConversationInput("Hello", [], []), ConversationDelivery(speak=speak),
        ))
        try:
            if outcome.startswith("cancel"):
                ready = generation_waiting if outcome in {"cancel_generation", "cancel_sentence"} else speech_started
                await asyncio.wait_for(ready.wait(), 1)
                response.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await response
            elif outcome == "error":
                with pytest.raises(RuntimeError, match="generation failed"):
                    await response
            else:
                await response
            pending = [task for task in asyncio.all_tasks() if task not in baseline and not task.done()]
            assert pending == []
            assert stream_closed.is_set()
        finally:
            pending = [task for task in asyncio.all_tasks() if task not in baseline and not task.done()]
            for child in pending:
                child.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(check())


def test_conversation_talking_waits_for_first_spoken_word():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock
    from agents.tasks.conversation import ConversationDelivery, ConversationInput, ConversationTask
    from config import ConfigAssistant
    from data.agents_dataclasses import AssistantStatus

    events = []

    async def stream(**kwargs):
        for text in ["[[tone:calm]]", "Hello there."]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
            await asyncio.sleep(0)

    async def speak(sentence, on_word):
        assert events == []
        await asyncio.sleep(0)
        assert events == []
        on_word("Hello")
        assert ("text", "Hello") in events
        assert ("tone", "calm") in events
        assert ("status", AssistantStatus.TALKING) in events
        on_word(sentence)

    task = ConversationTask(ConfigAssistant(), Mock(), backend=SimpleNamespace(stream=stream))
    asyncio.run(task.execute(
        ConversationInput("Hello", [], []),
        ConversationDelivery(
            speak=speak,
            on_text=lambda text: events.append(("text", text)),
            on_tone=lambda tone: events.append(("tone", tone)),
            on_status=lambda status: events.append(("status", status)),
        ),
    ))
    assert events[0] == ("text", "Hello")
    assert events[-2:] == [("tone", None), ("status", AssistantStatus.IDLE)]


def test_conversation_spoken_words_keep_previous_sentences_visible():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock
    from agents.tasks.conversation import ConversationDelivery, ConversationInput, ConversationTask
    from config import ConfigAssistant

    updates = []

    async def stream(**kwargs):
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content="[[tone:calm]]Hello there. Enjoy your drive. Stay safe."
        ))])

    async def speak(sentence, on_word):
        words = sentence.split()
        for count in range(1, len(words) + 1):
            on_word(" ".join(words[:count]))
            await asyncio.sleep(0)

    task = ConversationTask(ConfigAssistant(), Mock(), backend=SimpleNamespace(stream=stream))
    result = asyncio.run(task.execute(
        ConversationInput("Hello", [], []),
        ConversationDelivery(speak=speak, on_text=updates.append),
    ))

    assert updates == [
        "Hello", "Hello there.", "Hello there.",
        "Hello there. Enjoy", "Hello there. Enjoy your",
        "Hello there. Enjoy your drive.", "Hello there. Enjoy your drive.",
        "Hello there. Enjoy your drive. Stay",
        "Hello there. Enjoy your drive. Stay safe.",
        "Hello there. Enjoy your drive. Stay safe.",
    ]
    assert result.spoken == "Hello there. Enjoy your drive. Stay safe."


def test_wait_message_talking_waits_for_first_chunk():
    import asyncio
    from unittest.mock import Mock
    from data.agents_dataclasses import AssistantStatus

    async def check():
        events = []
        agent = make_routed_agent([])
        agent.set_assistant_status = Mock(side_effect=lambda status: events.append(("status", status)))
        agent.on_response_update = lambda text: events.append(("text", text))
        agent.on_response = Mock()
        agent.tts_manager = None
        chunks = asyncio.Queue()
        delivery = asyncio.create_task(agent.speak_wait_message_chunks(chunks))
        try:
            await asyncio.sleep(0)
            assert events == []
            chunks.put_nowait("Checking...")
            chunks.put_nowait(None)
            await delivery
            assert events == [
                ("text", "Checking..."),
                ("status", AssistantStatus.TALKING),
                ("status", AssistantStatus.IDLE),
            ]
        finally:
            delivery.cancel()
            await asyncio.gather(delivery, return_exceptions=True)

    asyncio.run(check())


def test_short_message_talking_waits_for_first_spoken_word():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock
    from data.agents_dataclasses import AssistantStatus, ToneType

    events = []
    agent = make_routed_agent([])
    agent.conversation_history = []
    agent.set_assistant_status = Mock(side_effect=lambda status: events.append(("status", status)))
    agent.on_speaking_tone_changed = lambda tone: events.append(("tone", tone))
    agent.on_response_update = lambda text: events.append(("text", text))
    agent.on_response = Mock()

    async def speak(message, on_word):
        assert events == []
        await asyncio.sleep(0)
        assert events == []
        on_word("Hello")
        assert ("text", "Hello") in events
        assert ("tone", "calm") in events
        assert ("status", AssistantStatus.TALKING) in events

    agent.tts_manager = SimpleNamespace(speak=speak)
    asyncio.run(agent.speak("Hello there.", tone=ToneType.CALM))
    assert events[0] == ("text", "Hello")
    assert events[-2:] == [("tone", None), ("status", AssistantStatus.IDLE)]


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
    system_prompt = create.await_args.kwargs["messages"][0]["content"]
    assert "Do not ask questions unless strictly necessary" in system_prompt
    assert "or obtain required consent" in system_prompt
    assert "Respond to greetings with one brief, friendly statement and no question." in system_prompt
    assert '"How is your journey going so far?"' in system_prompt
    assert "Before responding, silently remove every question that is not indispensable." in system_prompt
    assert "Once the request is answered, stop." in system_prompt
    ttft_calls = [call for call in logger.debug.call_args_list if call.args[0] == "[LLM] Time to first token"]
    assert len(ttft_calls) == 1
    assert ttft_calls[0].kwargs["llm_call"] == "conversation response"
    logger.info.assert_called_once()
    summary = logger.info.call_args.kwargs
    assert summary["response_text"] == "Hello. Take care."
    assert summary["status"] == "completed"
    assert summary["ttft_ms"] >= 0
    assert summary["duration_ms"] >= summary["ttft_ms"]


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
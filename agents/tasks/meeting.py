import json

from crewai import Agent, Crew, Process, Task
from agents.backends.crewai_backend import CrewAIBackend
from agents.tasks.decision import DECISION_PROMPT, DecisionInput, DecisionTask, NotificationDecision, enum_options
from agents.tasks.routing import TaskType
from data.agents_dataclasses import ActionType, InterventionType, SkillType, ToneType, UrgencyType


TASK = DecisionTask(
    task_type=TaskType.MEETING,
    skill=SkillType.CONVERSATION,
    context_fields=("DriverAgenda.late_for_meeting",),
    actions=(ActionType.NONE, ActionType.ASK_ATTEND_MEETING),
    instructions="""
This is the directly invoked late-for-meeting scenario, not general telemetry.
Changed facts explicitly say the driver is late for an upcoming meeting =>
ask_attend_meeting, intervention_type=act, urgency=high, suggestion_type=none.
spoken_message asks permission to attend on their behalf and brief them afterwards.
Empty user input is expected. Asking permission IS the action; do not suppress it
as routine telemetry. If not late, remain silent. Never claim a meeting was joined
or invent details. Actual attendance requires the driver's subsequent confirmation.
""",
)


class MeetingTask:
    spec = TASK

    def __init__(self, opt, on_usage, *, backend=None):
        self.backend = backend if backend is not None else CrewAIBackend(opt, on_usage)
        agent = Agent(role="Meeting Assistant", goal="Ask permission when the driver is late.",
                      backstory="Never invent meeting details or claim attendance without confirmation.",
                      llm=self.backend.llm, tools=[], verbose=False, allow_delegation=False)
        task = Task(description=DECISION_PROMPT, expected_output="One meeting permission decision.",
                    output_pydantic=NotificationDecision, agent=agent)
        self.crew = Crew(agents=[agent], tasks=[task], verbose=False, tracing=False,
                         process=Process.sequential, memory=None)

    async def execute(self, task_input: DecisionInput) -> NotificationDecision:
        inputs = {
            "event_name": task_input.event_name,
            "driver_name": task_input.driver_name,
            "pending_message_count": task_input.pending_message_count,
            "event_details": task_input.event_details,
            "changed_facts": task_input.changed_facts,
            "context": task_input.context,
            "task_instructions": self.spec.instructions,
            "action_options": enum_options(self.spec.actions),
            "suggestion_options": enum_options(self.spec.suggestions),
            "urgency_options": enum_options(UrgencyType),
            "tone_options": enum_options(ToneType),
            "intervention_options": enum_options(InterventionType),
        }
        decision = await self.backend.run(self.crew, inputs, NotificationDecision, "meeting")
        if decision.action not in self.spec.actions or decision.suggestion_type not in self.spec.suggestions:
            raise ValueError("meeting returned a capability outside its contract")
        decision.skill = SkillType.CONVERSATION if (
            decision.notify or decision.action is not ActionType.NONE
        ) else SkillType.NONE
        return decision


def meeting_reply_context(confirmed: bool | None) -> tuple[str, str]:
    return (
        "This response is to the driver's reply to a pending request to attend a meeting. "
        "Use the supplied confirmation result as authoritative. Generate a brief, natural reply "
        "from the actual conversation and vehicle context, not a canned acknowledgement. "
        "If confirmed is true, acknowledge permission and intended attendance; do not ask again "
        "or claim the meeting has already been attended. If false, acknowledge the refusal "
        "without offering to join again. If null, request clarification without treating it as "
        "acceptance or refusal. Never invent meeting details or completed actions.",
        "Meeting confirmation result: " + json.dumps({"confirmed": confirmed}),
    )
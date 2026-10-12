from crewai import Agent, Crew, Process, Task
from agents.backends.crewai_backend import CrewAIBackend
from agents.tasks.decision import DECISION_PROMPT, DecisionInput, DecisionTask, NotificationDecision, enum_options
from agents.tasks.routing import TaskType
from data.agents_dataclasses import ActionType, InterventionType, SkillType, ToneType, UrgencyType


TASK = DecisionTask(
    task_type=TaskType.MESSAGES,
    skill=SkillType.CONVERSATION,
    context_fields=(
        "VehicleState.privacy_mode", "DetectedObjects.people_inside",
        "DriverPhysicalState.attention_level", "DriverPhysicalState.fatigue_level",
    ),
    actions=(
        ActionType.NONE, ActionType.ASK_PERMISSION_TO_TALK,
        ActionType.POSTPONE_NOTIFICATION_DELIVERY, ActionType.READ_PENDING_MESSAGES,
    ),
    instructions="""
Handle incoming messages, reminders and recovery of pending delivery.
Classify incoming MESSAGE CONTENT independently of permission: serious for breakdowns,
safety issues or urgent help, enthusiastic for genuinely playful/positive content;
urgency=high for ASAP/immediate requests, stranded people or near deadlines, critical
only for immediate serious-injury risk. Bad news is not enthusiastic. Urgency and
tone must remain classified even when delivery is blocked.
For incoming_message_received, apply this exhaustive ordered table:
1. Manual privacy ON/unknown OR more than one occupant => ask_permission_to_talk.
2. Privacy explicitly OFF, <=1 occupant, high/very high fatigue OR low/very low
attention => postpone_notification_delivery.
3. Otherwise => read_pending_messages.
Use intervention_type=act, suggestion_type=none, spoken_message=null for all three.
Never reveal sender/content in spoken_message; the message manager delivers the queue.
Message urgency never changes this table. Heavy traffic alone is not a privacy veto.
For pending_messages_reminder, pending_count=0 => silent. Otherwise apply the same
table: urgency=low, discreet for permission/postpone, empathetic for read, no spoken text.
For knowledge_updated recovery, pending_count>0, privacy explicitly OFF, <=1 occupant,
and low/no fatigue OR high/very high attention => read_pending_messages, urgency=low,
tone=empathetic, with a short spoken introduction. A high fatigue/low attention fact
still vetoes reading. Favorable recovery may be supporting context, not the trigger.
Read the whole pending queue; unknown privacy or occupancy never authorizes recovery.
Otherwise remain silent. Never treat privacy as an emotion, urgency, or vehicle action.
""",
)


class MessagesTask:
    spec = TASK

    def __init__(self, opt, on_usage, *, backend=None):
        self.backend = backend if backend is not None else CrewAIBackend(opt, on_usage)
        agent = Agent(role="Message Delivery Specialist", goal="Classify content and respect delivery policy.",
                      backstory="Urgency never overrides privacy or driver readiness.",
                      llm=self.backend.llm, tools=[], verbose=False, allow_delegation=False)
        task = Task(description=DECISION_PROMPT, expected_output="One message delivery decision.",
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
        decision = await self.backend.run(self.crew, inputs, NotificationDecision, "messages")
        if decision.action not in self.spec.actions or decision.suggestion_type not in self.spec.suggestions:
            raise ValueError("messages returned a capability outside its contract")
        decision.skill = SkillType.CONVERSATION if (
            decision.notify or decision.action is not ActionType.NONE
        ) else SkillType.NONE
        return decision
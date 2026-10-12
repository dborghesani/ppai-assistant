from crewai import Agent, Crew, Process, Task
from agents.backends.crewai_backend import CrewAIBackend
from agents.tasks.decision import DECISION_PROMPT, DecisionInput, DecisionTask, NotificationDecision, enum_options
from agents.tasks.routing import TaskType
from data.agents_dataclasses import ActionType, InterventionType, SkillType, SuggestionType, ToneType, UrgencyType


TASK = DecisionTask(
    task_type=TaskType.WELLBEING,
    skill=SkillType.WELLBEING,
    context_fields=(
        "DriverPhysicalState", "DetectedObjects.children_inside",
        "DetectedObjects.people_inside", "DetectedObjects.animal_inside",
        "VehicleMotion.speed", "VehicleMotion.moving", "VehicleState.adas_target_speed",
        "VehicleState.following_distance_level", "VehicleState.adaptive_cruise_control_on",
        "VehicleState.lane_keep_assist_enabled", "VehicleState.blind_spot_monitor",
        "EnvironmentState.traffic", "EnvironmentState.time_of_day",
    ),
    actions=(ActionType.NONE, ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE),
    suggestions=(
        SuggestionType.NONE, SuggestionType.TAKE_BREAK,
        SuggestionType.RESTORE_ATTENTION, SuggestionType.REDUCE_DISTRACTION,
    ),
    instructions="""
Protect driver wellbeing and cabin occupant safety. Fatigue, attention and activities
are not emotions; never propose music or infer an emotion from tiredness.
Changed high/very high fatigue or low/very low attention => suggest a safe break,
urgency=low, action=none, suggestion_type=take_break.
Only explicitly persistent fatigue/low attention AND explicitly moving vehicle =>
apply_restrictive_adas_profile. Severity, night, traffic and repeated readings do not
establish persistence. If stationary or motion unknown, suggest a break without ADAS.
Detected phone use or sleeping while moving => warn and apply_restrictive_adas_profile.
Other explicitly distracting activities, such as eating, justify a concise warning
when relevant to driving, not automatic ADAS. Idle/normal activity remains silent.
Children out of place => calm high-priority warning to ensure supervision and that
they are clear of the vehicle before moving; if moving, advise a stop only when safe.
Do not invent their location or immediate danger. Mere child/animal presence is not
a hazard. Normal attention and low fatigue do not require a wellbeing announcement.
The simulated ADAS profile lowers target speed, increases following distance and
enables assistance; never claim physical actuation or change measured vehicle speed.
""",
)


class WellbeingTask:
    spec = TASK

    def __init__(self, opt, on_usage, *, backend=None):
        self.backend = backend if backend is not None else CrewAIBackend(opt, on_usage)
        agent = Agent(role="Driver Wellbeing Specialist", goal="Evaluate driver and cabin safety.",
                      backstory="Use only reported facts and supported actions.",
                      llm=self.backend.llm, tools=[], verbose=False, allow_delegation=False)
        task = Task(description=DECISION_PROMPT, expected_output="One wellbeing decision.",
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
        decision = await self.backend.run(self.crew, inputs, NotificationDecision, "wellbeing")
        if decision.action not in self.spec.actions or decision.suggestion_type not in self.spec.suggestions:
            raise ValueError("wellbeing returned a capability outside its contract")
        decision.skill = SkillType.WELLBEING if (
            decision.notify or decision.action is not ActionType.NONE
        ) else SkillType.NONE
        return decision
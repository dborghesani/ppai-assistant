from crewai import Agent, Crew, Process, Task
from agents.backends.crewai_backend import CrewAIBackend
from agents.tasks.decision import DECISION_PROMPT, DecisionInput, DecisionTask, NotificationDecision, enum_options
from agents.tasks.routing import TaskType
from data.agents_dataclasses import ActionType, InterventionType, SkillType, ToneType, UrgencyType


TASK = DecisionTask(
    task_type=TaskType.COMFORT,
    skill=SkillType.WELLBEING,
    context_fields=(
        "VehicleState.internal_temperature", "VehicleState.heating_on",
        "VehicleState.air_conditioning_on", "VehicleState.fan_speed",
        "VehicleState.air_recirculation_on", "VehicleState.seat_heating_on",
        "DriverPreferences.preferred_cabin_temperature",
    ),
    actions=(ActionType.NONE, ActionType.ENABLE_HEATING, ActionType.ENABLE_AIR_CONDITIONING),
    instructions="""
Handle cabin comfort. Relative cabin temperature already reflects the driver's
preference; do not recompute it or treat the preference as a measured temperature.
Changed high/too high cabin temperature + air conditioning explicitly off =>
enable_air_conditioning. Changed low/too low + cabin heating explicitly off =>
enable_heating, never cooling. These are useful actions even while stationary:
urgency=low, intervention_type=act, suggestion_type=none, brief acknowledgement.
Optimal temperature, no temperature trigger, unknown required control state,
or required control already on => silence, action=none. Never reverse on/off.
Heating and seat heating are different controls. Do not request a target value
or claim measured temperature changed or physical actuation was confirmed.
""",
)


class ComfortTask:
    spec = TASK

    def __init__(self, opt, on_usage, *, backend=None):
        self.backend = backend if backend is not None else CrewAIBackend(opt, on_usage)
        agent = Agent(role="Cabin Comfort Specialist", goal="Evaluate useful climate adjustments.",
                      backstory="Respect current controls and relative temperature facts.",
                      llm=self.backend.llm, tools=[], verbose=False, allow_delegation=False)
        task = Task(description=DECISION_PROMPT, expected_output="One comfort decision.",
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
        decision = await self.backend.run(self.crew, inputs, NotificationDecision, "comfort")
        if decision.action not in self.spec.actions or decision.suggestion_type not in self.spec.suggestions:
            raise ValueError("comfort returned a capability outside its contract")
        decision.skill = SkillType.WELLBEING if (
            decision.notify or decision.action is not ActionType.NONE
        ) else SkillType.NONE
        return decision
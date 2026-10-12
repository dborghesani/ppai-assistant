from crewai import Agent, Crew, Process, Task
from agents.backends.crewai_backend import CrewAIBackend
from agents.tasks.decision import DECISION_PROMPT, DecisionInput, DecisionTask, NotificationDecision, enum_options
from agents.tasks.routing import TaskType
from data.agents_dataclasses import ActionType, InterventionType, SkillType, SuggestionType, ToneType, UrgencyType


TASK = DecisionTask(
    task_type=TaskType.DRIVING,
    skill=SkillType.DRIVING,
    context_fields=(
        "VehicleMotion.speed", "VehicleMotion.moving", "VehicleState.turn_signal", "LaneTracing",
        "DriverDrivingStyle", "EnvironmentState", "DetectedObjects.traffic_signs",
        "DetectedObjects.dangerous_objects_around", "DetectedObjects.people_around",
        "DetectedObjects.vehicles_around", "VehicleState.doors_locked",
        "VehicleState.door_open_front_left", "VehicleState.door_open_front_right",
        "VehicleState.door_open_rear_left", "VehicleState.door_open_rear_right",
        "VehicleState.trunk_open", "VehicleState.lights_on_fog_lights",
        "VehicleState.lights_on_low_beams", "VehicleState.lights_on_sidelights",
    ),
    actions=(
        ActionType.NONE, ActionType.LOCK_DOORS,
        ActionType.ENABLE_FOG_LIGHTS, ActionType.ENABLE_LOW_BEAM_HEADLIGHTS,
    ),
    suggestions=(
        SuggestionType.NONE, SuggestionType.CALM_DRIVING, SuggestionType.REDUCE_SPEED,
        SuggestionType.USE_TURN_SIGNAL, SuggestionType.ADAPT_DRIVING_TO_CONDITIONS,
        SuggestionType.SECURE_VEHICLE, SuggestionType.PREPARE_FOR_WEATHER,
        SuggestionType.PREPARE_FOR_MANEUVER,
    ),
    instructions="""
Provide driving coaching, not wellbeing or music. Treat reported driving tension
as a driving-style signal; excessively aggressive/high tension warrants calm-driving
guidance, never an inferred emotion or playlist.
Warn about a lane change with turn signal Off. An active signal with no manoeuvre
requires explicit sustained duration before warning; a newly active signal is routine.
Never replace the actual Left/Right/Both value with an example.
Vehicle speed too high means >10 km/h above the detected limit: warn. High alone,
optimal/low/too low and fluctuations alone remain silent. An ADAS target is not a
speed limit. Without reliable evidence, never invent a limit or unsafe speed.
Warn when a door/trunk is open AND the vehicle is explicitly moving.
Dangerous object detected + doors explicitly unlocked => warn and lock_doors;
if already locked, do not repeat the action. Do not invent location or proximity.
Fog => enable fog lights only when explicitly off. Low visibility + low beams
explicitly off => enable low beams. Do not repeat already enabled controls.
Hazardous road/weather guidance needs explicit current evidence in changed facts.
Lane advice requires a confirmed deviation tied to an exit or specific risk.
Do not modify manual privacy mode. Prioritize hazards over convenience.
""",
)


class DrivingTask:
    spec = TASK

    def __init__(self, opt, on_usage, *, backend=None):
        self.backend = backend if backend is not None else CrewAIBackend(opt, on_usage)
        agent = Agent(role="Driving Coach", goal="Provide evidence-grounded driving guidance.",
                      backstory="Never invent road conditions, speed limits or hazards.",
                      llm=self.backend.llm, tools=[], verbose=False, allow_delegation=False)
        task = Task(description=DECISION_PROMPT, expected_output="One driving decision.",
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
        decision = await self.backend.run(self.crew, inputs, NotificationDecision, "driving")
        if decision.action not in self.spec.actions or decision.suggestion_type not in self.spec.suggestions:
            raise ValueError("driving returned a capability outside its contract")
        decision.skill = SkillType.DRIVING if (
            decision.notify or decision.action is not ActionType.NONE
        ) else SkillType.NONE
        return decision
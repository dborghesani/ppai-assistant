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
        ActionType.INCREASE_FOLLOWING_DISTANCE,
        ActionType.ENABLE_ADAPTIVE_CRUISE_CONTROL,
        ActionType.ENABLE_LANE_KEEP_ASSIST,
        ActionType.ENABLE_BLIND_SPOT_MONITOR,
        ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE,
    ),
    suggestions=(
        SuggestionType.NONE, SuggestionType.CALM_DRIVING, SuggestionType.REDUCE_SPEED,
        SuggestionType.USE_TURN_SIGNAL, SuggestionType.ADAPT_DRIVING_TO_CONDITIONS,
        SuggestionType.SECURE_VEHICLE, SuggestionType.PREPARE_FOR_WEATHER,
        SuggestionType.PREPARE_FOR_MANEUVER,
    ),
    instructions="""
Provide driving coaching, not wellbeing or music. Driving tension is a driving-style
signal, not an inferred emotion. Never use a naming/shaming tone or suggest music.
Changed facts identify the trigger. Supporting facts describe context, NOT changes.
Apply the following driving-tension table literally; high AND very high qualify:
- Changed facts reports high or very high driving tension WITHOUT "This condition
    has persisted for a while" => initial advice, NOT silence and NOT ADAS:
    urgency=medium, tone=calm, intervention_type=suggest, action=none,
    suggestion_type=calm_driving, spoken_message="You're driving with high tension.
    Try to relax and drive more smoothly."
    This advice is REQUIRED even if stationary/parked and no hazards are present.
    Persistence is NOT a prerequisite for advice; it is ONLY needed for escalation.
- Changed facts reports high or very high driving tension WITH "This condition
    has persisted for a while" => escalate to an ADAS action with a calm-driving
    suggestion. Prior advice does not prevent escalation. Available aids: increase
    following distance, adaptive cruise control, lane keep assist, blind spot monitor,
    restrictive ADAS profile. Do not replace this action with advice alone.
    Set intervention_type=act and provide a nonempty spoken_message acknowledging
    the selected ADAS action in the first person as an action being initiated.
    For apply_restrictive_adas_profile, for example: "I'm activating the restrictive
    driver-assistance profile to help you drive more smoothly."
    For increase_following_distance: "I'm increasing the following distance to
    give you more space." Name only the selected action, not unselected aids or
    invented settings. Do not claim completion before it is confirmed.
    Do NOT say only "Please reduce your speed and increase following distance"
    or ask the driver to manually perform the action the assistant is taking.
    A brief calm-driving suggestion may follow, but must not replace the acknowledgement.
- Changed facts reports only ADAS activation/settings, or says "No specific vehicle
    fact changed." => SILENT, even with persistent very high tension in Supporting
    facts. Do not move that tension into Changed facts or treat ADAS activation as
    worsening tension. Do not acknowledge activation with another coaching message.
Silent means urgency=none, intervention_type=none, action=none,
suggestion_type=none, spoken_message=null.
If supplied evidence says ADAS already addressed the same tension episode, do not
repeat the action or return to initial advice for unchanged tension. Never invent
previous interventions. Explicit worsening or a new episode after recovery may
justify a fresh intervention. A new hazard in Changed facts always takes priority
over these silence rules; assess it using the hazard rules below.
Warn about a lane change with turn signal Off. An active signal with no manoeuvre
requires explicit sustained duration before warning; a newly active signal is routine.
Never replace the actual Left/Right/Both value with an example.
Vehicle speed too high means >10 km/h above the detected limit: warn. Speed rated
high alone, optimal/low/too low and speed fluctuations alone remain silent.
These speed-only silence rules do NOT apply to high driving tension. An ADAS target is not a
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
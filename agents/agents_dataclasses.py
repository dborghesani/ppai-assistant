from enum import Enum


class UrgencyType(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return tuple(type(self)).index(self)

    @property
    def description(self) -> str:
        return {
            UrgencyType.NONE: "No intervention is needed.",
            UrgencyType.LOW: "Minor issue or optional assistance.",
            UrgencyType.MEDIUM: "Intervention is advisable soon.",
            UrgencyType.HIGH: "Prompt intervention is strongly advisable.",
            UrgencyType.CRITICAL: "Immediate attention is required to reduce a serious safety risk.",
        }[self]

class ToneType(str, Enum):
    CALM = "calm" # default
    ENTHUSIASTIC = "enthusiastic"
    SERIOUS = "serious"
    EMPATHETIC = "empathetic"
    DISCREET = "discreet"

    @property
    def description(self) -> str:
        return {
            ToneType.CALM: "Use a calm and reassuring tone.",
            ToneType.ENTHUSIASTIC: "Use a lively and encouraging tone.",
            ToneType.SERIOUS: "Use a serious and direct tone.",
            ToneType.EMPATHETIC: "Use a warm and understanding tone.",
            ToneType.DISCREET: "Use a discreet and non-intrusive tone.",
        }[self]

class ActionType(str, Enum):
    NONE = "none"
    INCREASE_TEMPERATURE = "increase_temperature"
    DECREASE_TEMPERATURE = "decrease_temperature"
    ENABLE_AIR_CONDITIONING = "enable_air_conditioning"
    DISABLE_AIR_CONDITIONING = "disable_air_conditioning"
    LOCK_DOORS = "lock_doors"
    UNLOCK_DOORS = "unlock_doors"
    START_RADIO = "start_radio"
    STOP_RADIO = "stop_radio"
    START_NAVIGATION = "start_navigation"
    STOP_NAVIGATION = "stop_navigation"
    ENABLE_SIDELIGHTS = "enable_sidelights"
    DISABLE_SIDELIGHTS = "disable_sidelights"
    ENABLE_LOW_BEAM_HEADLIGHTS = "enable_low_beam_headlights"
    DISABLE_LOW_BEAM_HEADLIGHTS = "disable_low_beam_headlights"
    ENABLE_HIGH_BEAM_HEADLIGHTS = "enable_high_beam_headlights"
    DISABLE_HIGH_BEAM_HEADLIGHTS = "disable_high_beam_headlights"
    ENABLE_FOG_LIGHTS = "enable_fog_lights"
    DISABLE_FOG_LIGHTS = "disable_fog_lights"
    FIND_REST_AREA = "find_rest_area"

    @property
    def description(self) -> str:
        return {
            ActionType.NONE: "No direct action.",
            ActionType.INCREASE_TEMPERATURE: "increase temperature",
            ActionType.DECREASE_TEMPERATURE: "decrease temperature",
            ActionType.ENABLE_AIR_CONDITIONING: "enable air conditioning",
            ActionType.DISABLE_AIR_CONDITIONING: "disable air conditioning",
            ActionType.LOCK_DOORS: "lock doors",
            ActionType.UNLOCK_DOORS: "unlock doors",
            ActionType.START_RADIO: "start radio",
            ActionType.STOP_RADIO: "stop radio",
            ActionType.START_NAVIGATION: "start navigation",
            ActionType.STOP_NAVIGATION: "stop navigation",
            ActionType.ENABLE_SIDELIGHTS: "enable sidelights",
            ActionType.DISABLE_SIDELIGHTS: "disable sidelights",
            ActionType.ENABLE_LOW_BEAM_HEADLIGHTS: "enable low beam headlights",
            ActionType.DISABLE_LOW_BEAM_HEADLIGHTS: "disable low beam headlights",
            ActionType.ENABLE_HIGH_BEAM_HEADLIGHTS: "enable high beam headlights",
            ActionType.DISABLE_HIGH_BEAM_HEADLIGHTS: "disable high beam headlights",
            ActionType.ENABLE_FOG_LIGHTS: "enable fog lights",
            ActionType.DISABLE_FOG_LIGHTS: "disable fog lights",
            ActionType.FIND_REST_AREA: "Find the next rest area.",
        }[self]

class SkillType(str, Enum):
    NONE = "none"
    CONVERSATION = "conversation"
    DRIVING = "driving"
    WELLBEING = "wellbeing"

    @property
    def description(self) -> str:
        return {
            SkillType.NONE: "No skill is required.",
            SkillType.CONVERSATION: "General conversation, social interaction, or dialogue management.",
            SkillType.DRIVING: "Driving, navigation, vehicle state, road safety or coaching.",
            SkillType.WELLBEING: "Fatigue, attention, emotion or comfort.",
        }[self]

class InterventionType(str, Enum):
    NONE = "none"
    SUGGEST = "suggest"
    ACT = "act"

    @property
    def description(self) -> str:
        return {
            InterventionType.NONE: "Do not intervene.",
            InterventionType.SUGGEST: "Communicate a contextual recommendation without directly controlling a function.",
            InterventionType.ACT: "Execute one supported reversible vehicle or comfort action.",
        }[self]

class SuggestionType(str, Enum):
    NONE = "none"
    TAKE_BREAK = "take_break"
    RESTORE_ATTENTION = "restore_attention"
    REDUCE_DISTRACTION = "reduce_distraction"
    REGULATE_EMOTIONAL_STATE = "regulate_emotional_state"
    CALM_DRIVING = "calm_driving"
    REDUCE_SPEED = "reduce_speed"
    USE_TURN_SIGNAL = "use_turn_signal"
    ADAPT_DRIVING_TO_CONDITIONS = "adapt_driving_to_conditions"
    SECURE_VEHICLE = "secure_vehicle"
    PREPARE_FOR_WEATHER = "prepare_for_weather"
    PREPARE_FOR_MANEUVER = "prepare_for_maneuver"

    @property
    def description(self) -> str:
        return {
            SuggestionType.NONE: "No textual suggestion is needed.",
            SuggestionType.TAKE_BREAK: "Recommend stopping at the next safe opportunity because of fatigue or sleepiness.",
            SuggestionType.RESTORE_ATTENTION: "Prompt the driver to restore complete attention.",
            SuggestionType.REDUCE_DISTRACTION: "Recommend stopping or postponing a distracting activity.",
            SuggestionType.REGULATE_EMOTIONAL_STATE: "Recommend calming down or taking a short pause when safe.",
            SuggestionType.CALM_DRIVING: "Recommend smoother, less tense and less aggressive driving.",
            SuggestionType.REDUCE_SPEED: "Recommend reducing speed to a safer or compliant level.",
            SuggestionType.USE_TURN_SIGNAL: "Recommend using the turn signal before a maneuver.",
            SuggestionType.ADAPT_DRIVING_TO_CONDITIONS: "Recommend adapting speed, following distance or driving style to road, traffic, visibility, weather or hazards.",
            SuggestionType.SECURE_VEHICLE: "Recommend securing doors, trunk or another relevant vehicle state.",
            SuggestionType.PREPARE_FOR_WEATHER: "Recommend preparing for current or forecast weather.",
            SuggestionType.PREPARE_FOR_MANEUVER: "Recommend preparing for an upcoming exit, lane change or navigation event.",
        }[self]
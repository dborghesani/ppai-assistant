from enum import Enum


class UrgencyType(str, Enum):
    NONE = "none: no intervention is needed"
    LOW = "low: minor issue or optional assistance"
    MEDIUM = "medium: intervention is advisable soon"
    HIGH = "high: prompt intervention is strongly advisable"
    CRITICAL = "critical: immediate attention is required to reduce a serious safety risk"

class ActionType(str, Enum):
    NONE = "Do not execute a direct action"
    INCREASE_TEMPERATURE = "Increase the cabin temperature"
    DECREASE_TEMPERATURE = "Decrease the cabin temperature"
    ENABLE_AIR_CONDITIONING = "Enable air conditioning"
    DISABLE_AIR_CONDITIONING = "Disable air conditioning"
    LOCK_DOORS = "Lock the vehicle doors"
    UNLOCK_DOORS = "Unlock the vehicle doors"
    START_RADIO = "Start the radio"
    STOP_RADIO = "Stop the radio"
    START_NAVIGATION = "Start navigation"
    STOP_NAVIGATION = "Stop navigation"
    ENABLE_SIDELIGHTS = "Enable sidelights"
    DISABLE_SIDELIGHTS = "Disable sidelights"
    ENABLE_LOW_BEAM_HEADLIGHTS = "Enable low beam headlights"
    DISABLE_LOW_BEAM_HEADLIGHTS = "Disable low beam headlights"
    ENABLE_HIGH_BEAM_HEADLIGHTS = "Enable high beam headlights"
    DISABLE_HIGH_BEAM_HEADLIGHTS = "Disable high beam headlights"
    ENABLE_FOG_LIGHTS = "Enable fog lights"
    DISABLE_FOG_LIGHTS = "Disable fog lights"


class SkillType(str, Enum):
    NONE = "No skill is required"
    DRIVING = ("Driving, navigation, vehicle state, road conditions, "
               "safety, or driving coaching")
    WELLBEING = ("Driver fatigue, attention, emotional state, "
                 "physical state, or comfort")

class InterventionType(str, Enum):
    NONE = "No assistant intervention is useful or appropriate."
    SUGGEST = (
        "Provide the driver with a contextual suggestion, recommendation, "
        "or warning without directly controlling a vehicle function"
    )
    ACT = (
        "Execute one supported direct vehicle, comfort, navigation, "
        "or infotainment action"
    )

class SuggestionType(str, Enum):
    NONE = "No textual suggestion is needed"
    TAKE_BREAK =(
           "Recommend stopping at the next safe opportunity "
           "to recover from fatigue or sleepiness"
    )
    RESTORE_ATTENTION = (
        "Prompt the driver to restore full attention before continuing"
    )
    REDUCE_DISTRACTION = (
        "Recommend stopping or postponing a distracting activity"
    )
    REGULATE_EMOTIONAL_STATE = (
        "Recommend calming down, pausing, or regulating the driver's "
        "emotional state when safe"
    )
    CALM_DRIVING = (
        "Recommend smoother and less aggressive driving behavior"
    )
    REDUCE_SPEED = (
        "Recommend reducing speed to a safer or compliant level"
    )
    USE_TURN_SIGNAL = (
        "Recommend using the turn signal before or during "
        "the appropriate maneuver"
    )
    ADAPT_DRIVING_TO_CONDITIONS = (
        "Recommend adapting speed, following distance, or driving style "
        "to road, weather, visibility, traffic, or detected hazards"
    )
    SECURE_VEHICLE = (
        "Recommend securing doors, trunk, or another relevant "
        "vehicle state before continuing"
    )
    PREPARE_FOR_WEATHER = (
        "Recommend preparing for current or predicted weather conditions"
    )
    PREPARE_FOR_MANEUVER = (
        "Recommend preparing for an upcoming exit, lane change, "
        "route event, or maneuver"
    )
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
    CALM = "calm"  # UI: blue (#78baff)
    ENTHUSIASTIC = "enthusiastic"  # UI: yellow (#ffd166)
    SERIOUS = "serious"  # UI: red (#ff3b30)
    EMPATHETIC = "empathetic"  # UI: pink (#efa2c7)
    DISCREET = "discreet"  # UI: periwinkle (#a9bcff)

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
    ENABLE_HEATING = "enable_heating"
    DISABLE_HEATING = "disable_heating"
    OPEN_WINDOWS = "open_windows"
    CLOSE_WINDOWS = "close_windows"
    OPEN_SUNROOF = "open_sunroof"
    CLOSE_SUNROOF = "close_sunroof"
    ENABLE_AIR_CONDITIONING = "enable_air_conditioning"
    DISABLE_AIR_CONDITIONING = "disable_air_conditioning"
    INCREASE_FAN_SPEED = "increase_fan_speed"
    DECREASE_FAN_SPEED = "decrease_fan_speed"
    ENABLE_AIR_RECIRCULATION = "enable_air_recirculation"
    DISABLE_AIR_RECIRCULATION = "disable_air_recirculation"
    ENABLE_SEAT_HEATING = "enable_seat_heating"
    DISABLE_SEAT_HEATING = "disable_seat_heating"
    LOCK_DOORS = "lock_doors"
    UNLOCK_DOORS = "unlock_doors"
    START_RADIO = "start_radio"
    STOP_RADIO = "stop_radio"
    PLAY_MUSIC = "play_music"
    PAUSE_MUSIC = "pause_music"
    RESUME_MUSIC = "resume_music"
    NEXT_MUSIC = "next_music"
    STOP_MUSIC = "stop_music"
    ACCEPT_MUSIC = "accept_music"
    DECLINE_MUSIC = "decline_music"
    INCREASE_AUDIO_VOLUME = "increase_audio_volume"
    DECREASE_AUDIO_VOLUME = "decrease_audio_volume"
    START_NAVIGATION = "start_navigation"
    STOP_NAVIGATION = "stop_navigation"
    REDUCE_TARGET_SPEED = "reduce_target_speed"
    INCREASE_TARGET_SPEED = "increase_target_speed"
    INCREASE_FOLLOWING_DISTANCE = "increase_following_distance"
    DECREASE_FOLLOWING_DISTANCE = "decrease_following_distance"
    ENABLE_ADAPTIVE_CRUISE_CONTROL = "enable_adaptive_cruise_control"
    DISABLE_ADAPTIVE_CRUISE_CONTROL = "disable_adaptive_cruise_control"
    ENABLE_LANE_KEEP_ASSIST = "enable_lane_keep_assist"
    DISABLE_LANE_KEEP_ASSIST = "disable_lane_keep_assist"
    ENABLE_BLIND_SPOT_MONITOR = "enable_blind_spot_monitor"
    DISABLE_BLIND_SPOT_MONITOR = "disable_blind_spot_monitor"
    ENABLE_PRIVACY_MODE = "enable_privacy_mode"
    DISABLE_PRIVACY_MODE = "disable_privacy_mode"
    ANNOUNCE_INCOMING_MESSAGE = "announce_incoming_message"
    ASK_PERMISSION_TO_TALK = "ask_permission_to_talk"
    READ_PENDING_MESSAGES = "read_pending_messages"
    APPLY_RESTRICTIVE_ADAS_PROFILE = "apply_restrictive_adas_profile"
    ENABLE_SIDELIGHTS = "enable_sidelights"
    DISABLE_SIDELIGHTS = "disable_sidelights"
    ENABLE_LOW_BEAM_HEADLIGHTS = "enable_low_beam_headlights"
    DISABLE_LOW_BEAM_HEADLIGHTS = "disable_low_beam_headlights"
    ENABLE_HIGH_BEAM_HEADLIGHTS = "enable_high_beam_headlights"
    DISABLE_HIGH_BEAM_HEADLIGHTS = "disable_high_beam_headlights"
    ENABLE_FOG_LIGHTS = "enable_fog_lights"
    DISABLE_FOG_LIGHTS = "disable_fog_lights"
    FIND_REST_AREA = "find_rest_area"
    ASK_ATTEND_MEETING = "ask_attend_meeting"

    @property
    def description(self) -> str:
        return {
            ActionType.NONE: "No direct action.",
            ActionType.ENABLE_HEATING: "enable cabin heating",
            ActionType.DISABLE_HEATING: "disable cabin heating",
            ActionType.OPEN_WINDOWS: "open the windows",
            ActionType.CLOSE_WINDOWS: "close the windows",
            ActionType.OPEN_SUNROOF: "open the sunroof",
            ActionType.CLOSE_SUNROOF: "close the sunroof",
            ActionType.ENABLE_AIR_CONDITIONING: "enable air conditioning",
            ActionType.DISABLE_AIR_CONDITIONING: "disable air conditioning",
            ActionType.INCREASE_FAN_SPEED: "increase fan speed by one step",
            ActionType.DECREASE_FAN_SPEED: "decrease fan speed by one step",
            ActionType.ENABLE_AIR_RECIRCULATION: "enable air recirculation",
            ActionType.DISABLE_AIR_RECIRCULATION: "disable air recirculation",
            ActionType.ENABLE_SEAT_HEATING: "enable seat heating",
            ActionType.DISABLE_SEAT_HEATING: "disable seat heating",
            ActionType.LOCK_DOORS: "lock doors",
            ActionType.UNLOCK_DOORS: "unlock doors",
            ActionType.START_RADIO: "start radio",
            ActionType.STOP_RADIO: "stop radio",
            ActionType.PLAY_MUSIC: "find and play a themed music playlist on explicit request",
            ActionType.PAUSE_MUSIC: "pause music playback",
            ActionType.RESUME_MUSIC: "resume music playback",
            ActionType.NEXT_MUSIC: "skip to the next music track",
            ActionType.STOP_MUSIC: "stop music playback and cancel pending music proposals",
            ActionType.ACCEPT_MUSIC: "accept the assistant's pending music proposal",
            ActionType.DECLINE_MUSIC: "decline the assistant's pending music proposal",
            ActionType.INCREASE_AUDIO_VOLUME: "increase audio volume by one step",
            ActionType.DECREASE_AUDIO_VOLUME: "decrease audio volume by one step",
            ActionType.START_NAVIGATION: "start navigation",
            ActionType.STOP_NAVIGATION: "stop navigation",
            ActionType.REDUCE_TARGET_SPEED: "reduce the ADAS target speed by one step",
            ActionType.INCREASE_TARGET_SPEED: "increase the ADAS target speed by one step",
            ActionType.INCREASE_FOLLOWING_DISTANCE: "increase the following distance by one step",
            ActionType.DECREASE_FOLLOWING_DISTANCE: "decrease the following distance by one step",
            ActionType.ENABLE_ADAPTIVE_CRUISE_CONTROL: "enable adaptive cruise control",
            ActionType.DISABLE_ADAPTIVE_CRUISE_CONTROL: "disable adaptive cruise control",
            ActionType.ENABLE_LANE_KEEP_ASSIST: "enable lane keeping assistance",
            ActionType.DISABLE_LANE_KEEP_ASSIST: "disable lane keeping assistance",
            ActionType.ENABLE_BLIND_SPOT_MONITOR: "enable blind spot monitoring",
            ActionType.DISABLE_BLIND_SPOT_MONITOR: "disable blind spot monitoring",
            ActionType.ENABLE_PRIVACY_MODE: "enable privacy mode",
            ActionType.DISABLE_PRIVACY_MODE: "disable privacy mode",
            ActionType.ANNOUNCE_INCOMING_MESSAGE: "announce an incoming message",
            ActionType.ASK_PERMISSION_TO_TALK: "ask permission to read queued messages",
            ActionType.READ_PENDING_MESSAGES: "read the queued messages after explicit permission",
            ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE: (
                "apply the restrictive ADAS safety profile: lower target speed, "
                "increase following distance, and enable available assistance"
            ),
            ActionType.ENABLE_SIDELIGHTS: "enable sidelights",
            ActionType.DISABLE_SIDELIGHTS: "disable sidelights",
            ActionType.ENABLE_LOW_BEAM_HEADLIGHTS: "enable low beam headlights",
            ActionType.DISABLE_LOW_BEAM_HEADLIGHTS: "disable low beam headlights",
            ActionType.ENABLE_HIGH_BEAM_HEADLIGHTS: "enable high beam headlights",
            ActionType.DISABLE_HIGH_BEAM_HEADLIGHTS: "disable high beam headlights",
            ActionType.ENABLE_FOG_LIGHTS: "enable fog lights",
            ActionType.DISABLE_FOG_LIGHTS: "disable fog lights",
            ActionType.FIND_REST_AREA: "Find the next rest area.",
            ActionType.ASK_ATTEND_MEETING: (
                "Ask the driver whether the assistant should attend their upcoming "
                "meeting on their behalf because they are running late."
            ),
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

class SimpleInterventionType(str, Enum):
    NONE = "none"
    SUGGEST = "suggest"

    @property
    def description(self) -> str:
        return {
            SimpleInterventionType.NONE: "No assistant intervention is useful or appropriate.",
            SimpleInterventionType.SUGGEST: (
                "Communicate with the driver because a suggestion, recommendation, "
                "or other helpful guidance would be appropriate."
            ),
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

class AssistantStatus(str, Enum):
    IDLE = "idle"  # UI: stationary white ball (gray while connecting/reconnecting)
    TALKING = "talking"  # UI: tone-colored organic wobble
    ASK_PERMISSION_TO_TALK = "ask_permission_to_talk"  # UI: stationary white ball; no dedicated animation
    ASK_PERMISSION_TO_ACT = "ask_permission_to_act"  # UI: stationary white ball; no dedicated animation
    ACT = "act"  # UI: brief enlargement pulse, then returns to idle size
    NOTIFY = "notify"  # UI: stationary white ball; no dedicated animation
    BACKGROUND_TASK_RUNNING = "background_task_running"  # UI: vertically focused wobble

    @property
    def description(self) -> str:
        return {
            AssistantStatus.IDLE: "The assistant is idle.",
            AssistantStatus.TALKING: "The assistant is talking.",
            AssistantStatus.ASK_PERMISSION_TO_TALK: "The assistant is asking for permission to talk.",
            AssistantStatus.ASK_PERMISSION_TO_ACT: "The assistant is asking for permission to act.",
            AssistantStatus.ACT: "The assistant is performing an action.",
            AssistantStatus.NOTIFY: "The assistant is notifying the user.",
            AssistantStatus.BACKGROUND_TASK_RUNNING: "The assistant is running a background task.",
        }[self]
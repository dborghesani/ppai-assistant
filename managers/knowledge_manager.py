import asyncio
import math
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, List, Mapping, get_args, get_type_hints

import structlog
from config import ConfigAssistant
from data import assistant_dataclasses
from data.database_manager import DatabaseManager
from data.events import CarEvent, EventName
from agents.agents_dataclasses import SkillType


@dataclass(frozen=True)
class KnowledgeState:
    value: Any
    trend: str | None = None


class KnowledgeManager:
    PERSISTENCE_CHECK_INTERVAL = 5.0
    _BOOLEAN_KNOWLEDGE_FORMATTERS: dict[tuple[str, str], Callable[[bool], str]] = {
        ("LaneTracing", "lane_crossing_left"): lambda value: (
            "The driver is crossing the lane to the left."
            if value
            else "No lane crossing to the left is currently detected."
        ),
        ("LaneTracing", "lane_crossing_right"): lambda value: (
            "The driver is crossing the lane to the right."
            if value
            else "No lane crossing to the right is currently detected."
        ),
        ("VehicleState", "doors_locked"): lambda value: (
            "The vehicle doors are locked."
            if value
            else "The vehicle doors are unlocked."
        ),
        ("VehicleState", "trunk_open"): lambda value: (
            "The trunk is open." if value else "The trunk is closed."
        ),
        ("VehicleState", "door_open_front_left"): lambda value: (
            "The front left door is open." if value else "The front left door is closed."
        ),
        ("VehicleState", "door_open_front_right"): lambda value: (
            "The front right door is open." if value else "The front right door is closed."
        ),
        ("VehicleState", "door_open_rear_left"): lambda value: (
            "The rear left door is open." if value else "The rear left door is closed."
        ),
        ("VehicleState", "door_open_rear_right"): lambda value: (
            "The rear right door is open." if value else "The rear right door is closed."
        ),
        ("VehicleState", "window_open_front_left"): lambda value: (
            "The front left window is open." if value else "The front left window is closed."
        ),
        ("VehicleState", "window_open_front_right"): lambda value: (
            "The front right window is open." if value else "The front right window is closed."
        ),
        ("VehicleState", "window_open_rear_left"): lambda value: (
            "The rear left window is open." if value else "The rear left window is closed."
        ),
        ("VehicleState", "window_open_rear_right"): lambda value: (
            "The rear right window is open." if value else "The rear right window is closed."
        ),
        ("VehicleState", "sunroof_open"): lambda value: (
            "The sunroof is open." if value else "The sunroof is closed."
        ),
        ("VehicleState", "air_conditioning_on"): lambda value: (
            "The air conditioning is on." if value else "The air conditioning is off."
        ),
        ("VehicleState", "heating_on"): lambda value: (
            "Cabin heating is on." if value else "Cabin heating is off."
        ),
        ("VehicleState", "air_recirculation_on"): lambda value: (
            "Cabin air recirculation is on." if value else "Cabin air recirculation is off."
        ),
        ("VehicleState", "seat_heating_on"): lambda value: (
            "Seat heating is on." if value else "Seat heating is off."
        ),
        ("VehicleState", "radio_on"): lambda value: (
            "The radio is on." if value else "The radio is off."
        ),
        ("VehicleState", "navigation_active"): lambda value: (
            "Navigation is active." if value else "Navigation is inactive."
        ),
        ("VehicleState", "adaptive_cruise_control_on"): lambda value: (
            "Adaptive cruise control is on." if value else "Adaptive cruise control is off."
        ),
        ("VehicleState", "lane_keep_assist_enabled"): lambda value: (
            "Lane keeping assistance is on." if value else "Lane keeping assistance is off."
        ),
        ("VehicleState", "lights_on_sidelights"): lambda value: (
            "The sidelights are on." if value else "The sidelights are off."
        ),
        ("VehicleState", "lights_on_low_beams"): lambda value: (
            "The low beam headlights are on." if value else "The low beam headlights are off."
        ),
        ("VehicleState", "lights_on_high_beams"): lambda value: (
            "The high beam headlights are on." if value else "The high beam headlights are off."
        ),
        ("VehicleState", "lights_on_fog_lights"): lambda value: (
            "The lights for fog are on." if value else "The lights for fog are off."
        ),
        ("VehicleState", "engine_on"): lambda value: (
            "The engine is on." if value else "The engine is off."
        ),
        ("VehicleState", "privacy_mode"): lambda value: (
            "Privacy mode is on." if value else "Privacy mode is off."
        ),
        ("DriverAgenda", "late_for_meeting"): lambda value: (
            "The driver is running late for an upcoming meeting."
            if value
            else "The driver is not late for any meeting."
        ),
        ("DetectedObjects", "animal_inside"): lambda value: (
            "An animal (dog or cat) is detected inside the vehicle."
            if value
            else "No dogs or cats are detected inside the vehicle."
        ),
        ("DetectedObjects", "dangerous_objects_around"): lambda value: (
            "A dangerous object is detected around the vehicle."
            if value
            else "No dangerous objects are detected around the vehicle."
        ),
    }
    _COUNT_KNOWLEDGE_FORMATTERS: dict[tuple[str, str], Callable[[int], str]] = {
        ("DetectedObjects", "people_around"): lambda value: (
            "No people are currently detected around the vehicle."
            if value == 0
            else f"People density around the vehicle is "
            f"{KnowledgeManager._density_level(value)}."
        ),
        ("DetectedObjects", "children_inside"): lambda value: (
            "No children are detected inside the vehicle."
            if value == 0
            else f"{value} {'child is' if value == 1 else 'children are'} detected inside the vehicle."
        ),
        ("DetectedObjects", "people_inside"): lambda value: (
            "No people are detected inside the vehicle."
            if value == 0
            else "One person is detected inside the vehicle."
            if value == 1
            else "Multiple people are detected inside the vehicle."
        ),
        ("DetectedObjects", "vehicles_around"): lambda value: (
            "No vehicles are currently detected around the vehicle."
            if value == 0
            else f"Traffic around the vehicle is "
            f"{KnowledgeManager._density_level(value)}."
        ),
        ("VehicleState", "fan_speed"): lambda value: (
            f"Cabin fan speed is level {value}."
        ),
        ("VehicleState", "audio_volume"): lambda value: (
            f"Audio volume is {value} percent."
        ),
        ("VehicleState", "adas_target_speed"): lambda value: (
            f"ADAS target speed is {value} km/h."
        ),
        ("VehicleState", "following_distance_level"): lambda value: (
            f"Following distance setting is level {value} of 5."
        ),
    }
    _CATEGORY_KNOWLEDGE_FORMATTERS: dict[tuple[str, str], Callable[[str], str]] = {
        ("LaneTracing", "highway_exit"): lambda value: {
            "No exit": "The vehicle is not approaching a highway exit.",
            "Right": "The vehicle is approaching a highway exit on the right.",
            "Left": "The vehicle is approaching a highway exit on the left.",
        }.get(value, f"The highway exit status is {value}."),
        ("EnvironmentState", "traffic"): lambda value: {
            "No traffic": "No traffic is currently detected around the vehicle.",
            "Light": "Current traffic is light.",
            "Medium": "Current traffic is medium.",
            "Heavy": "Current traffic is heavy.",
        }.get(value, f"Current traffic is {value}."),
        ("DriverPreferences", "preferred_cabin_temperature"): lambda value: (
            f"The driver's preferred cabin temperature is {value} C."
        ),
        ("DriverPreferences", "preferred_music"): lambda value: (
            f"The driver's music preference is {value}."
            if value
            else "The driver has not specified music preferences."
        ),
        ("DriverPhysicalState", "activity"): lambda value: {
            "Idle": "No distracting driver activity is detected.",
            "About to exit": "The driver is about to exit the vehicle.",
            "Driving": "Normal driving activity is detected.",
            "Talking": "The driver is talking, which may distract from driving.",
            "Eating": "The driver is eating, which distracts from driving.",
            "On the phone": "The driver is using a phone, which seriously distracts from driving.",
            "Sleeping": "The driver appears to be asleep and unable to drive safely.",
            "Children out of place": "Children are out of place near the vehicle.",
        }.get(value, f"The driver's current activity is {value}."),
        ("EnvironmentState", "forecast_weather"): lambda value: {
            "Sunny": "The weather forecast predicts sunny conditions.",
            "Cloudy": "The weather forecast predicts cloudy conditions.",
            "Rainy": "The weather forecast predicts rainy conditions.",
            "Snowy": "The weather forecast predicts snowy conditions.",
            "Foggy": "The weather forecast predicts foggy conditions.",
        }.get(value, f"The weather forecast is {value}."),
        ("EnvironmentState", "weather"): lambda value: {
            "Sunny": "The current weather is sunny.",
            "Cloudy": "The current weather is cloudy.",
            "Rainy": "The current weather is rainy.",
            "Snowy": "The current weather is snowy.",
            "Foggy": "The current weather is foggy.",
        }.get(value, f"The current weather is {value}."),
        ("EnvironmentState", "risk_level"): lambda value: {
            "None": "There is no current risk in this area.",
            "Low": "The current risk level in this area is low.",
            "Medium": "The current risk level in this area is medium.",
            "High": "The current risk level in this area is high.",
        }.get(value, f"The current risk level in this area is {value}."),
        ("EnvironmentState", "road_type"): lambda value: {
            "Urban": "The vehicle is currently driving on an urban road.",
            "Rural": "The vehicle is currently driving on a rural road.",
            "Highway": "The vehicle is currently driving on a highway.",
            "Residential": "The vehicle is currently driving on a residential road.",
        }.get(value, f"The current road type is {value}."),
    }
    _INTENSITY_KNOWLEDGE_FORMATTERS: dict[tuple[str, str], Callable[[float], str]] = {
        ("DriverPhysicalState", "fatigue_level"): lambda value: (
            "No fatigue detected." if value <= 0.0
            else f"Fatigue level is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverPhysicalState", "attention_level"): lambda value: (
            "No attention detected." if value <= 0.0
            else f"Attention level is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverEmotionState", "angry"): lambda value: (
            "No anger detected." if value <= 0.0
            else f"Anger level is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverEmotionState", "disgust"): lambda value: (
            "No disgust detected." if value <= 0.0
            else f"Disgust level is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverEmotionState", "fear"): lambda value: (
            "No fear detected." if value <= 0.0
            else f"Fear level is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverEmotionState", "happy"): lambda value: (
            "No happiness detected." if value <= 0.0
            else f"Happiness level is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverEmotionState", "sad"): lambda value: (
            "No sadness detected." if value <= 0.0
            else f"Sadness level is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverEmotionState", "surprise"): lambda value: (
            "No surprise detected." if value <= 0.0
            else f"Surprise level is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverEmotionState", "neutral"): lambda value: (
            "The driver seems emotionally neutral." if value >= 1.0
            else f"The driver's emotional neutrality is {KnowledgeManager._intensity_level(value)}."
        ),
        ("DriverDrivingStyle", "driving_tension"): lambda value: (
            "The driving style is calm and relaxed." if value <= 0.0
            else f"The driving style shows a {KnowledgeManager._intensity_level(value)} level of tension."
        ),
        ("EnvironmentState", "visibility"): lambda value: (
            "Driving visibility is optimal." if value >= 1.0
            else f"Driving visibility is {KnowledgeManager._intensity_level(value)}."
        ),
    }

    def __init__(
        self,
        database_manager: DatabaseManager,
        data_event_queue: asyncio.Queue,
        knowledge_event_queue: asyncio.Queue[CarEvent],
        opt: ConfigAssistant,
    ):
        self.logger = structlog.get_logger()
        self.database_manager = database_manager
        self.event_queue = data_event_queue
        self.knowledge_event_queue = knowledge_event_queue
        self.opt = opt
        self.context: dict[str, str] = {}
        self._last_evaluated: dict[str, KnowledgeState] = {}
        self._raw_values: dict[str, Any] = {}
        self.persistence_since: dict[str, float] = {}
        self.persistence_notified: set[str] = set()
        self.monotonic_clock: Callable[[], float] = time.monotonic
        self.on_context_updated: Callable[[], None] | None = None
        self.on_knowledge_changed: Callable[[list[str]], None] | None = None
        # The very first batch establishes the startup baseline: it must not
        # flood the model with every field's initial (mostly routine) value.
        self._is_first_batch = True

    @staticmethod
    def _format_value(value: float | int) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _density_level(count: int) -> str:
        if count > 10:
            return "high"
        if count > 5:
            return "medium"
        return "low"

    @staticmethod
    def _intensity_level(value: float) -> str:
        if value >= 0.8:
            return "very high"
        if value >= 0.6:
            return "high"
        if value >= 0.3:
            return "medium"
        return "low"

    @staticmethod
    def _display_measure(measure: str) -> str:
        return measure.replace("_", " ").replace(".", " ")

    @staticmethod
    def _boolean_state(measure: str, value: bool) -> str:
        if "door_open" in measure:
            return "open" if value else "closed"
        if "light" in measure or measure == "turn_signal":
            return "on" if value else "off"
        return "active" if value else "inactive"

    @staticmethod
    def _trend_label(mean_derivative: float, threshold: float = 1e-6) -> str:
        if mean_derivative > threshold:
            return "increasing"
        if mean_derivative < -threshold:
            return "decreasing"
        return "stable"

    def _numeric_knowledge(
        self,
        name: str,
        measure: str,
        value: float | int,
        window: str,
    ) -> tuple[str | None, str | None]:
        sample_count = self.database_manager.count(name, measure, window)
        # assuming that the variable names are semantically significant
        display_measure = self._display_measure(measure)
        if sample_count is None or sample_count < 3:
            return None, None

        mean_derivative = self.database_manager.mean_derivative(name, measure, window)
        if mean_derivative is None:
            return None, None

        stddev_derivative = self.database_manager.stddev_derivative(
            name, measure, window
        )
        trend = self._trend_label(mean_derivative)
        if trend == "stable":
            trend_description = "seems stable over time"
        else:
            has_fluctuations = (
                stddev_derivative is not None
                and abs(stddev_derivative) > abs(mean_derivative)
            )
            qualifier = "with fluctuations" if has_fluctuations else "steadily"
            historical_rate = self.database_manager.derivative_quantile(
                name, measure
            )
            unusually_fast = (
                historical_rate is not None
                and abs(mean_derivative) > historical_rate
            )
            speed_qualifier = " unusually fast" if unusually_fast else ""
            trend_description = f"is {trend}{speed_qualifier} {qualifier}"

        extracted_knowledge = (
            f"{display_measure.capitalize()} is {self._format_value(value)} and "
            f"{trend_description}."
        )
        self.logger.debug("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge, trend

    def _count_knowledge(self, name: str, measure: str, value: int) -> str:
        formatter = self._COUNT_KNOWLEDGE_FORMATTERS.get((name, measure))
        if formatter is not None:
            return formatter(value)

        display_measure = self._display_measure(measure)
        return f"The {display_measure} count is {value}."

    def _intensity_knowledge(self, name: str, measure: str, value: float) -> str:
        formatter = self._INTENSITY_KNOWLEDGE_FORMATTERS.get((name, measure))
        if formatter is not None:
            return formatter(value)

        display_measure = self._display_measure(measure)
        if value <= 0.0:
            return f"No {display_measure} detected."
        return f"{display_measure.capitalize()} level is {self._intensity_level(value)}."

    def _categorical_knowledge(
        self, name: str, measure: str, value: str
    ) -> str:
        formatter = self._CATEGORY_KNOWLEDGE_FORMATTERS.get((name, measure))
        if formatter is not None:
            return formatter(value)

        display_measure = self._display_measure(measure)
        # No real trend tracking for categorical values (only updated on
        # change): just report the current state.
        extracted_knowledge = f"The {display_measure} is {value}."

        self.logger.debug("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge

    def _boolean_knowledge(
        self, name: str, measure: str, value: bool
    ) -> str:
        formatter = self._BOOLEAN_KNOWLEDGE_FORMATTERS.get((name, measure))
        if formatter is not None:
            return formatter(value)

        display_measure = self._display_measure(measure)
        # No real trend tracking for boolean values (only updated on change):
        # just report the current state.
        current_state = self._boolean_state(measure, value)
        extracted_knowledge = f"The {display_measure} is {current_state}."

        self.logger.debug("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge

    def add_persistence_knowledge(
        self,
        key: str,
        value: Any,
        knowledge: str,
        metadata: Mapping[str, Any],
        now: float,
    ) -> tuple[str, bool]:
        persistence_levels = metadata.get("persistence_levels", ())
        persistence_since = getattr(self, "persistence_since", None)
        persistence_notified = getattr(self, "persistence_notified", None)
        if persistence_since is None:
            persistence_since = self.persistence_since = {}
        if persistence_notified is None:
            persistence_notified = self.persistence_notified = set()

        qualifies = (
            metadata.get("value_kind") == "intensity"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0.0
            and self._intensity_level(float(value)) in persistence_levels
        )
        if not qualifies:
            persistence_since.pop(key, None)
            persistence_notified.discard(key)
            return knowledge, False

        duration_seconds = metadata.get("persistence_seconds", 30.0)
        if key not in persistence_since:
            self.logger.info(
                "persistence timer started", key=key, seconds=duration_seconds
            )
        started_at = persistence_since.setdefault(key, now)
        if now - started_at < duration_seconds:
            return knowledge, False

        persistence_fact = (
            f"{knowledge} This condition has persisted for a while."
        )
        should_notify = key not in persistence_notified
        persistence_notified.add(key)
        if should_notify:
            self.logger.info("persistence threshold reached", key=key)
        return persistence_fact, should_notify

    @staticmethod
    def _knowledge_metadata(name: str, measure: str) -> Mapping[str, Any] | None:
        data_class = getattr(assistant_dataclasses, name, None)
        if data_class is None or not is_dataclass(data_class):
            return None

        path = measure.split(".")
        for index, part in enumerate(path):
            item = next((field for field in fields(data_class) if field.name == part), None)
            if item is None:
                return None
            if index == len(path) - 1:
                return item.metadata if item.metadata.get("knowledge", False) else None

            type_hint = get_type_hints(data_class).get(part)
            nested_types = get_args(type_hint)
            data_class = next(
                (nested_type for nested_type in nested_types if is_dataclass(nested_type)),
                type_hint if is_dataclass(type_hint) else None,
            )
            if data_class is None:
                return None
        return None

    @classmethod
    def _field_skills(cls, name: str, measure: str) -> tuple[SkillType, ...]:
        metadata = cls._knowledge_metadata(name, measure) or {}
        return tuple(metadata.get("skills", ()))

    def _is_significant_change(
        self,
        name: str,
        measure: str,
        key: str,
        value: Any,
        trend: str | None = None,
    ) -> bool:
        previous = self._last_evaluated.get(key)
        current = KnowledgeState(value=value, trend=trend)
        self._last_evaluated[key] = current

        # Booleans/categories are meaningful on their own: the first value
        # observed for a field introduced after startup can already matter
        # (e.g. weather set to "Foggy" from the UI), unlike the very first
        # startup batch, which is just the baseline and must stay quiet.
        if isinstance(value, bool) or isinstance(value, str):
            if previous is None:
                return not self._is_first_batch
            return value != previous.value

        if previous is None:
            return False

        if isinstance(value, (int, float)) and isinstance(previous.value, (int, float)):
            metadata = self._knowledge_metadata(name, measure) or {}
            trend_changed = (
                metadata.get("notify_on_trend_change", True) and trend != previous.trend
            )
            absolute_change = abs(value - previous.value)
            change_threshold = metadata.get("change_threshold")
            if isinstance(change_threshold, (int, float)):
                return trend_changed or absolute_change >= change_threshold

            scale = max(abs(value), abs(previous.value), 1.0)
            relative_change = absolute_change / scale
            change_ratio = metadata.get("change_ratio")
            if not isinstance(change_ratio, (int, float)):
                change_ratio = self.opt.knowledge_numeric_change_ratio
            return trend_changed or relative_change >= change_ratio

        return value != previous.value

    async def _notify_agent(self, changes: dict[str, Any]) -> None:
        if self.opt.use_laya:
            await self.knowledge_event_queue.put(
                CarEvent(
                    skill=SkillType.NONE,
                    event_name=EventName.KNOWLEDGE_UPDATED,
                    event_value=dict(changes),
                    context=list(self.context.values()),
                )
            )
            return

        changes_by_skill: dict[SkillType, dict[str, Any]] = {}
        for key, value in changes.items():
            name, separator, measure = key.partition(".")
            if not separator:
                continue
            skills = self._field_skills(name, measure)
            for skill in skills:
                changes_by_skill.setdefault(skill, {})[key] = value

        for skill, skill_changes in changes_by_skill.items():
            skill_context = {}
            for context_key, context_value in self.context.items():
                context_name, separator, context_measure = context_key.partition(".")
                if separator and skill in self._field_skills(
                    context_name, context_measure
                ):
                    skill_context[context_key] = context_value

            changed_context = [
                f"Changed just now: {skill_context[key]}"
                for key in skill_changes
                if key in skill_context
            ]
            supporting_context = [
                value for key, value in skill_context.items() if key not in skill_changes
            ]
            skill_context_all_values_as_list = changed_context + supporting_context
            await self.knowledge_event_queue.put(
                CarEvent(
                    skill=skill,
                    event_name=EventName.KNOWLEDGE_UPDATED,
                    event_value=dict(skill_changes),
                    context=skill_context_all_values_as_list,
                )
            )

    def _relative_knowledge(
        self, measure: str, value: Any, metadata: Mapping[str, Any]
    ) -> tuple[str, str] | None:
        reference = self._raw_values.get(".".join(metadata["reference"]))
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isinstance(reference, (int, float))
            or isinstance(reference, bool)
            or not math.isfinite(value)
            or not math.isfinite(reference)
        ):
            return None
        difference = value - reference
        tolerance = metadata["reference_tolerance"]
        extreme_threshold = metadata["reference_extreme_threshold"]
        if abs(difference) <= tolerance:
            level = "optimal"
        elif difference > extreme_threshold:
            level = "too high"
        elif difference < -extreme_threshold:
            level = "too low"
        else:
            level = "high" if difference > 0 else "low"
        label = metadata.get("reference_label") or self._display_measure(measure).capitalize()
        knowledge = f"{label} is {level}."
        unit = metadata.get("reference_unit")
        if unit:
            knowledge += (
                f" Current value is {self._format_value(value)} {unit}; "
                f"reference value is {self._format_value(reference)} {unit}."
            )
        return knowledge, level

    @staticmethod
    def _generates_knowledge(name: str, measure: str) -> bool:
        return KnowledgeManager._knowledge_metadata(name, measure) is not None

    @staticmethod
    def _collect_event(
        event: dict[str, Any], pending: dict[tuple[str, str], Any]
    ) -> None:
        if event.get("type") == "measurements_updated":
            name = event.get("name")
            values = event.get("values")
            if isinstance(name, str) and isinstance(values, dict):
                for measure, value in values.items():
                    if isinstance(
                        measure, str
                    ) and KnowledgeManager._generates_knowledge(name, measure):
                        pending[(name, measure)] = value
        """
        elif event.get("type") == "measure_updated":
            name = event.get("name")
            measure = event.get("measure")
            if (
                isinstance(name, str)
                and isinstance(measure, str)
                and KnowledgeManager._generates_knowledge(name, measure)
            ):
                pending[(name, measure)] = event.get("value")
        """

    async def seed_default_knowledge(self) -> None:
        """Publish known baseline values (e.g. doors closed, turn signal off)
        before any telemetry arrives, so the knowledge base and UI aren't
        empty and the first real reading is compared against a real baseline
        instead of being silently treated as the startup batch."""
        pending: dict[tuple[str, str], Any] = {}
        for attr_name in dir(assistant_dataclasses):
            data_class = getattr(assistant_dataclasses, attr_name)
            if not is_dataclass(data_class):
                continue
            for data_field in fields(data_class):
                if not data_field.metadata.get("knowledge", False):
                    continue
                if data_field.default is None:
                    continue
                pending[(attr_name, data_field.name)] = data_field.default

        for name, data in self.database_manager.current_state.items():
            if not is_dataclass(data):
                continue
            for data_field in fields(data):
                value = getattr(data, data_field.name)
                if data_field.metadata.get("knowledge", False) and value is not None:
                    pending[(name, data_field.name)] = value

        if pending:
            await self.process_pending(pending)

    async def run(self) -> None:
        next_persistence_check = time.monotonic() + self.PERSISTENCE_CHECK_INTERVAL
        while True:
            pending: dict[tuple[str, str], Any] = {}
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=max(0.0, next_persistence_check - time.monotonic()),
                )
            except asyncio.TimeoutError:
                pass
            else:
                self._collect_event(event, pending)
                await asyncio.sleep(
                    min(
                        self.opt.knowledge_update_interval,
                        max(0.0, next_persistence_check - time.monotonic()),
                    )
                )
            while True:
                try:
                    self._collect_event(self.event_queue.get_nowait(), pending)
                except asyncio.QueueEmpty:
                    break

            if time.monotonic() >= next_persistence_check:
                now = getattr(self, "monotonic_clock", time.monotonic)()
                due_pending = self._due_persistence_updates(now)
                pending = {**due_pending, **pending}
                next_persistence_check = (
                    time.monotonic() + self.PERSISTENCE_CHECK_INTERVAL
                )
            if pending:
                await self.process_pending(pending)

    def _due_persistence_updates(
        self, now: float
    ) -> dict[tuple[str, str], Any]:
        persistence_since = getattr(self, "persistence_since", {})
        persistence_notified: set[str] = getattr(
            self, "persistence_notified", set()
        )
        raw_values = getattr(self, "_raw_values", {})
        pending = {}
        for key, started_at in persistence_since.items():
            if key in persistence_notified or key not in raw_values:
                continue
            name, separator, measure = key.partition(".")
            if not separator:
                continue
            metadata = self._knowledge_metadata(name, measure) or {}
            duration = metadata.get("persistence_seconds", 30.0)
            if isinstance(duration, (int, float)) and now - started_at >= duration:
                pending[(name, measure)] = raw_values[key]
        monitored_values = {}
        for key, value in raw_values.items():
            name, separator, measure = key.partition(".")
            if separator:
                metadata = self._knowledge_metadata(name, measure) or {}
                if metadata.get("persistence_levels"):
                    monitored_values[key] = value
        self.logger.debug(
            "periodic persistence check",
            values=monitored_values,
            elapsed_seconds={
                key: round(now - started_at, 1)
                for key, started_at in persistence_since.items()
            },
            notified_keys=sorted(persistence_notified),
            due_keys=list(pending),
        )
        return pending

    async def process_pending(self, pending: dict[tuple[str, str], Any]) -> None:
        significant_changes: dict[str, dict[str, Any]] = {}
        window = "-10s"
        if not hasattr(self, "persistence_since"):
            self.persistence_since = {}
        if not hasattr(self, "persistence_notified"):
            self.persistence_notified = set()
        now = getattr(self, "monotonic_clock", time.monotonic)()
        pending = dict(pending)
        updated_keys = {f"{name}.{measure}" for name, measure in pending}
        for (name, measure), value in pending.items():
            self._raw_values[f"{name}.{measure}"] = value
        reference_keys: set[str] = set()
        for key, value in self._raw_values.items():
            name, separator, measure = key.partition(".")
            if not separator:
                continue
            metadata = self._knowledge_metadata(name, measure) or {}
            reference = metadata.get("reference")
            if reference:
                reference_key = ".".join(reference)
                reference_keys.add(reference_key)
                if reference_key in updated_keys:
                    pending.setdefault((name, measure), value)

        for (name, measure), value in pending.items():
            key = f"{name}.{measure}"
            trend: str | None = None
            metadata = self._knowledge_metadata(name, measure) or {}
            if metadata.get("value_kind") == "presence":
                value = bool(value)
            self._raw_values[key] = value
            if value is None and key in reference_keys:
                self.context.pop(key, None)
                self._last_evaluated.pop(key, None)
                continue
            relative_level: str | None = None
            knowledge: str | None
            if metadata.get("reference"):
                relative = self._relative_knowledge(measure, value, metadata)
                if relative is None:
                    self.context.pop(key, None)
                    self._last_evaluated.pop(key, None)
                    continue
                knowledge, relative_level = relative
            elif (
                isinstance(value, int)
                and not isinstance(value, bool)
                and metadata.get("value_kind") in {"count", "density"}
            ):
                knowledge = await asyncio.to_thread(
                    self._count_knowledge, name, measure, value
                )
            elif (
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float, str))
                and metadata.get("value_kind") == "category"
            ):
                label = self._format_value(value) if isinstance(value, (int, float)) else value
                knowledge = await asyncio.to_thread(
                    self._categorical_knowledge, name, measure, label
                )
            elif (
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and metadata.get("value_kind") == "intensity"
            ):
                knowledge = await asyncio.to_thread(
                    self._intensity_knowledge, name, measure, float(value)
                )
            elif (
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
            ):
                knowledge, trend = await asyncio.to_thread(
                    self._numeric_knowledge, name, measure, value, window
                )
            elif isinstance(value, bool):
                knowledge = await asyncio.to_thread(
                    self._boolean_knowledge, name, measure, value
                )
            elif isinstance(value, str):
                knowledge = await asyncio.to_thread(
                    self._categorical_knowledge, name, measure, value
                )
            else:
                continue

            if knowledge is None:
                continue

            knowledge, persistence_notification = self.add_persistence_knowledge(
                key, value, knowledge, metadata, now
            )
            self.context[key] = knowledge
            change_value: Any = value
            if relative_level is not None:
                change_value = relative_level
            elif metadata.get("value_kind") == "density":
                change_value = self._density_level(value)
            elif metadata.get("value_kind") == "category":
                # Compare as a label, not a magnitude: any change is significant.
                change_value = self._format_value(value) if isinstance(value, (int, float)) else str(value)
            elif metadata.get("value_kind") == "intensity":
                # Compare as a bucketed label, not the raw float; keep "none"
                # distinct so a zero-to-nonzero change is still flagged.
                numeric_value = float(value)
                change_value = "none" if numeric_value <= 0.0 else self._intensity_level(numeric_value)
            significant_change = self._is_significant_change(
                name, measure, key, change_value, trend
            )
            if (
                (significant_change or persistence_notification)
                and metadata.get("notify_on_change", True)
            ):
                self.logger.info(
                    "significant change detected",
                    name=name,
                    measure=measure,
                    value=change_value,
                    trend=trend,
                    persisted=persistence_notification,
                )
                significant_changes[key] = change_value

        self._is_first_batch = False

        # Notify listeners (e.g. the UI) of the fresh knowledge base before the
        # agent, and therefore any LLM call, is triggered below.
        if pending and self.on_context_updated is not None:
            self.on_context_updated()

        if significant_changes:
            on_knowledge_changed = getattr(self, "on_knowledge_changed", None)
            if on_knowledge_changed is not None:
                changed_knowledge = [
                    self.context[key]
                    for key in significant_changes
                    if key in self.context
                ]
                if changed_knowledge:
                    on_knowledge_changed(changed_knowledge)
            await self._notify_agent(significant_changes)

    def dump_knowledge(self) -> str:
        output_knowledge = ""
        for key, value in self.context.items():
            #output_knowledge += f"{key}: {value}\n\n"
            output_knowledge += f"{value}\n"
        return output_knowledge

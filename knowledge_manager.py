import asyncio
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, List, Mapping, get_args, get_type_hints

import structlog
from config import ConfigAssistant
from data import assistant_dataclasses
from data.database_manager import DatabaseManager
from events import CarEvent
from skill_manager import SkillType


@dataclass(frozen=True)
class KnowledgeState:
    value: Any
    trend: str | None = None


class KnowledgeManager:
    _SPEED_KEY = "VehicleMotion.speed"
    _SPEED_LIMIT_KEY = "DetectedObjects.traffic_signs.speed_limit"
    _SPEED_STATUS_KEY = "VehicleMotion.speed_status"
    event_skill_map: dict[str, List[SkillType]] = {}
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
    }
    _COUNT_KNOWLEDGE_FORMATTERS: dict[tuple[str, str], Callable[[int], str]] = {
        ("DetectedObjects", "people_around"): lambda value: (
            "No people are currently detected around the vehicle."
            if value == 0
            else f"People density around the vehicle is "
            f"{KnowledgeManager._density_level(value)}."
        ),
        ("DetectedObjects", "vehicles_around"): lambda value: (
            "No vehicles are currently detected around the vehicle."
            if value == 0
            else f"Traffic around the vehicle is "
            f"{KnowledgeManager._density_level(value)}."
        ),
        ("DetectedObjects", "dangerous_objects_around"): lambda value: (
            "No dangerous objects are currently detected around the vehicle."
            if value == 0
            else f"Dangerous object density around the vehicle is "
            f"{KnowledgeManager._density_level(value)}."
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
        ("DriverPhysicalState", "activity"): lambda value: {
            "Idle": "No distracting driver activity is detected.",
            "About to exit": "The driver is about to exit the vehicle.",
            "Driving": "Normal driving activity is detected.",
            "Talking": "The driver is talking, which may distract from driving.",
            "Eating": "The driver is eating, which distracts from driving.",
            "On the phone": "The driver is using a phone, which seriously distracts from driving.",
            "Sleeping": "The driver appears to be asleep and unable to drive safely.",
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
        self._last_speed_status: str | None = None
        self.on_context_updated: Callable[[], None] | None = None
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
    def _field_skills(cls, name: str, measure: str) -> tuple[str, ...]:
        metadata = cls._knowledge_metadata(name, measure) or {}
        skills = metadata.get("skills", ())
        return tuple(skill.value if isinstance(skill, SkillType) else skill for skill in skills)

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
                    skill="vehicle_assistance",
                    event_name="knowledge_updated",
                    event_value=dict(changes),
                    context=list(self.context.values()),
                )
            )
            return

        changes_by_skill: dict[str, dict[str, Any]] = {}
        for key, value in changes.items():
            name, separator, measure = key.partition(".")
            if not separator:
                continue
            # management of specific derived knowledge
            if key == self._SPEED_STATUS_KEY:
                skills = (SkillType.NAVIGATION_AND_COACHING.value,)
            else:
                skills = self._field_skills(name, measure)
            for skill in skills:
                changes_by_skill.setdefault(skill, {})[key] = value

        for skill, skill_changes in changes_by_skill.items():
            if self._SPEED_STATUS_KEY in changes:
                skill_context = dict(self.context)
            else:
                skill_context = {}
                for context_key, context_value in self.context.items():
                    context_name, separator, context_measure = context_key.partition(".")
                    if separator and skill in self._field_skills(
                        context_name, context_measure
                    ):
                        skill_context[context_key] = context_value

            skill_context_all_values_as_list = [
                f"{context_key}: {context_value}"
                for context_key, context_value in skill_context.items()
            ]
            await self.knowledge_event_queue.put(
                CarEvent(
                    skill=skill,
                    event_name="knowledge_updated",
                    event_value=dict(skill_changes),
                    context=skill_context_all_values_as_list,
                )
            )

    def _update_speed_status(self) -> tuple[str, str] | None:
        speed = self._raw_values.get(self._SPEED_KEY)
        speed_limit = self._raw_values.get(self._SPEED_LIMIT_KEY)
        if not isinstance(speed, (int, float)) or isinstance(speed, bool):
            speed = None
        if not isinstance(speed_limit, (int, float)) or isinstance(speed_limit, bool):
            speed_limit = None

        speed_metadata = self._knowledge_metadata("VehicleMotion", "speed") or {}
        near_margin = speed_metadata.get("change_threshold")
        if not isinstance(near_margin, (int, float)):
            near_margin = 0.0

        if speed is None or speed_limit is None:
            status = "limit_unknown"
            status_text = (
                "Speed status is limit_unknown because no verified speed limit "
                "is available."
            )
        elif speed > speed_limit and speed <= speed_limit + near_margin:
            status = "slightly_above_limit"
            status_text = (
                f"Speed status is slightly_above_limit. Current speed is "
                f"{self._format_value(speed)} km/h and the verified speed limit is "
                f"{self._format_value(speed_limit)} km/h."
            )
        elif speed > speed_limit:
            status = "above_limit"
            status_text = (
                f"Speed status is above_limit. Current speed is "
                f"{self._format_value(speed)} km/h and the verified speed limit is "
                f"{self._format_value(speed_limit)} km/h."
            )
        elif speed < speed_limit - near_margin:
            status = "below_limit"
            status_text = (
                f"Speed status is below_limit. Current speed is "
                f"{self._format_value(speed)} km/h and the verified speed limit is "
                f"{self._format_value(speed_limit)} km/h."
            )
        elif speed > speed_limit - near_margin and speed <= speed_limit:
            status = "slightly_below_limit"
            status_text = (
                f"Speed status is slightly_below_limit. Current speed is "
                f"{self._format_value(speed)} km/h and the verified speed limit is "
                f"{self._format_value(speed_limit)} km/h."
            )

        self.context[self._SPEED_STATUS_KEY] = status_text

        entered_above_limit = (
            status == "above_limit" and self._last_speed_status != "above_limit"
        )
        self._last_speed_status = status
        if not entered_above_limit:
            return None
        return self._SPEED_STATUS_KEY, status

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

        if pending:
            await self.process_pending(pending)

    async def run(self) -> None:
        while True:
            event = await self.event_queue.get()
            pending: dict[tuple[str, str], Any] = {}
            self._collect_event(event, pending)

            await asyncio.sleep(self.opt.knowledge_update_interval)
            while True:
                try:
                    self._collect_event(self.event_queue.get_nowait(), pending)
                except asyncio.QueueEmpty:
                    break

            await self.process_pending(pending)

    async def process_pending(self, pending: dict[tuple[str, str], Any]) -> None:
        significant_changes: dict[str, dict[str, Any]] = {}
        window = "-10s"

        for (name, measure), value in pending.items():
            key = f"{name}.{measure}"
            trend: str | None = None
            metadata = self._knowledge_metadata(name, measure) or {}
            self._raw_values[key] = value
            if (
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

            self.context[key] = knowledge
            change_value: Any = value
            if metadata.get("value_kind") == "density":
                change_value = self._density_level(value)
            elif metadata.get("value_kind") == "category":
                # Compare as a label, not a magnitude: any change is significant.
                change_value = self._format_value(value) if isinstance(value, (int, float)) else str(value)
            elif metadata.get("value_kind") == "intensity":
                # Compare as a bucketed label, not the raw float; keep "none"
                # distinct so a zero-to-nonzero change is still flagged.
                numeric_value = float(value)
                change_value = "none" if numeric_value <= 0.0 else self._intensity_level(numeric_value)
            if (
                self._is_significant_change(name, measure, key, change_value, trend)
            ):
                if metadata.get("notify_on_change", True):
                    self.logger.info(
                        "significant change detected",
                        name=name,
                        measure=measure,
                        value=change_value,
                        trend=trend,
                    )
                    significant_changes[key] = change_value

        self._is_first_batch = False

        derived_change = None
        if any(
            key in self._raw_values
            for key in (self._SPEED_KEY, self._SPEED_LIMIT_KEY)
        ):
            derived_change = self._update_speed_status()
        if derived_change is not None:
            key, value = derived_change
            significant_changes[key] = value

        # Notify listeners (e.g. the UI) of the fresh knowledge base before the
        # agent, and therefore any LLM call, is triggered below.
        if pending and self.on_context_updated is not None:
            self.on_context_updated()

        if significant_changes:
            await self._notify_agent(significant_changes)

    def dump_knowledge(self) -> str:
        output_knowledge = ""
        for key, value in self.context.items():
            #output_knowledge += f"{key}: {value}\n\n"
            output_knowledge += f"{value}\n"
        return output_knowledge

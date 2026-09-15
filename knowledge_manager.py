import asyncio
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List, Mapping

import structlog
from config import ConfigAssistant
from data import assistant_dataclasses
from data.database_manager import DatabaseManager
from data.skill_map import SkillMap
from events import CarEvent
from omegaconf import OmegaConf
from skill_manager import SkillType


@dataclass(frozen=True)
class KnowledgeState:
    value: Any
    trend: str | None = None


class KnowledgeManager:
    event_skill_map: dict[str, List[SkillType]] = {}

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
        self.skill_map = SkillMap()
        self.context: dict[str, str] = {}
        self._last_evaluated: dict[str, KnowledgeState] = {}

    @staticmethod
    def _format_value(value: float | int) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _display_measure(measure: str) -> str:
        names = {
            "speed": "vehicle speed",
            "acceleration_longitudinal": "longitudinal acceleration",
            "acceleration_lateral": "lateral acceleration",
            "steering_angle": "steering angle",
            "engine_rpm": "engine speed",
            "people_around": "nearby people",
            "vehicles_around": "nearby vehicles",
            "dangerous_objects_around": "nearby hazardous objects",
            "door_open_front_left": "front-left door",
            "door_open_front_right": "front-right door",
            "door_open_rear_left": "rear-left door",
            "door_open_rear_right": "rear-right door",
            "turn_signal": "turn signal",
            "lane_keep_assist": "lane-keeping assist",
            "blind_spot_monitor": "blind-spot monitor",
            "engine_on": "engine",
            "internal_temperature": "cabin temperature",
        }
        return names.get(measure, measure.replace("_", " "))

    @staticmethod
    def _window_label(window: str) -> str:
        return window.lstrip("-")

    @staticmethod
    def _duration_sentence(duration: int | None) -> str:
        if duration is None:
            return ""
        unit = "second" if int(duration) == 1 else "seconds"
        return f" The current state has lasted {int(duration)} {unit}."

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
    ) -> tuple[str, str | None]:
        sample_count = self.database_manager.count(name, measure, window)
        display_measure = self._display_measure(measure)
        if sample_count is None or sample_count < 3:
            return (
                f"Current {display_measure}: {self._format_value(value)}. "
                "There are not enough recent samples to determine a trend.",
                None,
            )

        minimum = self.database_manager.min(name, measure, window)
        maximum = self.database_manager.max(name, measure, window)
        mean = self.database_manager.mean(name, measure, window)
        stddev = self.database_manager.stddev(name, measure, window)
        mean_derivative = self.database_manager.mean_derivative(name, measure, window)
        stddev_derivative = self.database_manager.stddev_derivative(
            name, measure, window
        )

        if (
            minimum is None
            or maximum is None
            or mean is None
            or stddev is None
            or mean_derivative is None
            or stddev_derivative is None
        ):
            knowledge = (
                f"Current {display_measure}: {self._format_value(value)}. "
                "Recent statistics are incomplete."
            )
            self.logger.debug("extracted knowledge", knowledge=knowledge)
            return knowledge, None

        trend = self._trend_label(mean_derivative)
        if trend == "stable":
            trend_sentence = (
                f"It has remained stable over the last {self._window_label(window)}."
            )
        else:
            trend_description = (
                "steadily"
                if abs(stddev_derivative) <= abs(mean_derivative)
                else "with fluctuations"
            )
            trend_sentence = (
                f"It has been {trend} {trend_description} over the last "
                f"{self._window_label(window)}, ranging from "
                f"{self._format_value(minimum)} to {self._format_value(maximum)}."
            )
        variability = (
            f" Variation was approximately {self._format_value(stddev)}."
            if trend != "stable" and stddev > 0
            else ""
        )
        extracted_knowledge = f"Current {display_measure}: {self._format_value(value)}. {trend_sentence}{variability}"
        self.logger.debug("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge, trend

    def _categorical_knowledge(
        self, name: str, measure: str, value: str, window: str
    ) -> str:
        display_measure = self._display_measure(measure)
        first_value = self.database_manager.first_value(name, measure, window)
        current_value = self.database_manager.last_value(name, measure, window) or value
        value_counts = self.database_manager.value_counts(name, measure, window)
        current_duration = self.database_manager.current_value_duration(
            name, measure, current_value, window
        )

        if not value_counts:
            return f"The current {display_measure} is {current_value}. There are not enough recent samples for a temporal summary."

        sample_count = sum(value_counts.values())
        extracted_knowledge = f"The current {display_measure} is {current_value}. "
        if first_value is not None and first_value != current_value:
            extracted_knowledge += f"It changed from {first_value} during the last {self._window_label(window)}. "
        else:
            extracted_knowledge += (
                f"It remained unchanged during the last {self._window_label(window)}."
            )
        extracted_knowledge += self._duration_sentence(current_duration)

        self.logger.debug("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge

    def _boolean_knowledge(
        self, name: str, measure: str, value: bool, window: str
    ) -> str:
        display_measure = self._display_measure(measure)
        first_value = self.database_manager.first_value(name, measure, window)
        last_value = self.database_manager.last_value(name, measure, window)
        current_value = last_value if isinstance(last_value, bool) else value
        value_counts = self.database_manager.value_counts(name, measure, window)
        transitions = self.database_manager.value_transitions(name, measure, window)
        current_duration = self.database_manager.current_value_duration(
            name, measure, current_value, window
        )

        sample_count = sum(value_counts.values())
        if sample_count == 0:
            return (
                f"The current {display_measure} state is {str(current_value).lower()}. "
                "There are not enough recent samples for a temporal summary."
            )

        current_state = self._boolean_state(measure, current_value)
        extracted_knowledge = f"The {display_measure} is {current_state}."
        if isinstance(first_value, bool) and first_value != current_value:
            extracted_knowledge += (
                f" It changed from {self._boolean_state(measure, first_value)} during the last "
                f"{self._window_label(window)}."
            )
        elif transitions is not None:
            if transitions == 0:
                extracted_knowledge += f" It has remained {current_state} for the last {self._window_label(window)}."
            else:
                extracted_knowledge += (
                    f" It changed state {transitions} times during the last "
                    f"{self._window_label(window)}."
                )
        if transitions != 0:
            extracted_knowledge += self._duration_sentence(current_duration)
        elif current_duration is not None:
            unit = "second" if int(current_duration) == 1 else "seconds"
            extracted_knowledge += (
                f" It has remained {current_state} for {int(current_duration)} {unit}."
            )

        self.logger.debug("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge

    @staticmethod
    def _knowledge_metadata(name: str, measure: str) -> Mapping[str, Any] | None:
        data_class = getattr(assistant_dataclasses, name, None)
        if data_class is None or not is_dataclass(data_class):
            return None

        for item in fields(data_class):
            if item.name == measure and item.metadata.get("knowledge", False):
                return item.metadata
        return None

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
        if previous is None:
            return False

        if isinstance(value, bool) or isinstance(value, str):
            return value != previous.value

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

        if isinstance(value, str) and isinstance(previous.value, str):
            return value != previous.value

        return value != previous.value

    async def _notify_agent(self, name: str, changes: dict[str, Any]) -> None:
        skill_context = {
            context_key: context_value
            for context_key, context_value in self.context.items()
            if context_key.startswith(f"{name}.")
        }
        for skill in self.skill_map.get_skills_for_event(name):
            await self.knowledge_event_queue.put(
                CarEvent(
                    skill=skill.value if isinstance(skill, SkillType) else skill,
                    event_name="knowledge_updated",
                    event_value=changes,
                    context=skill_context,
                )
            )

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
        elif event.get("type") == "measure_updated":
            name = event.get("name")
            measure = event.get("measure")
            if (
                isinstance(name, str)
                and isinstance(measure, str)
                and KnowledgeManager._generates_knowledge(name, measure)
            ):
                pending[(name, measure)] = event.get("value")

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
            if (
                value is not None
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
            ):
                knowledge, trend = await asyncio.to_thread(
                    self._numeric_knowledge, name, measure, value, window
                )
            elif isinstance(value, bool):
                knowledge = await asyncio.to_thread(
                    self._boolean_knowledge, name, measure, value, window
                )
            elif isinstance(value, str):
                knowledge = await asyncio.to_thread(
                    self._categorical_knowledge, name, measure, value, window
                )
            else:
                continue

            self.context[key] = knowledge
            if self._is_significant_change(name, measure, key, value, trend):
                self.logger.info(
                    "significant change detected",
                    name=name,
                    measure=measure,
                    value=value,
                    trend=trend,
                )
                significant_changes.setdefault(name, {})[measure] = value

        for name, changes in significant_changes.items():
            await self._notify_agent(name, changes)

    def dump_knowledge(self) -> str:
        output_knowledge = ""
        for key, value in self.context.items():
            output_knowledge += f"{key}: {value}\n\n"
        return output_knowledge

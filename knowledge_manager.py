from typing import Any, List
from dataclasses import dataclass, fields, is_dataclass
from omegaconf import OmegaConf
import structlog
import asyncio

from data import assistant_dataclasses
from data.database_manager import DatabaseManager
from config import ConfigAssistant
from data.skill_map import SkillMap
from skill_manager import SkillType
from events import CarEvent


@dataclass(frozen=True)
class KnowledgeState:
    value: Any
    trend: str | None = None


class KnowledgeManager:
    event_skill_map: dict[str, List[SkillType]] = {}

    def __init__(self, database_manager: DatabaseManager, data_event_queue: asyncio.Queue,
                 knowledge_event_queue: asyncio.Queue[CarEvent], opt: ConfigAssistant):
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
        if sample_count is None or sample_count < 3:
            return (
                f"{measure} is currently {self._format_value(value)}; insufficient recent samples for a trend.",
                None,
            )

        minimum = self.database_manager.min(name, measure, window)
        maximum = self.database_manager.max(name, measure, window)
        mean = self.database_manager.mean(name, measure, window)
        stddev = self.database_manager.stddev(name, measure, window)
        mean_derivative = self.database_manager.mean_derivative(name, measure, window)
        stddev_derivative = self.database_manager.stddev_derivative(name, measure, window)

        if (
            minimum is None
            or maximum is None
            or mean is None
            or stddev is None
            or mean_derivative is None
            or stddev_derivative is None
        ):
            knowledge = f"{measure} is currently {self._format_value(value)}; recent statistics are incomplete."
            self.logger.info("extracted knowledge", knowledge=knowledge)
            return knowledge, None

        trend = self._trend_label(mean_derivative)
        variability = "steadily" if abs(stddev_derivative) <= abs(mean_derivative) else "with fluctuations"
        extracted_knowledge = (
            f"Over the last {window.lstrip('-')}, {measure} averaged {self._format_value(mean)} "
            f"(range {self._format_value(minimum)}-{self._format_value(maximum)}, "
            f"standard deviation {self._format_value(stddev)}, {int(sample_count)} samples). "
            f"It is {trend} {variability}. Current value: {self._format_value(value)}."
        )
        self.logger.info("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge, trend

    def _categorical_knowledge(self, name: str, measure: str, value: str, window: str) -> str:
        first_value = self.database_manager.first_value(name, measure, window)
        current_value = self.database_manager.last_value(name, measure, window) or value
        value_counts = self.database_manager.value_counts(name, measure, window)
        current_duration = self.database_manager.current_value_duration(name, measure, current_value, window)

        if not value_counts:
            return f"{measure} is currently {current_value}; insufficient recent samples for a temporal summary."

        sample_count = sum(value_counts.values())
        prevalent_value, prevalent_count = max(value_counts.items(), key=lambda item: item[1])
        extracted_knowledge = (
            f"Over the last {window.lstrip('-')}, {measure} was most often {prevalent_value} "
            f"({prevalent_count} of {sample_count} samples). "
        )
        if first_value is not None and first_value != current_value:
            extracted_knowledge += f"It changed from {first_value} to {current_value}. "
        else:
            extracted_knowledge += f"It remained {current_value}. "
        if current_duration is not None:
            extracted_knowledge += f"The current state has lasted {int(current_duration)} seconds."
        else:
            extracted_knowledge += f"Current value: {current_value}."

        self.logger.info("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge

    def _boolean_knowledge(self, name: str, measure: str, value: bool, window: str) -> str:
        first_value = self.database_manager.first_value(name, measure, window)
        last_value = self.database_manager.last_value(name, measure, window)
        current_value = last_value if isinstance(last_value, bool) else value
        value_counts = self.database_manager.value_counts(name, measure, window)
        transitions = self.database_manager.value_transitions(name, measure, window)
        current_duration = self.database_manager.current_value_duration(name, measure, current_value, window)

        sample_count = sum(value_counts.values())
        if sample_count == 0:
            return f"{measure} is currently {str(current_value).lower()}; insufficient recent samples for a temporal summary."

        true_count = value_counts.get(True, 0)
        extracted_knowledge = (
            f"Over the last {window.lstrip('-')}, {measure} was true for "
            f"{true_count} of {sample_count} samples. "
        )
        if isinstance(first_value, bool) and first_value != current_value:
            extracted_knowledge += (
                f"It changed from {str(first_value).lower()} to {str(current_value).lower()}. "
            )
        elif transitions is not None:
            extracted_knowledge += f"It changed state {transitions} times. "
        if current_duration is not None:
            extracted_knowledge += (
                f"It is currently {str(current_value).lower()} and has remained so for "
                f"{int(current_duration)} seconds."
            )
        else:
            extracted_knowledge += f"It is currently {str(current_value).lower()}."

        self.logger.info("extracted knowledge", knowledge=extracted_knowledge)
        return extracted_knowledge

    def _is_significant_change(self, key: str, value: Any, trend: str | None = None) -> bool:
        previous = self._last_evaluated.get(key)
        current = KnowledgeState(value=value, trend=trend)
        self._last_evaluated[key] = current
        if previous is None:
            return False

        if isinstance(value, bool) or isinstance(value, str):
            return value != previous.value

        if isinstance(value, (int, float)) and isinstance(previous.value, (int, float)):
            scale = max(abs(value), abs(previous.value), 1.0)
            relative_change = abs(value - previous.value) / scale
            return trend != previous.trend or relative_change >= self.opt.knowledge_numeric_change_ratio

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
            await self.knowledge_event_queue.put(CarEvent(
                skill=skill.value if isinstance(skill, SkillType) else skill,
                event_name="knowledge_updated",
                event_value=changes,
                context=skill_context,
            ))

    @staticmethod
    def _generates_knowledge(name: str, measure: str) -> bool:
        data_class = getattr(assistant_dataclasses, name, None)
        if data_class is None or not is_dataclass(data_class):
            return False

        return any(
            item.name == measure and item.metadata.get("knowledge", False)
            for item in fields(data_class)
        )

    @staticmethod
    def _collect_event(event: dict[str, Any], pending: dict[tuple[str, str], Any]) -> None:
        if event.get("type") == "measurements_updated":
            name = event.get("name")
            values = event.get("values")
            if isinstance(name, str) and isinstance(values, dict):
                for measure, value in values.items():
                    if (
                        isinstance(measure, str)
                        and KnowledgeManager._generates_knowledge(name, measure)
                    ):
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
            if value is not None and not isinstance(value, bool) and isinstance(value, (int, float)):
                knowledge, trend = self._numeric_knowledge(name, measure, value, window)
            elif isinstance(value, bool):
                knowledge = self._boolean_knowledge(name, measure, value, window)
            elif isinstance(value, str):
                knowledge = self._categorical_knowledge(name, measure, value, window)
            else:
                continue

            self.context[key] = knowledge
            self.logger.info("context updated", knowledge=knowledge)
            if self._is_significant_change(key, value, trend):
                significant_changes.setdefault(name, {})[measure] = value

        for name, changes in significant_changes.items():
            await self._notify_agent(name, changes)

if __name__ == "__main__":
    logger = structlog.get_logger()
    opt_cli = OmegaConf.from_cli()
    opt_default = OmegaConf.structured(ConfigAssistant())
    merged = OmegaConf.merge(opt_default, opt_cli)
    opt = OmegaConf.structured(merged)

    database_manager = DatabaseManager(None, opt)
    logger.info("Database manager initialized", database_manager=database_manager)

    name = "DriverState"
    measure = "fatigue_level"
    range = "-1h"

    # extract knowledge from the database
    if database_manager:
        mean_value = database_manager.mean(name, measure, range)
        max_value = database_manager.max(name, measure, range)
        min_value = database_manager.min(name, measure, range)
        stddev_value = database_manager.stddev(name, measure, range)
        count_value = database_manager.count(name, measure, range)
        logger.info("Extracted knowledge from the database",
                    mean_value=mean_value,
                    max_value=max_value,
                    min_value=min_value,
                    stddev_value=stddev_value,
                    count_value=count_value)
        derivative_value = database_manager.derivative(name, measure, range)
        mean_derivative_value = database_manager.mean_derivative(name, measure, range)
        stddev_derivative_value = database_manager.stddev_derivative(name, measure, range)
        logger.info("Extracted trend knowledge from the database",
                    derivative_value=derivative_value,
                    mean_derivative_value=mean_derivative_value,
                    stddev_derivative_value=stddev_derivative_value)

        if derivative_value is not None:
            if derivative_value > 0:
                logger.info(f"{measure} has increasing trend")
            elif derivative_value < 0:
                logger.info(f"{measure} has decreasing trend")
            else:
                logger.info(f"{measure} has stable trend")

        if mean_derivative_value is not None:
            if mean_derivative_value > 0:
                logger.info(f"{measure} has increasing trend on average")
            elif mean_derivative_value < 0:
                logger.info(f"{measure} has decreasing trend on average")
            else:
                logger.info(f"{measure} has stable trend on average")

        if stddev_derivative_value is not None:
            if stddev_derivative_value > 0:
                logger.info(f"{measure} has high variability in the trend")
            elif stddev_derivative_value < 0:
                logger.info(f"{measure} has decreasing variability in the trend")
            else:
                logger.info(f"{measure} has stable trend with no variability")

    database_manager.close()

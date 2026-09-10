import asyncio
from typing import Any, Dict, List, Tuple

from PySide6.QtCore import QObject, Slot

from config import ConfigAssistant
from data.database_manager import DatabaseManager
from skill_manager import SkillType
from agents import AutomotiveAgent
from events import CarEvent

class VehicleBridge(QObject):

    def __init__(
        self,
        agent: AutomotiveAgent,
        loop: asyncio.AbstractEventLoop,
        opt: ConfigAssistant,
        database_manager: DatabaseManager,
        parent=None,
    ):
        super().__init__(parent)
        self.opt = opt
        self.agent = agent
        self.loop = loop
        self.database_manager = database_manager

        # keep a queue of recent contexts for the assistant to use in its reasoning
        # 3 at most
        self.recent_contexts: Dict[SkillType,List[Any]] = {}

    """
    def send_event(self, event_type: str, value: Any):
        if not self.agent.is_listening:
            return

        skills = self.assistant_manager.get_skills_for_event(event_type)

        for skill in skills:

            # update the assistant state with the new value
            setattr(self.assistant_manager.assistant_state, event_type, value)
            context = self.assistant_manager.get_context_for_event(event_type)
                
            event = CarEvent(
                skill=skill.value,
                event_name=event_type,
                event_value=value if event_type != "user_input" else None,
                context=context,
                previous_contexts=self.recent_contexts.get(skill, []),
                user_input=value if event_type == "user_input" else None,
            )
            asyncio.run_coroutine_threadsafe(
                self.event_queue.put(event),
                self.loop,
            )

            # update the recent contexts queue
            if skill not in self.recent_contexts:
                self.recent_contexts[skill] = []
            self.recent_contexts[skill].append(context)
            if len(self.recent_contexts[skill]) > 3:
                self.recent_contexts[skill].pop(0)
    """

    def get_dataclass_from_ui_event_type(self, ui_event_type: str) -> Tuple[str, str] | None:
        # assume that the event_type corresponds to classname.classmember
        parts = ui_event_type.split(".")
        if len(parts) != 2:
            return None
        (class_name, member_name) = parts
        return (class_name, member_name)
        
    def send_event(self, ui_event_type: str, value: Any):
        if not self.agent.is_listening:
            return

        dataclass_info = self.get_dataclass_from_ui_event_type(ui_event_type)
        if dataclass_info is None:
            return

        dataclass_class_name, dataclass_member = dataclass_info
        self.database_manager.write_measure(dataclass_class_name, dataclass_member, value)

    @Slot(str)
    def userInput(self, text: str):
        text = text.strip()
        if not text:
            return
        self.send_event("user_input", text)

    @Slot(bool)
    def eventProcessingChanged(self, enabled: bool):
        self.agent.is_listening = enabled

    # DriverState

    @Slot(float)
    def fatigueChanged(self, value):
        self.send_event("DriverState.fatigue_level", value)

    @Slot(float)
    def attentionChanged(self, value):
        self.send_event("DriverState.attention_level", value)

    @Slot(float)
    def aggressivenessChanged(self, value):
        self.send_event("DriverState.aggressiveness_level", value)

    @Slot(str)
    def driverActivityChanged(self, value):
        self.send_event("DriverState.activity", value)

    @Slot(str)
    def driverMoodChanged(self, value):
        self.send_event("DriverState.mood", value)

    # EnvironmentState

    @Slot(str)
    def weatherChanged(self, value):
        self.send_event("EnvironmentState.weather", value)

    @Slot(str)
    def roadConditionChanged(self, value):
        self.send_event("EnvironmentState.road_condition", value)

    @Slot(str)
    def timeOfDayChanged(self, value):
        self.send_event("EnvironmentState.time_of_day", value)

    @Slot(str)
    def roadTypeChanged(self, value):
        self.send_event("EnvironmentState.road_type", value)

    @Slot(float)
    def externalTemperatureChanged(self, value):
        self.send_event(
            "EnvironmentState.external_temperature",
            float(value)
        )

    @Slot(str)
    def riskLevelChanged(self, value):
        self.send_event("EnvironmentState.risk_level", value)

    @Slot(str)
    def visibilityChanged(self, value):
        self.send_event("EnvironmentState.visibility", value)

    @Slot(str)
    def trafficChanged(self, value):
        self.send_event("EnvironmentState.traffic", value)

    # VehicleState

    @Slot(float)
    def temperatureChanged(self, value):
        self.send_event(
            "VehicleState.internal_temperature",
            float(value)
        )

    @Slot(bool)
    def engineChanged(self, value):
        self.send_event(
            "VehicleState.engine_running",
            bool(value)
        )

    @Slot(bool)
    def doorsChanged(self, value):
        self.send_event(
            "VehicleState.doors_unlocked",
            bool(value)
        )

    # VehicleMotion

    @Slot(float)
    def speedChanged(self, value):
        self.send_event(
            "VehicleMotion.speed",
            float(value)
        )

    # DetectedObjects

    @Slot(float)
    def peopleAroundChanged(self, value):
        self.send_event(
            "DetectedObjects.people_around",
            int(value)
        )

    @Slot(float)
    def carsAroundChanged(self, value):
        self.send_event(
            "DetectedObjects.cars_around",
            int(value)
        )


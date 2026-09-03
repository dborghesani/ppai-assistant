import asyncio
from typing import Any, Dict, List

from PySide6.QtCore import QObject, Slot

from assistant_manager import AssistantManager
from skill_manager import SkillType
from agents import AutomotiveAgent
from events import CarEvent

class VehicleBridge(QObject):

    def __init__(
        self,
        agent: AutomotiveAgent,
        loop: asyncio.AbstractEventLoop,
        shutdown_event: asyncio.Event,
        parent=None,
    ):
        super().__init__(parent)
        self._agent = agent
        self._event_queue = agent.event_queue
        self._user_event_queue = agent.user_event_queue
        self._loop = loop
        self._shutdown_event = shutdown_event

        self.assistant_manager = AssistantManager()

        # keep a queue of recent contexts for the assistant to use in its reasoning
        # 3 at most
        self.recent_contexts: Dict[SkillType,List[Any]] = {}


    def _send_event(self, event_type: str, value: Any):
        if not self._agent.is_listening:
            return

        skills = self.assistant_manager.get_skills_for_event(event_type)

        for skill in skills:

            # update the assistant state with the new value
            setattr(self.assistant_manager.assistant_state, event_type, value)
            context = self.assistant_manager.get_context_for_event(event_type)
                
            event = CarEvent(
                skill=skill.value,
                event_name=event_type,
                event_value=value,
                context=context,
                previous_contexts=self.recent_contexts.get(skill, []),
            )
            asyncio.run_coroutine_threadsafe(
                self._event_queue.put(event),
                self._loop,
            )

            # update the recent contexts queue
            if skill not in self.recent_contexts:
                self.recent_contexts[skill] = []
            self.recent_contexts[skill].append(context)
            if len(self.recent_contexts[skill]) > 3:
                self.recent_contexts[skill].pop(0)

    def _send_user_input(self, text: str):
        event = CarEvent(
            skill=SkillType.CONVERSATION.value,
            event_name="user_input",
            event_value=None,
            context={}, 
            user_input=text
        )

        asyncio.run_coroutine_threadsafe(
            self._user_event_queue.put(event),
            self._loop,
        )

    @Slot(float)
    def fatigueChanged(self, value):
        self._send_event("fatigue_level", value)

    @Slot(float)
    def attentionChanged(self, value):
        self._send_event("attention_level", value)

    @Slot(str)
    def vehicleStatusChanged(self, value):
        self._send_event("vehicle_status", value)

    @Slot(str)
    def drivingBehaviorChanged(self, value):
        self._send_event("driving_behavior", value)

    @Slot(str)
    def driverActivityChanged(self, value):
        self._send_event("driver_activity", value)

    @Slot(str)
    def driverMoodChanged(self, value):
        self._send_event("driver_mood", value)

    @Slot(str)
    def drivingGoalChanged(self, value):
        self._send_event("driving_goal", value)

    @Slot(str)
    def trafficChanged(self, value):
        self._send_event("traffic", value)

    @Slot(str)
    def roadTypeChanged(self, value):
        self._send_event("road_type", value)

    @Slot(str)
    def timeOfDayChanged(self, value):
        self._send_event("time_of_day", value)

    @Slot(str)
    def weatherChanged(self, value):
        self._send_event("weather", value)

    @Slot(float)
    def peopleAroundChanged(self, value):
        self._send_event(
            "people_around",
            int(value)
        )

    @Slot(float)
    def speedChanged(self, value):
        self._send_event(
            "speed",
            float(value)
        )

    @Slot(float)
    def temperatureChanged(self, value):
        self._send_event(
            "internal_temperature",
            float(value)
        )

    @Slot(float)
    def externalTemperatureChanged(self, value):
        self._send_event(
            "external_temperature",
            float(value)
        )

    @Slot(bool)
    def engineChanged(self, value):
        self._send_event(
            "engine_running",
            bool(value)
        )

    @Slot(bool)
    def doorsChanged(self, value):
        self._send_event(
            "doors_unlocked",
            bool(value)
        )
    
    @Slot(str)
    def detectedObjectsChanged(self, value):
        self._send_event(
            "detected_objects",
            value
        )

    @Slot(bool)
    def riskyAreaChanged(self, value):
        self._send_event(
            "risky_area",
            bool(value)
        )

    @Slot(str)
    def userInput(self, text: str):
        text = text.strip()
        if not text:
            return
        self._send_user_input(text)

    # app shutdown
    @Slot()
    def shutdown(self):
        self._loop.call_soon_threadsafe(
            self._shutdown_event.set,
        )

    @Slot(bool)
    def eventProcessingChanged(self, enabled: bool):
        self._agent.is_listening = enabled
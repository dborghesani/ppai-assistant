import asyncio
from typing import Any, Dict, List, Tuple

from PySide6.QtCore import QObject, Slot, Signal

from config import ConfigAssistant
from data.database_manager import DatabaseManager
from events import CarEvent
from knowledge_manager import KnowledgeManager
from skill_manager import SkillType
from automotive_agent import AutomotiveAgent
import structlog

class VehicleBridge(QObject):
    responseReceived = Signal(str)

    def __init__(
        self,
        agent: AutomotiveAgent,
        loop: asyncio.AbstractEventLoop,
        opt: ConfigAssistant,
        database_manager: DatabaseManager,
        knowledge_manager: KnowledgeManager,
        parent=None,
    ):
        super().__init__(parent)
        self.logger = structlog.get_logger()
        self.opt = opt
        self.agent = agent
        self.loop = loop
        self.database_manager = database_manager
        self.knowledge_manager = knowledge_manager
        self.agent.on_response = self.responseReceived.emit

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
        event = CarEvent(
            skill=SkillType.CONVERSATION.value,
            event_name="user_input",
            event_value=text,
            context=self.knowledge_manager.context,
            user_input=text,
        )
        self.agent.event_queue.put_nowait(event)

    @Slot(bool)
    def eventProcessingChanged(self, enabled: bool):
        self.agent.is_listening = enabled

    # generic slots
    @Slot(str, str, float)
    def floatChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    @Slot(str, str, int)
    def intChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    @Slot(str, str, str)
    def stringChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    @Slot(str, str, bool)
    def boolChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    @Slot(result=str)
    def dumpKnowledge(self) -> str:
        return self.knowledge_manager.dump_knowledge()

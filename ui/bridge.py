import asyncio
import json
from typing import Any, Dict, List, Tuple

from PySide6.QtCore import QObject, Slot, Signal, Property
from PySide6.QtGui import QGuiApplication, Qt

try:
    import aiomqtt
except ImportError:
    aiomqtt = None  # type: ignore[assignment]

from config import ConfigAssistant
from data.database_manager import DatabaseManager
from events import CarEvent
from knowledge_manager import KnowledgeManager
from skill_manager import SkillType
from automotive_agent import AutomotiveAgent
from voice.stt_manager import STTManager
import structlog

class VehicleBridge(QObject):
    responseReceived = Signal(str)
    isDarkModeChanged = Signal(bool)

    def __init__(
        self,
        agent: AutomotiveAgent,
        loop: asyncio.AbstractEventLoop,
        opt: ConfigAssistant,
        database_manager: DatabaseManager,
        knowledge_manager: KnowledgeManager,
        stt_manager: STTManager | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.logger = structlog.get_logger()
        self.opt = opt
        self.agent = agent
        self.loop = loop
        self.database_manager = database_manager
        self.knowledge_manager = knowledge_manager
        self.stt_manager = stt_manager
        self.agent.on_response = self.responseReceived.emit
        self._check_dark_mode()

        self._mqtt_queue: asyncio.Queue[tuple[str, str, str, Any]] = asyncio.Queue()
        if self.opt.mqtt_enabled and aiomqtt is not None:
            self.loop.create_task(self._mqtt_publisher_loop())

        style_hints = QGuiApplication.styleHints()
        if style_hints is not None:
            style_hints.colorSchemeChanged.connect(self._on_color_scheme_changed)

    def _check_dark_mode(self) -> bool:
        style_hints = QGuiApplication.styleHints()
        if style_hints is not None:
            self._is_dark_mode = style_hints.colorScheme() == Qt.ColorScheme.Dark
        else:
            self._is_dark_mode = False
        return self._is_dark_mode

    def _on_color_scheme_changed(self, scheme):
        is_dark = scheme == Qt.ColorScheme.Dark
        if is_dark != self._is_dark_mode:
            self._is_dark_mode = is_dark
            self.isDarkModeChanged.emit(self._is_dark_mode)

    @Property(bool, notify=isDarkModeChanged)
    def isDarkMode(self) -> bool:
        return self._is_dark_mode

    def get_dataclass_from_ui_event_type(self, ui_event_type: str) -> Tuple[str, str] | None:
        # assume that the event_type corresponds to classname.classmember
        parts = ui_event_type.split(".")
        if len(parts) != 2:
            return None
        (class_name, member_name) = parts
        return (class_name, member_name)

    async def _mqtt_publisher_loop(self):
        """Persistent connection for publishing UI telemetry events to the MQTT broker."""
        await asyncio.sleep(1.0)  # wait for broker startup
        while True:
            try:
                async with aiomqtt.Client(
                    hostname=self.opt.mqtt_host,
                    port=self.opt.mqtt_port,
                    username=self.opt.mqtt_username,
                    password=self.opt.mqtt_password,
                ) as client:
                    while True:
                        topic, payload, class_name, value = await self._mqtt_queue.get()
                        try:
                            await client.publish(topic, payload)
                            self.logger.debug("Published to MQTT broker", topic=topic, class_name=class_name)
                        except Exception as pub_err:
                            self.logger.warning("Failed to publish over MQTT, writing locally", error=str(pub_err))
                            self.database_manager.write_measure(class_name, json.loads(payload)["data"].keys(), value)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.debug("MQTT publisher reconnecting...", error=str(e))
                await asyncio.sleep(2.0)

    def _publish_mqtt(self, class_name: str, member_name: str, value: Any):
        topic = f"telemetry/{class_name}"
        payload = json.dumps({
            "name": class_name,
            "data": {member_name: value}
        })
        self._mqtt_queue.put_nowait((topic, payload, class_name, value))
        
    def send_event(self, ui_event_type: str, value: Any):
        if not self.agent.is_listening:
            return

        dataclass_info = self.get_dataclass_from_ui_event_type(ui_event_type)
        if dataclass_info is None:
            return

        dataclass_class_name, dataclass_member = dataclass_info

        # Route DriverDrivingStyle and DriverEmotionState through MQTT broker when enabled
        if dataclass_class_name in ("DriverDrivingStyle", "DriverEmotionState"):
            if self.opt.mqtt_enabled and aiomqtt is not None:
                self._publish_mqtt(dataclass_class_name, dataclass_member, value)
                return

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

    @Slot()
    def startVoiceInput(self):
        if not self.agent.is_listening or self.stt_manager is None:
            return
        self.stt_manager.start_recording()

    @Slot()
    def stopVoiceInput(self):
        if not self.agent.is_listening or self.stt_manager is None:
            return
        self.loop.create_task(self._transcribe_and_send())

    async def _transcribe_and_send(self):
        text = await asyncio.to_thread(self.stt_manager.stop_and_transcribe)
        if text:
            self.userInput(text)

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

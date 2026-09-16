import asyncio
import json
import time
from typing import Any, Dict, List, Tuple

import structlog
from automotive_agent import AutomotiveAgent
from config import ConfigAssistant
from data.database_manager import DatabaseManager
from data.mqtt_thread import MqttThreadClient
from events import CarEvent
from knowledge_manager import KnowledgeManager
from PySide6.QtCore import Property, QObject, Signal, Slot
from PySide6.QtGui import QGuiApplication, Qt
from skill_manager import SkillType
from voice.stt_manager import STTManager


class VehicleBridge(QObject):
    responseReceived = Signal(str)
    isDarkModeChanged = Signal(bool)
    conversationModeChanged = Signal(bool)

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

        self._mqtt_client: MqttThreadClient | None = None
        if self.opt.mqtt_enabled:
            self._mqtt_client = MqttThreadClient(
                host=self.opt.mqtt_host,
                port=self.opt.mqtt_port,
                username=self.opt.mqtt_username,
                password=self.opt.mqtt_password,
                client_id="ppai-assistant-ui",
            )
            self._mqtt_client.start()

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

    def get_dataclass_from_ui_event_type(
        self, ui_event_type: str
    ) -> Tuple[str, str] | None:
        # assume that the event_type corresponds to classname.classmember
        parts = ui_event_type.split(".")
        if len(parts) != 2:
            return None
        (class_name, member_name) = parts
        return (class_name, member_name)

    def _publish_mqtt(self, class_name: str, member_name: str, value: Any):
        topic = f"telemetry/{class_name}"
        payload = json.dumps({"name": class_name, "data": {member_name: value}})
        if self._mqtt_client is None or not self._mqtt_client.publish(topic, payload):
            self.logger.warning("MQTT unavailable, writing UI event locally")
            self.database_manager.write_measure(class_name, member_name, value)

    def close(self) -> None:
        """Stop the thread-backed MQTT publisher before the event loop closes."""
        if self._mqtt_client is not None:
            self._mqtt_client.stop()
            self._mqtt_client = None

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

        self.database_manager.write_measure(
            dataclass_class_name, dataclass_member, value
        )

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
        stt_start = time.time()
        text = await asyncio.to_thread(self.stt_manager.stop_and_transcribe)
        self.logger.info(
            f">>> STT transcription took {time.time() - stt_start:.2f}s", text=text
        )
        if text:
            self.userInput(text)

    @Slot()
    def startConversation(self):
        self.logger.info("Starting conversation mode")
        """Enter always-listening mode: turn-taking, auto-timeout and barge-in."""
        if not self.agent.is_listening:
            self.logger.warning(
                "Cannot start conversation: event processing is disabled"
            )
            return
        if self.stt_manager is None:
            self.logger.warning(
                "Cannot start conversation: STT manager is not configured"
            )
            return
        if not self.stt_manager.enabled:
            self.logger.warning(
                "Cannot start conversation: STT manager failed to load/is disabled"
            )
            return
        try:
            self.stt_manager.start_conversation(
                on_utterance=self._on_conversation_utterance,
                on_speech_start=self._on_conversation_speech_start,
                on_session_timeout=self._on_conversation_timeout,
                vad_head_index=self.opt.conversation_vad_head_index,
                vad_threshold=self.opt.conversation_vad_threshold,
                session_silence_timeout=self.opt.conversation_session_silence_timeout,
                pause_seconds=self.opt.conversation_pause_seconds,
                is_tts_speaking=lambda: bool(
                    self.agent.tts_manager is not None
                    and self.agent.tts_manager.is_speaking
                ),
                mute_mic_during_tts=self.opt.conversation_mute_mic_during_tts,
                barge_in_energy_threshold=self.opt.conversation_barge_in_energy_threshold,
                barge_in_min_frames=self.opt.conversation_barge_in_min_frames,
                get_tts_reference=(
                    self.agent.tts_manager.get_recent_playback
                    if self.agent.tts_manager is not None
                    else None
                ),
                echo_correlation_threshold=self.opt.conversation_echo_correlation_threshold,
            )
        except Exception as e:
            self.logger.error(
                "Failed to start conversation mode", error=str(e), exc_info=True
            )

    @Slot()
    def stopConversation(self):
        self.logger.info("Stopping conversation mode")
        if self.stt_manager is not None:
            self.stt_manager.stop_conversation()

    def _on_conversation_utterance(self, text: str):
        # Called from the STT background thread; hop back onto the event loop.
        self.loop.call_soon_threadsafe(self.userInput, text)

    def _on_conversation_speech_start(self):
        # Called from the STT background thread: barge-in, stop any TTS playback now.
        if self.agent.tts_manager is not None and self.agent.tts_manager.is_speaking:
            self.agent.tts_manager.stop()

    def _on_conversation_timeout(self):
        # Called from the STT background thread after prolonged silence.
        self.loop.call_soon_threadsafe(self._handle_conversation_timeout)

    def _handle_conversation_timeout(self):
        if self.stt_manager is not None:
            self.stt_manager.stop_conversation()
        self.conversationModeChanged.emit(False)

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

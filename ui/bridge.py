import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, DefaultDict, Tuple

import structlog
from automotive_agent import AutomotiveAgent
from config import ConfigAssistant
from data.database_manager import DatabaseManager
from data.mqtt_thread import MqttThreadClient
from events import CarEvent
from knowledge_manager import KnowledgeManager
from skill_manager import SkillType
from voice.stt_manager import STTManager


class VehicleBridge:
    def __init__(
        self,
        agent: AutomotiveAgent,
        loop: asyncio.AbstractEventLoop,
        opt: ConfigAssistant,
        database_manager: DatabaseManager,
        knowledge_manager: KnowledgeManager,
        stt_manager: STTManager | None = None,
    ):
        self.logger = structlog.get_logger()
        self.opt = opt
        self.agent = agent
        self.loop = loop
        self.database_manager = database_manager
        self.knowledge_manager = knowledge_manager
        self.knowledge_manager.on_context_updated = self._on_knowledge_updated
        self.stt_manager = stt_manager
        self._listeners: DefaultDict[str, list[Callable[..., None]]] = defaultdict(list)
        self.agent.on_response = lambda message: self._emit("responseReceived", message)
        self.agent.on_response_update = lambda message: self._emit(
            "responseUpdated", message
        )

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

    def on(self, event_name: str, callback: Callable[..., None]) -> None:
        self._listeners[event_name].append(callback)

    def _emit(self, event_name: str, *args: Any) -> None:
        for callback in tuple(self._listeners[event_name]):
            callback(*args)

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
        published = self._mqtt_client is not None and self._mqtt_client.publish(
            topic, payload
        )
        self.logger.debug(
            "MQTT publish", topic=topic, payload=payload, published=published
        )
        if not published:
            self.logger.warning("MQTT unavailable, writing UI event locally")
            self.database_manager.write_measure(class_name, member_name, value)

    def close(self) -> None:
        """Release audio and thread-backed resources before the event loop closes."""
        if self.stt_manager is not None:
            self.stt_manager.close()
        if self.agent.tts_manager is not None:
            self.agent.tts_manager.stop()
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
            if self.opt.mqtt_enabled:
                self._publish_mqtt(dataclass_class_name, dataclass_member, value)
                return

        self.database_manager.write_measure(
            dataclass_class_name, dataclass_member, value
        )

    def userInput(self, text: str):
        text = text.strip()
        if not text:
            return
        event = CarEvent(
            skill=SkillType.CONVERSATION.value,
            event_name="user_input",
            event_value=text,
            context=list(self.knowledge_manager.context.values()),
            user_input=text,
        )
        self.agent.event_queue.put_nowait(event)

    def eventProcessingChanged(self, enabled: bool):
        self.agent.is_listening = enabled

    def layaMinimumUrgencyChanged(self, urgency: str):
        try:
            self.agent.set_laya_minimum_urgency(urgency)
        except ValueError:
            self.logger.warning(
                "Ignoring invalid Laya urgency threshold", urgency=urgency
            )

    def startVoiceInput(self):
        if not self.agent.is_listening or self.stt_manager is None:
            return
        self.stt_manager.start_recording()

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
            self._submit_transcription(text)

    def startConversation(self):
        self.logger.info("Starting conversation mode")
        """Enter always-listening mode: turn-taking, auto-timeout and barge-in."""
        if not self.agent.is_listening:
            self.logger.warning(
                "Cannot start conversation: event processing is disabled"
            )
            self._emit("conversationModeChanged", False)
            return
        if self.stt_manager is None:
            self.logger.warning(
                "Cannot start conversation: STT manager is not configured"
            )
            self._emit("conversationModeChanged", False)
            return
        if not self.stt_manager.enabled:
            self.logger.warning(
                "Cannot start conversation: STT manager failed to load/is disabled"
            )
            self._emit("conversationModeChanged", False)
            return
        try:
            started = self.stt_manager.start_conversation(
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
            started = False
        # Keep the UI toggle in sync with whether the mic actually started.
        self._emit("conversationModeChanged", started)

    def stopConversation(self):
        self.logger.info("Stopping conversation mode")
        if self.stt_manager is not None:
            self.stt_manager.stop_conversation()

    def _on_conversation_utterance(self, text: str):
        # Called from the STT background thread; hop back onto the event loop.
        self.loop.call_soon_threadsafe(self._submit_transcription, text)

    def _submit_transcription(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._emit("userSpeechReceived", text)
        self.userInput(text)

    def _on_conversation_speech_start(self):
        # Called from the STT background thread: barge-in, stop any TTS playback now.
        self.loop.call_soon_threadsafe(self.agent.cancel_voice_response)
        if self.agent.tts_manager is not None and self.agent.tts_manager.is_speaking:
            self.agent.tts_manager.stop()

    def _on_conversation_timeout(self):
        # Called from the STT background thread after prolonged silence.
        self.loop.call_soon_threadsafe(self._handle_conversation_timeout)

    def _handle_conversation_timeout(self):
        if self.stt_manager is not None:
            self.stt_manager.stop_conversation()
        self._emit("conversationModeChanged", False)

    # generic slots
    def floatChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    def intChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    def stringChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    def boolChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    def _on_knowledge_updated(self) -> None:
        """Called synchronously by KnowledgeManager right after its context changes."""
        self._emit("knowledgeUpdated", self.dumpKnowledgeData())

    def dumpKnowledge(self) -> str:
        return self.knowledge_manager.dump_knowledge()

    def dumpKnowledgeData(self) -> dict[str, Any]:
        knowledge = {
            str(key): asdict(value) if is_dataclass(value) else value
            for key, value in self.knowledge_manager.context.items()
        }
        return json.loads(json.dumps(knowledge, default=str))

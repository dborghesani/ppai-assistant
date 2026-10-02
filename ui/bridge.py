import asyncio
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, DefaultDict, Tuple

import structlog
from agents.agents_dataclasses import ActionType
from agents.automotive_agent import AutomotiveAgent
from config import ConfigAssistant
from data.assistant_dataclasses import DetectedObjects, DriverPreferences, VehicleState
from data.database_manager import DatabaseManager
from data.mqtt_thread import MqttThreadClient
from data.events import CarEvent, EventName
from managers.knowledge_manager import KnowledgeManager
from managers.skill_manager import SkillType
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
        self.database_manager.current_state.setdefault(
            "DriverPreferences", DriverPreferences()
        )
        self.knowledge_manager = knowledge_manager
        self.knowledge_manager.on_context_updated = self._on_knowledge_updated
        self.stt_manager = stt_manager
        self._listeners: DefaultDict[str, list[Callable[..., None]]] = defaultdict(list)
        self.agent.on_response = lambda message: self._emit("responseReceived", message)
        self.agent.on_response_update = lambda message: self._emit(
            "responseUpdated", message
        )
        self.agent.on_speaking_tone_changed = lambda tone: self._emit(
            "speakingToneChanged", tone
        )
        self.agent.on_incoming_message_classified = lambda tone, urgency: self._emit(
            "incomingMessageClassified",
            {"tone": tone.value, "urgency": urgency.value},
        )
        self.agent.on_assistant_status_changed = lambda status: self._emit(
            "assistantStatusChanged", status.value
        )
        self.agent.action_manager.on_action = self.apply_vehicle_action

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

    def set_stt_manager(self, stt_manager: STTManager | None) -> None:
        self.stt_manager = stt_manager
        self._emit(
            "sttEnabledChanged",
            bool(stt_manager is not None and stt_manager.enabled),
        )

    def apply_vehicle_action(
        self, action_type: ActionType, parameters: dict[str, Any]
    ) -> None:
        window_fields = (
            "window_open_front_left",
            "window_open_front_right",
            "window_open_rear_left",
            "window_open_rear_right",
        )
        state_updates: dict[str, Any] = {
            ActionType.OPEN_WINDOWS: dict.fromkeys(window_fields, True),
            ActionType.CLOSE_WINDOWS: dict.fromkeys(window_fields, False),
            ActionType.OPEN_SUNROOF: {"sunroof_open": True},
            ActionType.CLOSE_SUNROOF: {"sunroof_open": False},
            ActionType.ENABLE_AIR_CONDITIONING: {"air_conditioning_on": True},
            ActionType.DISABLE_AIR_CONDITIONING: {"air_conditioning_on": False},
            ActionType.ENABLE_AIR_RECIRCULATION: {"air_recirculation_on": True},
            ActionType.DISABLE_AIR_RECIRCULATION: {"air_recirculation_on": False},
            ActionType.ENABLE_SEAT_HEATING: {"seat_heating_on": True},
            ActionType.DISABLE_SEAT_HEATING: {"seat_heating_on": False},
            ActionType.LOCK_DOORS: {"doors_locked": True},
            ActionType.UNLOCK_DOORS: {"doors_locked": False},
            ActionType.START_RADIO: {"radio_on": True},
            ActionType.STOP_RADIO: {"radio_on": False},
            ActionType.START_NAVIGATION: {"navigation_active": True},
            ActionType.STOP_NAVIGATION: {"navigation_active": False},
            ActionType.ENABLE_ADAPTIVE_CRUISE_CONTROL: {"adaptive_cruise_control_on": True},
            ActionType.DISABLE_ADAPTIVE_CRUISE_CONTROL: {"adaptive_cruise_control_on": False},
            ActionType.ENABLE_LANE_KEEP_ASSIST: {"lane_keep_assist_enabled": True},
            ActionType.DISABLE_LANE_KEEP_ASSIST: {"lane_keep_assist_enabled": False},
            ActionType.ENABLE_BLIND_SPOT_MONITOR: {"blind_spot_monitor": True},
            ActionType.DISABLE_BLIND_SPOT_MONITOR: {"blind_spot_monitor": False},
            ActionType.ENABLE_PRIVACY_MODE: {"privacy_mode": True},
            ActionType.DISABLE_PRIVACY_MODE: {"privacy_mode": False},
            ActionType.ENABLE_SIDELIGHTS: {"lights_on_sidelights": True},
            ActionType.DISABLE_SIDELIGHTS: {"lights_on_sidelights": False},
            ActionType.ENABLE_LOW_BEAM_HEADLIGHTS: {"lights_on_low_beams": True},
            ActionType.DISABLE_LOW_BEAM_HEADLIGHTS: {"lights_on_low_beams": False},
            ActionType.ENABLE_HIGH_BEAM_HEADLIGHTS: {"lights_on_high_beams": True},
            ActionType.DISABLE_HIGH_BEAM_HEADLIGHTS: {"lights_on_high_beams": False},
            ActionType.ENABLE_FOG_LIGHTS: {"lights_on_fog_lights": True},
            ActionType.DISABLE_FOG_LIGHTS: {"lights_on_fog_lights": False},
        }.get(action_type, {})

        vehicle_state = self.database_manager.current_state.get("VehicleState")
        if action_type is ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE:
            current_target_speed = getattr(vehicle_state, "adas_target_speed", None)
            current_following_distance = getattr(
                vehicle_state, "following_distance_level", None
            )
            current_target_speed = 90 if current_target_speed is None else current_target_speed
            current_following_distance = (
                3 if current_following_distance is None else current_following_distance
            )
            state_updates = {
                "adaptive_cruise_control_on": True,
                "adas_target_speed": max(0, current_target_speed - 10),
                "following_distance_level": min(5, current_following_distance + 1),
                "lane_keep_assist_enabled": True,
                "blind_spot_monitor": True,
            }
        elif action_type in {
            ActionType.REDUCE_TARGET_SPEED,
            ActionType.INCREASE_TARGET_SPEED,
        }:
            current_target_speed = getattr(vehicle_state, "adas_target_speed", None)
            current_target_speed = 90 if current_target_speed is None else current_target_speed
            target_speed_step = (
                10 if action_type is ActionType.INCREASE_TARGET_SPEED else -10
            )
            state_updates = {
                "adas_target_speed": min(160, max(0, current_target_speed + target_speed_step))
            }
        elif action_type in {
            ActionType.INCREASE_FOLLOWING_DISTANCE,
            ActionType.DECREASE_FOLLOWING_DISTANCE,
        }:
            current_following_distance = getattr(
                vehicle_state, "following_distance_level", None
            )
            current_following_distance = (
                3 if current_following_distance is None else current_following_distance
            )
            following_distance_step = (
                1 if action_type is ActionType.INCREASE_FOLLOWING_DISTANCE else -1
            )
            state_updates = {
                "following_distance_level": min(
                    5, max(1, current_following_distance + following_distance_step)
                )
            }
        elif action_type in {
            ActionType.INCREASE_TEMPERATURE,
            ActionType.DECREASE_TEMPERATURE,
        }:
            current_temperature = getattr(vehicle_state, "internal_temperature", None)
            if current_temperature is None:
                current_temperature = VehicleState().internal_temperature or 22.0
            temperature_step = (
                1.0 if action_type is ActionType.INCREASE_TEMPERATURE else -1.0
            )
            state_updates = {
                "internal_temperature": min(
                    40.0, max(10.0, current_temperature + temperature_step)
                )
            }
        elif action_type in {
            ActionType.INCREASE_FAN_SPEED,
            ActionType.DECREASE_FAN_SPEED,
        }:
            current_fan_speed = getattr(vehicle_state, "fan_speed", None)
            if current_fan_speed is None:
                current_fan_speed = VehicleState().fan_speed or 3
            fan_speed_step = (
                1 if action_type is ActionType.INCREASE_FAN_SPEED else -1
            )
            state_updates = {
                "fan_speed": min(7, max(0, current_fan_speed + fan_speed_step))
            }
        elif action_type in {
            ActionType.INCREASE_AUDIO_VOLUME,
            ActionType.DECREASE_AUDIO_VOLUME,
        }:
            current_audio_volume = getattr(vehicle_state, "audio_volume", None)
            if current_audio_volume is None:
                current_audio_volume = VehicleState().audio_volume or 20
            audio_volume_step = (
                5 if action_type is ActionType.INCREASE_AUDIO_VOLUME else -5
            )
            state_updates = {
                "audio_volume": min(100, max(0, current_audio_volume + audio_volume_step))
            }
        if not state_updates:
            return
        for measure, value in state_updates.items():
            self.database_manager.write_measure("VehicleState", measure, value)
        self._emit("vehicleStateChanged", state_updates)

    def _emit(self, event_name: str, *args: Any) -> None:
        for callback in tuple(self._listeners[event_name]):
            callback(*args)

    def get_dataclass_from_ui_event_type(
        self, ui_event_type: str
    ) -> Tuple[str, str] | None:
        # assume that the event_type corresponds to classname.classmember
        class_name, separator, member_name = ui_event_type.partition(".")
        if not separator or not member_name:
            return None
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
        self.agent.cancel_message_tasks()
        if self.stt_manager is not None:
            self.stt_manager.close()
        if self.agent.tts_manager is not None:
            self.agent.tts_manager.stop()
        if self._mqtt_client is not None:
            self._mqtt_client.stop()
            self._mqtt_client = None

    def startMessageSimulation(self) -> bool:
        started = self.agent.start_message_simulation()
        self._emit("friendMessageSimulationChanged", self.agent.message_simulator.active)
        return started

    async def stopMessageSimulation(self) -> bool:
        stopped = await self.agent.stop_message_simulation()
        self._emit("friendMessageSimulationChanged", self.agent.message_simulator.active)
        return stopped

    def startFriendMessageSimulation(self) -> bool:
        return self.startMessageSimulation()

    async def stopFriendMessageSimulation(self) -> bool:
        return await self.stopMessageSimulation()

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

    def setDriverPreference(self, name: str, value: Any) -> None:
        if name == "preferred_cabin_temperature":
            try:
                temperature = float(value)
            except (TypeError, ValueError):
                self.logger.warning("Ignoring invalid preferred temperature", value=value)
                return
            if not math.isfinite(temperature):
                self.logger.warning("Ignoring invalid preferred temperature", value=value)
                return
            value = min(30.0, max(16.0, temperature))
        elif name == "preferred_music":
            if not isinstance(value, str):
                self.logger.warning("Ignoring invalid music preference", value=value)
                return
            value = value.strip()[:500]
        else:
            self.logger.warning("Ignoring unknown driver preference", name=name)
            return
        self.database_manager.write_measure("DriverPreferences", name, value)
        self._emit("driverPreferencesChanged", self.dumpDriverPreferences())

    def dumpDriverPreferences(self) -> dict[str, Any]:
        preferences = self.database_manager.current_state.get(
            "DriverPreferences", DriverPreferences()
        )
        return {
            "preferred_cabin_temperature": preferences.preferred_cabin_temperature,
            "preferred_music": preferences.preferred_music,
        }

    def userInput(self, text: str):
        text = text.strip()
        if not text:
            return

        context = list(self.knowledge_manager.context.values())
        event = CarEvent(
            skill=SkillType.CONVERSATION,
            event_name=EventName.USER_INPUT,
            event_value=text,
            context=context,
            user_input=text,
        )
        self.agent.event_queue.put_nowait(event)

    def eventProcessingChanged(self, enabled: bool):
        self.agent.is_listening = enabled

    def minimumUrgencyChanged(self, urgency: str):
        try:
            self.agent.set_minimum_urgency(urgency)
        except ValueError:
            self.logger.warning(
                "Ignoring invalid Laya urgency threshold", urgency=urgency
            )

    @property
    def duplicate_suppression_enabled(self) -> bool:
        return self.agent.llm_agent.duplicate_suppression_enabled

    def duplicateSuppressionChanged(self, enabled: bool) -> None:
        self.agent.llm_agent.duplicate_suppression_enabled = bool(enabled)

    # Registry of demo scenarios: scenario_id -> (knowledge class, field, active value, inactive value).
    _SCENARIOS: dict[str, tuple[str, str, Any, Any]] = {
        "late_for_meeting": ("DriverAgenda", "late_for_meeting", True, False),
    }

    def setScenario(self, scenario_id: str, active: bool) -> None:
        """Toggle a demo scenario on/off; activating one deactivates all the others,
        including their knowledge signal, so at most one scenario is ever active."""
        if scenario_id not in self._SCENARIOS:
            self.logger.warning("Unknown scenario", scenario_id=scenario_id)
            return
        for other_id, (cls_name, field, _active_value, inactive_value) in self._SCENARIOS.items():
            if other_id != scenario_id:
                self.database_manager.write_measure(cls_name, field, inactive_value)
        cls_name, field, active_value, inactive_value = self._SCENARIOS[scenario_id]
        self.database_manager.write_measure(cls_name, field, active_value if active else inactive_value)

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
        if classname == "DetectedObjects" and varname == "dangerous_objects_around":
            count = max(0, min(10, int(value)))
            self.database_manager.write_measure(
                "DetectedObjects", "dangerous_objects_around", count
            )
            return
        self.send_event(f"{classname}.{varname}", value)

    def stringChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    def boolChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    def _on_knowledge_updated(self) -> None:
        """Called synchronously by KnowledgeManager right after its context changes."""
        self._emit("knowledgeUpdated", self.dumpKnowledgeData())
        self._emit("detectedObjectsStateChanged", self.dumpDetectedObjectsState())

    def dumpKnowledge(self) -> str:
        return self.knowledge_manager.dump_knowledge()

    def dumpKnowledgeData(self) -> dict[str, Any]:
        knowledge = {
            str(key): asdict(value) if is_dataclass(value) else value
            for key, value in self.knowledge_manager.context.items()
        }
        return json.loads(json.dumps(knowledge, default=str))

    def dumpVehicleState(self) -> dict[str, Any]:
        state = self.database_manager.current_state.get("VehicleState")
        values = asdict(VehicleState())
        if isinstance(state, VehicleState):
            values.update(
                {
                    key: value
                    for key, value in asdict(state).items()
                    if value is not None
                }
            )
        return values

    def dumpDetectedObjectsState(self) -> dict[str, Any]:
        state = self.database_manager.current_state.get("DetectedObjects")
        values = asdict(DetectedObjects())
        if isinstance(state, DetectedObjects):
            values.update(
                {
                    key: value
                    for key, value in asdict(state).items()
                    if value is not None
                }
            )
        return values

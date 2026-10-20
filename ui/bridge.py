import asyncio
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Tuple

import structlog
from data.agents_dataclasses import ActionType
from agents.automotive_agent import AutomotiveAgent
from config import ConfigAssistant
from data.assistant_dataclasses import DetectedObjects, DriverAgenda, DriverPreferences, VehicleState
from managers.database_manager import DatabaseManager
from managers.mqtt_thread import MqttThreadClient
from data.events import CarEvent, EventName
from managers.knowledge_manager import KnowledgeManager
from data.agents_dataclasses import SkillType
from agents.tasks.routing import TaskType

if TYPE_CHECKING:
    from voice.stt_manager import STTManager


class VehicleBridge:
    def __init__(
        self,
        agent: AutomotiveAgent,
        loop: asyncio.AbstractEventLoop,
        opt: ConfigAssistant,
        database_manager: DatabaseManager,
        knowledge_manager: KnowledgeManager,
        stt_manager: "STTManager | None" = None,
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
        self.knowledge_manager.on_knowledge_changed = self.on_knowledge_changed
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
        self.agent.on_assistant_status_changed = lambda flags: self._emit(
            "assistantStatusChanged", {
                "status": self.agent.assistant_status.value, "flags": flags,
            }
        )
        self.agent.action_manager.on_action = self.apply_vehicle_action
        self.agent.action_manager.on_meeting_completed = self._on_meeting_completed
        self.agent.on_music_update = lambda value: self._emit("musicUpdated", value)
        self.agent.music_preferences_provider = lambda: self.dumpDriverPreferences().get("preferred_music") or ""
        self.agent.driver_preferences_provider = self.dumpDriverPreferences
        self.agent.driving_frame_provider = self.capture_driving_frame

        self._driving_video_path: Path | None = None
        self._driving_capture_result: dict | None = None
        self._driving_capture_event: asyncio.Event | None = None
        self._last_driving_time: dict | None = None

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

    def set_stt_manager(self, stt_manager: "STTManager | None") -> None:
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
            ActionType.ENABLE_HEATING: {"heating_on": True},
            ActionType.DISABLE_HEATING: {"heating_on": False},
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
        self.agent.action_manager.close()
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

    def startDrivingSimulation(self, requested: str | None = None) -> str:
        """Show a local driving recording in the Web UI; resolves under recordings/."""
        path = self._resolve_recording(requested)
        if path is None:
            self._emit("drivingSimulationChanged", {"active": False, "video_url": ""})
            return ""
        self._driving_video_path = path
        self._emit("drivingSimulationChanged", {
            "active": True, "video_url": f"/recordings/{path.name}",
        })
        return path.name

    def _resolve_recording(self, requested: str | None) -> Path | None:
        """Map a bare name, file name or path to an existing recording file."""
        return self.agent.frame_capture.resolve_recording(requested, self.opt.driving_video)

    @property
    def recordings_dir(self) -> Path:
        """Directory of the current recording (for serving it to the UI)."""
        if self._driving_video_path is not None:
            return self._driving_video_path.parent
        return Path(__file__).resolve().parent.parent / "recordings"

    def stopDrivingSimulation(self) -> bool:
        self._driving_video_path = None
        self._driving_capture_result = None
        self._emit("drivingSimulationChanged", {"active": False, "video_url": ""})
        return True

    async def capture_driving_frame(self) -> dict | None:
        """Ask the Web UI for the current recording and playback time."""
        if not self._driving_video_path:
            return None
        self._driving_capture_result = None
        self._driving_capture_event = asyncio.Event()
        self._emit("drivingVideoCaptureRequested", {})
        try:
            await asyncio.wait_for(self._driving_capture_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Driving video capture time not reported by the UI")
        result = self._driving_capture_result or {}
        fallback = self._last_driving_time or {}
        seconds = result.get("seconds")
        if seconds is None:
            seconds = fallback.get("seconds")
        # Without a reported time the recording is read from its start.
        return {"video_path": str(self._driving_video_path), "seconds": seconds or 0.0}

    def reportDrivingVideoTime(self, video_id: str | None, seconds: float | None) -> bool:
        self._last_driving_time = {"video_id": video_id or None, "seconds": seconds}
        self._driving_capture_result = self._last_driving_time
        if self._driving_capture_event is not None:
            self._driving_capture_event.set()
        return True

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
        elif name == "driver_name":
            if not isinstance(value, str):
                return
            value = " ".join(value.split())[:80] or DriverPreferences().driver_name
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
            "driver_name": preferences.driver_name or DriverPreferences().driver_name,
            "preferred_cabin_temperature": preferences.preferred_cabin_temperature,
            "preferred_music": preferences.preferred_music,
        }

    def userInput(self, text: str):
        text = text.strip()
        if not text:
            return

        # Typed input acts like voice barge-in, but we let the word currently
        # being spoken finish so the cut never sounds like an audio glitch.
        tts_manager = self.agent.tts_manager
        if tts_manager is not None and tts_manager.is_speaking:
            self.loop.call_soon_threadsafe(tts_manager.stop, True)
        self.loop.call_soon_threadsafe(self.agent.cancel_voice_response)

        context = list(self.knowledge_manager.context.values())
        event = CarEvent(
            skill=SkillType.CONVERSATION,
            event_name=EventName.USER_INPUT,
            event_value=text,
            context=context,
            user_input=text,
        )
        self.agent.event_queue.put_nowait(event)

    def clearConversation(self) -> bool:
        self.agent.clear_conversation()
        self._emit("conversationCleared")
        return True

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
        return self.agent.duplicate_suppression_enabled

    def duplicateSuppressionChanged(self, enabled: bool) -> None:
        self.agent.duplicate_suppression_enabled = bool(enabled)

    # Registry of demo scenarios: scenario_id -> (knowledge class, field, active value, inactive value).
    _SCENARIOS: dict[str, tuple[str, str, Any, Any]] = {
        "late_for_meeting": ("DriverAgenda", "meeting_timing", "late", "none"),
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
                barge_in_vad_threshold=self.opt.conversation_barge_in_vad_threshold,
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
        if classname == "DriverAgenda":
            allowed = {
                "meeting_status": {"none", "prepare", "started", "finished"},
                "meeting_timing": {"none", "on_time", "late", "arrived"},
            }
            if varname not in allowed or value not in allowed[varname]:
                raise ValueError("Invalid meeting context")
            if varname == "meeting_timing" and value == "none":
                agenda = self.dumpDriverAgendaState()
                if agenda["meeting_status"] == "started" and agenda["meeting_timing"] == "late":
                    value = "arrived"
        self.send_event(f"{classname}.{varname}", value)
        if classname == "DriverAgenda":
            return value

    def boolChanged(self, classname, varname, value):
        self.send_event(f"{classname}.{varname}", value)

    def _on_knowledge_updated(self) -> None:
        """Called synchronously by KnowledgeManager right after its context changes."""
        self._emit("knowledgeUpdated", self.dumpKnowledgeData())
        self._emit("detectedObjectsStateChanged", self.dumpDetectedObjectsState())
        self.loop.call_soon_threadsafe(self._sync_driver_agenda)

    def _sync_driver_agenda(self) -> None:
        values = self.dumpDriverAgendaState()
        status = values["meeting_status"]
        self.agent.action_manager.set_meeting_status(status)
        self.agent.action_manager.set_meeting_timing(values["meeting_timing"])
        if values["meeting_timing"] != "late":
            self.agent.tasks[TaskType.MEETING].reset()
        self._emit("driverAgendaChanged", values)

    def _on_meeting_completed(self) -> None:
        self.database_manager.write_measure("DriverAgenda", "meeting_status", "finished")
        self._sync_driver_agenda()

    def dumpMusicState(self) -> dict[str, Any]:
        if self.agent.pending_music_selection is not None:
            return {
                "status": "proposal", "autoplay": False,
                **self.agent.pending_music_selection,
            }
        return {**self.agent.music_task.current, "autoplay": False}

    def dumpAssistantStatus(self) -> dict[str, Any]:
        """Current assistant status plus the raw status bitmask for the UI."""
        return {"status": self.agent.assistant_status.value, "flags": self.agent.assistant_flags}

    def musicControl(self, command: str) -> None:
        if command == "stop":
            self.agent.music_task.clear_pending_selection()
            self.agent.music_task.reset()
        self.agent.music_task.control_playback(command)

    def on_knowledge_changed(self, changes: list[str]) -> None:
        self._emit("knowledgeChanged", changes)

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

    def dumpDriverAgendaState(self) -> dict[str, Any]:
        state = self.database_manager.current_state.get("DriverAgenda")
        values = asdict(DriverAgenda())
        if isinstance(state, DriverAgenda):
            values.update({key: value for key, value in asdict(state).items() if value is not None})
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

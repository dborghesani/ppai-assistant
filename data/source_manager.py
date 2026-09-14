
import asyncio
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import structlog
import websockets
try:
    import aiomqtt
except ImportError:
    aiomqtt = None  # type: ignore[assignment]
try:
    from amqtt.broker import Broker
except ImportError:
    Broker = None  # type: ignore[assignment]

from data import assistant_dataclasses
from config import ConfigAssistant
from data.assistant_dataclasses import (
    GPSData, 
    VehicleMotion, 
    GPSIMUData, 
    RadarObject, 
    VisionObject, 
    DetectedObjects,
    GPSIMUState,
    TrafficSigns,
    LaneTracing,
    VehicleState,
    DriverEmotionState,
    DriverDrivingStyle,
)


class SourceManager:
    def __init__(self, data_event_queue: asyncio.Queue, opt: ConfigAssistant):
        self.data_event_queue = data_event_queue
        self.websocket_url = opt.data_websocket_url
        self.opt = opt
        self.previous_timestamp: int | None = None
        self.logger = structlog.get_logger()
        self.server_ready = asyncio.Event()

    @staticmethod
    def deserialize(payload: dict[str, Any]) -> Any | None:
        name = payload.get("name")
        values = payload.get("data")
        if name is None and len(payload) == 1:
            name, values = next(iter(payload.items()))
        if not isinstance(name, str) or not isinstance(values, dict):
            return None

        data_class = getattr(assistant_dataclasses, name, None)
        if data_class is None or not is_dataclass(data_class):
            return None

        valid_fields = {field.name for field in fields(data_class)}
        data = {key: value for key, value in values.items() if key in valid_fields}
        try:
            return data_class(**data)
        except TypeError:
            return None

    @staticmethod
    def deserialize_vehicle_motion(values: Any) -> VehicleMotion | None:
        if not isinstance(values, dict):
            return None

        valid_fields = {field.name for field in fields(VehicleMotion)}
        normalized = {
            key: value
            for key, value in values.items()
            if key in valid_fields
        }
        nested_fields = {
            "wheel_speed": {
                "front_left": "wheel_speed_front_left",
                "front_right": "wheel_speed_front_right",
                "rear_left": "wheel_speed_rear_left",
                "rear_right": "wheel_speed_rear_right",
            },
            "acceleration": {
                "longitudinal": "acceleration_longitudinal",
                "lateral": "acceleration_lateral",
            },
            "acceleration_g": {
                "x": "acceleration_g_x",
                "y": "acceleration_g_y",
                "z": "acceleration_g_z",
            },
            "angular_velocity_dps": {
                "x": "angular_velocity_dps_x",
                "y": "angular_velocity_dps_y",
                "z": "angular_velocity_dps_z",
            },
            "orientation_degrees": {
                "yaw": "orientation_degrees_yaw",
                "pitch": "orientation_degrees_pitch",
                "roll": "orientation_degrees_roll",
            },
        }
        for group_name, field_mapping in nested_fields.items():
            group = values.get(group_name)
            if isinstance(group, dict):
                for source_name, target_name in field_mapping.items():
                    if source_name in group:
                        normalized[target_name] = group[source_name]

        return VehicleMotion(**normalized)

    @staticmethod
    def deserialize_gpsimu_data(values: Any) -> GPSIMUData | None:
        if not isinstance(values, dict):
            return None

        valid_fields = {field.name for field in fields(GPSIMUData)}
        normalized = {
            key: value
            for key, value in values.items()
            if key in valid_fields
        }
        magnetic_field = values.get("magnetic_field")
        if isinstance(magnetic_field, dict):
            for axis in ("x", "y", "z"):
                if axis in magnetic_field:
                    normalized[f"magnetic_field_{axis}"] = magnetic_field[axis]

        return GPSIMUData(**normalized)

    @staticmethod
    def deserialize_vehicle_state(values: Any) -> VehicleState | None:
        if not isinstance(values, dict):
            return None

        valid_fields = {field.name for field in fields(VehicleState)}
        normalized = {
            key: value
            for key, value in values.items()
            if key in valid_fields
        }
        nested_fields = {
            "door_open": {
                "front_left": "door_open_front_left",
                "front_right": "door_open_front_right",
                "rear_left": "door_open_rear_left",
                "rear_right": "door_open_rear_right",
            },
            "lights_on": {
                "sidelights": "lights_on_sidelights",
                "low_beams": "lights_on_low_beams",
                "high_beams": "lights_on_high_beams",
                "fog_lights": "lights_on_fog_lights",
            },
        }
        for group_name, field_mapping in nested_fields.items():
            group = values.get(group_name)
            if isinstance(group, dict):
                for source_name, target_name in field_mapping.items():
                    if source_name in group:
                        normalized[target_name] = group[source_name]

        boolean_fields = {
            "door_open_front_left",
            "door_open_front_right",
            "door_open_rear_left",
            "door_open_rear_right",
            "doors_unlocked",
            "lights_on_sidelights",
            "lights_on_low_beams",
            "lights_on_high_beams",
            "lights_on_fog_lights",
            "trunk_open",
            "blind_spot_monitor",
            "engine_on",
        }
        for field_name in boolean_fields & normalized.keys():
            value = normalized[field_name]
            if isinstance(value, (bool, int)):
                normalized[field_name] = bool(value)

        return VehicleState(**normalized)

    async def process_message(self, message: str | bytes) -> None:
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.logger.warning("Ignoring invalid WebSocket JSON payload")
            return

        if not isinstance(payload, dict):
            self.logger.warning("Ignoring WebSocket payload that is not an object")
            return

        #data = self.deserialize(payload)
        # enumerate payload
        if "gps_data" in payload:
            gps_data = GPSData(**payload["gps_data"])  # type: ignore[name-defined]
            await self.data_event_queue.put({"type": "data_received", "data": gps_data})
        if "vehicle_motion" in payload:
            vehicle_motion = self.deserialize_vehicle_motion(payload["vehicle_motion"])
            if vehicle_motion is not None:
                # vehicle speed is the average of the wheel speeds if available
                wheel_speeds = [
                    vehicle_motion.wheel_speed_front_left,
                    vehicle_motion.wheel_speed_front_right,
                    vehicle_motion.wheel_speed_rear_left,
                    vehicle_motion.wheel_speed_rear_right,
                ]
                valid_wheel_speeds: list[float] = [ws for ws in wheel_speeds if ws is not None]
                if valid_wheel_speeds:
                    vehicle_motion.speed = sum(valid_wheel_speeds) / len(valid_wheel_speeds)
                await self.data_event_queue.put({"type": "data_received", "data": vehicle_motion})
        if "gpsimu_data" in payload:
            gpsimu_data = self.deserialize_gpsimu_data(payload["gpsimu_data"])
            if gpsimu_data is not None:
                await self.data_event_queue.put({"type": "data_received", "data": gpsimu_data})
        if "gpsimu_state" in payload:
            gpsimu_state = GPSIMUState(**payload["gpsimu_state"])  # type: ignore[name-defined]
            await self.data_event_queue.put({"type": "data_received", "data": gpsimu_state})
        if "radar_objects" in payload or "objects" in payload or "traffic_signs" in payload:
            radar_objects: list[RadarObject] = []
            vision_objects: list[VisionObject] = []
            if "radar_objects" in payload:
                for sensor_objects in payload["radar_objects"].values():
                    if not isinstance(sensor_objects, list):
                        continue
                    for obj_value in sensor_objects:
                        if isinstance(obj_value, dict):
                            radar_objects.append(RadarObject(**obj_value))
            if "objects" in payload:
                for obj_value in payload["objects"]:
                    if isinstance(obj_value, dict):
                        vision_objects.append(VisionObject(**obj_value))
                # add car_around, people_around, dangerous_objects_around
                vehicles_around = 0
                people_around = 0
                dangerous_objects_around = 0
                for vision_object in vision_objects:
                    if vision_object.category == "Car" or vision_object.category == "Truck":
                        vehicles_around += 1
                    elif vision_object.category == "Pedestrian":
                        people_around += 1
                    elif vision_object.category == "Gun":
                        dangerous_objects_around += 1
            detected_objects = DetectedObjects(
                radar_objects=radar_objects,
                vision_objects=vision_objects,
                vehicles_around=vehicles_around if "objects" in payload else None,
                people_around=people_around if "objects" in payload else None,
                dangerous_objects_around=dangerous_objects_around if "objects" in payload else None,
            )
            if "traffic_signs" in payload:
                traffic_signs = TrafficSigns(**payload["traffic_signs"])  # type: ignore[name-defined]
                detected_objects.traffic_signs = traffic_signs
            # TODO: futher analyze detected objects if necessary
            await self.data_event_queue.put({"type": "data_received", "data": detected_objects})
        if "lane_tracing" in payload:
            lane_tracing = LaneTracing(**payload["lane_tracing"])  # type: ignore[name-defined]
            # TODO: verify if nesting works
            await self.data_event_queue.put({"type": "data_received", "data": lane_tracing})
        if "vehicle_state" in payload:
            vehicle_state = self.deserialize_vehicle_state(payload["vehicle_state"])
            if vehicle_state is not None:
                await self.data_event_queue.put({"type": "data_received", "data": vehicle_state})
        if "driver_emotion_state" in payload or "DriverEmotionState" in payload or payload.get("name") == "DriverEmotionState":
            data = payload.get("driver_emotion_state") or payload.get("DriverEmotionState") or payload.get("data")
            if isinstance(data, dict):
                valid_fields = {f.name for f in fields(DriverEmotionState)}
                driver_emotion = DriverEmotionState(**{k: v for k, v in data.items() if k in valid_fields})
                await self.data_event_queue.put({"type": "data_received", "data": driver_emotion})
        if "driver_driving_style" in payload or "DriverDrivingStyle" in payload or payload.get("name") == "DriverDrivingStyle":
            data = payload.get("driver_driving_style") or payload.get("DriverDrivingStyle") or payload.get("data")
            if isinstance(data, dict):
                valid_fields = {f.name for f in fields(DriverDrivingStyle)}
                driver_style = DriverDrivingStyle(**{k: v for k, v in data.items() if k in valid_fields})
                await self.data_event_queue.put({"type": "data_received", "data": driver_style})
        if "name" in payload and "data" in payload:
            generic_obj = self.deserialize(payload)
            if generic_obj is not None:
                await self.data_event_queue.put({"type": "data_received", "data": generic_obj})

    async def _send_replay(
        self,
        websocket: websockets.ServerConnection,
        files: list[Path],
    ) -> None:
        previous_timestamp: int | None = None
        self.logger.info("Replay consumer connected...", file_count=len(files))
        try:
            for index, path in enumerate(files):
                timestamp = int(path.stem.split("_")[-1])
                if previous_timestamp is not None:
                    await asyncio.sleep(max(0, timestamp - previous_timestamp) / 1e6)
                payload = json.loads(path.read_text(encoding="utf-8"))
                await websocket.send(json.dumps(payload))
                self.logger.debug(f"[{index + 1}/{len(files)}] Sent {path.name}")
                previous_timestamp = timestamp
            self.logger.info("Finished sending all replay files")
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Replay consumer disconnected")

    async def run_replay(self) -> None:
        files = sorted(
            Path(self.opt.data_replay_folder).glob("*.json"), key=lambda path: int(path.stem.split("_")[-1])
        )
        if not files:
            self.logger.warning(f"No JSON files found in {self.opt.data_replay_folder}")
            return

        parsed_url = urlparse(self.websocket_url)
        if parsed_url.scheme != "ws" or parsed_url.hostname is None:
            raise ValueError(f"Replay WebSocket URL must use ws:// and include a host: {self.websocket_url}")

        port = parsed_url.port or 80
        self.logger.info("Starting replay WebSocket server", host=parsed_url.hostname, port=port)
        async with websockets.serve(
            lambda websocket: self._send_replay(websocket, files),
            parsed_url.hostname,
            port,
            ping_interval=20,
        ):
            self.server_ready.set()
            self.logger.info("Replay WebSocket server listening", url=self.websocket_url)
            try:
                await asyncio.Future()
            finally:
                self.server_ready.clear()

    async def run_embedded_broker(self) -> None:
        if Broker is None:
            self.logger.error("amqtt is not installed, cannot start embedded MQTT broker")
            return

        broker_config = {
            "listeners": {
                "default": {
                    "type": "tcp",
                    "bind": f"{self.opt.mqtt_host}:{self.opt.mqtt_port}",
                }
            },
            "sys_interval": 0,
            "plugins": {
                "amqtt.plugins.authentication.AnonymousAuthPlugin": {
                    "allow_anonymous": True
                }
            },
            "topic_check": {
                "enabled": False
            },
        }
        self.logger.info("Starting embedded MQTT broker", host=self.opt.mqtt_host, port=self.opt.mqtt_port)
        broker = Broker(broker_config)
        await broker.start()
        self.logger.info("Embedded MQTT broker running", host=self.opt.mqtt_host, port=self.opt.mqtt_port)
        try:
            await asyncio.Future()
        finally:
            await broker.shutdown()

    async def run_mqtt(self) -> None:
        if aiomqtt is None:
            self.logger.error("aiomqtt is not installed, cannot start MQTT client")
            return

        if self.opt.mqtt_embedded_broker:
            # Short grace period to ensure the embedded broker listener is bound
            await asyncio.sleep(0.5)

        while True:
            try:
                self.logger.info(
                    "Connecting to MQTT broker",
                    host=self.opt.mqtt_host,
                    port=self.opt.mqtt_port,
                    topic=self.opt.mqtt_topic,
                )
                async with aiomqtt.Client(
                    hostname=self.opt.mqtt_host,
                    port=self.opt.mqtt_port,
                    username=self.opt.mqtt_username,
                    password=self.opt.mqtt_password,
                ) as client:
                    await client.subscribe(self.opt.mqtt_topic)
                    self.logger.info("Connected to MQTT broker and subscribed", topic=self.opt.mqtt_topic)
                    async for message in client.messages:
                        payload = message.payload
                        msg_str: str | bytes
                        if isinstance(payload, bytes):
                            msg_str = payload.decode("utf-8")
                        elif isinstance(payload, str):
                            msg_str = payload
                        else:
                            msg_str = str(payload)
                        await self.process_message(msg_str)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.logger.warning("MQTT broker disconnected or error occurred", error=str(error))
                await asyncio.sleep(5)

    async def run(self) -> None:
        if self.opt.data_replay_folder:
            await self.server_ready.wait()

        while True:
            try:
                self.logger.info("Connecting to WebSocket source", url=self.websocket_url)
                async with websockets.connect(self.websocket_url, ping_interval=20) as websocket:
                    self.logger.info("Connected to WebSocket source", url=self.websocket_url)
                    async for message in websocket:
                        await self.process_message(message)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.logger.warning("WebSocket source disconnected", error=str(error))
                await asyncio.sleep(5)
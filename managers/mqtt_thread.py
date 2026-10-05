"""Thread-backed MQTT client for asyncio applications.

Paho owns its socket polling loop in a normal thread. The public methods are
safe to call from either the application thread or a background worker.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt
import structlog

MessageCallback = Callable[[str, bytes], None]


class MqttThreadClient:
    """Run one Paho MQTT connection outside the application event loop."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str | None = None,
        password: str | None = None,
        on_message: MessageCallback | None = None,
        client_id: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.on_message = on_message
        self.client_id = client_id
        self.logger = structlog.get_logger()
        self._client: mqtt.Client | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._topic: str | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self, topic: str | None = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="mqtt-client",
            daemon=True,
        )
        self._topic = topic
        self._thread.start()

    def _new_client(self) -> mqtt.Client:
        # VERSION2 is available in current paho-mqtt.  Keep the fallback for
        # older installations so the worker remains usable on Linux too.
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id or "ppai-assistant",
            )
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=self.client_id or "ppai-assistant")

        if self.username:
            client.username_pw_set(self.username, self.password)
        client.reconnect_delay_set(min_delay=1, max_delay=5)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        return client

    def _run(self) -> None:
        while not self._stop.is_set():
            client = self._new_client()
            with self._lock:
                self._client = client
            try:
                self.logger.info(
                    "Connecting to MQTT broker",
                    host=self.host,
                    port=self.port,
                )
                client.connect(self.host, self.port, keepalive=60)
                # Paho performs all socket reads/writes in this worker thread.
                client.loop_forever(retry_first_connection=False)
            except Exception as error:
                if not self._stop.is_set():
                    self.logger.warning(
                        "MQTT broker disconnected or error occurred",
                        error=str(error),
                    )
            finally:
                self._connected.clear()
                with self._lock:
                    if self._client is client:
                        self._client = None
            if not self._stop.wait(2.0):
                continue

    def _on_connect(
        self, client: mqtt.Client, userdata: Any, flags: Any, rc: Any, *args: Any
    ) -> None:
        if getattr(rc, "value", rc) not in (0, "0"):
            self.logger.warning("MQTT connection refused", reason=str(rc))
            return
        self._connected.set()
        if self._topic:
            client.subscribe(self._topic)
            self.logger.info(
                "Connected to MQTT broker and subscribed", topic=self._topic
            )
        else:
            self.logger.info("Connected to MQTT broker")

    def _on_disconnect(
        self, client: mqtt.Client, userdata: Any, flags: Any, rc: Any, *args: Any
    ) -> None:
        self._connected.clear()

    def _on_message(
        self, client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage
    ) -> None:
        if self.on_message is not None:
            self.on_message(message.topic, bytes(message.payload))

    def publish(self, topic: str, payload: str | bytes) -> bool:
        with self._lock:
            client = self._client
        if client is None or not self.is_connected:
            return False
        try:
            result = client.publish(topic, payload)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as error:
            self.logger.warning("Failed to publish over MQTT", error=str(error))
            return False

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            client = self._client
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        self._thread = None
        self._connected.clear()

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog
from aiohttp import WSMsgType, web
from ui.bridge import VehicleBridge


class WebUIServer:
    _METHODS = {
        "boolChanged",
        "dumpKnowledge",
        "duplicateSuppressionChanged",
        "eventProcessingChanged",
        "floatChanged",
        "startMessageSimulation",
        "stopMessageSimulation",
        "startFriendMessageSimulation",
        "stopFriendMessageSimulation",
        "intChanged",
        "minimumUrgencyChanged",
        "musicConsent",
        "musicControl",
        "setScenario",
        "setDriverPreference",
        "startConversation",
        "startVoiceInput",
        "stopConversation",
        "stopVoiceInput",
        "stringChanged",
        "userInput",
    }

    def __init__(
        self,
        bridge: VehicleBridge,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.bridge = bridge
        self.host = host
        self.port = port
        self.logger = structlog.get_logger()
        self._clients: set[web.WebSocketResponse] = set()
        self._runner: web.AppRunner | None = None
        self._loop = asyncio.get_running_loop()

        static_dir = Path(__file__).with_name("web")
        self._app = web.Application(middlewares=[self._no_cache_middleware])
        self._app.router.add_get("/", self._index)
        self._app.router.add_get("/ws", self._websocket)
        self._app.router.add_get("/health", self._health)
        self._app.router.add_static("/static", static_dir)

        for event_name in (
            "responseReceived",
            "responseUpdated",
            "speakingToneChanged",
            "incomingMessageClassified",
            "assistantStatusChanged",
            "sttEnabledChanged",
            "vehicleStateChanged",
            "detectedObjectsStateChanged",
            "driverPreferencesChanged",
            "userSpeechReceived",
            "conversationModeChanged",
            "knowledgeUpdated",
            "knowledgeChanged",
            "friendMessageSimulationChanged",
            "musicUpdated",
        ):
            bridge.on(
                event_name,
                lambda *args, name=event_name: self._queue_broadcast(name, *args),
            )

    @property
    def url(self) -> str:
        browser_host = "localhost" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{browser_host}:{self.port}"

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.logger.info("Web UI ready", url=self.url)

    async def close(self) -> None:
        clients = tuple(self._clients)
        for client in clients:
            await client.close(code=1001, message=b"Server shutting down")
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(Path(__file__).with_name("web") / "index.html")

    @staticmethod
    @web.middleware
    async def _no_cache_middleware(
        request: web.Request, handler: Any
    ) -> web.StreamResponse:
        """Always force revalidation: this UI is edited on disk while the server keeps running."""
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "clients": len(self._clients)})

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        self._clients.add(socket)
        await socket.send_json(
            {
                "event": "ready",
                "data": {
                    "knowledge": self.bridge.dumpKnowledgeData(),
                    "vehicleState": self.bridge.dumpVehicleState(),
                    "detectedObjectsState": self.bridge.dumpDetectedObjectsState(),
                    "driverPreferences": self.bridge.dumpDriverPreferences(),
                    "musicState": getattr(self.bridge, "dumpMusicState", lambda: {"status": "disabled"})(),
                    "musicDuckingFactor": getattr(getattr(self.bridge, "opt", None), "music_ducking_factor", 0.25),
                    "sttEnabled": bool(
                        self.bridge.stt_manager is not None
                        and self.bridge.stt_manager.enabled
                    ),
                    "friendMessageSimulationActive": bool(
                        getattr(
                            getattr(self.bridge, "friend_message_agent", None),
                            "active",
                            False,
                        )
                    ),
                    "duplicateSuppressionEnabled": bool(
                        getattr(self.bridge, "duplicate_suppression_enabled", False)
                    ),
                },
            }
        )

        try:
            async for message in socket:
                if message.type == WSMsgType.TEXT:
                    await self._handle_message(socket, message.data)
                elif message.type == WSMsgType.ERROR:
                    self.logger.warning(
                        "Web UI socket error", error=str(socket.exception())
                    )
        finally:
            self._clients.discard(socket)
        return socket

    async def _handle_message(
        self, socket: web.WebSocketResponse, raw_message: str
    ) -> None:
        request_id: Any = None
        try:
            payload = json.loads(raw_message)
            request_id = payload.get("id")
            method_name = payload.get("method")
            params = payload.get("params", [])
            if method_name not in self._METHODS:
                raise ValueError(f"Unsupported bridge method: {method_name}")
            if not isinstance(params, list):
                raise ValueError("params must be an array")

            result = getattr(self.bridge, method_name)(*params)
            if asyncio.iscoroutine(result):
                result = await result
            if request_id is not None:
                await socket.send_json({"id": request_id, "result": result})
        except Exception as error:
            self.logger.warning("Invalid Web UI request", error=str(error))
            if request_id is not None:
                await socket.send_json({"id": request_id, "error": str(error)})

    def _queue_broadcast(self, event_name: str, *args: Any) -> None:
        def schedule() -> None:
            asyncio.create_task(self._broadcast(event_name, list(args)))

        self._loop.call_soon_threadsafe(schedule)

    async def _broadcast(self, event_name: str, data: list[Any]) -> None:
        if not self._clients:
            return
        payload = {"event": event_name, "data": data}
        clients = tuple(self._clients)
        results = await asyncio.gather(
            *(client.send_json(payload) for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                self._clients.discard(client)

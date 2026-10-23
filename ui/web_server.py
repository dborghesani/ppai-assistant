import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import structlog
from aiohttp import WSMsgType, web
from ui.bridge import VehicleBridge
from config import PROJECT_ROOT


class WebUIServer:
    _METHODS = {
        "boolChanged",
        "clearConversation",
        "dumpKnowledge",
        "detectActivity",
        "deleteFaceCluster",
        "renameFaceCluster",
        "duplicateSuppressionChanged",
        "eventProcessingChanged",
        "floatChanged",
        "startMessageSimulation",
        "stopMessageSimulation",
        "startFriendMessageSimulation",
        "stopFriendMessageSimulation",
        "startDrivingSimulation",
        "stopDrivingSimulation",
        "reportDrivingVideoTime",
        "startActivityVideo",
        "stopActivityVideo",
        "waitCaptureBufferReady",
        "setCaptureSource",
        "intChanged",
        "minimumUrgencyChanged",
        "musicControl",
        "setScenario",
        "setDriverPreference",
        "selectFaceCluster",
        "recognizeFace",
        "startFaceRecognition",
        "mergeFaceClusters",
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
        self._app.router.add_get("/face-clusters/{identifier}.jpg", self._face_thumbnail)
        self._app.router.add_get("/face-preview.mjpeg", self._face_preview_stream)
        self._app.router.add_static("/static", static_dir)
        driving_video = getattr(getattr(bridge, "opt", None), "driving_video", "") or "recordings/rome.mp4"
        recordings_dir = PROJECT_ROOT / Path(driving_video).parent
        recordings_dir.mkdir(parents=True, exist_ok=True)
        self._app.router.add_static("/recordings", recordings_dir)
        video_capture = getattr(getattr(bridge, "agent", None), "video_capture", None)
        activity_recordings_dir = getattr(video_capture, "recordings_dir", PROJECT_ROOT / "recordings")
        self._app.router.add_static("/activity-recordings", activity_recordings_dir)

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
            "driverAgendaChanged",
            "userSpeechReceived",
            "conversationModeChanged",
            "conversationCleared",
            "knowledgeUpdated",
            "knowledgeChanged",
            "friendMessageSimulationChanged",
            "scenarioCompleted",
            "musicUpdated",
            "drivingSimulationChanged",
            "drivingVideoCaptureRequested",
            "activityDetectionChanged",
            "driverPhysicalStateChanged",
            "vehicleSecurityStateChanged",
            "captureSourcesChanged",
            "activityVideoStateChanged",
            "faceClustersChanged",
        ):
            bridge.on(
                event_name,
                lambda *args, name=event_name: self._queue_broadcast(name, *args),
            )
        self.bridge.on("waitCaptureBufferReady", self._queue_broadcast)

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
        detection = getattr(self.bridge, "_activity_detection_task", None)
        if detection is not None and not detection.done():
            detection.cancel()
            await asyncio.gather(detection, return_exceptions=True)
        face_detection = getattr(self.bridge, "_face_detection_task", None)
        if face_detection is not None and not face_detection.done():
            face_detection.cancel()
            await asyncio.gather(face_detection, return_exceptions=True)
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

    async def _face_thumbnail(self, request: web.Request) -> web.FileResponse:
        store = getattr(self.bridge, "face_clusters", None)
        if store is None:
            raise web.HTTPNotFound()
        try:
            path = store.thumbnail_path(request.match_info["identifier"])
        except ValueError as error:
            raise web.HTTPNotFound() from error
        if path is None:
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    async def _face_preview_stream(self, request: web.Request) -> web.StreamResponse:
        task = getattr(self.bridge, "_face_detection_task", None)
        video_capture = getattr(getattr(self.bridge, "agent", None), "video_capture", None)
        if task is None or task.done() or video_capture is None:
            raise web.HTTPNotFound()
        if video_capture.buffer_source("internal") is None:
            raise web.HTTPNotFound()

        response = web.StreamResponse(
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)
        last_frame_seconds = None
        try:
            while video_capture.buffer_source("internal") is not None:
                frame = video_capture.latest_buffer_frame("internal")
                if frame is not None and frame.seconds != last_frame_seconds:
                    image = base64.b64decode(frame.image_base64)
                    part = (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(image)}\r\n\r\n".encode("ascii")
                        + image
                        + b"\r\n"
                    )
                    await response.write(part)
                    last_frame_seconds = frame.seconds
                await asyncio.sleep(0.03)
        except ConnectionResetError:
            pass
        finally:
            try:
                await response.write(b"--frame--\r\n")
                await response.write_eof()
            except ConnectionResetError:
                pass
        return response

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        self._clients.add(socket)
        activity_detection_task = getattr(self.bridge, "_activity_detection_task", None)
        activity_detection_busy = (
            activity_detection_task is not None and not activity_detection_task.done()
        )
        await socket.send_json(
            {
                "event": "ready",
                "data": {
                    "knowledge": self.bridge.dumpKnowledgeData(),
                    "vehicleState": self.bridge.dumpVehicleState(),
                    "driverPhysicalState": getattr(self.bridge, "dumpDriverPhysicalState", lambda: {})(),
                    "activityDetectionBusy": activity_detection_busy,
                    "detectedObjectsState": self.bridge.dumpDetectedObjectsState(),
                    "vehicleSecurityState": getattr(
                        self.bridge, "dumpVehicleSecurityState", lambda: {
                            "sentry_mode_enabled": False,
                            "activity": "No activity detected",
                        },
                    )(),
                    "captureSources": getattr(self.bridge, "dumpCaptureSources", lambda: {
                        "internal": "camera_stream",
                        "external": "video_file",
                        "scene": "video_file",
                    })(),
                    "activityRecordings": getattr(self.bridge, "dumpActivityRecordings", lambda: {
                        "internal": [], "external": [],
                    })(),
                    "driverPreferences": self.bridge.dumpDriverPreferences(),
                    "faceClusters": getattr(self.bridge, "dumpFaceClusters", lambda: [])(),
                    "selectedFaceClusterId": getattr(self.bridge, "dumpSelectedFaceClusterId", lambda: None)(),
                    "driverAgendaState": getattr(self.bridge, "dumpDriverAgendaState", lambda: {
                        "meeting_status": "none", "meeting_timing": "none",
                    })(),
                    "musicState": getattr(self.bridge, "dumpMusicState", lambda: {"status": "disabled"})(),
                    "assistantStatus": getattr(self.bridge, "dumpAssistantStatus", lambda: "idle")(),
                    "musicDuckingFactor": getattr(getattr(self.bridge, "opt", None), "music_ducking_factor", 0.25),
                    "permissionReminderSeconds": getattr(
                        getattr(self.bridge, "opt", None),
                        "ui_permission_reminder_seconds", 10,
                    ),
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

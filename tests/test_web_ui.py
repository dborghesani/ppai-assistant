import sys
import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from aiohttp.test_utils import TestClient, TestServer
from data.agents_dataclasses import AssistantStatus
from agents.automotive_agent import AutomotiveAgent
from ui.bridge import VehicleBridge
from ui.web_server import WebUIServer


class FakeBridge:
    stt_manager = None

    def dumpAssistantStatus(self) -> str:
        return AssistantStatus.IDLE.value

    def __init__(self) -> None:
        self.listeners = {}
        self.friend_simulation_running = False
        self.conversation_cleared = False
        self.duplicate_suppression_enabled = False
        self.driver_preferences = {
            "preferred_cabin_temperature": None,
            "preferred_music": "",
        }

    def on(self, event_name, callback) -> None:
        self.listeners[event_name] = callback

    def dumpKnowledge(self) -> str:
        return '{"VehicleState": {"engine_on": false}}'

    def dumpKnowledgeData(self) -> dict:
        return {"VehicleState": {"engine_on": False}}

    def dumpVehicleState(self) -> dict:
        return {"engine_on": False, "privacy_mode": False}

    def dumpDetectedObjectsState(self) -> dict:
        return {
            "dangerous_objects_around": 0,
            "children_inside": 0,
            "people_inside": 1,
        }

    def dumpVehicleSecurityState(self) -> dict:
        return {
            "sentry_mode_enabled": False,
            "activity": "No activity detected",
        }

    def dumpCaptureSources(self) -> dict:
        return {
            "internal": "camera_stream",
            "external": "video_file",
            "scene": "video_file",
        }

    def dumpActivityRecordings(self) -> dict:
        return {
            "internal": ["activity_intl_driver_eating.mp4"],
            "external": ["activity_ext_vandalism.mp4"],
        }

    def dumpDriverPreferences(self) -> dict:
        return self.driver_preferences

    def setDriverPreference(self, name: str, value) -> None:
        self.driver_preferences[name] = value
        self.listeners["driverPreferencesChanged"](self.driver_preferences)

    def startFriendMessageSimulation(self) -> bool:
        self.friend_simulation_running = True
        return self.friend_simulation_running

    def duplicateSuppressionChanged(self, enabled: bool) -> None:
        self.duplicate_suppression_enabled = enabled

    def clearConversation(self) -> bool:
        self.conversation_cleared = True
        self.listeners["conversationCleared"]()
        return True


class WebUIServerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.web_ui = WebUIServer(FakeBridge())
        self.client = TestClient(TestServer(self.web_ui._app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_serves_dashboard_assets_and_health(self) -> None:
        for path, content_type in (
            ("/", "text/html"),
            ("/static/app.js", "text/javascript"),
            ("/static/style.css", "text/css"),
        ):
            response = await self.client.get(path)
            self.assertEqual(response.status, 200)
            self.assertTrue(response.content_type.startswith(content_type))

        app_response = await self.client.get("/static/app.js")
        app_script = await app_response.text()
        index_response = await self.client.get("/")
        index_html = await index_response.text()
        self.assertIn('data-tab="activities"', index_html)
        self.assertIn('data-tab="scenario"', index_html)
        self.assertIn('"driver_eating"', app_script)
        self.assertIn('"children_unattended"', app_script)
        self.assertIn('"animal_unattended"', app_script)
        self.assertIn('title: "Vehicle security", type: "toggles", cls: "VehicleSecurityState"', app_script)
        self.assertIn('["Sentry mode", "sentry_mode_enabled", false, "On", "Off"]', app_script)
        self.assertIn('data-capture-scope="${group.scope}"', app_script)
        self.assertIn('type: "captureSource", scope: "scene"', app_script)
        self.assertIn('buttonLabel = group.scope === "external" ? "Bump" : "Detect"', app_script)
        self.assertIn("let activityDetectionBusy = { internal: false, external: false };", app_script)
        self.assertIn("const buttonBusy = Boolean(activityDetectionBusy[buttonScope]);", app_script)
        self.assertIn('.activity-detect-button[aria-pressed="true"]:disabled', (await (await self.client.get("/static/style.css")).text()))
        self.assertNotIn("getUserMedia", app_script)
        self.assertIn("const CHANGED_EVENT_VISIBLE_MS = 5000;", app_script)
        self.assertIn('empty.textContent = "No recent changes";', app_script)
        self.assertIn("video[data-activity-video]", app_script)
        self.assertIn('bridge.call("startActivityVideo"', app_script)
        self.assertIn('bridge.call("stopActivityVideo"', app_script)
        self.assertIn(
            '["People inside", "people_inside", 0, 5, 1, 1, "integer"]',
            app_script,
        )

        response = await self.client.get("/health")
        self.assertEqual(await response.json(), {"status": "ok", "clients": 0})

    async def test_websocket_sends_state_and_handles_bridge_calls(self) -> None:
        socket = await self.client.ws_connect("/ws")
        ready = await socket.receive_json()
        self.assertEqual(ready["event"], "ready")
        self.assertEqual(ready["data"]["assistantStatus"], AssistantStatus.IDLE.value)
        self.assertFalse(ready["data"]["sttEnabled"])
        self.assertFalse(ready["data"]["duplicateSuppressionEnabled"])
        self.assertEqual(
            ready["data"]["vehicleState"],
            {"engine_on": False, "privacy_mode": False},
        )
        self.assertEqual(
            ready["data"]["detectedObjectsState"],
            {
                "dangerous_objects_around": 0,
                "children_inside": 0,
                "people_inside": 1,
            },
        )
        self.assertEqual(
            ready["data"]["vehicleSecurityState"],
            {
                "sentry_mode_enabled": False,
                "activity": "No activity detected",
            },
        )
        self.assertEqual(
            ready["data"]["captureSources"],
            {
                "internal": "camera_stream",
                "external": "video_file",
                "scene": "video_file",
            },
        )
        self.assertEqual(
            ready["data"]["activityRecordings"],
            {
                "internal": ["activity_intl_driver_eating.mp4"],
                "external": ["activity_ext_vandalism.mp4"],
            },
        )
        self.assertEqual(
            ready["data"]["driverPreferences"],
            {"preferred_cabin_temperature": None, "preferred_music": ""},
        )
        self.assertEqual(
            ready["data"]["knowledge"],
            {"VehicleState": {"engine_on": False}},
        )

        await socket.send_json({"id": 1, "method": "dumpKnowledge", "params": []})
        response = await socket.receive_json()
        self.assertEqual(response["id"], 1)
        self.assertIn("VehicleState", response["result"])
        await socket.close()

    async def test_activity_recordings_are_served_from_capture_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            recording = Path(temporary_directory) / "activity_intl_driver_eating.mp4"
            recording.write_bytes(b"activity video")
            bridge = FakeBridge()
            bridge.agent = SimpleNamespace(video_capture=SimpleNamespace(
                recordings_dir=Path(temporary_directory),
            ))
            server = WebUIServer(bridge)
            client = TestClient(TestServer(server._app))
            await client.start_server()
            try:
                response = await client.get("/activity-recordings/activity_intl_driver_eating.mp4")
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.read(), b"activity video")
            finally:
                await client.close()

    async def test_websocket_ready_restores_blocked_messages_status(self) -> None:
        self.web_ui.bridge.dumpAssistantStatus = lambda: {
            "status": AssistantStatus.IDLE.value,
            "flags": AssistantStatus.PENDING_MESSAGES_BLOCKED.bit,
        }

        socket = await self.client.ws_connect("/ws")
        ready = await socket.receive_json()

        self.assertEqual(
            ready["data"]["assistantStatus"],
            {
                "status": AssistantStatus.IDLE.value,
                "flags": AssistantStatus.PENDING_MESSAGES_BLOCKED.bit,
            },
        )
        await socket.close()

    async def test_websocket_updates_duplicate_suppression(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()
        await socket.send_json(
            {"id": 4, "method": "duplicateSuppressionChanged", "params": [True]}
        )

        response = await socket.receive_json()
        self.assertEqual(response, {"id": 4, "result": None})
        self.assertTrue(cast(Any, self.web_ui.bridge).duplicate_suppression_enabled)
        await socket.close()

    async def test_websocket_clears_conversation(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()
        await socket.send_json({"id": 7, "method": "clearConversation", "params": []})

        messages = [await socket.receive_json(), await socket.receive_json()]
        self.assertIn({"id": 7, "result": True}, messages)
        self.assertIn({"event": "conversationCleared", "data": []}, messages)
        self.assertTrue(cast(Any, self.web_ui.bridge).conversation_cleared)
        await socket.close()

    async def test_websocket_accepts_legacy_friend_simulation_method(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()
        await socket.send_json(
            {"id": 3, "method": "startFriendMessageSimulation", "params": []}
        )

        response = await socket.receive_json()
        self.assertEqual(response, {"id": 3, "result": True})
        await socket.close()

    async def test_websocket_forwards_vehicle_state_changes(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()

        cast(Any, self.web_ui.bridge).listeners["vehicleStateChanged"](
            {"window_open_front_left": True}
        )
        message = await socket.receive_json()

        self.assertEqual(
            message,
            {
                "event": "vehicleStateChanged",
                "data": [{"window_open_front_left": True}],
            },
        )
        await socket.close()

    async def test_websocket_forwards_transient_knowledge_changes(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()

        changes = ["Attention level is low."]
        cast(Any, self.web_ui.bridge).listeners["knowledgeChanged"](changes)
        message = await socket.receive_json()

        self.assertEqual(
            message,
            {"event": "knowledgeChanged", "data": [changes]},
        )
        await socket.close()

    async def test_websocket_forwards_detected_objects_state(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()

        cast(Any, self.web_ui.bridge).listeners["detectedObjectsStateChanged"](
            {"children_inside": 2}
        )
        message = await socket.receive_json()

        self.assertEqual(
            message,
            {"event": "detectedObjectsStateChanged", "data": [{"children_inside": 2}]},
        )
        await socket.close()

    async def test_websocket_updates_driver_preferences(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()

        await socket.send_json(
            {
                "id": 2,
                "method": "setDriverPreference",
                "params": ["preferred_music", "classic 1970s rock"],
            }
        )
        response = await socket.receive_json()
        update = await socket.receive_json()

        self.assertEqual(response, {"id": 2, "result": None})
        self.assertEqual(
            update,
            {
                "event": "driverPreferencesChanged",
                "data": [
                    {
                        "preferred_cabin_temperature": None,
                        "preferred_music": "classic 1970s rock",
                    }
                ],
            },
        )
        await socket.close()

    async def test_websocket_forwards_stt_readiness_changes(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()

        self.web_ui.bridge.listeners["sttEnabledChanged"](True)
        message = await socket.receive_json()

        self.assertEqual(
            message,
            {"event": "sttEnabledChanged", "data": [True]},
        )
        await socket.close()

    async def test_websocket_forwards_recognized_speech(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()

        self.web_ui.bridge.listeners["userSpeechReceived"]("Bonjour voiture")
        message = await socket.receive_json()

        self.assertEqual(
            message,
            {"event": "userSpeechReceived", "data": ["Bonjour voiture"]},
        )
        await socket.close()

    async def test_websocket_forwards_response_before_playback(self) -> None:
        socket = await self.client.ws_connect("/ws")
        await socket.receive_json()

        self.web_ui.bridge.listeners["responseUpdated"]("Je cherche un trajet.")
        message = await socket.receive_json()

        self.assertEqual(
            message,
            {"event": "responseUpdated", "data": ["Je cherche un trajet."]},
        )
        await socket.close()

    def test_qt_is_not_loaded(self) -> None:
        self.assertNotIn("PySide6", sys.modules)
        self.assertNotIn("qasync", sys.modules)


class VehicleBridgeMeetingTest(unittest.TestCase):
    def test_no_phase_does_not_clear_pending_meeting_permission_when_late(self) -> None:
        from agents.tasks.routing import TaskType
        from data.assistant_dataclasses import DriverAgenda

        bridge = cast(Any, VehicleBridge.__new__(VehicleBridge))
        bridge.database_manager = SimpleNamespace(current_state={
            "DriverAgenda": DriverAgenda(meeting_status="none", meeting_timing="late"),
        })
        meeting = Mock()
        bridge.agent = SimpleNamespace(action_manager=Mock(), tasks={TaskType.MEETING: meeting})
        bridge._listeners = defaultdict(list)

        bridge._sync_driver_agenda()

        meeting.reset.assert_not_called()
        bridge.agent.action_manager.set_meeting_timing.assert_called_once_with("late")

    def test_clearing_late_during_started_publishes_driver_arrival(self) -> None:
        from data.assistant_dataclasses import DriverAgenda

        bridge = cast(Any, VehicleBridge.__new__(VehicleBridge))
        bridge.database_manager = SimpleNamespace(current_state={
            "DriverAgenda": DriverAgenda(meeting_status="started", meeting_timing="late"),
        })
        bridge.send_event = Mock()

        result = bridge.stringChanged("DriverAgenda", "meeting_timing", "none")

        self.assertEqual(result, "arrived")
        bridge.send_event.assert_called_once_with("DriverAgenda.meeting_timing", "arrived")

    def test_clearing_late_before_started_still_means_no_scheduled_meeting(self) -> None:
        from data.assistant_dataclasses import DriverAgenda

        bridge = cast(Any, VehicleBridge.__new__(VehicleBridge))
        bridge.database_manager = SimpleNamespace(current_state={
            "DriverAgenda": DriverAgenda(meeting_status="prepare", meeting_timing="late"),
        })
        bridge.send_event = Mock()

        result = bridge.stringChanged("DriverAgenda", "meeting_timing", "none")

        self.assertEqual(result, "none")
        bridge.send_event.assert_called_once_with("DriverAgenda.meeting_timing", "none")

    def test_agenda_sync_passes_arrival_to_simulation(self) -> None:
        from agents.tasks.routing import TaskType
        from data.assistant_dataclasses import DriverAgenda

        bridge = cast(Any, VehicleBridge.__new__(VehicleBridge))
        bridge.database_manager = SimpleNamespace(current_state={
            "DriverAgenda": DriverAgenda(meeting_status="started", meeting_timing="arrived"),
        })
        bridge.agent = SimpleNamespace(action_manager=Mock(), tasks={TaskType.MEETING: Mock()})
        bridge._listeners = defaultdict(list)

        bridge._sync_driver_agenda()

        bridge.agent.action_manager.set_meeting_status.assert_called_once_with("started")
        bridge.agent.action_manager.set_meeting_timing.assert_called_once_with("arrived")


class VehicleBridgeTranscriptionTest(unittest.TestCase):
    def test_transcription_is_displayed_and_submitted_once(self) -> None:
        bridge = VehicleBridge.__new__(VehicleBridge)
        bridge._listeners = defaultdict(list)
        displayed = []
        submitted = []
        bridge.on("userSpeechReceived", displayed.append)
        bridge.userInput = submitted.append

        bridge._submit_transcription("  Bonjour voiture  ")

        self.assertEqual(displayed, ["Bonjour voiture"])
        self.assertEqual(submitted, ["Bonjour voiture"])


class AgentResponseTimingTest(unittest.IsolatedAsyncioTestCase):
    async def test_response_is_published_before_tts_speaks(self) -> None:
        order = []

        class FakeStream:
            def __init__(self) -> None:
                self.chunks = iter(("Bonjour. ", "Comment allez-vous?"))
                self.usage = None

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    token = next(self.chunks)
                except StopIteration as error:
                    raise StopAsyncIteration from error
                delta = SimpleNamespace(content=token)
                return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

            async def close(self) -> None:
                pass

        class FakeTTS:
            async def speak(self, text: str, on_word=None) -> None:
                order.append(("speak", text))

        async def create(**kwargs):
            return FakeStream()

        agent = AutomotiveAgent.__new__(AutomotiveAgent)
        agent._assistant_flags = AssistantStatus.IDLE.bit
        agent._assistant_status = AssistantStatus.IDLE
        agent.on_assistant_status_changed = None
        agent.on_speaking_tone_changed = None
        agent.log_llm_usage = Mock()
        agent.conversation_history = []
        agent.voice_llm = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        agent.opt = SimpleNamespace(
            ollama_model="test",
            max_tokens=1024,
            context_window_size=4096,
            conversation_attention_poll_seconds=0.01,
        )
        agent.tts_manager = FakeTTS()
        agent.on_response_update = lambda text: order.append(("ui", text))
        agent.on_response = lambda text: order.append(("final", text.strip()))
        event = SimpleNamespace(context=[], user_input="Bonjour")

        await agent.stream_user_response(event)

        self.assertEqual(order[0], ("ui", "Bonjour."))
        self.assertEqual(order[1], ("speak", "Bonjour."))
        self.assertEqual(order[2], ("ui", "Bonjour. Comment allez-vous?"))
        self.assertEqual(order[3], ("speak", "Comment allez-vous?"))


if __name__ == "__main__":
    unittest.main()

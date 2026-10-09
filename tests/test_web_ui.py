import sys
import unittest
from collections import defaultdict
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

    async def test_websocket_ready_restores_pending_permission_status(self) -> None:
        self.web_ui.bridge.dumpAssistantStatus = lambda: AssistantStatus.ASK_PERMISSION_TO_TALK.value

        socket = await self.client.ws_connect("/ws")
        ready = await socket.receive_json()

        self.assertEqual(
            ready["data"]["assistantStatus"],
            AssistantStatus.ASK_PERMISSION_TO_TALK.value,
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
            async def speak(self, text: str) -> None:
                order.append(("speak", text))

        async def create(**kwargs):
            return FakeStream()

        agent = AutomotiveAgent.__new__(AutomotiveAgent)
        agent._assistant_status = AssistantStatus.IDLE
        agent.on_assistant_status_changed = None
        agent.on_speaking_tone_changed = None
        agent.log_llm_usage = Mock()
        agent.skill_manager = SimpleNamespace(get_skill=lambda _: "")
        agent.conversation_history = []
        agent.voice_llm = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        agent.opt = SimpleNamespace(ollama_model="test", max_tokens=1024)
        from agents.llm_backend import LLMBackend
        from data.agents_dataclasses import ToneType
        agent.llm_backend = LLMBackend.__new__(LLMBackend)
        agent.llm_backend.opt = agent.opt
        agent.llm_backend.classify_conversation_tone = AsyncMock(return_value=ToneType.CALM)
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

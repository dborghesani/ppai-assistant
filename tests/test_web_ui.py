import sys
import unittest
from collections import defaultdict
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer
from automotive_agent import AutomotiveAgent
from ui.bridge import VehicleBridge
from ui.web_server import WebUIServer


class FakeBridge:
    stt_manager = None

    def __init__(self) -> None:
        self.listeners = {}

    def on(self, event_name, callback) -> None:
        self.listeners[event_name] = callback

    def dumpKnowledge(self) -> str:
        return '{"VehicleState": {"engine_on": false}}'

    def dumpKnowledgeData(self) -> dict:
        return {"VehicleState": {"engine_on": False}}


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

        response = await self.client.get("/health")
        self.assertEqual(await response.json(), {"status": "ok", "clients": 0})

    async def test_websocket_sends_state_and_handles_bridge_calls(self) -> None:
        socket = await self.client.ws_connect("/ws")
        ready = await socket.receive_json()
        self.assertEqual(ready["event"], "ready")
        self.assertFalse(ready["data"]["sttEnabled"])
        self.assertEqual(
            ready["data"]["knowledge"],
            {"VehicleState": {"engine_on": False}},
        )

        await socket.send_json({"id": 1, "method": "dumpKnowledge", "params": []})
        response = await socket.receive_json()
        self.assertEqual(response["id"], 1)
        self.assertIn("VehicleState", response["result"])
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
        agent.skill_manager = SimpleNamespace(get_skill=lambda _: "")
        agent._conversation_history = []
        agent._voice_llm = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        agent.opt = SimpleNamespace(ollama_model="ollama/test")
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

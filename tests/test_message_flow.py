import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from data.agents_dataclasses import (
    ActionType,
    AssistantStatus,
    InterventionType,
    SkillType,
    SuggestionType,
    ToneType,
    UrgencyType,
)
from agents.automotive_agent import AutomotiveAgent
from agents.llm_backend import NotificationDecision
from data.events import CarEvent, IncomingMessage
from managers.action_manager import ActionManager
from managers.message_manager import PendingMessage
from data.assistant_dataclasses import DriverPreferences
from sim.friend_message_agent import FriendMessageAgent


def test_driver_name_preference_defaults_to_david():
    assert DriverPreferences().driver_name == "David"


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = [
            "Did you hear what happened at dinner?",
            "No way!",
            "Apparently Giulia left early after an unexpected phone call.",
            "Do you know who she was talking to?",
        ]

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = self.responses.pop(0)
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        )


def make_agent(context: list[str] | None = None) -> AutomotiveAgent:
    agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
    agent.logger = Mock()
    agent.opt = SimpleNamespace(
        ollama_model="test-model", context_window_size=4096
    )
    summary_create = AsyncMock(
        return_value=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "David, Giulia left dinner early after an unexpected phone call. "
                            "Luca also has news that Marco is planning a surprise party for Sara."
                        )
                    )
                )
            ],
        )
    )
    agent.voice_llm = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=summary_create))
    )
    agent.knowledge_facts_provider = lambda: dict(context or {})
    agent.is_processing_event = False
    agent.event_queue = asyncio.Queue()
    agent._assistant_status = AssistantStatus.IDLE
    agent.on_assistant_status_changed = Mock()
    agent.on_incoming_message_classified = Mock()
    agent.on_response = None
    agent.on_speaking_tone_changed = None
    agent.tts_manager = None
    agent.conversation_history = []
    agent.action_manager = ActionManager(agent)
    agent.action_manager.message_manager.simulator = SimpleNamespace(
        submit_driver_reply=Mock(return_value=False),
        cancel=Mock(),
    )
    return agent


class MessageSimulatorTest(IsolatedAsyncioTestCase):
    async def test_generates_english_message_with_personality_and_logs_usage(self):
        usage = SimpleNamespace(prompt_tokens=20, completion_tokens=12, total_tokens=32)
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Did you hear that Giulia is planning a surprise picnic?"
                    )
                )
            ],
            usage=usage,
        )
        create = AsyncMock(return_value=response)
        llm_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        logged_usage: list[tuple[str, Any | None]] = []
        simulator = FriendMessageAgent(
            llm_client=cast(Any, llm_client),
            model="test-model",
            on_message=AsyncMock(),
            on_usage=lambda name, value: logged_usage.append((name, value)),
        )

        message = await simulator._generate_message()

        self.assertEqual(
            message,
            IncomingMessage(
                sender="Luca",
                text="Did you hear that Giulia is planning a surprise picnic?",
            ),
        )
        request = create.await_args.kwargs
        self.assertEqual(request["model"], "test-model")
        self.assertIn(
            "Return only the message text", request["messages"][0]["content"]
        )
        self.assertIn("frivolous", request["messages"][1]["content"])
        self.assertEqual(request["max_tokens"], 512)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertNotIn("response_format", request)
        self.assertEqual(logged_usage, [("message generation", usage)])

    async def test_serious_scenario_prompt_requests_unmistakably_urgent_message(self):
        response = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Please call me now about an important booking issue."
                    )
                )
            ],
        )
        create = AsyncMock(return_value=response)
        simulator = FriendMessageAgent(
            llm_client=cast(
                Any,
                SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
            ),
            model="test-model",
            on_message=AsyncMock(),
            on_usage=Mock(),
        )
        simulator._message_scenario = "serious"

        message = await simulator._generate_message()

        scenario_prompt = create.await_args.kwargs["messages"][1]["content"]
        self.assertIn("clearly time-sensitive", scenario_prompt)
        self.assertIn("immediately", scenario_prompt)
        self.assertIn("do not imply a crash", scenario_prompt)
        self.assertEqual(message.text, "Please call me now about an important booking issue.")

    async def test_generated_message_leaves_classification_unset(self):
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="Call me when you can."))],
        )
        simulator = FriendMessageAgent(
            llm_client=cast(
                Any,
                SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=AsyncMock(return_value=response))
                    )
                ),
            ),
            model="test-model",
            on_message=AsyncMock(),
            on_usage=Mock(),
        )

        message = await simulator._generate_message()

        self.assertEqual(message, IncomingMessage(sender="Luca", text="Call me when you can."))

    async def test_simulation_start_selects_new_tone_and_preserves_recent_history(self):
        simulator = FriendMessageAgent(
            llm_client=cast(Any, SimpleNamespace()),
            model="unused-in-this-test",
            on_message=AsyncMock(),
            on_usage=Mock(),
        )
        previous_history = [
            {"role": "assistant", "content": f"Earlier message {index}."}
            for index in range(8)
        ]
        simulator._history.extend(previous_history)

        with patch(
            "sim.friend_message_agent.random.choice",
            side_effect=["serious", "frivolous"],
        ):
            self.assertTrue(simulator.start())
            self.assertEqual(simulator._message_scenario, "serious")
            self.assertEqual(simulator._history, previous_history)
            await simulator.stop()
            self.assertTrue(simulator.start())
            self.assertEqual(simulator._message_scenario, "frivolous")
            self.assertEqual(simulator._history, previous_history)
            await simulator.stop()

    async def test_retries_when_model_returns_empty_content(self):
        empty_response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="stop")
            ],
            usage=None,
        )
        valid_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Any news from Marco?"
                    )
                )
            ],
            usage=None,
        )
        create = AsyncMock(side_effect=[empty_response, valid_response])
        logger = Mock()
        simulator = FriendMessageAgent(
            llm_client=cast(
                Any,
                SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
            ),
            model="test-model",
            on_message=AsyncMock(),
            on_usage=Mock(),
        )
        simulator.logger = logger

        message = await simulator._generate_message()

        self.assertEqual(
            message,
            IncomingMessage(
                sender="Luca",
                text="Any news from Marco?",
            ),
        )
        self.assertEqual(create.await_count, 2)
        logger.warning.assert_called_once_with(
            "Message model returned empty content",
            model="test-model",
            attempt=1,
            finish_reason="stop",
        )

    async def test_simulation_waits_for_driver_reply_between_turns(self):
        incoming: asyncio.Queue[IncomingMessage] = asyncio.Queue()
        simulator = FriendMessageAgent(
            llm_client=cast(Any, SimpleNamespace()),
            model="unused-in-this-test",
            on_message=incoming.put,
            on_usage=Mock(),
            interval_seconds=0.01,
        )
        simulator._generate_message = AsyncMock(
            side_effect=[
                IncomingMessage(sender="Luca", text="First"),
                IncomingMessage(sender="Luca", text="Second"),
                IncomingMessage(sender="Luca", text="Third"),
            ]
        )

        with patch("sim.friend_message_agent.random.randint", return_value=2):
            self.assertTrue(simulator.start())
            first = await asyncio.wait_for(incoming.get(), timeout=1)
            second = await asyncio.wait_for(incoming.get(), timeout=1)
            self.assertEqual([first.text, second.text], ["First", "Second"])
            self.assertTrue(simulator.waiting_for_reply)
            self.assertEqual(simulator._generate_message.await_count, 2)

            self.assertTrue(simulator.submit_driver_reply("What happened next?"))
            third = await asyncio.wait_for(incoming.get(), timeout=1)
            self.assertEqual(third.text, "Third")
            self.assertEqual(simulator._generate_message.await_count, 3)

            self.assertTrue(await simulator.stop())
            self.assertFalse(simulator.active)

    async def test_three_message_burst_keeps_one_topic(self):
        completions = FakeCompletions()
        incoming: asyncio.Queue[IncomingMessage] = asyncio.Queue()
        simulator = FriendMessageAgent(
            llm_client=cast(
                Any,
                SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            ),
            model="test-model",
            on_message=incoming.put,
            on_usage=Mock(),
            interval_seconds=0.001,
        )

        with patch("sim.friend_message_agent.random.randint", return_value=3):
            simulator.start()
            messages = [
                await asyncio.wait_for(incoming.get(), timeout=1) for _ in range(3)
            ]

            self.assertEqual(len(completions.calls), 3)
            third_prompt = completions.calls[2]["messages"]
            self.assertIn(
                "Did you hear what happened at dinner?",
                [message["content"] for message in third_prompt],
            )
            self.assertIn("No way!", [message["content"] for message in third_prompt])
            self.assertIn(
                "final message (3 of 3)",
                third_prompt[-1]["content"],
            )
            self.assertIn("Add one new detail", third_prompt[-1]["content"])
            self.assertTrue(simulator.waiting_for_reply)
            self.assertEqual(
                [message.text for message in messages],
                [
                    "Did you hear what happened at dinner?",
                    "No way!",
                    "Apparently Giulia left early after an unexpected phone call.",
                ],
            )

            self.assertTrue(simulator.submit_driver_reply("Who called her?"))
            fourth = await asyncio.wait_for(incoming.get(), timeout=1)
            self.assertEqual(fourth.text, "Do you know who she was talking to?")
            await simulator.stop()


class AutomotiveAgentMessageFlowTest(IsolatedAsyncioTestCase):
    async def test_pending_classification_tracks_remaining_messages(self):
        agent = make_agent()
        serious = IncomingMessage(sender="Luca", text="Call me urgently")
        casual = IncomingMessage(sender="Luca", text="A harmless rumor")
        agent.action_manager.message_manager.pending_messages.extend((PendingMessage(serious), PendingMessage(casual)))
        agent.remember_pending_message_classification(
            {"sender": serious.sender, "text": serious.text},
            ToneType.SERIOUS,
            UrgencyType.HIGH,
        )
        agent.remember_pending_message_classification(
            {"sender": casual.sender, "text": casual.text},
            ToneType.ENTHUSIASTIC,
            UrgencyType.NONE,
        )

        self.assertEqual(
            agent.most_urgent_pending_message_classification(),
            (ToneType.SERIOUS, UrgencyType.HIGH),
        )
        agent.handle_pending_message_action(ActionType.READ_PENDING_MESSAGES)
        self.assertIsNone(agent.most_urgent_pending_message_classification())

    async def test_stopping_simulation_discards_queued_luca_events_only(self):
        agent = make_agent()
        pending_message = IncomingMessage(sender="Luca", text="Call me soon")
        agent.action_manager.message_manager.pending_messages.append(PendingMessage(pending_message))
        agent.message_simulator.stop = AsyncMock(return_value=True)
        luca_event = CarEvent(
            SkillType.CONVERSATION,
            "incoming_message_received",
            {"sender": "Luca", "text": pending_message.text},
            [],
        )
        other_event = CarEvent(
            SkillType.CONVERSATION,
            "incoming_message_received",
            {"sender": "Mara", "text": "Another message"},
            [],
        )
        reminder_event = CarEvent(
            SkillType.CONVERSATION,
            "pending_messages_reminder",
            {"pending_count": 1},
            [],
        )
        for event in (luca_event, other_event, reminder_event):
            agent.event_queue.put_nowait(event)

        self.assertTrue(await agent.stop_message_simulation())

        self.assertEqual(list(agent.action_manager.message_manager.pending_messages), [PendingMessage(pending_message)])
        self.assertEqual(
            [agent.event_queue.get_nowait(), agent.event_queue.get_nowait()],
            [other_event, reminder_event],
        )

    async def test_repeated_permission_status_is_emitted_to_restart_animation(self):
        agent = cast(Any, AutomotiveAgent.__new__(AutomotiveAgent))
        agent._assistant_status = AssistantStatus.ASK_PERMISSION_TO_TALK
        agent.on_assistant_status_changed = Mock()

        agent.set_assistant_status(AssistantStatus.ASK_PERMISSION_TO_TALK)

        agent.on_assistant_status_changed.assert_called_once_with(
            AssistantStatus.ASK_PERMISSION_TO_TALK
        )

    async def test_agent_queues_message_with_current_knowledge_context(self):
        for privacy_fact in ("Privacy mode is off.", "Privacy mode is on."):
            agent = make_agent([privacy_fact])
            message = IncomingMessage(
                sender="Luca", text="A harmless rumor"
            )

            await agent._receive_incoming_message(message)

            self.assertEqual(list(agent.action_manager.message_manager.pending_messages), [PendingMessage(message)])
            event = agent.event_queue.get_nowait()
            self.assertEqual(event.event_name, "incoming_message_received")
            self.assertEqual(
                event.event_value,
                {"sender": "Luca", "text": "A harmless rumor"},
            )
            self.assertIn(privacy_fact, event.context)
            agent.cancel_message_tasks()
            await asyncio.sleep(0)

    async def test_permission_keeps_pending_messages_until_read_action(self):
        agent = make_agent()
        first = IncomingMessage(sender="Luca", text="First private message")
        second = IncomingMessage(sender="Luca", text="Second private message")
        agent.action_manager.message_manager.pending_messages.extend((PendingMessage(first), PendingMessage(second)))

        agent.action_manager.handle_decision(ActionType.ASK_PERMISSION_TO_TALK)
        self.assertEqual(
            agent.assistant_status, AssistantStatus.ASK_PERMISSION_TO_TALK
        )

        self.assertEqual(
            list(agent.action_manager.message_manager.pending_messages),
            [PendingMessage(first), PendingMessage(second)],
        )
        agent.action_manager.handle_decision(ActionType.READ_PENDING_MESSAGES)
        read_task = agent.action_manager.message_manager.read_task
        self.assertIsNotNone(read_task)
        self.assertEqual(agent.action_manager.message_manager.pending_count, 2)
        await read_task
        self.assertFalse(agent.action_manager.message_manager.pending_messages)


    async def test_read_action_interprets_messages_for_named_driver(self):
        agent = make_agent()
        first = IncomingMessage(
            sender="Luca",
            text="Giulia left dinner early after an unexpected phone call.",
        )
        second = IncomingMessage(
            sender="Luca",
            text="Marco is planning a surprise party for Sara.",
        )
        agent.action_manager.message_manager.pending_messages.extend((PendingMessage(first), PendingMessage(second)))

        agent.action_manager.handle_decision(ActionType.READ_PENDING_MESSAGES)
        read_task = agent.action_manager.message_manager.read_task
        self.assertIsNotNone(read_task)
        self.assertEqual(agent.action_manager.message_manager.pending_count, 2)
        agent.action_manager.handle_decision(ActionType.READ_PENDING_MESSAGES)
        self.assertIs(agent.action_manager.message_manager.read_task, read_task)
        await read_task
        self.assertFalse(agent.action_manager.message_manager.pending_messages)
        self.assertEqual(agent.voice_llm.chat.completions.create.await_count, 1)
        summary_request = agent.voice_llm.chat.completions.create.await_args.kwargs
        self.assertIn("Do not use report formulas", summary_request["messages"][0]["content"])
        self.assertIn('"recipient_name": "David"', summary_request["messages"][1]["content"])
        self.assertIn("unexpected phone call", summary_request["messages"][1]["content"])
        self.assertIn("surprise party for Sara", summary_request["messages"][1]["content"])
        self.assertEqual(
            [line for line in agent.conversation_history if line["role"] == "assistant"],
            [
                {
                    "role": "assistant",
                    "content": (
                        "David, Giulia left dinner early after an unexpected phone call. "
                        "Luca also has news that Marco is planning a surprise party for Sara."
                    ),
                }
            ],
        )

    async def test_postponed_delivery_acknowledges_delay_and_leaves_private_message_queued(self):
        agent = make_agent()
        private_message = IncomingMessage(sender="Luca", text="A private message")
        postponed_message = IncomingMessage(sender="Mara", text="A message sent while tired")
        manager = agent.action_manager.message_manager
        manager.pending_messages.extend((PendingMessage(private_message), PendingMessage(postponed_message)))

        agent.action_manager.handle_decision(
            ActionType.POSTPONE_NOTIFICATION_DELIVERY,
            {
                "event_value": {"sender": postponed_message.sender, "text": postponed_message.text},
                "context": ["Fatigue level is high."],
            },
        )

        self.assertEqual(manager.postponed_count, 1)
        self.assertEqual(manager.permission_pending_count, 1)
        self.assertEqual(manager.postponed_reason_summary, "fatigue")
        self.assertEqual(manager.pending_messages[1].postponed_reason, "fatigue")
        acknowledgement = "I delayed this message because you were tired."
        agent.action_manager.handle_decision(
            ActionType.READ_PENDING_MESSAGES,
            {"postponed_only": True, "acknowledgement": acknowledgement},
        )

        self.assertEqual(manager.pending_count, 2)
        self.assertEqual(manager.postponed_count, 0)
        await manager.read_task
        self.assertEqual(list(manager.pending_messages), [PendingMessage(private_message)])
        self.assertEqual(manager.postponed_count, 0)
        self.assertTrue(agent.conversation_history[-1]["content"].startswith(acknowledgement))
        self.assertEqual(agent.conversation_history[-1]["content"].count(acknowledgement), 1)
        self.assertIn("Giulia left dinner early", agent.conversation_history[-1]["content"])

    async def test_fatigued_pending_reminder_postpones_all_waiting_messages(self):
        agent = make_agent()
        messages = [
            IncomingMessage(sender="Luca", text="First pending message"),
            IncomingMessage(sender="Mara", text="Second pending message"),
        ]
        manager = agent.action_manager.message_manager
        manager.pending_messages.extend(PendingMessage(message) for message in messages)

        agent.action_manager.handle_decision(
            ActionType.POSTPONE_NOTIFICATION_DELIVERY,
            {
                "postpone_pending": True,
                "context": ["Privacy mode is off.", "Fatigue level is high."],
            },
        )

        self.assertEqual(manager.postponed_count, 2)
        self.assertEqual(manager.permission_pending_count, 0)
        self.assertEqual(manager.postponed_reason_summary, "fatigue")

    async def test_pending_message_prompt_uses_custom_driver_name(self):
        agent = make_agent()
        agent.driver_preferences_provider = lambda: {"driver_name": "Alex"}
        manager = agent.action_manager.message_manager
        manager.pending_messages.append(PendingMessage(IncomingMessage(sender="Luca", text="Can you call me now?")))
        manager.handle_action(ActionType.READ_PENDING_MESSAGES)
        await manager.read_task
        request = agent.voice_llm.chat.completions.create.await_args.kwargs
        self.assertIn('"recipient_name": "Alex"', request["messages"][1]["content"])
        self.assertIn("Do not use report formulas", request["messages"][0]["content"])
        self.assertEqual(request["reasoning_effort"], "none")

    async def test_read_action_speaks_original_messages_when_summary_is_empty(self):
        agent = make_agent()
        message = IncomingMessage(sender="Luca", text="Giulia has a new ride.")
        agent.action_manager.message_manager.pending_messages.append(PendingMessage(message))
        agent.voice_llm.chat.completions.create.return_value.choices[
            0
        ].message.content = ""

        agent.action_manager.handle_decision(ActionType.READ_PENDING_MESSAGES)
        await agent.action_manager.message_manager.read_task

        self.assertEqual(
            agent.conversation_history[-1],
            {"role": "assistant", "content": "Message from Luca: Giulia has a new ride."},
        )
        self.assertEqual(agent.pending_message_count, 0)

    async def test_failed_message_speech_keeps_queue_pending(self):
        agent = make_agent()
        message = IncomingMessage(sender="Luca", text="A message to deliver")
        manager = agent.action_manager.message_manager
        manager.pending_messages.append(PendingMessage(message))
        agent.speak = AsyncMock(side_effect=RuntimeError("speech unavailable"))

        agent.action_manager.handle_decision(ActionType.READ_PENDING_MESSAGES)
        await manager.read_task

        self.assertEqual(list(manager.pending_messages), [PendingMessage(message)])

    async def test_user_reply_goes_to_simulator_only_without_pending_messages(self):
        agent = make_agent()
        agent.message_simulator.submit_driver_reply.return_value = True
        agent.process_event_llm = AsyncMock()
        reply = CarEvent(
            SkillType.CONVERSATION, "user_input", "Tell me more", [], "Tell me more"
        )

        await agent.process_event(reply)

        agent.message_simulator.submit_driver_reply.assert_called_once_with(
            "Tell me more"
        )
        agent.process_event_llm.assert_not_awaited()

        agent.action_manager.message_manager.pending_messages.append(
            PendingMessage(IncomingMessage(sender="Luca", text="Private rumor"))
        )
        await agent.process_event(reply)
        agent.process_event_llm.assert_awaited_once_with(reply)

    async def test_reminder_enqueues_reasoner_event_only_when_agent_is_idle(self):
        agent = make_agent(["Privacy mode is on."])
        agent.action_manager.message_manager.reminder_interval = 0.01
        agent.action_manager.message_manager.pending_messages.append(
            PendingMessage(IncomingMessage(sender="Luca", text="Do not leak this text"))
        )
        agent.is_processing_event = True
        task = asyncio.create_task(agent.action_manager.message_manager.remind())
        agent.action_manager.message_manager.reminder_task = task

        await asyncio.sleep(0.025)
        self.assertTrue(agent.event_queue.empty())

        agent.is_processing_event = False
        event = await asyncio.wait_for(agent.event_queue.get(), timeout=1)
        self.assertEqual(event.event_name, "pending_messages_reminder")
        self.assertEqual(event.event_value, {"pending_count": 1})
        self.assertNotIn("Do not leak this text", " ".join(event.context))

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
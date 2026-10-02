import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from data.events import IncomingMessage
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


class FriendMessageAgent:
    _PERSONALITY = """
You are Luca, the driver's old friend and a familiar contact in a car messaging demo.
Generate content that fits the current simulation scenario. Keep it kind and plausible; never
invent humiliating, dangerous, or genuinely private claims. The mutual friends are fictional:
Giulia, Marco, Sara, Paolo, and Chiara. Send one short message at a time, then wait for the
driver's response before starting a new turn. A turn contains one, two, or three messages.
Follow-up messages must continue the same topic.

Always write in English. Return only the message text, with no labels or quotes.
""".strip()

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        model: str,
        on_message: Callable[[IncomingMessage], Awaitable[None]],
        on_usage: Callable[[str, Any | None], None],
        interval_seconds: float = 2.0,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.on_message = on_message
        self.on_usage = on_usage
        self.interval_seconds = interval_seconds
        self.logger = structlog.get_logger()
        self._task: asyncio.Task[None] | None = None
        self._message_scenario = "frivolous"
        self._history: list[ChatCompletionMessageParam] = []
        self._reply_future: asyncio.Future[str] | None = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def waiting_for_reply(self) -> bool:
        return self._reply_future is not None and not self._reply_future.done()

    def start(self) -> bool:
        if self.active:
            return False
        self._message_scenario = random.choice(("frivolous", "serious"))
        self.logger.info("Friend-message simulation scenario selected", scenario=self._message_scenario)
        self._task = asyncio.create_task(self._run())
        return True

    async def stop(self) -> bool:
        task = self._task
        if task is None:
            return False
        self._task = None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    def cancel(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()

    def submit_driver_reply(self, text: str) -> bool:
        reply = text.strip()
        if not reply or not self.waiting_for_reply or self._reply_future is None:
            return False
        self._reply_future.set_result(reply)
        return True

    async def _generate_message(
        self, message_index: int = 0, message_count: int = 1
    ) -> IncomingMessage | None:
        if message_count == 1:
            turn_instruction = (
                "This is a one-message turn. Open one conversation thread and "
                "ask at most one brief question."
            )
        elif message_index == 0:
            turn_instruction = (
                f"This is message 1 of {message_count} in one turn. Open a single "
                "thread with one brief question or detail; leave room to add "
                "new related details in the follow-up messages."
            )
        elif message_index == message_count - 1:
            turn_instruction = (
                f"This is the final message ({message_index + 1} of {message_count}) "
                "in the same turn. Add one new detail that follows directly from the "
                "existing thread. Do not repeat the question or recap earlier messages."
            )
        else:
            turn_instruction = (
                f"This is follow-up message {message_index + 1} of {message_count} "
                "in the same turn. Add a new, related detail; do not change topic, "
                "repeat the question, or summarize what you already said."
            )

        scenario_instruction = (
            "This simulation is serious: send a clearly time-sensitive message that asks "
            "the driver to call you immediately about an important fictional matter. Make "
            "the urgency unmistakable, but do not imply a crash, illness, or immediate danger. "
            "This scenario only guides message content."
            if self._message_scenario == "serious"
            else "This simulation is frivolous: send a light, harmless, low-stakes message "
            "about amusing fictional gossip. Do not imply urgency or ask for immediate action. "
            "This scenario only guides message content."
        )

        for attempt in range(2):
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._PERSONALITY},
                    {"role": "system", "content": scenario_instruction},
                    *self._history[-6:],
                    {
                        "role": "user",
                        "content": f"{turn_instruction} Write one short, natural "
                        "message to the driver. Return the message text only.",
                    },
                ],
                temperature=0.7 if attempt else 0.85,
                max_tokens=512,
                reasoning_effort="none",
            )
            self.on_usage("message generation", response.usage)
            choices = getattr(response, "choices", None) or []
            raw_content = (
                getattr(getattr(choices[0], "message", None), "content", None)
                if choices
                else None
            )
            text = raw_content.strip() if isinstance(raw_content, str) else ""
            if text:
                return IncomingMessage(sender="Luca", text=text)
            self.logger.warning(
                "Message model returned empty content",
                model=self.model,
                attempt=attempt + 1,
                finish_reason=getattr(choices[0], "finish_reason", None)
                if choices
                else None,
            )
        return None

    async def _run(self) -> None:
        try:
            while True:
                reply_future = asyncio.get_running_loop().create_future()
                self._reply_future = reply_future
                message_limit = random.randint(1, 3)
                sent_count = 0

                for message_index in range(message_limit):
                    try:
                        message = await self._generate_message(
                            message_index=message_index,
                            message_count=message_limit,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        self.logger.warning(
                            "Friend message generation failed", error=str(error)
                        )
                        self.on_usage("friend message generation", None)
                        break

                    if message is None:
                        break
                    self._history.append(
                        {"role": "assistant", "content": message.text}
                    )
                    sent_count += 1
                    await self.on_message(message)
                    if reply_future.done():
                        break
                    if message_index + 1 < message_limit:
                        await asyncio.sleep(self.interval_seconds)

                if sent_count == 0:
                    self._reply_future = None
                    await asyncio.sleep(self.interval_seconds)
                    continue

                driver_reply = await reply_future
                self._history.append({"role": "user", "content": driver_reply})
        except asyncio.CancelledError:
            raise
        finally:
            self._reply_future = None
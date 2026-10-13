from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable

import structlog
from data.agents_dataclasses import ActionType, AssistantStatus
from agents.tasks.mood_music import MUSIC_ACTIONS, MUSIC_ACTION_GUIDANCE
from sim.meeting_sim import DEFAULT_TOPIC, build_crew

if TYPE_CHECKING:
    from agents.automotive_agent import AutomotiveAgent

class ActionManager:
    """Single dispatch point for decided actions (LLM or Laya): launches actuations and hands
    their outcome to the agent's output layer (speech/text). No LLM or conversational logic here."""

    def __init__(self, agent: "AutomotiveAgent"):
        self.agent = agent
        self.logger = structlog.get_logger()
        self._meeting_task: asyncio.Task | None = None
        self._awaiting_meeting_confirmation = False
        self.on_action: Callable[[ActionType, dict[str, Any]], None] | None = None
        self.on_meeting_completed: Callable[[], None] | None = None
        self._registered_handlers: dict[ActionType, Callable[[ActionType, str], Awaitable[None]]] = {}
        self._registered_guidance: list[str] = []
        self.register_actions(
            MUSIC_ACTIONS,
            lambda action, request: self.agent.music_task.execute_music_action(
                action, request, self.agent.opt.music_enabled,
            ),
            MUSIC_ACTION_GUIDANCE,
        )

    def close(self) -> None:
        self.agent.music_task.close_playback()

    @property
    def registered_actions(self) -> frozenset[ActionType]:
        return frozenset(self._registered_handlers)

    @property
    def registered_action_options(self) -> str:
        return "\n".join(f"- {action.value}: {action.description}" for action in ActionType if action in self._registered_handlers)

    @property
    def registered_action_guidance(self) -> str:
        return "\n".join(self._registered_guidance)

    def register_actions(self, actions: Iterable[ActionType], handler: Callable[[ActionType, str], Awaitable[None]], guidance: str) -> None:
        for action in actions:
            if action in self._registered_handlers:
                raise ValueError(f"Action already registered: {action.value}")
            self._registered_handlers[action] = handler
        self._registered_guidance.append(guidance)

    async def execute_registered_action(self, action: ActionType, request: str) -> None:
        await self._registered_handlers[action](action, request)

    def handle_decision(
        self, action_type: ActionType, parameters: dict[str, Any] | None = None
    ) -> None:
        if action_type is ActionType.ASK_PERMISSION_TO_TALK:
            self.agent.set_assistant_status(AssistantStatus.ASK_PERMISSION_TO_TALK)
        elif action_type in {
            ActionType.READ_PENDING_MESSAGES,
            ActionType.POSTPONE_NOTIFICATION_DELIVERY,
        }:
            self.agent.messages_task.handle_action(action_type, parameters)
        elif action_type is ActionType.ASK_ATTEND_MEETING:
            self.mark_awaiting_meeting_confirmation()
        elif action_type is not ActionType.NONE:
            self.agent.set_assistant_status(AssistantStatus.ACT)
            try:
                self.logger.info(
                    f"[action] executing {action_type} with parameters: {parameters}"
                )
                if self.on_action is not None:
                    self.on_action(action_type, parameters or {})
            finally:
                self.agent.set_assistant_status(AssistantStatus.IDLE)

    @property
    def awaiting_confirmation(self) -> bool:
        return self._awaiting_meeting_confirmation

    def mark_awaiting_meeting_confirmation(self) -> None:
        if self._meeting_task is not None and not self._meeting_task.done():
            return
        self._awaiting_meeting_confirmation = True
        self.logger.info("[meeting] asked driver, awaiting confirmation reply")

    def confirm_attend_meeting(self) -> None:
        """Called by the agent once it has interpreted the driver's reply as a yes."""
        self._awaiting_meeting_confirmation = False
        self.logger.info("[meeting] confirmed by driver, launching simulation")
        self.agent.set_assistant_status(AssistantStatus.BACKGROUND_TASK_RUNNING)
        self._meeting_task = asyncio.create_task(self.attend_meeting())

    def decline_attend_meeting(self) -> None:
        self._awaiting_meeting_confirmation = False
        self.agent.set_assistant_status(AssistantStatus.IDLE)
        self.logger.info("[meeting] declined by driver, no simulation launched")

    async def attend_meeting(self) -> None:
        """Run the simulated work meeting crew and hand its outcome to the agent to speak."""
        summary = ""
        try:
            crew, driver_summary_task = build_crew(
                self.agent.llm, DEFAULT_TOPIC, verbose=self.agent.opt.crewai_verbose
            )
            await asyncio.to_thread(crew.kickoff)
            if driver_summary_task.output is not None:
                summary = driver_summary_task.output.raw.strip()
        except Exception as error:
            self.logger.error(f"Meeting simulation failed: {error}")
        finally:
            self._meeting_task = None
            self.agent.set_assistant_status(AssistantStatus.IDLE)
        await self.agent.speak(
            summary or "Sorry, something went wrong while I was attending the meeting."
        )
        self.logger.info("[meeting] scenario completed; ending it")
        if self.on_meeting_completed is not None:
            self.on_meeting_completed()

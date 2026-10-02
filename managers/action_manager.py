from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

import structlog
from agents.agents_dataclasses import ActionType, AssistantStatus
from sim.meeting_sim import DEFAULT_TOPIC, build_crew

if TYPE_CHECKING:
    from agents.automotive_agent import AutomotiveAgent


class ActionManager:
    """Single dispatch point for decided actions (LLM or Laya): launches actuations and hands
    their outcome to the agent's output layer (speech/text). No LLM or conversational logic here."""

    def __init__(self, agent: "AutomotiveAgent"):
        self.agent = agent
        self.logger = structlog.get_logger()
        self.awaiting_confirmation = False
        self._meeting_task: asyncio.Task | None = None
        self.on_action: Callable[[ActionType, dict[str, Any]], None] | None = None

    def handle_decision(
        self, action_type: ActionType, parameters: dict[str, Any] | None = None
    ) -> None:
        if action_type is ActionType.ASK_PERMISSION_TO_TALK:
            self.agent.set_assistant_status(AssistantStatus.ASK_PERMISSION_TO_TALK)
        elif action_type in {
            ActionType.ANNOUNCE_INCOMING_MESSAGE,
            ActionType.READ_PENDING_MESSAGES,
        }:
            self.agent.handle_pending_message_action(action_type)
        elif action_type is ActionType.ASK_ATTEND_MEETING:
            self._mark_awaiting_meeting_confirmation()
        elif action_type is not ActionType.NONE:
            self.agent.set_assistant_status(AssistantStatus.ACT)
            try:
                self.logger.info(
                    f">>> [action] executing {action_type} with parameters: {parameters}"
                )
                if self.on_action is not None:
                    self.on_action(action_type, parameters or {})
            finally:
                self.agent.set_assistant_status(AssistantStatus.IDLE)

    def _mark_awaiting_meeting_confirmation(self) -> None:
        if self._meeting_task is not None and not self._meeting_task.done():
            return
        self.logger.info(">>> [meeting] asked driver, awaiting confirmation reply")
        self.awaiting_confirmation = True

    def confirm_attend_meeting(self) -> None:
        """Called by the agent once it has interpreted the driver's reply as a yes."""
        self.awaiting_confirmation = False
        self.logger.info(">>> [meeting] confirmed by driver, launching simulation")
        self.agent.set_assistant_status(AssistantStatus.BACKGROUND_TASK_RUNNING)
        self._meeting_task = asyncio.create_task(self._attend_meeting())

    def decline_attend_meeting(self) -> None:
        self.awaiting_confirmation = False
        self.logger.info(">>> [meeting] declined by driver, no simulation launched")

    async def _attend_meeting(self) -> None:
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

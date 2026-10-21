from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable

import structlog
from data.agents_dataclasses import ActionType, AssistantStatus
from agents.tasks.mood_music import MUSIC_ACTIONS, MUSIC_ACTION_GUIDANCE
from sim.meeting_sim import (
    DEFAULT_TOPIC, DEFAULT_TURN_PAUSE_SECONDS, MeetingInterrupted, MeetingProgress,
    build_arrival_summary_crew, build_crew, build_preparation_crew,
)

if TYPE_CHECKING:
    from agents.automotive_agent import AutomotiveAgent

class ActionManager:
    """Single dispatch point for decided actions (LLM or Laya): launches actuations and hands
    their outcome to the agent's output layer (speech/text). No LLM or conversational logic here."""

    def __init__(self, agent: "AutomotiveAgent"):
        self.agent = agent
        self.logger = structlog.get_logger()
        self._meeting_task: asyncio.Task | None = None
        self._meeting_preparation_task: asyncio.Task | None = None
        self._awaiting_meeting_confirmation = False
        self._meeting_status = "none"
        self._meeting_timing = "none"
        self._meeting_authorized = False
        self._meeting_progress: MeetingProgress | None = None
        self._meeting_arrival_transcript: list[dict[str, str]] | None = None
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
        if self._meeting_progress is not None:
            self._meeting_progress.stop()
        for task in (self._meeting_preparation_task, self._meeting_task):
            if task is not None and not task.done():
                task.cancel()
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
        if action_type is ActionType.BLOCK_PENDING_MESSAGES:
            if self.agent.messages_task.pending_count:
                self.agent.set_assistant_status(AssistantStatus.PENDING_MESSAGES_BLOCKED)
            else:
                self.agent.clear_blocked_messages_status()
        elif action_type is ActionType.READ_PENDING_MESSAGES:
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
        self._meeting_authorized = True
        self.logger.info("[meeting] confirmed by driver", meeting_status=self._meeting_status)
        self._start_authorized_meeting()

    def set_meeting_status(self, status: str) -> None:
        if status not in {"none", "prepare", "started", "finished"}:
            raise ValueError(f"Unknown meeting status: {status}")
        if status == self._meeting_status:
            return
        self._meeting_status = status
        if status in {"none", "prepare", "finished"}:
            if self._meeting_preparation_task is not None and not self._meeting_preparation_task.done():
                self._meeting_preparation_task.cancel()
            self._meeting_preparation_task = None
        if status == "prepare":
            self._meeting_preparation_task = asyncio.create_task(self.prepare_meeting())
        if status != "started":
            if status == "finished":
                self._meeting_authorized = False
                self._awaiting_meeting_confirmation = False
            if self._meeting_progress is not None:
                self._meeting_progress.stop()
            if self._meeting_task is not None and not self._meeting_task.done():
                self._meeting_task.cancel()
        self._start_authorized_meeting()

    def set_meeting_timing(self, timing: str) -> None:
        if timing not in {"none", "on_time", "late", "arrived"}:
            raise ValueError(f"Unknown meeting timing: {timing}")
        previous = self._meeting_timing
        self._meeting_timing = timing
        if timing != "late":
            self._meeting_authorized = False
            self._awaiting_meeting_confirmation = False
        if (
            previous == "late" and timing in {"none", "on_time", "arrived"}
            and self._meeting_status == "started"
            and self._meeting_task is not None and not self._meeting_task.done()
            and self._meeting_arrival_transcript is None
        ):
            self._meeting_authorized = False
            self._meeting_arrival_transcript = (
                self._meeting_progress.stop() if self._meeting_progress is not None else []
            )
            self.logger.info("[meeting] driver arrived; stopping attendance", completed_contributions=len(self._meeting_arrival_transcript))

    def _start_authorized_meeting(self) -> None:
        if self._meeting_status != "started" or not self._meeting_authorized:
            return
        if self._meeting_task is not None and not self._meeting_task.done():
            return
        self.logger.info("[meeting] started and authorized, launching simulation")
        self._meeting_progress = None
        self._meeting_arrival_transcript = None
        self.agent.set_assistant_status(AssistantStatus.BACKGROUND_TASK_RUNNING)
        self._meeting_task = asyncio.create_task(self.attend_meeting())

    def decline_attend_meeting(self) -> None:
        self._awaiting_meeting_confirmation = False
        self._meeting_authorized = False
        self.agent.set_assistant_status(AssistantStatus.IDLE)
        self.logger.info("[meeting] declined by driver, no simulation launched")

    async def prepare_meeting(self) -> None:
        current_task = asyncio.current_task()
        self.agent.set_assistant_status(AssistantStatus.BACKGROUND_TASK_RUNNING)
        try:
            crew, briefing_task = build_preparation_crew(
                self.agent.llm, DEFAULT_TOPIC, verbose=self.agent.opt.crewai_verbose,
            )
            await asyncio.to_thread(crew.kickoff)
            briefing = briefing_task.output.raw.strip() if briefing_task.output is not None else ""
            if not briefing:
                raise ValueError("Meeting preparation returned no agenda briefing")
            await self.agent.speak(briefing)
        except Exception as error:
            self.logger.error("Meeting preparation failed", error=str(error))
            await self.agent.speak("Sorry, I couldn't prepare the meeting agenda.")
        finally:
            if self._meeting_preparation_task is current_task:
                self._meeting_preparation_task = None
                self.agent.set_assistant_status(AssistantStatus.IDLE)

    async def attend_meeting(self) -> None:
        """Run the simulated work meeting crew and hand its outcome to the agent to speak."""
        current_task = asyncio.current_task()
        summary = ""
        try:
            if self._meeting_preparation_task is not None:
                await self._meeting_preparation_task
            self.agent.set_assistant_status(AssistantStatus.BACKGROUND_TASK_RUNNING)
            if self._meeting_arrival_transcript is None:
                crew, driver_summary_task = build_crew(
                    self.agent.llm, DEFAULT_TOPIC, verbose=self.agent.opt.crewai_verbose
                )
                crew_tasks = getattr(crew, "tasks", None)
                self._meeting_progress = MeetingProgress(
                    pause_seconds=DEFAULT_TURN_PAUSE_SECONDS,
                    expected_contributions=(
                        len(crew_tasks) if isinstance(crew_tasks, (list, tuple)) else None
                    ),
                )
                crew.task_callback = self._meeting_progress.record
                try:
                    await asyncio.to_thread(crew.kickoff)
                except MeetingInterrupted:
                    pass
                if self._meeting_arrival_transcript is None and driver_summary_task.output is not None:
                    summary = driver_summary_task.output.raw.strip()
            if self._meeting_arrival_transcript is not None:
                crew, handover_task = build_arrival_summary_crew(
                    self.agent.llm, self._meeting_arrival_transcript,
                    verbose=self.agent.opt.crewai_verbose,
                    driver_name=self.agent.driver_name,
                )
                await asyncio.to_thread(crew.kickoff)
                if handover_task.output is not None:
                    summary = handover_task.output.raw.strip()
        except Exception as error:
            self.logger.error(f"Meeting simulation failed: {error}")
        finally:
            if self._meeting_task is current_task:
                self._meeting_task = None
                self._meeting_progress = None
                self.agent.set_assistant_status(AssistantStatus.IDLE)
        await self.agent.speak(
            summary or "Sorry, something went wrong while I was attending the meeting."
        )
        self.logger.info("[meeting] scenario completed; ending it")
        if self.on_meeting_completed is not None:
            self.on_meeting_completed()

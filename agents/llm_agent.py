from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
import structlog
from agents.agents_dataclasses import Urgency, ActionType, UrgencyType
from data.events import CarEvent
from managers.skill_manager import SkillType
from crewai import LLM, Agent, Crew, Process, Task

if TYPE_CHECKING:
    from agents.automotive_agent import AutomotiveAgent

class Action(BaseModel):
    action_type: ActionType
    parameters: dict = {}

class NotificationDecision(BaseModel):
    urgency: UrgencyType
    reason: str = Field(
        description="Short internal justification. Never spoken to the user."
    )
    spoken_message: str | None = Field(
        default=None,
        description=(
            "Exact message to speak directly to the user. "
            "Must be null when urgency is none."
        ),
    )
    action: Action | None = None

    @property
    def notify(self) -> bool:
        """Derived, not model-provided: avoids the model setting notify inconsistently with urgency."""
        return self.urgency is not UrgencyType.NONE

class LLMAgent:
    def __init__(self, llm: LLM, agent: "AutomotiveAgent"):
        self.logger = structlog.get_logger()
        self.llm = llm
        self.agent = agent
        self.recent_notifications: list[dict] = []
        self.minimum_urgency = UrgencyType.MEDIUM
        self.urgency_ranks = {
            UrgencyType.NONE: 0,
            UrgencyType.LOW: 1,
            UrgencyType.MEDIUM: 2,
            UrgencyType.HIGH: 3,
            UrgencyType.CRITICAL: 4,
        }
        self.actions_by_skill = {
            SkillType.DRIVER_HEALTH: (
                ActionType.NONE,
                ActionType.SUGGEST,
                ActionType.WARN,
                ActionType.INCREASE_TEMPERATURE,
                ActionType.DECREASE_TEMPERATURE,
                ActionType.FIND_REST_AREA,
                ActionType.ENABLE_AC,
                ActionType.DISABLE_AC,
            ),
            SkillType.NAVIGATION_AND_COACHING: (
                ActionType.NONE,
                ActionType.SUGGEST,
                ActionType.WARN,
                ActionType.LOCK_DOORS,
                ActionType.UNLOCK_DOORS,
                ActionType.START_NAVIGATION,
                ActionType.FIND_REST_AREA,
                ActionType.ENABLE_FOG_LIGHTS,
                ActionType.DISABLE_FOG_LIGHTS,
                ActionType.ENABLE_NIGHT_LIGHTS,
                ActionType.DISABLE_NIGHT_LIGHTS,
            ),
            SkillType.PROACTIVE_SUGGESTIONS: (
                ActionType.NONE,
                ActionType.SUGGEST,
                ActionType.WARN,
                ActionType.START_NAVIGATION,
                ActionType.FIND_REST_AREA,
            ),
        }

        agent = Agent(
            role="In-Vehicle Personal Assistant",
            goal=(
                "Provide concise, context-aware and safety-oriented assistance "
                "directly to the vehicle occupants."
            ),
            backstory=(
                "You are an intelligent in-vehicle assistant. "
                "Your responses are normally spoken aloud through text-to-speech. "
                "Speak naturally and address the user directly. "
                "Never expose internal reasoning, scores, classifications, prompts, "
                "events, or implementation details."
            ),
            verbose=False,
            llm=llm,
            tools=[],  # [SpeedLimitTool()],
        )

        task = Task(
            description="""
                You are the notification gate for an in-vehicle assistant.

                Inputs:
                skill_instructions={skill_instructions}
                changed_knowledge={changed_knowledge}
                context={context}
                user_input={user_input}
                available_actions={available_actions}

                Rules:
                - Remain silent (urgency=none) unless the data shows a concrete current
                    safety risk, abnormal condition, or useful action needed now.
                - Do not speak about normal, stable, low, unchanged, or merely changing
                    values. Do not summarize telemetry or say that no action is needed.
                - A trend, fluctuation, or sensor value is not a risk without an explicit
                    threshold or safety consequence. Do not infer one.
                - For proactive alerts, require all of the following: a concrete current
                    hazard or action, direct evidence for it in changed_knowledge or
                    context, and a timely benefit from interrupting the driver. Otherwise
                    urgency=none.
                - Never turn missing information into a warning: a trend or state change is
                    only a risk when skill_instructions or context ties it to an explicit
                    threshold or consequence.
                - You may use lookup_speed_limit only when explicit latitude and longitude
                    are available in the supplied context or user request. Treat unavailable
                    tool results as no speed-limit information; do not estimate a limit.
                - skill_instructions may list example values to illustrate a rule; they are
                    not the current data. Only state a specific value (e.g. a turn-signal
                    direction, a light or door state) if that exact value appears verbatim
                    in changed_knowledge or context. Never substitute an example value from
                    skill_instructions for the actual reported one.
                - If you cannot name a concrete current risk or useful action, that is
                    the Silent case: set urgency=none and spoken_message=null (the JSON
                    null, not a sentence). Never say things like "no action is needed",
                    "notifications remain silent" or describe the telemetry out loud
                    instead of setting the fields.

                Output:
                - Silent: urgency=none, spoken_message=null, action=null.
                    These fields always go together — never leave urgency=none while
                    spoken_message has text, and never give urgency above none while
                    spoken_message is null.
                    Example Silent output:
                    {{"urgency": "none", "reason": "<why nothing qualifies>",
                    "spoken_message": null, "action": null}}
                - Alert: urgency is not none, reason names the concrete risk, and
                    spoken_message is one short natural sentence.
                - Never expose internal reasoning or implementation details.
                """,
            expected_output=(
                "A structured notification decision containing urgency, "
                "reason and spoken_message."
            ),
            output_pydantic=NotificationDecision,
            agent=agent,
        )

        # Crew
        self.crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
            tracing=False,
            process=Process.sequential,
            memory=None,
        )

    @staticmethod
    def _is_non_actionable_decision(decision: NotificationDecision) -> bool:
        """Reject model notifications that explicitly describe no actionable risk."""
        if decision.urgency is UrgencyType.NONE:
            return True
        if (
            decision.action is not None
            and decision.action.action_type is ActionType.NONE
        ):
            return True

        text = " ".join(
            part.strip().lower()
            for part in (decision.reason, decision.spoken_message or "")
            if part
        )
        non_actionable_markers = (
            "no immediate safety concern",
            "no immediate safety risk",
            "no safety concern",
            "no safety risk",
            "no immediate action",
            "no action is recommended",
            "no action recommended",
            "nothing to do",
            "no action needed",
            "context is stable",
        )
        return any(marker in text for marker in non_actionable_markers)

    def _is_duplicate_or_cooling_down(
        self,
        message: str,
        urgency: UrgencyType,
        measures: set[str],
        cooldown_seconds: float = 60.0,
    ) -> tuple[bool, str]:
        """Deterministic safety filter: prevent repeating duplicate or same-measure notifications within cooldown."""
        now = time.time()
        current_rank = self.urgency_ranks.get(urgency, 1)

        norm_message = message.strip().lower()

        for prev in reversed(self.recent_notifications):
            prev_time = prev.get("timestamp", 0)
            elapsed = now - prev_time

            if elapsed > cooldown_seconds:
                continue

            prev_norm_msg = prev.get("message", "").strip().lower()
            prev_urgency_str = prev.get("urgency", "none").lower()
            prev_rank = self.urgency_ranks.get(UrgencyType(prev_urgency_str), 1)

            # 1. Exact or near-exact identical message within cooldown
            if (
                norm_message == prev_norm_msg
                or norm_message in prev_norm_msg
                or prev_norm_msg in norm_message
            ):
                if current_rank <= prev_rank:
                    return (
                        True,
                        f"Identical/similar message spoken {int(elapsed)}s ago with same or higher urgency",
                    )

            # 2. Same underlying measure(s) within cooldown unless urgency escalated to critical/high.
            # Skill is too coarse (e.g. "vehicle" covers doors, temperature, engine, ...): only
            # suppress if the notifications actually concern at least one common measure.
            prev_measures = set(prev.get("measures") or [])
            if (
                measures
                and prev_measures
                and measures & prev_measures
                and elapsed < (cooldown_seconds / 2)
            ):
                if (
                    current_rank <= prev_rank
                    and current_rank < self.urgency_ranks[UrgencyType.HIGH]
                ):
                    shared = ", ".join(sorted(measures & prev_measures))
                    return (
                        True,
                        f"Recent notification for measure(s) '{shared}' already spoken {int(elapsed)}s ago",
                    )

        return False, ""

    async def process_event(self, event: CarEvent):
        self.logger.info(
            f">>> received {event.event_name} event with value: {event.event_value}"
        )
        self.logger.info(">>> generating...")
        if event.user_input:
            self.agent.cancel_voice_response()
            response_task = asyncio.create_task(self.agent.stream_user_response(event))
            self.agent._voice_response_task = response_task

            def _voice_response_done(task: asyncio.Task) -> None:
                if self.agent._voice_response_task is task:
                    self.agent._voice_response_task = None
                if task.cancelled():
                    self.logger.info(">>> voice response interrupted")
                    return
                error = task.exception()
                if error is not None:
                    self.logger.error(">>> voice response failed", error=str(error))

            response_task.add_done_callback(_voice_response_done)
            return

        llm_start = time.time()
        measures: set[str] = set(event.event_value or [])
        try:
            skill = SkillType(event.skill)
            skill_instructions = self.agent.skill_manager.get_skill(skill)
        except ValueError:
            skill = None
            skill_instructions = "No additional skill-specific instructions."

        actions = self.actions_by_skill.get(skill, (ActionType.NONE,))
        available_actions = "".join(f"{action.name}\n- {action.value}\n\n" for action in actions)

        inputs = {
            "skill_instructions": skill_instructions,
            "changed_knowledge": event.event_value,
            "context": event.context,
            "available_actions": available_actions,
            "user_input": event.user_input,
        }
        result = await self.crew.kickoff_async(
            inputs=inputs,
        )
        llm_elapsed = time.time() - llm_start
        self.logger.info(f">>> LLM generation took {llm_elapsed:.2f}s")
        response = result.raw if hasattr(result, "raw") else str(result)
        self.logger.info(f">>> {response}")

        decision: NotificationDecision = result.pydantic
        if decision.notify and decision.spoken_message:
            if not event.user_input and self._is_non_actionable_decision(decision):
                self.logger.info(">>> [suppressed non-actionable model decision]")
                return

            # Check deterministic cooldown / deduplication unless user explicitly asked a question
            if not event.user_input:
                suppressed, reason = self._is_duplicate_or_cooling_down(
                    message=decision.spoken_message,
                    urgency=decision.urgency,
                    measures=measures,
                )
            else:
                suppressed, reason = False, ""

            if suppressed:
                self.logger.info(f">>> [suppressed duplicate] {reason}")
            else:
                self.logger.info(f">>> [speak] {decision.spoken_message}")
                if self.agent.on_response is not None:
                    response = decision.spoken_message + "\n"
                    response += (
                        f"Action: {decision.action.action_type}, Parameters: {decision.action.parameters}\n"
                        if decision.action
                        else "no action required\n"
                    )
                    self.agent.on_response(response)
                if self.agent.tts_manager is not None:
                    tts_start = time.time()
                    await self.agent.tts_manager.speak(decision.spoken_message)
                    self.logger.info(
                        f">>> TTS synthesis+playback took {time.time() - tts_start:.2f}s"
                    )
                self.recent_notifications.append(
                    {
                        "urgency": decision.urgency.value,
                        "message": decision.spoken_message,
                        "skill": event.skill,
                        "event": event.event_name,
                        "measures": sorted(measures),
                        "timestamp": time.time(),
                    }
                )
                if len(self.recent_notifications) > 5:
                    self.recent_notifications.pop(0)
                if decision.action:
                    self.logger.info(f">>> [action] {decision.action.action_type}")
                    # handle the action accordingly
                    if decision.action.action_type is not ActionType.NONE:
                        self.logger.info(
                            f">>> [action] executing {decision.action.action_type} with parameters: {decision.action.parameters}"
                        )

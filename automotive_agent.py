import asyncio
import time
from typing import Any, Callable

import structlog
from skill_manager import SkillManager, SkillType
from events import CarEvent
from tools.speed_limit_tool import SpeedLimitTool
from voice.tts_manager import TTSManager
from crewai import Agent, Task, Crew
from crewai.process import Process
from crewai import LLM, Agent, Task, Crew

logger = structlog.get_logger()

from enum import Enum
from pydantic import BaseModel, Field


class Urgency(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActionType(str, Enum):
    NONE = "no action required"
    SUGGEST = "suggest the driver"
    WARN = "warn the driver"
    INCREASE_TEMPERATURE = "increase internal temperature"
    DECREASE_TEMPERATURE = "decrease internal temperature"
    LOCK_DOORS = "lock doors"
    UNLOCK_DOORS = "unlock doors"
    START_NAVIGATION = "start navigation"
    FIND_REST_AREA = "find rest area"
    ENABLE_RECIRCULATION = "enable air recirculation"

class Action(BaseModel):

    action_type: ActionType

    parameters: dict = {}


class NotificationDecision(BaseModel):
    notify: bool = Field(
        description="Whether the user should be notified now."
    )
    urgency: Urgency
    reason: str = Field(
        description="Short internal justification. Never spoken to the user."
    )
    spoken_message: str | None = Field(
        default=None,
        description=(
            "Exact message to speak directly to the user. "
            "Must be null when notify is false."
        ),
    )
    action: Action | None = None

class AutomotiveAgent:
    def __init__(self, llm: LLM, tts_manager: TTSManager | None = None):
        self.llm = llm
        self.is_active = True
        self.event_queue: asyncio.Queue[CarEvent] = asyncio.Queue()
        self.is_listening = False
        self.tts_manager = tts_manager

        self.skill_manager = SkillManager()

        self.on_response: Callable[[str], None] | None = None

        self.recent_notifications: list[dict[str, Any]] = []

        self.actions_by_skill = {
            SkillType.DRIVER_HEALTH: (
                ActionType.NONE,
                ActionType.SUGGEST,
                ActionType.WARN,
                ActionType.INCREASE_TEMPERATURE,
                ActionType.DECREASE_TEMPERATURE,
                ActionType.FIND_REST_AREA,
                ActionType.ENABLE_RECIRCULATION,
            ),
            SkillType.NAVIGATION_AND_COACHING: (
                ActionType.NONE,
                ActionType.SUGGEST,
                ActionType.WARN,
                ActionType.LOCK_DOORS,
                ActionType.UNLOCK_DOORS,
                ActionType.START_NAVIGATION,
                ActionType.FIND_REST_AREA,
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
            tools=[]#[SpeedLimitTool()],
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
                - If user_input is not empty, answer that request. Set notify=true.
                - Otherwise, remain silent unless the data shows a concrete current
                    safety risk, abnormal condition, or useful action needed now.
                - Do not speak about normal, stable, low, unchanged, or merely changing
                    values. Do not summarize telemetry or say that no action is needed.
                - A trend, fluctuation, or sensor value is not a risk without an explicit
                    threshold or safety consequence. Do not infer one.
                - For proactive alerts, require all of the following: a concrete current
                    hazard or action, direct evidence for it in changed_knowledge or
                    context, and a timely benefit from interrupting the driver. Otherwise
                    choose Silent.
                - In particular, choose Silent for a speed increase or fluctuation without
                    a known speed-limit exceedance or explicitly unsafe manoeuvre; an
                    indicator being off without an explicitly reported turn or lane change;
                    or a vague lane assessment not tied to a reported dangerous deviation
                    or upcoming exit. Never turn missing information into a warning.
                - You may use lookup_speed_limit only when explicit latitude and longitude
                    are available in the supplied context or user request. Treat unavailable
                    tool results as no speed-limit information; do not estimate a limit.

                Output:
                - Silent: notify=false, urgency=none, spoken_message=null, action=null.
                - Alert: notify=true, urgency is not none, reason names the concrete
                    risk, and spoken_message is one short natural sentence.
                - Never expose internal reasoning or implementation details.
                """,
            expected_output=(
                "A structured notification decision containing notify, urgency, "
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

    async def run(self):
        """Main processing loop"""
        while self.is_active:
            if not self.is_listening:
                await asyncio.sleep(0.1)
                continue
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                continue

            try:
                await self._process_event(event)
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")

    def _is_duplicate_or_cooling_down(
        self,
        message: str,
        urgency: Urgency,
        measures: set[str],
        cooldown_seconds: float = 60.0,
    ) -> tuple[bool, str]:
        """Deterministic safety filter: prevent repeating duplicate or same-measure notifications within cooldown."""
        now = time.time()
        urgency_ranks = {
            Urgency.NONE: 0,
            Urgency.LOW: 1,
            Urgency.MEDIUM: 2,
            Urgency.HIGH: 3,
            Urgency.CRITICAL: 4,
        }
        current_rank = urgency_ranks.get(urgency, 1)

        norm_message = message.strip().lower()

        for prev in reversed(self.recent_notifications):
            prev_time = prev.get("timestamp", 0)
            elapsed = now - prev_time

            if elapsed > cooldown_seconds:
                continue

            prev_norm_msg = prev.get("message", "").strip().lower()
            prev_urgency_str = prev.get("urgency", "none").lower()
            prev_rank = urgency_ranks.get(Urgency(prev_urgency_str), 1)

            # 1. Exact or near-exact identical message within cooldown
            if norm_message == prev_norm_msg or norm_message in prev_norm_msg or prev_norm_msg in norm_message:
                if current_rank <= prev_rank:
                    return True, f"Identical/similar message spoken {int(elapsed)}s ago with same or higher urgency"

            # 2. Same underlying measure(s) within cooldown unless urgency escalated to critical/high.
            # Skill is too coarse (e.g. "vehicle" covers doors, temperature, engine, ...): only
            # suppress if the notifications actually concern at least one common measure.
            prev_measures = set(prev.get("measures") or [])
            if measures and prev_measures and measures & prev_measures and elapsed < (cooldown_seconds / 2):
                if current_rank <= prev_rank and current_rank < urgency_ranks[Urgency.HIGH]:
                    shared = ", ".join(sorted(measures & prev_measures))
                    return True, f"Recent notification for measure(s) '{shared}' already spoken {int(elapsed)}s ago"

        return False, ""

    @staticmethod
    def _is_non_actionable_decision(decision: NotificationDecision) -> bool:
        """Reject model notifications that explicitly describe no actionable risk."""
        if decision.urgency is Urgency.NONE:
            return True
        if decision.action is not None and decision.action.action_type is ActionType.NONE:
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

    async def _process_event(self, event: CarEvent):
        logger.info(f">>> received {event.event_name} event with value: {event.event_value}")
        logger.info(">>> generating...")
        llm_start = time.time()
        measures: set[str] = set(event.event_value or [])
        try:
            skill = SkillType(event.skill)
            skill_instructions = self.skill_manager.get_skill(skill)
        except ValueError:
            skill = None
            skill_instructions = "No additional skill-specific instructions."

        available_actions = self._format_available_actions(skill)

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
        logger.info(f">>> LLM generation took {llm_elapsed:.2f}s")
        response = result.raw if hasattr(result, 'raw') else str(result)
        logger.info(f">>> {response}")

        decision: NotificationDecision = result.pydantic
        if decision.notify and decision.spoken_message:
            if not event.user_input and self._is_non_actionable_decision(decision):
                logger.info(">>> [suppressed non-actionable model decision]")
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
                logger.info(f">>> [suppressed duplicate] {reason}")
            else:
                logger.info(f">>> [speak] {decision.spoken_message}")
                if self.tts_manager is not None:
                    tts_start = time.time()
                    await self.tts_manager.speak(decision.spoken_message)
                    logger.info(f">>> TTS synthesis+playback took {time.time() - tts_start:.2f}s")
                self.recent_notifications.append({
                    "urgency": decision.urgency.value,
                    "message": decision.spoken_message,
                    "skill": event.skill,
                    "event": event.event_name,
                    "measures": sorted(measures),
                    "timestamp": time.time(),
                })
                if len(self.recent_notifications) > 5:
                    self.recent_notifications.pop(0)
                if decision.action:
                    logger.info(f">>> [action] {decision.action.action_type}")
                    # handle the action accordingly
                    if decision.action.action_type is not ActionType.NONE:
                        logger.info(f">>> [action] executing {decision.action.action_type} with parameters: {decision.action.parameters}")

                if self.on_response is not None:
                    response = decision.spoken_message + "\n"
                    response += f"Action: {decision.action.action_type}, Parameters: {decision.action.parameters}\n" if decision.action else "no action required\n"
                    self.on_response(response)

    def _format_available_actions(self, skill: SkillType | None) -> str:
        actions = self.actions_by_skill.get(skill, (ActionType.NONE,))
        return "".join(f"{action.name}\n- {action.value}\n\n" for action in actions)
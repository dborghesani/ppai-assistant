import asyncio
import time
from typing import Any, Callable

import structlog
from skill_manager import SkillManager, SkillType
from events import CarEvent
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
    NONE = "none"
    SUGGEST = "suggest"

    INCREASE_TEMPERATURE = "increase_temperature"
    DECREASE_TEMPERATURE = "decrease_temperature"

    LOCK_DOORS = "lock_doors"

    START_NAVIGATION = "start_navigation"

    FIND_REST_AREA = "find_rest_area"

    ENABLE_RECIRCULATION = "enable_recirculation"

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

        self.available_actions = """
            NONE
            - do nothing

            INCREASE_TEMPERATURE
            - increase cabin temperature

            DECREASE_TEMPERATURE
            - decrease cabin temperature

            LOCK_DOORS
            - lock vehicle doors

            START_NAVIGATION
            - start route guidance

            FIND_REST_AREA
            - search for a nearby rest area
            """

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
            tools=[],
        )

        task = Task(
            description="""
                You are the notification gate of an in-vehicle assistant.

                Active skill:
                {skill}

                Skill instructions:
                {skill_instructions}

                Triggering event:
                {event}

                Values that triggered this evaluation:
                {value}

                Current extracted knowledge:
                {context}

                Direct user request, if any:
                {user_input}

                Recent notifications:
                {recent_notifications}

                Available actions:
                {available_actions}

                ABSOLUTE RULE, no exceptions: if "Direct user request" above is not empty, the
                user directly spoke or typed to you. You MUST set notify=true and produce a
                spoken_message that responds to their request. This overrides "Default to
                silence", the deduplication rules, and every other instruction below. Never
                output notify=false when user_input is non-empty.

                The rest of this section (default to silence, deduplication) applies only
                when user_input is empty, i.e. this evaluation was triggered by a knowledge/
                vehicle-event update and not by something the user said.

                Default to silence. A knowledge update requests an evaluation, not a spoken
                response. Normal, safe, stable, informational or non-actionable conditions
                must produce notify=false.

                Set notify=true when at least one condition applies:
                - user_input is non-empty (always, per the absolute rule above);
                - there is an immediate or developing safety risk;
                - a vehicle condition requires timely attention;
                - the occupant can take a useful, time-sensitive action now.

                Do not notify merely because a value changed, to confirm normal operation,
                to report that no problem was detected, or to say that no action is required.
                Routine controls such as turn signals, lights and normal engine state should
                remain silent unless their duration or surrounding context indicates a concrete
                risk. Stable or optimal conditions should remain silent.

                Deduplication & Repetition Rules (only when user_input is empty):
                - ALWAYS examine "Recent notifications" FIRST before deciding to notify.
                - If the condition, topic, or advice has already been communicated in "Recent notifications", you MUST set notify=false.
                - Repeating warnings or advice creates dangerous driver distraction and alert fatigue.
                - Minor value variations or ongoing persistent states of an already-notified condition MUST NOT trigger a new notification.
                - These deduplication rules NEVER apply when user_input is non-empty: a direct
                  user request always gets notify=true and a real answer, even if the topic or
                  wording resembles a recent notification.
                - In all other duplicate cases (user_input empty), output notify=false, urgency=none, spoken_message=null, reason="Condition already notified recently."

                Before setting notify=true, identify the concrete risk, required attention or
                useful immediate action that justifies interrupting the occupant. If none exists,
                set notify=false.

                When notifying, set notify to true, choose an urgency, give a brief internal
                reason, and produce one concise spoken message. When not notifying, set notify
                to false, urgency to none, and spoken_message to null.

                Consistency requirements:
                - notify=false requires urgency=none and spoken_message=null;
                - notify=true requires urgency other than none and a concrete justification;
                - action_type=none with notify=true is valid only for a warning that still
                    requires the occupant's attention;
                - never notify with a message meaning that everything is normal, no urgent
                    condition exists, or no action is recommended.

                Negative example 1 (Routine update): a turn signal active for ten seconds while all other
                conditions are normal results in notify=false, urgency=none and no spoken
                message.
                Negative example 2 (Duplicate notification):
                Recent notifications:
                - [Urgency: LOW] Topic/Skill: vehicle (knowledge_updated) | Spoken: "The cabin temperature is twenty-two degrees Celsius."
                Current context: internal_temperature is 22.3°C.
                Output: notify=false, urgency=none, spoken_message=null, reason="Temperature condition was already communicated recently and has not escalated in urgency."
                Positive example: a turn signal that remains active for several
                minutes after a completed turn may justify a low-urgency reminder.
                Positive example (user_input overrides everything): user_input is "What's the
                cabin temperature?" and Recent notifications already contains an identical
                temperature notification from moments ago. Output: notify=true, urgency=low,
                spoken_message="It's twenty-two degrees Celsius in the cabin.", reason="Direct
                user request, always answered regardless of recent notifications."

                The spoken message is sent directly to text-to-speech. Address the occupant
                naturally; do not include analysis, scores, variable names, or implementation
                details.

                Requests provided as user_input are high-priority and MUST always receive a
                notify=true response with a real, on-topic spoken_message — never silence, and
                never a generic acknowledgement that avoids answering the request.
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

    def _format_recent_notifications(self) -> str:
        if not self.recent_notifications:
            return "None (no recent notifications spoken yet)"
        lines = []
        now = time.time()
        for n in self.recent_notifications:
            urgency = n.get("urgency", "unknown").upper()
            skill = n.get("skill", "general")
            event = n.get("event", "update")
            measures = n.get("measures") or []
            measures_str = f", measure(s): {', '.join(measures)}" if measures else ""
            msg = n.get("message", "")
            elapsed = int(now - n.get("timestamp", now))
            lines.append(
                f"- [{elapsed}s ago | Urgency: {urgency}] Topic/Skill: {skill} ({event}{measures_str}) | Spoken: \"{msg}\""
            )
        return "\n".join(lines)

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

    async def _process_event(self, event: CarEvent):
        logger.info(f">>> received {event.event_name} event with value: {event.event_value}")
        logger.info(">>> generating...")
        llm_start = time.time()
        measures: set[str] = (
            set(event.event_value.keys()) if isinstance(event.event_value, dict) else set()
        )
        try:
            skill_instructions = self.skill_manager.get_skill(SkillType(event.skill))
        except ValueError:
            skill_instructions = "No additional skill-specific instructions."

        inputs = {
            "skill": event.skill,
            "skill_instructions": skill_instructions,
            "event": event.event_name,
            "context": event.context,
            "value": event.event_value,
            "recent_notifications": self._format_recent_notifications(),
            "available_actions": self.available_actions,
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
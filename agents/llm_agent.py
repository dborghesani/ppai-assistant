from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
import structlog
from agents.agents_dataclasses import (
    ActionType,
    InterventionType,
    SkillType,
    SuggestionType,
    ToneType,
    UrgencyType,
)
from data.events import CarEvent
from crewai import LLM, Agent, Crew, Process, Task

if TYPE_CHECKING:
    from agents.automotive_agent import AutomotiveAgent


def _enum_options(enum_type: type) -> str:
    return "\n".join(
        f"- {member.value}: {member.description}" for member in enum_type
    )


class NotificationDecision(BaseModel):
    urgency: UrgencyType = Field(description="Urgency of the assistant's intervention.")
    tone: ToneType = Field(
        description="Tone to use when speaking; this describes delivery, not the driver's emotional state."
    )
    intervention_type: InterventionType = Field(
        description="Whether to stay silent, make a suggestion, or perform an action."
    )
    skill: SkillType = Field(description="Skill area relevant to the intervention.")
    action: ActionType = Field(description="Supported action to perform, or none.")
    suggestion_type: SuggestionType = Field(
        description="Suggestion category to communicate, or none."
    )
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

    @property
    def notify(self) -> bool:
        """Derived, not model-provided: avoids the model setting notify inconsistently with urgency."""
        return self.urgency is not UrgencyType.NONE

class LLMAgent:
    def __init__(self, agent: "AutomotiveAgent"):
        self.logger = structlog.get_logger()
        self.agent = agent
        self.recent_notifications: list[dict] = []
        self.minimum_urgency = UrgencyType.MEDIUM

        opt = agent.opt
        self.llm = LLM(
            model=opt.ollama_model,
            base_url=f"http://{opt.ollama_host}:{opt.ollama_port}",
            timeout=opt.ollama_timeout,
            max_tokens=opt.max_tokens,
        )
        self.decision_options = {
            "urgency_options": _enum_options(UrgencyType),
            "tone_options": _enum_options(ToneType),
            "intervention_options": _enum_options(InterventionType),
            "skill_options": _enum_options(SkillType),
            "action_options": _enum_options(ActionType),
            "suggestion_options": _enum_options(SuggestionType),
        }
        self.meeting_decision_guidance = (
            f"urgency={UrgencyType.HIGH.value}, tone={ToneType.DISCREET.value}, "
            f"intervention_type={InterventionType.ACT.value}, "
            f"skill={SkillType.CONVERSATION.value}, "
            f"action={ActionType.ASK_ATTEND_MEETING.value}, "
            f"suggestion_type={SuggestionType.NONE.value}"
        )
        self.silent_decision_guidance = (
            f"urgency={UrgencyType.NONE.value}, tone={ToneType.DISCREET.value}, "
            f"intervention_type={InterventionType.NONE.value}, "
            f"skill={SkillType.NONE.value}, action={ActionType.NONE.value}, "
            f"suggestion_type={SuggestionType.NONE.value}"
        )

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
            llm=self.llm,
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
                urgency_options={urgency_options}
                tone_options={tone_options}
                intervention_options={intervention_options}
                skill_options={skill_options}
                action_options={action_options}
                suggestion_options={suggestion_options}
                meeting_decision_guidance={meeting_decision_guidance}
                silent_decision_guidance={silent_decision_guidance}

                Rules:
                - Exception that always applies first: if changed_knowledge or context
                    indicates the driver is running late for a meeting, this is always a
                    useful action needed now, regardless of the general silence rule below.
                    Never use the silent decision for this case. Use
                    meeting_decision_guidance, and
                    spoken_message asking, in these words or close to them, "Would you like
                    me to attend the meeting on your behalf and brief you once it is over?".
                    Do not claim the meeting has already been joined; only ask.
                - Choose every structured field only from its matching options list above;
                    do not invent or rename enum values.
                - Choose tone using tone_options and make spoken_message consistent with its
                    description. Tone describes delivery, not the driver's emotional state.
                - Keep intervention_type, skill, action and suggestion_type mutually
                    consistent. Use the corresponding none option when a field does not apply.
                - Outside of that exception, remain silent (urgency=none) unless the data
                    shows a concrete current safety risk, abnormal condition, or useful
                    action needed now.
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
                    the Silent case: use silent_decision_guidance and set spoken_message=null
                    (the JSON null, not a sentence). Never say things like "no action is needed",
                    "notifications remain silent" or describe the telemetry out loud
                    instead of setting the fields.
                - Never close spoken_message with a generic filler question such as "is
                    there anything else I can help with?" or "let me know if you need
                    anything else". Only ask a question when the driver's answer is
                    actually needed to proceed (like ask_attend_meeting above).

                Output:
                - Silent: follow silent_decision_guidance and set spoken_message=null.
                    Never pair a silent urgency with spoken text, or an active urgency with
                    a null spoken_message.
                - Alert: urgency is not none, reason names the concrete risk, and
                    spoken_message is one short natural sentence. Populate all six
                    structured decision fields, even when some are none.
                - Never expose internal reasoning or implementation details.
                """,
            expected_output=(
                "A structured notification decision containing urgency, tone, "
                "intervention_type, skill, action, suggestion_type, reason and spoken_message."
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
        if (
            decision.urgency is UrgencyType.NONE
            or decision.intervention_type is InterventionType.NONE
            or (
                decision.action is ActionType.NONE
                and decision.suggestion_type is SuggestionType.NONE
            )
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
        current_rank = UrgencyType(urgency).rank

        norm_message = message.strip().lower()

        for prev in reversed(self.recent_notifications):
            prev_time = prev.get("timestamp", 0)
            elapsed = now - prev_time

            if elapsed > cooldown_seconds:
                continue

            prev_norm_msg = prev.get("message", "").strip().lower()
            prev_urgency_str = prev.get("urgency", "none").lower()
            prev_rank = UrgencyType(prev_urgency_str).rank

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
                    and current_rank < UrgencyType(UrgencyType.HIGH).rank
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
            if await self.agent.maybe_handle_meeting_confirmation(event.user_input):
                return
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

        inputs = {
            "skill_instructions": skill_instructions,
            "changed_knowledge": event.event_value,
            "context": event.context,
            "user_input": event.user_input,
            **self.decision_options,
            "meeting_decision_guidance": self.meeting_decision_guidance,
            "silent_decision_guidance": self.silent_decision_guidance,
        }
        # This decision is a classification (urgency/action), not creative writing: lower the
        # shared LLM's temperature just for this call, then restore it for other uses.
        default_temperature = self.llm.temperature
        self.llm.temperature = 0.1
        try:
            result = await self.crew.kickoff_async(inputs=inputs)
        finally:
            self.llm.temperature = default_temperature
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
                await self.agent.speak(decision.spoken_message, tone=decision.tone)
                self.recent_notifications.append(
                    {
                        "urgency": decision.urgency.value,
                        "tone": decision.tone.value,
                        "intervention_type": decision.intervention_type.value,
                        "skill": decision.skill.value,
                        "action": decision.action.value,
                        "suggestion_type": decision.suggestion_type.value,
                        "message": decision.spoken_message,
                        "event": event.event_name,
                        "measures": sorted(measures),
                        "timestamp": time.time(),
                    }
                )
                if len(self.recent_notifications) > 5:
                    self.recent_notifications.pop(0)
                if decision.action is not ActionType.NONE:
                    self.logger.info(f">>> [action] {decision.action}")
                    self.agent.action_manager.handle_decision(
                        decision.action, {}
                    )

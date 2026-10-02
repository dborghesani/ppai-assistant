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
from data.events import CarEvent, EventName
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
            "Must be null when urgency is none or when permission is requested "
            "through the visual assistant status."
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
        self.duplicate_suppression_enabled = False

        opt = agent.opt
        self.llm = LLM(
            model=opt.ollama_model,
            base_url=f"http://{opt.ollama_host}:{opt.ollama_port}",
            timeout=opt.ollama_timeout,
            max_tokens=opt.max_tokens,
            additional_params={"reasoning_effort": "none"},
        )
        self.decision_options = {
            "urgency_options": _enum_options(UrgencyType),
            "tone_options": _enum_options(ToneType),
            "intervention_options": _enum_options(InterventionType),
            "skill_options": _enum_options(SkillType),
            "action_options": _enum_options(ActionType),
            "suggestion_options": _enum_options(SuggestionType),
        }
        self.direct_vehicle_actions = tuple(
            action
            for action in ActionType
            if action
            not in {
                ActionType.NONE,
                ActionType.FIND_REST_AREA,
                ActionType.ASK_ATTEND_MEETING,
                ActionType.ANNOUNCE_INCOMING_MESSAGE,
                ActionType.ASK_PERMISSION_TO_TALK,
                ActionType.READ_PENDING_MESSAGES,
            }
        )
        self.direct_action_options = "\n".join(
            f"- {action.value}: {action.description}"
            for action in self.direct_vehicle_actions
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
                vehicle_action_guidance={vehicle_action_guidance}
                Changed vehicle facts (trigger for this decision):
                {changed_facts}
                Supporting current vehicle context (not a new trigger):
                {context}
                Event name: {event_name}
                Event details: {event_details}
                user_input={user_input}
                urgency_options={urgency_options}
                tone_options={tone_options}
                intervention_options={intervention_options}
                skill_options={skill_options}
                action_options={action_options}
                suggestion_options={suggestion_options}
                silent_decision_guidance={silent_decision_guidance}

                Rules:
                - Incoming message workflow has priority over the general silent/proactive
                    notification rules below. An incoming_message_received event is an
                    explicit message delivery event even when user_input is empty; never
                    choose the silent decision only because the driver did not request it.
                - For incoming_message_received, event details include the sender's
                    raw message text. Classify its tone and urgency from that content. Tone
                    is serious for road breakdowns, safety problems, alarming news, or
                    practical requests for urgent help; enthusiastic only for genuinely
                    positive or playful content, never bad news or requests for help. Urgency
                    is high for explicit ASAP/immediate requests, a stranded person needing
                    help, or a near deadline; critical only for immediate risk of serious
                    injury. Do not infer one value from the other. For example, "Giulia is
                    down on Highway 402 with a mechanical issue and needs help ASAP" is
                    serious/high; a ten-minute booking deadline can be calm/high; harmless
                    gossip is enthusiastic/none.
                - For incoming_message_received, apply this strict priority. First, privacy
                    mode ON or unknown is an absolute veto: choose ask_permission_to_talk
                    regardless of attention, fatigue, or traffic. Never let favorable driver
                    conditions override privacy mode. Second, if privacy mode is OFF but
                    fatigue is high/very high, attention is low, or traffic is explicitly
                    heavy, choose ask_permission_to_talk. In either ask case keep the
                    message's classified tone and urgency; use intervention_type=act,
                    skill=conversation, action=ask_permission_to_talk, suggestion_type=none
                    and spoken_message=null. This action is conveyed
                    only by the brief visual ASK_PERMISSION_TO_TALK animation. Do not speak,
                    name the sender, or reveal any part of the message.
                - Choose announce_incoming_message only when privacy mode is explicitly OFF,
                    fatigue is not high/very high, attention is not low, and traffic is not
                    explicitly heavy. Keep the message's classified tone and urgency; use
                    intervention_type=act, skill=conversation, suggestion_type=none. Relay the message in third
                    person and attribute it to its sender (for example, "Luca says that…").
                    Retell every distinct detail, name, timeframe, qualifier, and question;
                    keep roughly the same level of detail as the original instead of giving
                    only its gist. Do not answer the message as if it were addressed to you,
                    or speak in the sender's voice. Use up to three natural sentences if
                    needed. Do not ask permission in this case.
                - Contrastive examples: privacy ON + high attention + no fatigue =>
                    ask_permission_to_talk; privacy OFF + high attention + no fatigue +
                    light traffic => announce_incoming_message. Use only explicit current
                    facts for these checks.
                - For pending_messages_reminder with pending_count greater than zero, choose
                    ask_permission_to_talk, urgency=low, intervention_type=act,
                    skill=conversation, action=ask_permission_to_talk,
                    suggestion_type=none, spoken_message=null. Do not announce that messages
                    are pending or ask verbally; use only the brief visual status animation.
                    If pending_count is zero, remain silent.
                - For a knowledge_updated event, an empty user_input is expected: there was
                    no direct utterance. Do not call it unintelligible; decide from the
                    changed vehicle facts and skill instructions.
                - Base the intervention on changed vehicle facts. Supporting context may
                    clarify or constrain that response, but must not introduce an unrelated
                    action on its own.
                - Facts prefixed "Changed just now:" are the trigger for this event; evaluate
                    them first. Use remaining vehicle facts only as supporting context and do
                    not choose an unrelated action because of them.
                - Classify a direct user request before context-driven interventions; unrelated
                    facts must not replace the user's explicit intent.
                - Choose ask_attend_meeting only when the current vehicle facts explicitly state
                    that the driver is late for an upcoming meeting. Ask permission; never claim
                    the meeting has already been joined.
                - Choose every structured field only from its matching options list above;
                    do not invent or rename enum values.
                - Choose tone using tone_options and make spoken_message consistent with its
                    description. Tone describes delivery, not the driver's emotional state.
                - Keep intervention_type, skill, action and suggestion_type mutually
                    consistent. Use the corresponding none option when a field does not apply.
                - When skill_instructions prescribe a supported safety action for an
                    explicitly reported current condition, include that action in the
                    structured decision as well as the spoken warning.
                - For a direct request that matches vehicle_action_guidance, this is the
                    action decision, not a conversational fallback: choose the matching
                    ActionType, intervention_type=act, urgency=low, suggestion_type=none,
                    and its driving or wellbeing skill. Use the catalog's defaults and
                    limits; don't ask for optional parameters. Use action=none only for an
                    unsupported operation or a capability question that doesn't request it.
                    Acknowledge briefly without claiming physical vehicle confirmation.
                - Outside of that exception, remain silent (urgency=none) unless the data
                    shows a concrete current safety risk, abnormal condition, or useful
                    action needed now.
                - Do not speak about normal, stable, low, unchanged, or merely changing
                    values. Do not summarize telemetry or say that no action is needed.
                - A trend, fluctuation, or sensor value is not a risk without an explicit
                    threshold or safety consequence. Do not infer one.
                - For proactive alerts, require all of the following: a concrete current
                    hazard or action, direct evidence in the changed vehicle facts, and a
                    timely benefit from interrupting the driver. Otherwise
                    urgency=none.
                - Never turn missing information into a warning: a trend or state change is
                    only a risk when skill_instructions or the changed vehicle facts tie it to an explicit
                    threshold or consequence.
                - You may use lookup_speed_limit only when explicit latitude and longitude
                    are available in the changed vehicle facts or user request. Treat unavailable
                    tool results as no speed-limit information; do not estimate a limit.
                - skill_instructions may list example values to illustrate a rule; they are
                    not the current data. Only state a specific value (e.g. a turn-signal
                    direction, a light or door state) if that exact value appears verbatim
                    in the changed vehicle facts. Never substitute an example value from
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
                    spoken_message is one short natural sentence. For
                    incoming_message_received, follow its higher-priority relay rule instead
                    and use up to three sentences to preserve the full message. Populate all
                    six structured decision fields, even when some are none.
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
        direct_action_task = Task(
            description="""
                Classify this direct user request as one simulated vehicle action or no action.

                User request: {user_input}
                Vehicle action reference: {vehicle_action_guidance}
                Supported vehicle actions: {action_options}
                Pending private messages: {pending_message_count}
                Skills: {skill_options}
                Tones: {tone_options}

                Rules:
                - Consider only the user's requested operation and the action reference.
                    Do not use telemetry, meeting guidance, or proactive notification rules.
                - If pending private messages are available, choose read_pending_messages
                    only when the user explicitly asks to hear/read them or asks what the
                    pending message says (for example "what do you want to tell me?",
                    "tell me the pending messages", "che cosa mi vuoi dire?", or
                    "dimmi pure i messaggi in sospeso"). A bare "yes", "okay" or similar
                    reply is not explicit permission. The action reads the existing queue;
                    do not repeat its message content in spoken_message.
                - If the user requests a supported operation, select its exact action,
                    intervention_type=act, urgency=low, suggestion_type=none, and its
                    driving or wellbeing skill. This includes polite forms such as "Can you..."
                    and "Could you...".
                - Use the catalog's default step and limits. Do not ask for optional values.
                - If the request is not a supported operation, use action=none,
                    intervention_type=none, urgency=none, skill=none,
                    suggestion_type=none, and spoken_message=null.
                - Keep the spoken acknowledgement short and do not claim physical vehicle
                    confirmation; this app updates simulated UI state only.
                - Fill every structured field using only the allowed enums.
                """,
            expected_output=(
                "A structured direct-action decision with urgency, tone, intervention_type, "
                "skill, action, suggestion_type, reason and spoken_message."
            ),
            output_pydantic=NotificationDecision,
            agent=agent,
        )
        self.direct_action_crew = Crew(
            agents=[agent],
            tasks=[direct_action_task],
            verbose=False,
            tracing=False,
            process=Process.sequential,
            memory=None,
        )

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

    def message_simulation_stopped(self, event: CarEvent) -> bool:
        if (
            event.event_name != "incoming_message_received"
            or not isinstance(event.event_value, dict)
            or event.event_value.get("sender") != "Luca"
        ):
            return False
        simulator = getattr(self.agent, "message_simulator", None)
        return simulator is not None and not getattr(simulator, "active", True)

    async def process_event(self, event: CarEvent):
        if (
            event.event_name is EventName.PENDING_MESSAGES_REMINDER
            and getattr(self.agent, "pending_message_count", 0) == 0
        ):
            return
        self.logger.info(
            f">>> received {event.event_name} event with value: {event.event_value}"
        )
        self.logger.info(">>> generating...")
        if event.user_input:
            if await self.agent.maybe_handle_meeting_confirmation(event.user_input):
                return
            self.agent.cancel_voice_response()
            await self._process_direct_user_input(event)
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
            "event_name": event.event_name,
            "event_details": (
                str(event.event_value)
                if event.event_name
                in {
                    EventName.INCOMING_MESSAGE_RECEIVED,
                    EventName.PENDING_MESSAGES_REMINDER,
                }
                else "No additional event-specific details."
            ),
            "skill_instructions": skill_instructions,
            "vehicle_action_guidance": getattr(
                self.agent.skill_manager, "vehicle_action_guidance", ""
            ),
            "changed_facts": "\n".join(
                f"- {fact.removeprefix('Changed just now: ')}"
                for fact in event.context
                if fact.startswith("Changed just now:")
            ) or "No specific vehicle fact changed.",
            "context": "\n".join(
                f"- {fact}"
                for fact in event.context
                if not fact.startswith("Changed just now:")
            ) or "No additional vehicle context is available.",
            "user_input": event.user_input,
            **self.decision_options,
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
        self.agent.log_llm_usage(
            "notification decision", getattr(result, "token_usage", None)
        )
        if self.message_simulation_stopped(event):
            return
        llm_elapsed = time.time() - llm_start
        self.logger.info(f">>> LLM generation took {llm_elapsed:.2f}s")
        response = result.raw if hasattr(result, "raw") else str(result)
        self.logger.info(f">>> {response}")

        decision: NotificationDecision = result.pydantic
        if event.event_name is EventName.INCOMING_MESSAGE_RECEIVED:
            self.agent.remember_pending_message_classification(
                event.event_value, decision.tone, decision.urgency
            )
            self.agent.notify_incoming_message_classification(
                decision.tone, decision.urgency
            )
        elif event.event_name is EventName.PENDING_MESSAGES_REMINDER:
            classification = self.agent.most_urgent_pending_message_classification()
            if classification is not None:
                decision.tone, decision.urgency = classification
            self.agent.notify_incoming_message_classification(
                decision.tone, decision.urgency
            )

        if decision.action in {
            ActionType.ASK_PERMISSION_TO_TALK,
            ActionType.ANNOUNCE_INCOMING_MESSAGE,
        }:
            if (
                decision.action is ActionType.ANNOUNCE_INCOMING_MESSAGE
                and decision.spoken_message
            ):
                await self.agent.speak(decision.spoken_message, tone=decision.tone)
            self.agent.action_manager.handle_decision(decision.action, {})
            return

        if decision.notify and decision.spoken_message:
            if getattr(self, "duplicate_suppression_enabled", False):
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

    async def _process_direct_user_input(self, event: CarEvent) -> None:
        pending_message_count = getattr(self.agent, "pending_message_count", 0)
        action_options = self.direct_action_options
        if pending_message_count:
            action_options += (
                f"\n- {ActionType.READ_PENDING_MESSAGES.value}: read queued private "
                "messages after the driver's explicit request"
            )
        inputs = {
            "user_input": event.user_input,
            "vehicle_action_guidance": getattr(
                self.agent.skill_manager, "vehicle_action_guidance", ""
            ),
            "action_options": action_options,
            "pending_message_count": pending_message_count,
            "skill_options": self.decision_options["skill_options"],
            "tone_options": self.decision_options["tone_options"],
        }
        default_temperature = self.llm.temperature
        self.llm.temperature = 0.1
        try:
            result = await self.direct_action_crew.kickoff_async(inputs=inputs)
        finally:
            self.llm.temperature = default_temperature
        self.agent.log_llm_usage(
            "direct action classification", getattr(result, "token_usage", None)
        )

        decision: NotificationDecision = result.pydantic
        if (
            decision.action is ActionType.READ_PENDING_MESSAGES
            and pending_message_count > 0
        ):
            self.agent.action_manager.handle_decision(decision.action, {})
            return

        action_selected = (
            decision.intervention_type is InterventionType.ACT
            and decision.action in self.direct_vehicle_actions
        )
        if action_selected:
            self.logger.info(
                "Direct user request classified as vehicle action",
                user_input=event.user_input,
                action=decision.action.value,
                skill=decision.skill.value,
                reason=decision.reason,
            )
            self.agent.action_manager.handle_decision(decision.action, {})
            await self.agent.speak(
                decision.spoken_message or "I received the action request.",
                tone=decision.tone,
            )
            return

        fallback_reason = (
            f"action={decision.action.value} is not a supported simulated vehicle action"
            if decision.action is not ActionType.NONE
            else "model selected action=none"
        )
        self.logger.warning(
            "Direct user request was not classified as a vehicle action; falling back to conversation",
            user_input=event.user_input,
            fallback_reason=fallback_reason,
            intervention_type=decision.intervention_type.value,
            action=decision.action.value,
            skill=decision.skill.value,
            suggestion_type=decision.suggestion_type.value,
            decision_reason=decision.reason,
            spoken_message=decision.spoken_message,
        )
        self._start_conversation_response(event)

    def _start_conversation_response(self, event: CarEvent) -> None:
        response_task = asyncio.create_task(self.agent.stream_user_response(event))
        self.agent.voice_response_task = response_task

        def _voice_response_done(task: asyncio.Task) -> None:
            if self.agent.voice_response_task is task:
                self.agent.voice_response_task = None
            if task.cancelled():
                self.logger.info(">>> voice response interrupted")
                return
            error = task.exception()
            if error is not None:
                self.logger.error(">>> voice response failed", error=str(error))

        response_task.add_done_callback(_voice_response_done)

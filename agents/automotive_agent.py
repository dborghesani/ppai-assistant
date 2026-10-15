import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import structlog
from agents.tasks.decision import DecisionInput, NotificationDecision
from agents.tasks import comfort, driving, meeting, messages, mood_music, vehicle_command, wellbeing
from agents.backends.crewai_backend import CrewAIBackend
from agents.backends.decision_backend import DecisionBackend, KevBackend, LayaBackend
from agents.backends.ollama_backend import OllamaBackend
from agents.backends.llm_logging import usage_log_fields
from agents.tasks.conversation import ConversationDelivery, ConversationInput, ConversationTask, WaitMessageTask
from agents.tasks import mood_music
from agents.tasks.pre_classify import PreClassification, PreClassificationUpdate, PreClassificationInput, RequestPreClassifier, ResponseSource
from agents.tasks.routing import RoutingInput, TaskType, TelemetryRouter
from agents.tasks.vehicle_manual import response_context as manual_response_context
from config import ConfigAssistant
from crewai import LLM
from data.events import CarEvent, EventName
from openai import AsyncOpenAI
from managers.vehicle_manual_manager import VehicleManualManager
from sim.friend_message_agent import FriendMessageAgent

if TYPE_CHECKING:
    from voice.tts_manager import TTSManager

logger = structlog.get_logger()

DECISION_TASKS = {task.task_type: task for task in (
    wellbeing.TASK, driving.TASK, comfort.TASK, mood_music.TASK, messages.TASK, meeting.TASK,
)}

from data.agents_dataclasses import ActionType, AssistantStatus, InterventionType, SkillType, ToneType, UrgencyType
from managers.action_manager import ActionManager
from data.assistant_dataclasses import DriverPreferences


@dataclass
class DirectInputPreparation:
    pre_classification: PreClassification
    classifier_task: asyncio.Task[NotificationDecision] | None
    wait_task: asyncio.Task[None] | None
    response_task: asyncio.Task[None] | None
    response_gate: asyncio.Event
    wait_message_claimed: asyncio.Event = field(default_factory=asyncio.Event)


class AutomotiveAgent:
    def __init__(
        self, opt: ConfigAssistant, tts_manager: "TTSManager | None" = None
    ):
        if opt.music_cooldown_seconds < 0:
            raise ValueError("Music cooldown must be non-negative")
        if opt.wait_message_delay_seconds < 0:
            raise ValueError("wait_message_delay_seconds must be non-negative")
        if opt.routing_backend not in {"crew", "llm", "decision"}:
            raise ValueError("routing_backend must be 'crew', 'llm' or 'decision'")
        if opt.pre_classify_backend not in {"crew", "llm", "decision"}:
            raise ValueError("pre_classify_backend must be 'crew', 'llm' or 'decision'")
        if opt.decision_model not in {"laya", "kev"}:
            raise ValueError("decision_model must be 'laya' or 'kev'")
        self.logger = logger
        self.opt = opt
        self.is_active = True
        self.is_processing_event = False
        self.event_queue: asyncio.Queue[CarEvent] = asyncio.Queue()
        self.is_listening = False
        self.tts_manager = tts_manager
        self.voice_response_task: asyncio.Task | None = None
        self.recent_notifications: list[dict] = []
        self.minimum_urgency = UrgencyType.MEDIUM
        self.duplicate_suppression_enabled = False
        self.conversation_history: list[dict[str, str]] = []
        self.manual_manager = VehicleManualManager(opt) if opt.rag_enabled else None
        self.response_context_providers: dict[ResponseSource, Callable[[str, list[dict[str, str]]], Awaitable[tuple[str, str]]]] = {
            ResponseSource.VEHICLE_MANUAL: self.manual_response_context,
        }
        self.ollama_backend = OllamaBackend(opt, self.log_llm_usage)
        self.voice_llm = self.ollama_backend.client
        self.knowledge_facts_provider: Callable[[], dict[str, str]] = lambda: {}

        self.action_manager = ActionManager(self)
        self._message_simulator = None
        self.on_music_update: Callable[[dict[str, Any]], None] | None = None
        self.music_preferences_provider: Callable[[], str] = lambda: ""
        self.driver_preferences_provider: Callable[[], dict[str, Any]] = lambda: {"driver_name": DriverPreferences().driver_name}

        self.crewai_backend = CrewAIBackend(opt, self.log_llm_usage)
        self.tasks = {
            TaskType.WELLBEING: wellbeing.WellbeingTask(opt, self.log_llm_usage, backend=self.crewai_backend),
            TaskType.DRIVING: driving.DrivingTask(opt, self.log_llm_usage, backend=self.crewai_backend),
            TaskType.COMFORT: comfort.ComfortTask(opt, self.log_llm_usage, backend=self.crewai_backend),
            TaskType.MOOD_MUSIC: mood_music.MoodMusicTask(
                opt, self.log_llm_usage, backend=self.crewai_backend,
                respond=self._music_respond,
                speak=self.speak,
                notify_music_update=self._notify_music_update,
                preferences_provider=lambda: self.music_preferences_provider(),
                emotion_provider=lambda: self.emotion_state(),
                set_status=self.set_assistant_status,
            ),
            TaskType.MESSAGES: messages.MessagesTask(
                opt, self.log_llm_usage, backend=self.crewai_backend,
                agent=self,
            ),
            TaskType.MEETING: meeting.MeetingTask(
                opt, self.log_llm_usage, backend=self.crewai_backend,
                speak=self.speak,
                on_confirm=self.action_manager.confirm_attend_meeting,
                on_decline=self.action_manager.decline_attend_meeting,
            ),
        }
        self.command_task = vehicle_command.VehicleCommandTask(
            opt, self.log_llm_usage, backend=self.crewai_backend,
            registered_actions=self.action_manager.registered_actions,
        )
        def build_decision_backend(model: str | None) -> DecisionBackend:
            if opt.decision_model == "kev":
                return KevBackend(opt, self.log_llm_usage, model=model)
            return LayaBackend(opt.laya_model, self.log_llm_usage)

        if opt.routing_backend == "decision":
            router_backend = build_decision_backend(opt.routing_model)
        elif opt.routing_backend == "crew":
            router_backend = CrewAIBackend(opt, self.log_llm_usage, model=opt.routing_model)
        else:
            router_backend = self.ollama_backend
        self.telemetry_router = TelemetryRouter(opt, self.log_llm_usage, backend=router_backend)
        if opt.pre_classify_backend == "decision":
            if (
                opt.routing_backend == "decision"
                and opt.pre_classify_model == opt.routing_model
            ):
                pre_classifier_backend = router_backend
            else:
                pre_classifier_backend = build_decision_backend(opt.pre_classify_model)
        else:
            # "llm" keeps streaming Ollama (the wait-message path needs incremental
            # output); "crew" selects the Crew engine with streaming disabled.
            pre_classifier_backend = self.ollama_backend
        self.request_pre_classifier = RequestPreClassifier(
            opt, self.log_llm_usage,
            backend=pre_classifier_backend,
        )

        self.on_response: Callable[[str], None] | None = None
        self.on_response_update: Callable[[str], None] | None = None
        self.on_speaking_tone_changed: Callable[[str | None], None] | None = None
        self.on_incoming_message_classified: Callable[[ToneType, UrgencyType], None] | None = None
        self.on_assistant_status_changed: Callable[[AssistantStatus], None] | None = None
        self._assistant_status = AssistantStatus.IDLE

    def _notify_music_update(self, value: dict[str, Any]) -> None:
        if self.on_music_update is not None:
            self.on_music_update(value)

    @property
    def music_task(self) -> mood_music.MoodMusicTask:
        return self.tasks[TaskType.MOOD_MUSIC]

    @property
    def pending_music_selection(self) -> dict[str, Any] | None:
        return self.music_task.pending_selection

    @property
    def messages_task(self):
        return self.tasks[TaskType.MESSAGES]

    async def _music_respond(self, user_input: str, reply_context: tuple[str, str]) -> str:
        event = CarEvent(
            skill=SkillType.CONVERSATION, event_name=EventName.USER_INPUT,
            event_value=user_input, context=self.knowledge_context(), user_input=user_input,
        )
        return await self.stream_user_response(event, response_context=reply_context)

    def notify_incoming_message_classification(
        self, tone: ToneType, urgency: UrgencyType
    ) -> None:
        if self.on_incoming_message_classified is not None:
            self.on_incoming_message_classified(tone, urgency)

    @property
    def llm(self) -> LLM:
        """The shared CrewAI LLM built and owned by LLMBackend."""
        return self.crewai_backend.llm

    @property
    def assistant_status(self) -> AssistantStatus:
        return self._assistant_status

    @assistant_status.setter
    def assistant_status(self, status: AssistantStatus) -> None:
        self.set_assistant_status(status)

    def set_assistant_status(self, status: AssistantStatus) -> None:
        if self.assistant_status is status:
            if (
                status is AssistantStatus.ASK_PERMISSION_TO_TALK
                and self.on_assistant_status_changed is not None
            ):
                self.on_assistant_status_changed(status)
            return
        self._assistant_status = status
        if self.on_assistant_status_changed is not None:
            self.on_assistant_status_changed(status)

    @property
    def pending_message_count(self) -> int:
        return self.messages_task.pending_count

    @property
    def driver_name(self) -> str:
        provider = getattr(self, "driver_preferences_provider", lambda: {"driver_name": DriverPreferences().driver_name})
        name = provider().get("driver_name")
        return name.strip() if isinstance(name, str) and name.strip() else "David"

    @property
    def message_simulator(self):
        """The external friend-message simulation source (kept out of the task)."""
        if self._message_simulator is None:
            self._message_simulator = FriendMessageAgent(
                llm_client=self.voice_llm,
                model=self.opt.ollama_model,
                on_message=self.messages_task.receive,
                on_usage=self.log_llm_usage,
            )
        return self._message_simulator

    def set_knowledge_facts_provider(
        self, provider: Callable[[], dict[str, str]]
    ) -> None:
        self.knowledge_facts_provider = provider

    def knowledge_context(self) -> list[str]:
        """Fact strings without their keys, for prompts that only need the text."""
        return list(self.knowledge_facts_provider().values())

    def emotion_state(self) -> dict[str, str]:
        """DriverEmotionState measures from the knowledge base, keyed by measure name."""
        emotion_state: dict[str, str] = {}
        for key, fact in self.knowledge_facts_provider().items():
            name, separator, measure = key.partition(".")
            if separator and name == "DriverEmotionState":
                emotion_state[measure] = fact
        return emotion_state

    def attention_is_low(self) -> bool:
        """True while the knowledge-base attention fact reads low/very low
        (e.g. "Attention level is low.") or no attention is detected."""
        fact = self.knowledge_facts_provider().get("DriverPhysicalState.attention_level", "")
        lowered = fact.lower()
        return "low" in lowered or "no attention" in lowered

    def start_message_simulation(self) -> bool:
        return self.message_simulator.start()

    async def stop_message_simulation(self) -> bool:
        stopped = await self.message_simulator.stop()
        return stopped

    def cancel_message_tasks(self) -> None:
        if self._message_simulator is not None:
            self._message_simulator.cancel()
        self.messages_task.close()

    def log_llm_usage(self, call_name: str, usage: Any | None) -> None:
        self.logger.debug("[LLM] Token usage", llm_call=call_name,
                          **usage_log_fields(usage, self.opt.context_window_size))



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

            if event.user_input:
                event = self.take_latest_user_event(event)

            self.is_processing_event = True
            try:
                await self.process_event(event)
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
            finally:
                self.is_processing_event = False

    def take_latest_user_event(self, event: CarEvent) -> CarEvent:
        """Drop stale queued user turns while preserving system events."""
        latest_event = event
        deferred_events: list[CarEvent] = []
        dropped_count = 0

        while True:
            try:
                queued_event = self.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if queued_event.user_input:
                latest_event = queued_event
                dropped_count += 1
            else:
                deferred_events.append(queued_event)

        for deferred_event in deferred_events:
            self.event_queue.put_nowait(deferred_event)

        if dropped_count:
            logger.info("Dropped stale queued user requests", count=dropped_count)
        return latest_event

    async def prepare_user_response(
        self, event: CarEvent, source: ResponseSource,
        *, response_gate: asyncio.Event | None = None,
        wait_message_claimed: asyncio.Event | None = None,
    ) -> None:
        provider = self.response_context_providers.get(source, self.empty_response_context)
        if source is ResponseSource.VEHICLE_MANUAL:
            context_task = asyncio.create_task(provider(event.user_input, self.conversation_history))
            wait_task = asyncio.create_task(self.generate_wait_message(
                event, source, context_task, response_gate=response_gate,
                wait_message_claimed=wait_message_claimed,
            ))
            try:
                response_context = await context_task
                await wait_task
            finally:
                for task in (context_task, wait_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(context_task, wait_task, return_exceptions=True)
        else:
            response_context = await provider(event.user_input, self.conversation_history)
        if source is not ResponseSource.VEHICLE_STATE:
            event = CarEvent(
                event.skill, event.event_name, event.event_value, [], event.user_input,
            )
        if response_gate is None:
            await self.stream_user_response(event, response_context=response_context)
        else:
            await self.stream_user_response(
                event, response_context=response_context, response_gate=response_gate,
            )

    @staticmethod
    async def empty_response_context(question: str, history: list[dict[str, str]]) -> tuple[str, str]:
        return "", ""

    async def manual_response_context(self, question: str, history: list[dict[str, str]]) -> tuple[str, str]:
        return await manual_response_context(self.manual_manager, question, history)

    async def stream_user_response(
        self, event: CarEvent, *, response_context: tuple[str, str] = ("", ""),
        response_gate: asyncio.Event | None = None,
    ) -> str:
        async def deliver_speech(sentence: str, on_word=None) -> None:
            if self.tts_manager is not None:
                await self.tts_manager.speak(sentence, on_word)

        def interrupt_speech() -> None:
            if self.tts_manager is not None:
                # Natural interruption: let the word currently being spoken
                # finish before halting playback.
                self.tts_manager.stop(finish_word=True)

        task = ConversationTask(self.opt, self.log_llm_usage, client=self.voice_llm,
                    backend=getattr(self, "ollama_backend", None))
        result = await task.execute(
            ConversationInput(event.user_input, list(self.conversation_history), list(event.context),
                              response_context, response_gate),
            ConversationDelivery(
                deliver_speech, self.on_response_update or (lambda _: None),
                self.on_speaking_tone_changed or (lambda _: None), self.set_assistant_status,
                attention_low=self.attention_is_low,
                interrupt_speech=interrupt_speech,
            ),
        )
        self.conversation_history.extend([
            {"role": "user", "content": event.user_input},
            # Keep history and chat aligned with what was actually spoken;
            # un-spoken text dropped during attention pauses is not persisted.
            {"role": "assistant", "content": result.spoken or result.text},
        ])
        self.conversation_history = self.conversation_history[-8:]
        if self.on_response is not None:
            self.on_response((result.spoken or result.text) + "\n")
        return result.text


    def cancel_voice_response(self) -> None:
        task = self.voice_response_task
        if task is not None and not task.done():
            task.cancel()

    def clear_conversation(self) -> None:
        self.cancel_voice_response()
        self.tasks[TaskType.MEETING].reset()
        self.music_task.reset()
        self.conversation_history.clear()

    def set_minimum_urgency(self, urgency: str) -> None:
        self.minimum_urgency = UrgencyType(urgency)

    def is_duplicate_or_cooling_down(self, message: str, urgency: UrgencyType, measures: set[str],
                                         cooldown_seconds: float = 60.0) -> tuple[bool, str]:
        now = time.time()
        current_rank = UrgencyType(urgency).rank
        norm_message = message.strip().lower()
        for previous in reversed(self.recent_notifications):
            elapsed = now - previous.get("timestamp", 0)
            if elapsed > cooldown_seconds:
                continue
            previous_message = previous.get("message", "").strip().lower()
            previous_rank = UrgencyType(previous.get("urgency", "none").lower()).rank
            if (norm_message == previous_message or norm_message in previous_message or previous_message in norm_message) and current_rank <= previous_rank:
                return True, f"Identical/similar message spoken {int(elapsed)}s ago with same or higher urgency"
            previous_measures = set(previous.get("measures") or [])
            if (measures and measures & previous_measures and elapsed < cooldown_seconds / 2
                    and current_rank <= previous_rank and current_rank < UrgencyType.HIGH.rank):
                shared = ", ".join(sorted(measures & previous_measures))
                return True, f"Recent notification for measure(s) '{shared}' already spoken {int(elapsed)}s ago"
        return False, ""
    
    async def process_event(self, event: CarEvent) -> None:
        self.logger.info("Received event", event_name=event.event_name, event_value=event.event_value)
        if event.user_input:
            await self.process_direct_user_input(event)
        else:
            await self.process_car_event(event)

    async def process_car_event(self, event: CarEvent) -> None:
        facts = dict(event.facts or getattr(self, "knowledge_facts_provider", lambda: {})())
        changes = dict(event.event_value) if isinstance(event.event_value, dict) else {}
        changed_text = [
            fact.removeprefix("Changed just now: ") for fact in event.context
            if fact.startswith("Changed just now:")
        ]
        changed_facts = {
            key: facts.get(key, changed_text[index] if index < len(changed_text) else f"{key}: {value}")
            for index, (key, value) in enumerate(changes.items())
        }
        facts.update(changed_facts)
        route = await self.telemetry_router.execute(RoutingInput(changed_facts))
        task_types = route.tasks
        self.logger.info("[LLM] Event dispatched", tasks=[task.value for task in task_types])
        decisions = []
        for task_type in task_types:
            selected_facts = DECISION_TASKS[task_type].select_facts(facts)
            selected_changes = {key: value for key, value in selected_facts.items() if key in changes}
            supporting = [value for key, value in selected_facts.items() if key not in changes]
            if not event.facts and not getattr(self, "knowledge_facts_provider", lambda: {})():
                supporting = [fact for fact in event.context if not fact.startswith("Changed just now:")]
            decision = await self.tasks[task_type].execute(DecisionInput(**{
                "event_name": event.event_name, "driver_name": self.driver_name,
                "pending_message_count": self.pending_message_count,
                "event_details": (json.dumps(
                    {"changes": changes, "newest_message": self.messages_task.newest_message},
                    default=str) if task_type is TaskType.MESSAGES
                    else "No additional event-specific details."),
                "changed_facts": "\n".join(f"- {fact}" for fact in selected_changes.values())
                    or "No specific vehicle fact changed.",
                "context": "\n".join(f"- {fact}" for fact in supporting)
                    or "No additional vehicle context is available.",
            }))
            decisions.append((task_type, decision))
            break   # Only process the first task decision for now
        # Determine if there is an acute hazard based on the urgency of the decisions
        acute_hazard = any(
            task_type in {TaskType.DRIVING, TaskType.WELLBEING} and decision.urgency.rank >= UrgencyType.HIGH.rank
            for task_type, decision in decisions
        )
        executed_actions = set()
        for task_type, decision in sorted(decisions, key=lambda item: item[1].urgency.rank, reverse=True):
            # Skip processing if there is an acute hazard and the task is not critical
            if task_type is TaskType.MOOD_MUSIC and (
                acute_hazard or self.music_task.active
            ):
                continue
            # Skip processing if the action has already been executed
            if decision.action is not ActionType.NONE and decision.action in executed_actions:
                continue
            self.logger.info(
                "[LLM] Task decision", task=task_type.value, action=decision.action.value,
                urgency=decision.urgency.value, reason=decision.reason,
            )
            await self.handle_notification_decision(event, decision, task_type=task_type)
            executed_actions.add(decision.action)

    async def process_direct_user_input(self, event: CarEvent) -> None:
        if self.music_task.awaiting_reply:
            await self.music_task.handle_user_input(event.user_input, self.conversation_history)
        elif self.tasks[TaskType.MEETING].awaiting_reply:
            await self.tasks[TaskType.MEETING].handle_user_input(event.user_input, self.conversation_history)
        else:
            await self.handle_direct_user_input(event)

    async def handle_direct_user_input(self, event: CarEvent) -> None:
        pending_count = self.pending_message_count
        preparation = await self.preclassify_direct_input(event, pending_count)
        if not preparation.pre_classification.requires_action_classification:
            self.logger.info(
                "Direct input routed to conversation without full action classification",
                response_source=preparation.pre_classification.response_source.value,
            )
            self.release_prepared_response(preparation)
            return
        else:
            decision = await self.resolve_action_decision(preparation)
            await self.dispatch_direct_decision(event, decision, preparation, pending_count=pending_count)

    def release_prepared_response(self, preparation: DirectInputPreparation) -> None:
        """Allow the prepared response to reach the UI and TTS."""
        assert preparation.response_task is not None
        preparation.response_gate.set()

    async def preclassify_direct_input(self, event: CarEvent, pending_count: int) -> DirectInputPreparation:
        """Preclassify the request and start its speculative response and optional action tasks."""
        command_task = self.command_task
        options = command_task.direct_action_options + "\n" + self.action_manager.registered_action_options
        if pending_count:
            options += f"\n- {ActionType.READ_PENDING_MESSAGES.value}: read queued private messages after the driver's explicit request"
        classifier_inputs = {
            "user_input": event.user_input,
            "vehicle_action_guidance": command_task.vehicle_action_guidance,
            "action_options": options, "pending_message_count": pending_count,
            "conversation_history": json.dumps(self.conversation_history[-8:]),
            "registered_action_guidance": self.action_manager.registered_action_guidance,
            "skill_options": command_task.decision_options["skill_options"],
            "tone_options": command_task.decision_options["tone_options"],
        }

        classifier_task: asyncio.Task | None = None
        wait_task: asyncio.Task | None = None
        response_task: asyncio.Task | None = None
        response_gate = asyncio.Event()
        wait_message_claimed = asyncio.Event()
        wait_chunks: asyncio.Queue[str | None] = asyncio.Queue()
        wait_chunk_count = 0
        requires_action_classification = False

        def start_classifier(requires_action_classification: bool) -> None:
            nonlocal classifier_task
            if requires_action_classification and classifier_task is None:
                self.logger.info(
                    "[LLM] Starting direct action classification",
                    model=self.opt.ollama_model,
                )
                classifier_task = asyncio.create_task(command_task.execute(classifier_inputs))

        def start_wait_message_stream() -> None:
            nonlocal wait_task
            if wait_task is None:
                wait_task = asyncio.create_task(self.speak_wait_message_chunks(wait_chunks))

        def start_response(source: ResponseSource) -> None:
            nonlocal response_task
            if response_task is not None:
                return
            self.logger.info(
                "[LLM] Starting speculative response preparation",
                response_source=source.value,
            )
            response_task = asyncio.create_task(self.prepare_user_response(
                event, source, response_gate=response_gate,
                wait_message_claimed=wait_message_claimed,
            ))
            self.track_voice_response_task(response_task)

        def handle_preclassification_update(update: PreClassificationUpdate) -> None:
            nonlocal requires_action_classification, wait_chunk_count
            if update.requires_action_classification is not None:
                requires_action_classification = update.requires_action_classification
                start_classifier(requires_action_classification)
            if update.response_source is not None:
                start_response(update.response_source)
            if update.wait_message_chunk and update.wait_message_chunk.strip() and requires_action_classification:
                wait_message_claimed.set()
                start_wait_message_stream()
                wait_chunks.put_nowait(update.wait_message_chunk)
                if update.wait_message_chunk.strip():
                    wait_chunk_count += 1

        pre_classification = await self.request_pre_classifier.execute(
            PreClassificationInput(event.user_input, self.conversation_history, pending_count),
            on_update=handle_preclassification_update,
        )

        if not pre_classification.requires_action_classification:
            if response_task is None:
                start_response(pre_classification.response_source)
            return DirectInputPreparation(
                pre_classification, None, None, response_task, response_gate, wait_message_claimed,
            )

        requires_action_classification = True
        start_classifier(requires_action_classification)
        if classifier_task is None:
            classifier_task = asyncio.create_task(command_task.execute(classifier_inputs))
        if response_task is None:
            start_response(pre_classification.response_source)
        if wait_chunk_count == 0 and not pre_classification.wait_message.strip():
            wait_task = asyncio.create_task(self.generate_wait_message(
                event, pre_classification.response_source, classifier_task,
                wait_message_claimed=wait_message_claimed, action_pending=True,
            ))
        else:
            wait_message_claimed.set()
            start_wait_message_stream()
            if wait_chunk_count == 0:
                wait_chunks.put_nowait(pre_classification.wait_message.strip())
            wait_chunks.put_nowait(None)
        return DirectInputPreparation(
            pre_classification, classifier_task, wait_task, response_task, response_gate, wait_message_claimed,
        )

    async def resolve_action_decision(self, preparation: DirectInputPreparation) -> NotificationDecision:
        """Wait for classification and acknowledgement, cleaning up on failure or interruption."""
        assert preparation.classifier_task is not None
        assert preparation.response_task is not None
        self.logger.info(
            "Starting action classification for user input",
            response_source=preparation.pre_classification.response_source.value,
        )
        try:
            decision = await preparation.classifier_task
            if preparation.wait_task is not None:
                # Finish any acknowledgement before releasing the final reply.
                await preparation.wait_task
        except (Exception, asyncio.CancelledError):
            await self.cancel_direct_input(preparation)
            raise
        return decision

    async def cancel_direct_input(self, preparation: DirectInputPreparation) -> None:
        pending_tasks = [task for task in (
            preparation.classifier_task, preparation.wait_task, preparation.response_task,
        ) if task is not None]
        for task in pending_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    async def cancel_prepared_response(self, preparation: DirectInputPreparation) -> None:
        if preparation.response_task is not None:
            preparation.response_task.cancel()
            await asyncio.gather(preparation.response_task, return_exceptions=True)

    async def dispatch_direct_decision(
        self, event: CarEvent, decision: NotificationDecision, preparation: DirectInputPreparation,
        *, pending_count: int,
    ) -> None:
        """Reuse the prepared reply, switch its information source, or execute the selected action."""
        backend = self.command_task
        pre_classification = preparation.pre_classification

        # The full classifier can revise the provisional source: discard the
        # prepared reply rather than answer from the wrong context.
        if (
            decision.action is ActionType.NONE
            and decision.response_source is not pre_classification.response_source
        ):
            await self.cancel_prepared_response(preparation)
            self.start_conversation_response(
                event, source=decision.response_source, wait_message_claimed=preparation.wait_message_claimed,
            )
            return

        if decision.action is ActionType.NONE:
            self.release_prepared_response(preparation)
            return

        # A real action replaces the speculative informational reply.
        await self.cancel_prepared_response(preparation)
        if decision.action in self.action_manager.registered_actions and decision.intervention_type is InterventionType.ACT:
            self.start_conversation_response(event, action=decision.action)
            return
        if decision.action is ActionType.READ_PENDING_MESSAGES and pending_count > 0:
            self.action_manager.handle_decision(decision.action, {})
            return
        if decision.action is ActionType.PLAY_MUSIC:
            self.start_conversation_response(event, action=decision.action)
            return
        if decision.intervention_type is InterventionType.ACT and decision.action in backend.direct_vehicle_actions:
            self.logger.info("[LLM] Direct user request classified as vehicle action", user_input=event.user_input,
                            action=decision.action.value, skill=decision.skill.value, reason=decision.reason)
            self.action_manager.handle_decision(decision.action, {})
            if decision.spoken_message:
                await self.speak(decision.spoken_message, tone=decision.tone)
            return
        fallback_reason = (f"action={decision.action.value} is not a supported simulated vehicle action"
                        if decision.action is not ActionType.NONE else "model selected action=none")
        self.logger.warning(
            "[LLM] Direct user request was not classified as a vehicle action; falling back to conversation",
            user_input=event.user_input, fallback_reason=fallback_reason,
            intervention_type=decision.intervention_type.value, action=decision.action.value,
            skill=decision.skill.value, suggestion_type=decision.suggestion_type.value,
            decision_reason=decision.reason, spoken_message=decision.spoken_message,
        )
        self.start_conversation_response(
            event, source=decision.response_source, wait_message_claimed=preparation.wait_message_claimed,
        )

    async def handle_notification_decision(
        self, event: CarEvent, decision: NotificationDecision,
        *, task_type: TaskType | None = None,
    ) -> None:
        if task_type is TaskType.MESSAGES and decision.action is not ActionType.ASK_PERMISSION_TO_TALK:
            await self.messages_task.handle_decision(
                event, decision,
                on_classified=self.notify_incoming_message_classification,
            )
            return
        if task_type is TaskType.MESSAGES:
            self.notify_incoming_message_classification(decision.tone, decision.urgency)
            self.action_manager.handle_decision(decision.action, {})
            return
        if decision.action is ActionType.POSTPONE_NOTIFICATION_DELIVERY:
            self.action_manager.handle_decision(decision.action, {})
            return
        if decision.action is ActionType.READ_PENDING_MESSAGES:
            parameters = {}
            if event.event_name is EventName.KNOWLEDGE_UPDATED:
                parameters = {
                    "acknowledgement": decision.spoken_message or "",
                }
            self.action_manager.handle_decision(decision.action, parameters)
            return
        if decision.action is ActionType.ASK_PERMISSION_TO_TALK:
            self.action_manager.handle_decision(decision.action, {})
            return
        if decision.action in ActionType.ASK_ATTEND_MEETING:
            meeting_task = self.tasks[TaskType.MEETING]
            if meeting_task.awaiting_reply:
                self.logger.info("[LLM] Meeting permission already pending; skipping repeated ask")
                return
            if decision.spoken_message:
                await self.speak(decision.spoken_message, tone=decision.tone)
            await meeting_task.on_asked()
            return
        if decision.action in ActionType.PROPOSE_MUSIC:
            if not self.music_task.should_offer(
                self.music_task.emotion, self.music_task.preferences,
                self.opt.music_cooldown_seconds,
            ):
                return
            await self.speak(await self.music_task.compose_suggestion(self.driver_name), tone=decision.tone)
            self.music_task.start_offering()
            return
        if not decision.notify:
            return
        measures = set(event.event_value or [])
        if decision.spoken_message and self.duplicate_suppression_enabled:
            suppressed, reason = self.is_duplicate_or_cooling_down(decision.spoken_message, decision.urgency, measures)
            if suppressed:
                self.logger.info(f"[suppressed duplicate] {reason}")
                return
        if decision.spoken_message:
            await self.speak(decision.spoken_message, tone=decision.tone)
            self.recent_notifications.append({
                "urgency": decision.urgency.value, "tone": decision.tone.value,
                "intervention_type": decision.intervention_type.value, "skill": decision.skill.value,
                "action": decision.action.value, "suggestion_type": decision.suggestion_type.value,
                "message": decision.spoken_message, "event": event.event_name,
                "measures": sorted(measures), "timestamp": time.time(),
            })
            self.recent_notifications = self.recent_notifications[-5:]
        if decision.action is not ActionType.NONE:
            self.action_manager.handle_decision(decision.action, {})

    def start_conversation_response(self, event: CarEvent, action: ActionType | None = None,
                                     source: ResponseSource = ResponseSource.GENERAL,
                                     *, wait_message_claimed: asyncio.Event | None = None) -> None:
        response = (self.action_manager.execute_registered_action(action, event.user_input) if action is not None
                    else self.prepare_user_response(event, source, wait_message_claimed=wait_message_claimed))
        response_task = asyncio.create_task(response)
        self.track_voice_response_task(response_task)

    def track_voice_response_task(self, response_task: asyncio.Task) -> None:
        self.voice_response_task = response_task

        def response_done(task: asyncio.Task) -> None:
            if self.voice_response_task is task:
                self.voice_response_task = None
            if task.cancelled():
                self.logger.info("[LLM] Voice response interrupted")
            elif task.exception() is not None:
                self.logger.error("[LLM] Voice response failed", error=str(task.exception()))

        response_task.add_done_callback(response_done)

    async def generate_wait_message(
        self, event: CarEvent, source: ResponseSource, ready_task: asyncio.Task,
        *, response_gate: asyncio.Event | None = None,
        wait_message_claimed: asyncio.Event | None = None,
        action_pending: bool = False,
    ) -> None:
        generation_task: asyncio.Task | None = None
        gate_task: asyncio.Task | None = None
        claimed_here = False
        delivery_started = False
        try:
            if response_gate is not None and not response_gate.is_set():
                gate_task = asyncio.create_task(response_gate.wait())
                done, _ = await asyncio.wait({ready_task, gate_task}, return_when=asyncio.FIRST_COMPLETED)
                if ready_task in done:
                    self.logger.debug("Wait message skipped", reason="work finished before route confirmation")
                    return
            done, _ = await asyncio.wait({ready_task}, timeout=self.opt.wait_message_delay_seconds)
            if done:
                self.logger.debug("Wait message skipped", reason="work finished before delay")
                return
            if wait_message_claimed is not None:
                if wait_message_claimed.is_set():
                    self.logger.debug("Wait message skipped", reason="turn already covered")
                    return
                wait_message_claimed.set()
                claimed_here = True
            purpose = (
                "Reviewing the requested action; do not promise execution or success, and do not name an"
                " information source such as the vehicle manual or live vehicle state."
                if action_pending
                else (
                    "Looking up the vehicle manual; do not claim to have found information yet."
                    if source is ResponseSource.VEHICLE_MANUAL
                    else "Reviewing the requested action; do not promise execution or success."
                )
            )
            task = WaitMessageTask(self.opt, self.ollama_backend)
            self.logger.debug("Starting wait message generation", response_source=source.value)
            generation_task = asyncio.create_task(task.execute(
                event.user_input, list(self.conversation_history), purpose=purpose,
            ))
            done, _ = await asyncio.wait({ready_task, generation_task}, return_when=asyncio.FIRST_COMPLETED)
            if ready_task in done:
                self.logger.debug("Wait message cancelled", reason="work finished during generation")
                return
            message = await generation_task
            if ready_task.done():
                return
            chunks: asyncio.Queue[str | None] = asyncio.Queue()
            chunks.put_nowait(message)
            chunks.put_nowait(None)
            delivery_started = True
            await self.speak_wait_message_chunks(chunks)
        except Exception as error:
            self.logger.warning("Wait message generation failed; continuing without it", error=str(error))
        finally:
            tasks = [task for task in (generation_task, gate_task) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if claimed_here and not delivery_started and wait_message_claimed is not None:
                wait_message_claimed.clear()

    async def speak_wait_message_chunks(
        self, chunks: asyncio.Queue[str | None]
    ) -> None:
        displayed_message = ""
        while (chunk := await chunks.get()) is not None:
            chunk = chunk.strip()
            if not chunk:
                continue
            displayed_message = f"{displayed_message} {chunk}".strip()
            on_response_update = getattr(self, "on_response_update", None)
            if on_response_update is not None:
                on_response_update(displayed_message)
            tts_manager = getattr(self, "tts_manager", None)
            if tts_manager is not None:
                await tts_manager.speak(chunk)
        if displayed_message:
            on_response = getattr(self, "on_response", None)
            if on_response is not None:
                on_response(displayed_message + "\n")

    async def speak(
        self, message: str, ui_suffix: str = "", tone: ToneType | None = None
    ) -> None:
        """Single place that keeps chat history, UI text and TTS in sync for a one-shot message."""
        self.conversation_history.append({"role": "assistant", "content": message})
        self.conversation_history = self.conversation_history[-8:]
        self.set_assistant_status(AssistantStatus.TALKING)
        if self.on_speaking_tone_changed is not None:
            self.on_speaking_tone_changed(tone.value if tone is not None else None)
        try:
            if self.on_response is not None:
                self.on_response(f"{message}\n{ui_suffix}" if ui_suffix else message + "\n")
            if self.tts_manager is not None:
                await self.tts_manager.speak(message)
        finally:
            if self.on_speaking_tone_changed is not None:
                self.on_speaking_tone_changed(None)
            self.set_assistant_status(AssistantStatus.IDLE)

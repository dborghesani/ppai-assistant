import json
import time
from typing import Any

from crewai import Agent, Crew, Process, Task
from agents.backends.crewai_backend import CrewAIBackend
from agents.tasks.decision import DECISION_PROMPT, DecisionInput, DecisionTask, NotificationDecision, enum_options
from agents.tasks.routing import TaskType
from data.agents_dataclasses import ActionType, InterventionType, SkillType, SuggestionType, ToneType, UrgencyType


TASK = DecisionTask(
    task_type=TaskType.MOOD_MUSIC,
    skill=SkillType.WELLBEING,
    context_fields=("DriverEmotionState", "DriverPreferences.preferred_music"),
    actions=(ActionType.NONE, ActionType.PROPOSE_MUSIC),
    suggestions=(SuggestionType.NONE, SuggestionType.REGULATE_EMOTIONAL_STATE),
    instructions="""
Respond only to explicitly CHANGED driver emotions, never fatigue, attention,
activities or driving tension. Neutral and intensity none remain silent.
Non-neutral low/medium emotion => brief warm acknowledgement, no music proposal.
Anger/fear/disgust at medium intensity may warrant a gentle calming suggestion;
happy/sad/surprise alone do not justify a driving-safety warning.
Non-neutral high/very high/extreme emotion => propose_music, intervention_type=act,
urgency=low, suggestion_type=none. Acknowledge the emotion and ask permission in
spoken_message. Never claim selection or playback has started. Prefer a
safety-relevant emotion when multiple emotions are reported. Playback/selection
and confirmation belong to the music manager, not this decision.
""",
)


class MoodMusicTask:
    spec = TASK

    def __init__(self, opt, on_usage, *, backend=None):
        self.backend = backend if backend is not None else CrewAIBackend(opt, on_usage)
        agent = Agent(role="Mood and Music Specialist", goal="Offer support based on changed emotions.",
                      backstory="Fatigue and driving style are not emotions; playback requires permission.",
                      llm=self.backend.llm, tools=[], verbose=False, allow_delegation=False)
        task = Task(description=DECISION_PROMPT, expected_output="One mood decision.",
                    output_pydantic=NotificationDecision, agent=agent)
        self.crew = Crew(agents=[agent], tasks=[task], verbose=False, tracing=False,
                         process=Process.sequential, memory=None)

    async def execute(self, task_input: DecisionInput) -> NotificationDecision:
        inputs = {
            "event_name": task_input.event_name,
            "driver_name": task_input.driver_name,
            "pending_message_count": task_input.pending_message_count,
            "event_details": task_input.event_details,
            "changed_facts": task_input.changed_facts,
            "context": task_input.context,
            "task_instructions": self.spec.instructions,
            "action_options": enum_options(self.spec.actions),
            "suggestion_options": enum_options(self.spec.suggestions),
            "urgency_options": enum_options(UrgencyType),
            "tone_options": enum_options(ToneType),
            "intervention_options": enum_options(InterventionType),
        }
        decision = await self.backend.run(self.crew, inputs, NotificationDecision, "mood_music")
        if decision.action not in self.spec.actions or decision.suggestion_type not in self.spec.suggestions:
            raise ValueError("mood_music returned a capability outside its contract")
        decision.skill = SkillType.WELLBEING if (
            decision.notify or decision.action is not ActionType.NONE
        ) else SkillType.NONE
        return decision

def _notify_music_update(session, value: dict[str, Any]) -> None:
    if session.on_music_update is not None:
        session.on_music_update(value)


def publish_pending_music_selection(session) -> None:
    if session.pending_music_selection is None:
        return
    session._notify_music_update({
        "status": "proposal", "autoplay": False,
        **session.pending_music_selection,
    })


def clear_pending_music_selection(session, *, publish: bool = True) -> None:
    had_selection = session.pending_music_selection is not None
    session.pending_music_selection = None
    session.pending_music_prompt = None
    if (session.pending_confirmation is not None
            and session.pending_confirmation.action is ActionType.EVALUATE_MUSIC_PROPOSAL):
        session.pending_confirmation = None
    if publish and had_selection:
        session._notify_music_update({"status": "idle"})


def confirm_pending_music_selection(session, accepted: bool) -> bool:
    selection = session.pending_music_selection
    if selection is None:
        return False
    if accepted:
        last_assistant = next((item["content"] for item in reversed(session.conversation_history)
                               if item["role"] == "assistant"), None)
        if last_assistant != session.pending_music_prompt:
            session.clear_pending_music_selection()
            return False
    session.clear_pending_music_selection(publish=not accepted)
    if accepted and not session.action_manager.music_manager.play(selection):
        session._notify_music_update({"status": "error"})
        return False
    return True


async def execute_music_action(session, action: ActionType, request: str) -> None:
    music_manager = session.action_manager.music_manager
    if action is ActionType.STOP_MUSIC:
        session.clear_pending_music_selection()
    if action is ActionType.PLAY_MUSIC:
        music_manager.close()
        if not session.opt.music_enabled or not music_manager.client_id:
            message = "Music playback isn't configured yet."
        else:
            result = await music_manager.request(
                session.music_preferences_provider(), request=request, autoplay=True,
                history=session.conversation_history,
            )
            message = "I've queued a themed playlist for you." if result else "I couldn't find music right now."
    else:
        commands = {ActionType.PAUSE_MUSIC: "pause", ActionType.RESUME_MUSIC: "resume",
                    ActionType.NEXT_MUSIC: "next", ActionType.STOP_MUSIC: "stop"}
        music_manager.control(commands[action])
        message = {"pause": "I've requested a music pause.", "resume": "I'll resume the music.",
                   "next": "I'll skip to the next track.", "stop": "I've stopped the music."}[commands[action]]
    await session.speak(message)


def emotion_state(session) -> dict[str, str]:
    """DriverEmotionState measures from the knowledge base, keyed by measure name."""
    emotion_state: dict[str, str] = {}
    for key, fact in session.knowledge_facts_provider().items():
        name, separator, measure = key.partition(".")
        if separator and name == "DriverEmotionState":
            emotion_state[measure] = fact
    return emotion_state


def should_offer_music_proposal(session) -> bool:
    selection = json.dumps({
        "emotion": session.emotion_state(),
        "preferences": session.driver_preferences_provider(),
    }, sort_keys=True, default=str)
    now = time.monotonic()
    if (selection == session._last_music_proposal_selection
            or now - session._last_music_proposal_attempt < session.opt.music_cooldown_seconds):
        return False
    session._last_music_proposal_selection = selection
    session._last_music_proposal_attempt = now
    return True


def music_proposal_reply_context(
    confirmed: bool | None, result: dict[str, Any] | None = None,
    *, asking_selection_confirmation: bool = False,
) -> tuple[str, str]:
    if asking_selection_confirmation:
        instructions = (
            "The driver agreed to a music proposal and the selected playlist is now ready for review. "
            "When a title is supplied, start with a neutral factual presentation using the exact title: "
            "'I found the playlist <title>.' Then ask one concise question: 'Does this selection work "
            "for you, and would you like me to start it?' Do not praise the choice, call it a perfect "
            "match, infer the driver's current mood, or address the driver by name. This is a second, "
            "separate confirmation; do not imply playback has started. If no title is supplied, explain "
            "briefly that no suitable selection was found and do not ask approval for a nonexistent "
            "playlist. Never invent music details."
        )
    else:
        instructions = (
            "This response is to the driver's reply to a pending music selection confirmation. "
            "Use the supplied confirmation result as authoritative. If confirmed is true and a "
            "playlist title is supplied, acknowledge the accepted choice by name without asking "
            "again or claiming the tracks are already playing. If false, acknowledge the refusal "
            "and, when a playlist title is supplied, explicitly confirm it was removed from the "
            "queue. Do not offer to play again. Never invent music details or completed actions."
        )
    return (
        "Generate a brief, natural reply from the actual conversation and vehicle context, not a "
        "canned acknowledgement. " + instructions,
        "Music proposal confirmation result: " + json.dumps({
            "confirmed": confirmed,
            "queued_playlist_title": (result or {}).get("title"),
            "asking_selection_confirmation": asking_selection_confirmation,
        }),
    )

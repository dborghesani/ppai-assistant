import asyncio
from typing import Any

import structlog
from skill_manager import SkillManager
from events import CarEvent
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

class AutomotiveAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.is_active = True
        self.event_queue: asyncio.Queue[CarEvent] = asyncio.Queue()
        self.user_event_queue: asyncio.Queue[CarEvent] = asyncio.Queue()
        self.is_listening = False

        self.skill_manager = SkillManager()

        self.recent_notifications: list[Any] = []

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
                You are the notification decision component of an in-vehicle assistant.

                Active skill:
                {skill}

                Triggering event:
                {event}

                Previous states:
                {previous_contexts}

                Current state:
                {context}

                Recent notifications:
                {recent_notifications}

                Evaluate whether the current situation provides enough value or urgency
                to interrupt the vehicle occupants now.

                Reason contextually. Do not notify merely because a value changed.

                Consider:
                - safety impact;
                - urgency;
                - magnitude and direction of the change;
                - interaction between multiple context variables;
                - whether the information is actionable;
                - whether the user already knows it;
                - whether a similar notification was recently delivered;
                - whether speaking could unnecessarily distract the driver.

                Examples of contextual reasoning:
                - Moderate fatigue alone may not justify an interruption.
                - Moderate fatigue combined with low attention, nighttime driving and
                highway speed may justify an immediate notification.
                - Rain alone may not justify a notification.
                - Rain combined with high speed and poor attention may justify one.
                - An unlocked door while parked may require a notification.
                - The same state repeatedly received without meaningful change should
                normally not generate another notification.

                If notification is appropriate:
                - set notify to true;
                - assign an urgency level;
                - provide a short internal reason;
                - generate one concise message addressed directly to the user.

                If notification is not appropriate:
                - set notify to false;
                - set urgency to none;
                - provide a short internal reason;
                - set spoken_message to null.

                The spoken message will be passed directly to text-to-speech.
                Never include analysis, scores, variable names or implementation details
                inside spoken_message.
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

                if hasattr(self, 'user_event_queue') and not self.user_event_queue.empty():
                    user_input: CarEvent = await self.user_event_queue.get()
                    await self._process_event(user_input)

                if hasattr(self, 'event_queue') and not self.event_queue.empty():
                    event: CarEvent = await self.event_queue.get()
                    if event.event_name == "shutdown":
                        self.is_active = False
                        break
                    await self._process_event(event)

                # check for proactive events
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                

    async def _process_event(self, event: CarEvent):
        logger.info(f">>> received {event.event_name} event with value: {event.event_value}")
        logger.info(">>> generating...")
        inputs = {
            "skill": event.skill,
            "context": event.context,
            "previous_contexts": event.previous_contexts,
            "event": event.event_name,
            "value": event.event_value,
            "user_input": event.user_input,
            "recent_notifications": self.recent_notifications,
        }
        result = await self.crew.kickoff_async(
            inputs=inputs,
        )
        response = result.raw if hasattr(result, 'raw') else str(result)
        logger.info(f">>> {response}")

        decision: NotificationDecision = result.pydantic
        if decision.notify and decision.spoken_message:
            logger.info(f">>> [speak] {decision.spoken_message}")
            #await self.tts.speak(decision.spoken_message)
            self.recent_notifications.append(decision.spoken_message)
            if len(self.recent_notifications) > 3:
                self.recent_notifications.pop(0)

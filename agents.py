import asyncio

import structlog
from skill_manager import SkillManager
from events import CarEvent
from crewai import Agent, Task, Crew
from crewai.process import Process
from crewai import LLM, Agent, Task, Crew

logger = structlog.get_logger()

class AutomotiveAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.is_active = True
        self.event_queue: asyncio.Queue[CarEvent] = asyncio.Queue()
        self.user_event_queue: asyncio.Queue[CarEvent] = asyncio.Queue()
        self.is_listening = False

        self.skill_manager = SkillManager()

        agent = Agent(
            role="Automotive Personal Assistant",
            goal="""
            Help the driver by understanding
            user requests, vehicle events and
            contextual information.

            Select the appropriate tools and
            skills when necessary.
            """,
            backstory="""
            You are an intelligent in-vehicle assistant.

            You can:
            - answer user questions
            - interpret vehicle events
            - monitor driver wellbeing
            - provide proactive suggestions
            - operate vehicle services

            Be concise and safety-oriented.
            """,
            llm=llm,
            verbose=True,
        )

        task = Task(
            description="""
            Current skill:
            {skill}

            Current context:
            {context}

            Current event:
            {event}

            User input:
            {user_input}

            Determine the most useful response or action. 
            If no action is needed, respond with a concise acknowledgment.
            """,
            expected_output="""
            A concise response or recommendation.
            """,
            agent=agent,
        )
    
        # Crew
        self.crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
            tracing=False,
            process=Process.sequential
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
            "event": event.event_name,
            "value": event.event_value,
            "user_input": event.user_input
        }
        result = await self.crew.kickoff_async(
            inputs=inputs,
        )
        response = result.raw if hasattr(result, 'raw') else str(result)
        logger.info(f">>> {response}")

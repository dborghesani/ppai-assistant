from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import laya
import structlog

from agents.agents_dataclasses import (
    ActionType,
    InterventionType,
    SkillType,
    SuggestionTarget,
    SuggestionType,
    Urgency,
    UrgencyType,
)
from data.events import CarEvent

if TYPE_CHECKING:
    from agents.automotive_agent import AutomotiveAgent


class LayaAgent:
    def __init__(self, agent: "AutomotiveAgent"):
        self.agent = agent
        self.logger = structlog.get_logger()
        self.laya_td = laya.load(
            "convaiinnovations/laya", subfolder="typed-decisions"
        )
        self.urgency_ranks = {
            UrgencyType.NONE: 0,
            UrgencyType.LOW: 1,
            UrgencyType.MEDIUM: 2,
            UrgencyType.HIGH: 3,
            UrgencyType.CRITICAL: 4,
        }
        self.minimum_urgency = UrgencyType.MEDIUM

    def format_laya_answers(self, answers: dict, precision: int = 3) -> list[str]:
        lines = []

        for name, answer in answers.items():
            answer_type = answer.get("type", "unknown")
            confidence = answer.get("confidence")

            if answer_type == "noul":
                value = answer.get("noul")

                text = f"{name}: {value:.{precision}f}"

            elif answer_type == "score":
                score = answer.get("score")
                legend = answer.get("legend", {})
                probabilities = answer.get("probabilities", {})

                text = f"{name}: {score:.{precision}f}"

                if probabilities:
                    probs_str = ", ".join(
                        f"{legend.get(k, k)}[{k}]={v:.{precision}f}"
                        for k, v in probabilities.items()
                    )
                    text += f" | {probs_str}"

            elif answer_type == "choice":
                choice = answer.get("choice")
                probabilities = answer.get("probabilities", {})

                text = f"{name}: {choice}"

                if probabilities:
                    probs_str = ", ".join(
                        f"{k}={v:.{precision}f}" for k, v in probabilities.items()
                    )
                    text += f" | {probs_str}"

            else:
                # Fallback nel caso Laya introduca un tipo che non gestiamo
                text = f"{name}: {answer}"

            if confidence is not None:
                text += f" | confidence={confidence:.{precision}f}"

            lines.append(text)

        return lines

    async def process_event(self, event: CarEvent) -> None:
        self.logger.info(f">>> [LAYA] Processing event: {event.event_value}")
        laya_start = time.time()
        # format events for laya
        state = {
            "event": event.event_value,
            "state": event.context,
        }
        laya_questions = {
            "urgency": {
                "instructions": (
                    "How urgent is an intervention for the current automotive situation?"
                ),
                "type": "score",
                "criteria": {k.name.lower(): k.value for k in UrgencyType},
            },

            "intervention_type": {
                "instructions": (
                    "What type of intervention is most appropriate for the current situation?"
                ),
                "type": "choice",
                "criteria": {k.name.lower(): k.value for k in InterventionType},
            },

            "skill": {
                "instructions": (
                    "Which assistant skill is most appropriate for handling the situation?"
                ),
                "type": "choice",
                "criteria": {k.name.lower(): k.value for k in SkillType},
            },

            "action": {
                "instructions": (
                    "Which direct action should be executed? "
                    "Select none when intervention_type is not act."
                ),
                "type": "choice",
                "criteria": {k.name.lower(): k.value for k in ActionType},
            },

            "suggestion_type": {
                "instructions": (
                    "What semantic suggestion should be communicated to the driver? "
                    "Select none when intervention_type is not suggest. "
                    "Select the intent of the suggestion, not the final driver-facing wording."
                ),
                "type": "choice",
                "criteria": {k.name.lower(): k.value for k in SuggestionType},
            },
        }

        result = self.laya_td.predict(state, laya_questions)
        laya_elapsed = time.time() - laya_start
        answers = result["answers"]
        output_lines = self.format_laya_answers(answers)
        for line in output_lines:
            self.logger.info(f">>> [LAYA] {line}")
        self.logger.info(
            f">>> finished processing intervention analysis in {laya_elapsed:.3f} seconds"
        )

        # LLM output will be generated based on this decision.
        decision = {
            "urgency": answers["urgency"]["legend"][
                str(round(answers["urgency"]["score"]))
            ],
            "intervention": answers["intervention_type"]["choice"],
            "action": answers["action"]["choice"],
            "suggestion": answers["suggestion_type"]["choice"],
        }

        intervention = InterventionType[decision["intervention"].upper()]
        urgency = UrgencyType[decision["urgency"].upper()]
        # Skip decisions below the urgency selected in the UI.
        if (
            intervention == InterventionType.NONE
            or self.urgency_ranks[urgency] < self.urgency_ranks[self.minimum_urgency]
        ):
            return

        system_message = """
            You are an in-vehicle assistant.

            Your task is to communicate naturally with the driver based on the current
            driving context and the decision provided by the decision model. If not 
            specified, prefer English as the default language.

            The decision has already been made. Do not override or reinterpret it.
            Treat the intervention and suggestion_target values as internal instructions. Never
            mention, repeat, or prefix the response with decision labels or values
            such as "intervention", "suggestion_target", "suggest", "driving", or
            "wellbeing".

            Interpret the intervention as follows:
            - none: no response should be generated.
            - suggest: suggest an appropriate behavior or response based on the triggering
            event, current context, and suggestion_target.

            The suggestion_target field identifies the aspect that the suggestion should
            address. Use it to focus the response, but do not name the category itself.

            Do not mention internal models, scores, probabilities, confidence values,
            or internal reasoning.

            Prefer concise and clear communication.
            Respond directly to the driver in one or two short sentences.
            """.strip()

        user_message = f"""
            Triggering event:
            {event.event_value}

            Current context:
            {event.context}

            Decision:
            {decision}

            Driver input:
            {event.user_input or "None"}

            Generate the appropriate response to the driver.
        """.strip()

        messages = [
            {"role": "system", "content": system_message},
            *self.agent._conversation_history[-8:],
            {"role": "user", "content": user_message},
        ]

        full_response = ""
        displayed_response = ""
        sentence_buffer = ""
        stream = None
        try:
            stream = await self.agent._voice_llm.chat.completions.create(
                model=self.agent.opt.ollama_model.removeprefix("ollama/"),
                messages=messages,
                stream=True,
                temperature=0.3,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue
                full_response += token
                sentence_buffer += token

                split = re.search(r"[.!?](?:\s|$)", sentence_buffer)
                while split is not None:
                    sentence = sentence_buffer[: split.end()].strip()
                    sentence_buffer = sentence_buffer[split.end() :]
                    if sentence:
                        displayed_response = f"{displayed_response} {sentence}".strip()
                        if self.agent.on_response_update is not None:
                            self.agent.on_response_update(displayed_response)
                        if self.agent.tts_manager is not None:
                            await self.agent.tts_manager.speak(sentence)
                    split = re.search(r"[.!?](?:\s|$)", sentence_buffer)
        finally:
            if stream is not None:
                await stream.close()

        trailing_sentence = sentence_buffer.strip()
        if trailing_sentence:
            displayed_response = f"{displayed_response} {trailing_sentence}".strip()
            if self.agent.on_response_update is not None:
                self.agent.on_response_update(displayed_response)
            if self.agent.tts_manager is not None:
                await self.agent.tts_manager.speak(trailing_sentence)

        full_response = full_response.strip()
        if full_response:
            self.agent._conversation_history.extend(
                [
                    {"role": "user", "content": event.user_input},
                    {"role": "assistant", "content": full_response},
                ]
            )
            self.agent._conversation_history = self.agent._conversation_history[-8:]
            if self.agent.on_response is not None:
                self.agent.on_response(full_response + "\n")
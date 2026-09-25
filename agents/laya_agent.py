from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import laya
import structlog

from agents.agents_dataclasses import (
    ActionType,
    InterventionType,
    SkillType,
    SuggestionType,
    UrgencyType,
)
from data.events import CarEvent

if TYPE_CHECKING:
    from agents.automotive_agent import AutomotiveAgent


class LayaAgent:
    def __init__(self, agent: "AutomotiveAgent"):
        self.agent = agent
        self.logger = structlog.get_logger()
        custom_model = getattr(agent.opt, "laya_custom_model", None)
        if custom_model:
            model_source = Path(custom_model).expanduser()
            if not model_source.is_dir():
                raise FileNotFoundError(
                    f"Custom Laya model directory does not exist: {model_source}"
                )
            model_subfolder = None
        else:
            model_source = "convaiinnovations/laya"
            model_subfolder = "typed-decisions"
        self.laya_td = laya.load(
            model_source,
            subfolder=model_subfolder,
        )
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

    @staticmethod
    def _criteria(enum_type: type[SkillType]) -> dict[str, str]:
        return {
            member.name.lower(): member.description
            for member in enum_type
        }

    async def process_event(self, event: CarEvent) -> None:
        self.logger.info(f">>> [LAYA] Processing event: {event.event_value}")
        laya_start = time.time()
        # Keep the inference payload identical to the fine-tuning dataset schema.
        state = {
            "knowledge": list(event.context or []),
            "skill_scope": SkillType.NONE.value,
        }
        laya_questions = {
            "urgency": {
                "instructions": (
                    "How urgent is an intervention for the current automotive situation?"
                ),
                "type": "choice",
                "criteria": self._criteria(UrgencyType),
            },

            "intervention_type": {
                "instructions": "What type of intervention is appropriate?",
                "type": "choice",
                "criteria": self._criteria(InterventionType),
            },

            "skill": {
                "instructions": "Which assistant skill should handle the situation?",
                "type": "choice",
                "criteria": self._criteria(SkillType),
            },

            "action": {
                "instructions": "Which direct action should be executed? Select none unless intervention_type is act.",
                "type": "choice",
                "criteria": self._criteria(ActionType),
            },

            "suggestion_type": {
                "instructions": "Which semantic suggestion should be communicated? Select none unless intervention_type is suggest.",
                "type": "choice",
                "criteria": self._criteria(SuggestionType),
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
            "urgency": answers["urgency"]["choice"],
            "intervention": answers["intervention_type"]["choice"],
            "action": answers["action"]["choice"],
            "suggestion": answers["suggestion_type"]["choice"],
            "skill": answers["skill"]["choice"],
        }

        intervention = InterventionType[decision["intervention"].upper()]
        urgency = UrgencyType[decision["urgency"].upper()]
        # Skip decisions below the urgency selected in the UI.
        if (
            intervention == InterventionType.NONE
            or urgency.rank < self.minimum_urgency.rank
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
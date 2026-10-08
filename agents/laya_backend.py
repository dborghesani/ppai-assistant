from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import laya
import structlog

from data.agents_dataclasses import (
    ActionType,
    AssistantStatus,
    InterventionType,
    SimpleInterventionType,
    SkillType,
    SuggestionType,
    ToneType,
    UrgencyType,
)
from data.events import CarEvent
from managers.knowledge_manager import KnowledgeManager

if TYPE_CHECKING:
    from agents.automotive_agent import AutomotiveAgent


class LayaBackend:
    def __init__(self, agent: "AutomotiveAgent"):
        self.agent = agent
        self.logger = structlog.get_logger()
        self.custom_model = getattr(agent.opt, "laya_custom_model", None)
        if self.custom_model:
            model_source = Path(self.custom_model).expanduser()
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

    @staticmethod
    def _skill_scope_for_event(event: CarEvent) -> str:
        if not isinstance(event.event_value, dict):
            return SkillType.NONE.value

        for key in event.event_value:
            name, separator, measure = key.partition(".")
            if not separator:
                continue
            skills = KnowledgeManager._field_skills(name, measure)
            if skills:
                return skills[0].value

        return SkillType.NONE.value

    async def process_event(self, event: CarEvent) -> None:
        self.logger.info(f">>> [LAYA] Processing event: {event.event_value}")
        laya_start = time.time()
        # Keep the inference payload identical to the fine-tuning dataset schema.
        if self.custom_model == "":
            state = {
                "knowledge": list(event.context or []),
                #"skill_scope": self._skill_scope_for_event(event),
            }
            laya_questions = {
                "urgency": {
                    "instructions": "Define how urgent the situation is.",
                    "type": "choice",
                    "criteria": self._criteria(UrgencyType),
                },

                "tone": {
                    "instructions": "Define the communication tone appropriate for the situation.",
                    "type": "choice",
                    "criteria": self._criteria(ToneType),
                },

                "intervention_type": {
                    "instructions": "Decide what the assistant should do.",
                    "type": "choice",
                    "criteria": self._criteria(InterventionType),
                },

                "skill": {
                    "instructions": "Define a scope for the potential skills the assistant should have to face the situation.",
                    "type": "choice",
                    "criteria": self._criteria(SkillType),
                },

                "action": {
                    "instructions": "Define what the assistant should actuate if an action is required.",
                    "type": "choice",
                    "criteria": self._criteria(ActionType),
                },

                "suggestion_type": {
                    "instructions": "Define what the assistant should suggest vocally to the driver if a suggestion is required.",
                    "type": "choice",
                    "criteria": self._criteria(SuggestionType),
                },
            }
        else:
            state = {
                "event": event.event_value,
                "state": event.context,
            }
            laya_questions = {
                "urgency": {
                    "type": "choice",
                    "instructions": (
                        "Considering the triggering event, the driver state, and the overall context, "
                        "how urgent is an assistant intervention?"
                    ),
                    "criteria": {k.name.lower(): k.value for k in UrgencyType},
                },
                "intervention_type": {
                    "type": "choice",
                    "instructions": (
                        "Considering the triggering event and the overall driving context, "
                        "including the driver's emotional, physical, attentional, and driving state, "
                        "would it be useful to provide the driver with a suggestion?"
                    ),
                    "criteria": {k.name.lower(): k.value for k in SimpleInterventionType},
                },
                "suggestion_type": {
                    "type": "choice",
                    "instructions": (
                        "If a suggestion is useful, which aspect should it primarily address? "
                        "Choose the most relevant target from the available criteria."
                    ),
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
        decision = {}
        if "urgency" in answers:
            decision["urgency"] = answers["urgency"]["choice"]
        if "intervention_type" in answers:
            decision["intervention_type"] = answers["intervention_type"]["choice"]
        if "suggestion_type" in answers:
            decision["suggestion_type"] = answers["suggestion_type"]["choice"]
        if "action" in answers:
            decision["action"] = answers["action"]["choice"]
        if "tone" in answers:
            decision["tone"] = answers["tone"]["choice"]
        if "skill" in answers:
            decision["skill"] = answers["skill"]["choice"]

        try:
            output_tone = ToneType(decision.get("tone", ToneType.CALM.value))
        except ValueError:
            output_tone = ToneType.CALM
    
        urgency = UrgencyType[decision["urgency"].upper()]
        # Skip decisions below the urgency selected in the UI.
        if not self.custom_model:
            if decision["suggestion_type"].upper() == SuggestionType.NONE.name:
                return
            if urgency.rank < self.minimum_urgency.rank:
                return
        else:
            if decision["intervention_type"].upper() == InterventionType.NONE.name:
                return
            if urgency.rank < self.minimum_urgency.rank:
                return

        if "action" in decision:
            try:
                self.agent.action_manager.handle_decision(ActionType(decision["action"]))
            except ValueError:
                self.logger.warning("Unknown action decided by Laya", action=decision["action"])

        system_message = """
            You are an in-vehicle assistant.

            Your task is to communicate naturally with the driver based on the current
            driving context and the decision provided by the decision model. If not 
            specified, prefer English as the default language.

            The decision has already been made. Do not override or reinterpret it.
            Treat the intervention, suggestion_target and action values as internal
            instructions. Never mention, repeat, or prefix the response with decision
            labels or values such as "intervention", "suggestion_target", "action",
            "suggest", "act", "driving", or "wellbeing".

            Interpret the intervention as follows:
            - none: no response should be generated.
            - suggest: suggest an appropriate behavior or response based on the triggering
            event, current context, and suggestion_target.
            - act: if the action name starts with "ask_", ask the driver that question
            naturally. Otherwise, naturally inform the driver that the action has been
            carried out, based on the triggering event, current context, and action.

            The suggestion_target field identifies the aspect that the suggestion should
            address. Use it to focus the response, but do not name the category itself.

            The action field, when present, identifies the vehicle or comfort function
            that was executed. Use it to describe what was done, but do not name the
            internal field itself.

            Do not mention internal models, scores, probabilities, confidence values,
            or internal reasoning.

            Never close with a generic filler question such as "is there anything else
            I can help with?" or "let me know if you need anything else". Only ask a
            question when the driver's answer is actually needed to proceed (e.g. when
            the action starts with "ask_").

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
            *self.agent.conversation_history[-8:],
            {"role": "user", "content": user_message},
        ]

        full_response = ""
        displayed_response = ""
        sentence_buffer = ""
        stream = None
        self.agent.set_assistant_status(AssistantStatus.TALKING)
        if self.agent.on_speaking_tone_changed is not None:
            self.agent.on_speaking_tone_changed(output_tone.value)
        try:
            llm_started = time.perf_counter()
            stream = await self.agent.voice_llm.chat.completions.create(
                model=self.agent.opt.ollama_model,
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
                if not full_response:
                    self.logger.info(
                        "LLM time to first token",
                        llm_call="laya response",
                        model=self.agent.opt.ollama_model,
                        ttft_ms=round((time.perf_counter() - llm_started) * 1000, 2),
                    )
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
            trailing_sentence = sentence_buffer.strip()
            if trailing_sentence:
                displayed_response = f"{displayed_response} {trailing_sentence}".strip()
                if self.agent.on_response_update is not None:
                    self.agent.on_response_update(displayed_response)
                if self.agent.tts_manager is not None:
                    await self.agent.tts_manager.speak(trailing_sentence)
        finally:
            if stream is not None:
                await stream.close()
            if self.agent.on_speaking_tone_changed is not None:
                self.agent.on_speaking_tone_changed(None)
            self.agent.set_assistant_status(AssistantStatus.IDLE)

        full_response = full_response.strip()
        if full_response:
            self.agent.conversation_history.extend(
                [
                    {"role": "user", "content": event.user_input},
                    {"role": "assistant", "content": full_response},
                ]
            )
            self.agent.conversation_history = self.agent.conversation_history[-8:]
            if self.agent.on_response is not None:
                self.agent.on_response(full_response + "\n")
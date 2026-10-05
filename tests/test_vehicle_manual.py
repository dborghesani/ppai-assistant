import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pymupdf
import pytest

from config import ConfigAssistant
from managers.vehicle_manual_manager import ManualPassage, VehicleManualManager


def make_manager(tmp_path):
    manual = tmp_path / "manual.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Activate adaptive cruise control using the steering wheel button.")
        page = document.new_page()
        page.insert_text((72, 72), "Check tyre pressure when the tyres are cold.")
        document.save(manual)
    return VehicleManualManager(ConfigAssistant(
        rag_manual_directory=str(tmp_path),
        rag_manual_filename=manual.name,
        rag_index_directory=str(tmp_path / "index"),
    ))


def test_index_retrieve_and_reuse(tmp_path):
    async def check():
        manager = make_manager(tmp_path)
        manager._embed = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
        await manager.prepare()
        manager._embed.assert_awaited_once()
        manager._embed = AsyncMock(return_value=[[1.0, 0.0]])
        passages = await manager.retrieve("How do I activate cruise control?")
        assert len(passages) == 1
        assert passages[0].page == 1
        assert passages[0].source == "manual.pdf"
        assert "steering wheel" in passages[0].text
        cached = VehicleManualManager(manager.opt)
        cached._embed = AsyncMock()
        await cached.prepare()
        cached._embed.assert_not_awaited()
    asyncio.run(check())


def test_context_uses_history_and_handles_failures(tmp_path):
    async def check():
        manager = make_manager(tmp_path)
        manager._embed = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
        await manager.prepare()
        manager._embed = AsyncMock(return_value=[[1.0, 0.0]])
        context = await manager.context_for("How do I activate it?", [
            {"role": "user", "content": "What is adaptive cruise control?"},
        ])
        assert "PDF page 1" in context
        assert "adaptive cruise control" in manager._embed.call_args.args[0][0]
        manager._embed = AsyncMock(return_value=[[-1.0, 0.0]])
        assert "No sufficiently relevant" in await manager.context_for("Unknown", [])
        manager._embed = AsyncMock(side_effect=RuntimeError("Ollama unavailable"))
        assert "unavailable" in await manager.context_for("Cruise control", [])
    asyncio.run(check())


def test_context_budget_does_not_cut_passages(tmp_path):
    manager = make_manager(tmp_path)
    passage = ManualPassage("Complete warning and procedure.", 1, "manual.pdf", 0.9)
    block = "Source: manual.pdf; PDF page 1\nComplete warning and procedure."
    manager.retrieve = AsyncMock(return_value=[passage, passage])
    manager.opt.rag_context_max_chars = len(block)
    assert asyncio.run(manager.context_for("Question", [])) == block


def test_chunk_overlap_and_empty_pdf(tmp_path):
    manager = make_manager(tmp_path)
    manager.opt.rag_chunk_words = 5
    manager.opt.rag_chunk_overlap_words = 2
    _, chunks = manager._read_chunks()
    assert chunks[0]["text"].split()[-2:] == chunks[1]["text"].split()[:2]
    with pymupdf.open() as document:
        document.new_page()
        document.save(manager.manual_path)
    with pytest.raises(ValueError, match="OCR"):
        manager._read_chunks()


def test_invalid_chunk_configuration():
    with pytest.raises(ValueError, match="overlap"):
        VehicleManualManager(ConfigAssistant(rag_chunk_words=10, rag_chunk_overlap_words=10))


def test_response_source_is_structured_and_defaults_to_general():
    from agents.llm_backend import NotificationDecision, ResponseSource
    payload = {"urgency": "none", "tone": "calm", "intervention_type": "none", "skill": "none",
               "action": "none", "suggestion_type": "none", "reason": "Informational question"}
    assert NotificationDecision(**payload).response_source is ResponseSource.GENERAL
    assert NotificationDecision(**payload, response_source="vehicle_manual").response_source is ResponseSource.VEHICLE_MANUAL
    with pytest.raises(ValueError):
        NotificationDecision(**payload, response_source="invented_source")


@pytest.mark.parametrize("rag_enabled", [False, True])
@pytest.mark.parametrize("manual_question", [False, True])
def test_manual_passages_reach_ollama_without_entering_history(rag_enabled, manual_question):
    from data.agents_dataclasses import AssistantStatus, SkillType, ToneType
    from agents.automotive_agent import AutomotiveAgent
    from data.events import CarEvent
    from agents.llm_backend import ResponseSource

    class FakeStream:
        async def __aiter__(self):
            yield SimpleNamespace(usage=None, choices=[
                SimpleNamespace(delta=SimpleNamespace(content="Use the steering wheel button."))
            ])

        async def close(self):
            pass

    async def check():
        agent = AutomotiveAgent.__new__(AutomotiveAgent)
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.skill_manager = SimpleNamespace(get_skill=Mock(return_value="Be concise."))
        agent._classify_conversation_tone = AsyncMock(return_value=ToneType.CALM)
        agent.manual_manager = SimpleNamespace(context_for=AsyncMock(
            return_value="Source: manual.pdf; PDF page 1\nUse the steering wheel button."
        )) if rag_enabled else None
        agent.response_context_providers = {ResponseSource.VEHICLE_MANUAL: VehicleManualManager.unavailable_response_context}
        if agent.manual_manager is not None:
            async def manual_context(question, history):
                context = await agent.manual_manager.context_for(question, history)
                return VehicleManualManager.RESPONSE_INSTRUCTIONS, f"Retrieved vehicle manual passages:\n{context}"
            agent.response_context_providers[ResponseSource.VEHICLE_MANUAL] = manual_context
        create = AsyncMock(return_value=FakeStream())
        agent.voice_llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        agent._assistant_status = AssistantStatus.IDLE
        agent.on_assistant_status_changed = None
        agent.on_speaking_tone_changed = None
        agent.on_response_update = Mock()
        agent.on_response = Mock()
        agent.tts_manager = SimpleNamespace(speak=AsyncMock())
        agent.log_llm_usage = Mock()
        from agents.llm_backend import LLMBackend
        agent.llm_backend = LLMBackend.__new__(LLMBackend)
        agent.llm_backend.opt = agent.opt
        agent.llm_backend.client = agent.voice_llm
        agent.llm_backend.on_usage = agent.log_llm_usage
        agent.llm_backend.logger = Mock()
        event = CarEvent(SkillType.CONVERSATION, "user_input", "How does cruise control work?", [], "How does cruise control work?")
        source = ResponseSource.VEHICLE_MANUAL if manual_question else ResponseSource.GENERAL
        await agent.prepare_user_response(event, source)
        messages = create.call_args.kwargs["messages"]
        assert create.call_args.kwargs["reasoning_effort"] == "none"
        assert create.call_args.kwargs["max_tokens"] == agent.opt.max_tokens
        assert ("Retrieved vehicle manual passages" in messages[-1]["content"]) is (rag_enabled and manual_question)
        if rag_enabled and manual_question:
            assert "PDF page 1" in messages[-1]["content"]
            assert "never invent" in messages[0]["content"]
            agent.manual_manager.context_for.assert_awaited_once()
        elif rag_enabled:
            agent.manual_manager.context_for.assert_not_awaited()
        if manual_question and not rag_enabled:
            assert "No verified manual passages" in messages[-1]["content"]
            assert "never invent" in messages[0]["content"]
        assert agent.conversation_history[0]["content"] == event.user_input
        assert "Retrieved" not in str(agent.conversation_history)
        agent.tts_manager.speak.assert_awaited_once_with("Use the steering wheel button.")
        assert agent.assistant_status == AssistantStatus.IDLE
    asyncio.run(check())


@pytest.mark.parametrize("content, expected_tone", [
    ('{"tone": "serious"}', "serious"),
    ('{}', "calm"),
    ('', "calm"),
    ('{"tone": "unknown"}', "calm"),
])
def test_tone_classification_requests_schema_without_reasoning(content, expected_tone):
    from data.agents_dataclasses import SkillType
    from agents.automotive_agent import AutomotiveAgent
    from data.events import CarEvent

    async def check():
        agent = AutomotiveAgent.__new__(AutomotiveAgent)
        agent.opt = ConfigAssistant()
        agent.conversation_history = []
        agent.log_llm_usage = Mock()
        response = SimpleNamespace(usage=None, choices=[
            SimpleNamespace(message=SimpleNamespace(content=content))
        ])
        create = AsyncMock(return_value=response)
        agent.voice_llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        event = CarEvent(SkillType.CONVERSATION, "user_input", "How do I activate cruise control?", [], "How do I activate cruise control?")
        from agents.llm_backend import LLMBackend
        agent.llm_backend = LLMBackend.__new__(LLMBackend)
        agent.llm_backend.opt = agent.opt
        agent.llm_backend.client = agent.voice_llm
        agent.llm_backend.on_usage = agent.log_llm_usage
        agent.llm_backend.logger = Mock()
        tone = await agent._classify_conversation_tone(event, "")
        assert tone.value == expected_tone
        assert create.call_args.kwargs["reasoning_effort"] == "none"
        schema = create.call_args.kwargs["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["tone"]
        assert expected_tone in schema["properties"]["tone"]["enum"]
    asyncio.run(check())
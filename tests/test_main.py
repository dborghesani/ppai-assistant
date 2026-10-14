import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from config import ConfigAssistant
from main import _stop_ollama_model, preload_ollama_models


@pytest.mark.parametrize("level", ["INFO", "debug"])
def test_configured_log_level_filters_debug_details(level, capsys):
    import logging
    import structlog
    from main import configure_logging

    original_config = structlog.get_config()
    original_level = logging.getLogger().level
    try:
        configure_logging(level)
        test_logger = structlog.get_logger()
        test_logger.debug("debug detail")
        test_logger.info("final summary")
        output = capsys.readouterr().out
        assert "final summary" in output
        assert ("debug detail" in output) == (level.upper() == "DEBUG")
    finally:
        structlog.configure(**original_config)
        logging.getLogger().setLevel(original_level)


def test_invalid_log_level_is_rejected():
    from main import configure_logging

    with pytest.raises(ValueError, match="log_level"):
        configure_logging("verbose")


def test_mqtt_client_waits_until_embedded_broker_is_ready():
    import asyncio
    from contextlib import suppress
    from unittest.mock import AsyncMock
    from managers.source_manager import SourceManager

    async def check():
        manager = SourceManager(asyncio.Queue(), ConfigAssistant(mqtt_embedded_broker=True))
        start_entered = asyncio.Event()
        allow_start = asyncio.Event()

        async def start():
            start_entered.set()
            await allow_start.wait()

        broker = Mock(start=AsyncMock(side_effect=start), shutdown=AsyncMock())
        with (
            patch("managers.source_manager.Broker", return_value=broker),
            patch("managers.source_manager.MqttThreadClient") as client_type,
        ):
            client_task = asyncio.create_task(manager.run_mqtt())
            broker_task = asyncio.create_task(manager.run_embedded_broker())
            try:
                await asyncio.wait_for(start_entered.wait(), timeout=1)
                client_type.assert_not_called()
                allow_start.set()
                await asyncio.wait_for(manager._mqtt_broker_ready.wait(), timeout=1)
                await asyncio.sleep(0)
                client_type.return_value.start.assert_called_once_with(manager.opt.mqtt_topic)
            finally:
                for task in (client_task, broker_task):
                    task.cancel()
                for task in (client_task, broker_task):
                    with suppress(asyncio.CancelledError):
                        await task
            assert not manager._mqtt_broker_ready.is_set()
            broker.shutdown.assert_awaited_once()
            client_type.return_value.stop.assert_called_once()

    asyncio.run(check())


def test_embedded_mqtt_broker_reports_missing_dependency():
    import asyncio
    from managers.source_manager import SourceManager

    async def check():
        manager = SourceManager(asyncio.Queue(), ConfigAssistant())
        with patch("managers.source_manager.Broker", None):
            with pytest.raises(RuntimeError, match="amqtt is not installed"):
                await manager.run_embedded_broker()
        assert not manager._mqtt_broker_ready.is_set()

    asyncio.run(check())


@pytest.fixture
def running_models():
    with patch("main.urlopen") as request:
        response = request.return_value.__enter__.return_value
        response.read.return_value = json.dumps({"models": []}).encode()
        yield request, response


@pytest.mark.parametrize("model", ["qwen2.5:3b-instruct", "another-model", "ollama/another-model"])
def test_laya_stops_configured_preclassification_model_on_the_configured_server(model, running_models):
    opt = ConfigAssistant(
        ollama_model="qwen3.5:4b", pre_classify_model=model,
        routing_backend="decision", pre_classify_backend="decision",
        ollama_host="inference-server", ollama_port=12345,
    )
    request, response = running_models
    response.read.return_value = json.dumps({"models": [{"name": model.removeprefix("ollama/")}]}).encode()
    with patch("main.subprocess.run") as stop:
        _stop_ollama_model(opt)
    assert stop.call_args.args[0] == ["ollama", "stop", model.removeprefix("ollama/")]
    assert stop.call_args.kwargs["env"]["OLLAMA_HOST"] == "http://inference-server:12345"
    assert stop.call_args.kwargs["timeout"] == 30
    assert stop.call_args.kwargs["check"] is True
    assert request.call_args.args[0].full_url == "http://inference-server:12345/api/ps"


@pytest.mark.parametrize("routing,preclassification,preload_small_model", [
    ("llm", "llm", True),
    ("decision", "decision", False),
    ("decision", "llm", True),
    # No dedicated routing_model: routing falls back to the main model, so the
    # small model is warmed only when the pre-classifier itself needs it.
    ("crew", "llm", True),
    ("llm", "decision", False),
    ("crew", "decision", False),
    ("other", "other", False),
])
def test_preload_always_loads_main_and_conditionally_loads_or_stops_preclassification(
    routing, preclassification, preload_small_model
):
    opt = ConfigAssistant(
        routing_backend=routing, pre_classify_backend=preclassification,
        ollama_model="main-model", pre_classify_model="ollama/small-model",
    )
    with (
        patch("main._warmup_ollama_model", side_effect=["main-model-ctx8192", "small-model"]) as warmup,
        patch("main._stop_ollama_model") as stop,
    ):
        preload_ollama_models(opt)
    assert opt.ollama_model == "main-model-ctx8192"
    assert warmup.call_args_list[0].args == (opt, "main-model")
    assert warmup.call_args_list[0].kwargs == {"configure_context_window": True}
    if preload_small_model:
        assert warmup.call_count == 2
        assert warmup.call_args_list[1].args == (opt, "small-model")
        stop.assert_not_called()
    else:
        warmup.assert_called_once()
        stop.assert_called_once_with(opt)


def test_ollama_warmup_performs_bounded_inference_not_just_model_loading():
    from main import _warmup_ollama_model

    opt = ConfigAssistant()
    with patch("main.urlopen") as request:
        response = request.return_value.__enter__.return_value
        response.read.return_value = json.dumps({"response": "Hello!", "done": True, "eval_count": 2}).encode()
        assert _warmup_ollama_model(opt, opt.ollama_model) == opt.ollama_model
    payload = json.loads(request.call_args.args[0].data)
    assert payload["prompt"]
    assert payload["raw"] is True
    assert payload["options"] == {"num_predict": 2, "temperature": 0}
    assert payload["stream"] is False
    assert payload["keep_alive"] == "30m"
    response.read.assert_called_once()


@pytest.mark.parametrize("payload", [b"invalid JSON", b"[]", b'{"error":"failed to generate"}'])
def test_ollama_warmup_reports_invalid_inference_response(payload):
    from main import _warmup_ollama_model

    opt = ConfigAssistant()
    with patch("main.urlopen") as request, patch("main.logger") as test_logger:
        request.return_value.__enter__.return_value.read.return_value = payload
        assert _warmup_ollama_model(opt, opt.ollama_model) == opt.ollama_model
    test_logger.warning.assert_called_once()
    test_logger.info.assert_not_called()


@pytest.mark.parametrize("used_by", ["conversation", "routing"])
def test_preload_does_not_unload_a_model_still_used_after_warmup(used_by):
    opt = ConfigAssistant(
        ollama_model="shared-model", pre_classify_backend="decision",
        pre_classify_model="ollama/shared-model" if used_by == "conversation" else "ollama/router-model",
        routing_backend="crew", routing_model="router-model" if used_by == "routing" else None,
    )
    with (
        patch("main._warmup_ollama_model", side_effect=lambda _opt, model, **kwargs: model),
        patch("main._stop_ollama_model") as stop,
    ):
        preload_ollama_models(opt)
    stop.assert_not_called()


@pytest.mark.parametrize("backend_name", ["KevBackend", "LayaBackend"])
def test_decision_backend_warmup_runs_harmless_inference(backend_name):
    import asyncio
    from unittest.mock import AsyncMock
    from agents.backends import decision_backend

    backend = Mock(spec=getattr(decision_backend, backend_name))
    backend.execute_decision = AsyncMock(return_value={"answers": {}})
    asyncio.run(decision_backend.DecisionBackend.warmup(backend))
    backend.execute_decision.assert_awaited_once()
    state, questions = backend.execute_decision.await_args.args
    assert state == {"user_input": "Hello", "conversation_history": [], "pending_message_count": 0}
    assert len(questions) == 2
    assert all(question["type"] == "choice" for question in questions.values())
    assert backend.execute_decision.await_args.kwargs["call_name"] == "decision warmup"


@pytest.mark.parametrize("shared", [False, True])
def test_main_warms_each_unique_decision_backend_once(shared):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from agents.backends.decision_backend import DecisionBackend
    from main import warmup_decision_models

    router_backend = Mock(spec=DecisionBackend)
    request_backend = router_backend if shared else Mock(spec=DecisionBackend)
    router_backend.warmup = AsyncMock()
    if not shared:
        request_backend.warmup = AsyncMock()
    agent = SimpleNamespace(
        telemetry_router=SimpleNamespace(backend=router_backend),
        request_pre_classifier=SimpleNamespace(backend=request_backend),
    )
    asyncio.run(warmup_decision_models(agent))
    router_backend.warmup.assert_awaited_once()
    request_backend.warmup.assert_awaited_once()


def test_main_skips_decision_warmup_for_llm_backends():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from agents.backends.ollama_backend import OllamaBackend
    from main import warmup_decision_models

    backend = Mock(spec=OllamaBackend)
    warmup = AsyncMock()
    backend.warmup = warmup
    agent = SimpleNamespace(
        telemetry_router=SimpleNamespace(backend=backend),
        request_pre_classifier=SimpleNamespace(backend=backend),
    )
    asyncio.run(warmup_decision_models(agent))
    warmup.assert_not_awaited()


def test_decision_warmup_failure_is_not_reported_as_ready():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from agents.backends.decision_backend import DecisionBackend
    from main import warmup_decision_models

    backend = Mock(spec=DecisionBackend)
    backend.warmup = AsyncMock(side_effect=RuntimeError("warmup failed"))
    agent = SimpleNamespace(
        telemetry_router=SimpleNamespace(backend=backend),
        request_pre_classifier=SimpleNamespace(backend=backend),
    )
    with patch("main.logger") as test_logger:
        with pytest.raises(RuntimeError, match="warmup failed"):
            asyncio.run(warmup_decision_models(agent))
    assert all(call.args[0] != "Decision backend ready" for call in test_logger.info.call_args_list)


def test_main_waits_for_decision_warmup_before_starting_ui():
    import asyncio
    from unittest.mock import AsyncMock
    from main import main

    with (
        patch("main.preload_ollama_models"),
        patch("main.DatabaseManager"),
        patch("main.AutomotiveAgent") as agent_type,
        patch("main.warmup_decision_models", new_callable=AsyncMock,
              side_effect=RuntimeError("warmup failed")) as warmup,
        patch("main.WebUIServer") as ui_type,
    ):
        with pytest.raises(RuntimeError, match="warmup failed"):
            asyncio.run(main(ConfigAssistant()))
    warmup.assert_awaited_once_with(agent_type.return_value)
    ui_type.assert_not_called()


def test_stop_stops_active_model_when_requested(running_models):
    opt = ConfigAssistant(
        routing_backend="decision", pre_classify_backend="decision",
        ollama_model="same-model", pre_classify_model="same-model",
    )
    running_models[1].read.return_value = json.dumps({"models": [{"model": "same-model:latest"}]}).encode()
    with patch("main.subprocess.run") as stop:
        _stop_ollama_model(opt)
    stop.assert_called_once()
    assert stop.call_args.args[0] == ["ollama", "stop", "same-model"]


@pytest.mark.parametrize("models", [[], [{"name": "another-model"}], [{"name": "qwen2.5:3b-instruct-ctx16384"}]])
def test_stop_skips_models_not_active_on_ollama(models, running_models):
    opt = ConfigAssistant(pre_classify_model="qwen2.5:3b-instruct", pre_classify_backend="decision")
    running_models[1].read.return_value = json.dumps({"models": models}).encode()
    with patch("main.subprocess.run") as stop:
        _stop_ollama_model(opt)
    stop.assert_not_called()


@pytest.mark.parametrize("error", [
    FileNotFoundError("ollama not installed"),
    subprocess.CalledProcessError(1, ["ollama", "stop", "qwen2.5:3b-instruct"]),
    subprocess.TimeoutExpired(["ollama", "stop", "qwen2.5:3b-instruct"], 30),
])
def test_stop_failure_is_logged_without_aborting_startup(error, running_models):
    opt = ConfigAssistant(ollama_model="qwen3.5:4b", pre_classify_model="qwen2.5:3b-instruct",
                          routing_backend="decision", pre_classify_backend="decision")
    logger = Mock()
    running_models[1].read.return_value = json.dumps({"models": [{"name": "qwen2.5:3b-instruct"}]}).encode()
    with patch("main.subprocess.run", side_effect=error), patch("main.logger", logger):
        _stop_ollama_model(opt)
    logger.warning.assert_called_once()


@pytest.mark.parametrize("payload", [b"invalid json", b"{}", b"[]", b'{"models": null}'])
def test_invalid_running_models_response_does_not_stop_any_model(payload, running_models):
    opt = ConfigAssistant(pre_classify_model="qwen2.5:3b-instruct", pre_classify_backend="decision")
    running_models[1].read.return_value = payload
    with patch("main.subprocess.run") as stop, patch("main.logger") as logger:
        _stop_ollama_model(opt)
    stop.assert_not_called()
    logger.warning.assert_called_once()
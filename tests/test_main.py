import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from config import ConfigAssistant
from main import _stop_ollama_model, preload_ollama_models


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
        routing_backend="laya", pre_classify_backend="laya",
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
    ("laya", "laya", False),
    ("laya", "llm", True),
    # No dedicated routing_model: routing falls back to the main model, so the
    # small model is warmed only when the pre-classifier itself needs it.
    ("crew", "llm", True),
    ("llm", "laya", False),
    ("crew", "laya", False),
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


def test_stop_stops_active_model_when_requested(running_models):
    opt = ConfigAssistant(
        routing_backend="laya", pre_classify_backend="laya",
        ollama_model="same-model", pre_classify_model="same-model",
    )
    running_models[1].read.return_value = json.dumps({"models": [{"model": "same-model:latest"}]}).encode()
    with patch("main.subprocess.run") as stop:
        _stop_ollama_model(opt)
    stop.assert_called_once()
    assert stop.call_args.args[0] == ["ollama", "stop", "same-model"]


@pytest.mark.parametrize("models", [[], [{"name": "another-model"}], [{"name": "qwen2.5:3b-instruct-ctx16384"}]])
def test_stop_skips_models_not_active_on_ollama(models, running_models):
    opt = ConfigAssistant(pre_classify_model="qwen2.5:3b-instruct", pre_classify_backend="laya")
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
                          routing_backend="laya", pre_classify_backend="laya")
    logger = Mock()
    running_models[1].read.return_value = json.dumps({"models": [{"name": "qwen2.5:3b-instruct"}]}).encode()
    with patch("main.subprocess.run", side_effect=error), patch("main.logger", logger):
        _stop_ollama_model(opt)
    logger.warning.assert_called_once()


@pytest.mark.parametrize("payload", [b"invalid json", b"{}", b"[]", b'{"models": null}'])
def test_invalid_running_models_response_does_not_stop_any_model(payload, running_models):
    opt = ConfigAssistant(pre_classify_model="qwen2.5:3b-instruct", pre_classify_backend="laya")
    running_models[1].read.return_value = payload
    with patch("main.subprocess.run") as stop, patch("main.logger") as logger:
        _stop_ollama_model(opt)
    stop.assert_not_called()
    logger.warning.assert_called_once()
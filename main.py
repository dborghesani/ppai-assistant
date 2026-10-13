import asyncio
import json
import logging
import shutil
import subprocess
import os

os.environ["OTEL_SDK_DISABLED"] = "true"

# fake openAI api key
import webbrowser
from datetime import datetime, timezone
from logging import warning
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import structlog
from agents.automotive_agent import AutomotiveAgent
from config import ConfigAssistant
from crewai_core.printer import set_suppress_console_output
from managers.database_manager import DatabaseManager
from managers.source_manager import SourceManager
from managers.knowledge_manager import KnowledgeManager
from omegaconf import OmegaConf
from ui.bridge import VehicleBridge
from ui.web_server import WebUIServer

os.environ["OPENAI_API_KEY"] = "sk-fakeapikey"

logging.basicConfig(level=logging.INFO)
# Silence noisy third-party libraries
logging.getLogger("transitions").setLevel(logging.WARNING)
logging.getLogger("amqtt").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

logger = structlog.get_logger()


def browser_already_running() -> bool:
    if not Path("/proc").is_dir():
        return False
    browser_processes = {"msedge", "firefox", "chrome", "chromium", "chromium-browser"}
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            process_name = (process_dir / "comm").read_text().strip()
        except OSError:
            continue
        if process_name in browser_processes:
            return True
    return False


def open_browser_tab(url: str) -> None:
    if browser_already_running():
        logger.info("Browser already running; reusing the existing UI tab", url=url)
        return

    browser = next(
        (
            path
            for name in (
                "microsoft-edge-stable",
                "microsoft-edge",
                "google-chrome",
                "chromium",
                "chromium-browser",
                "firefox",
            )
            if (path := shutil.which(name)) is not None
        ),
        None,
    )
    if browser is None:
        webbrowser.open(url, new=2)
        return
    subprocess.Popen(
        [browser, "--new-tab", url],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_ollama_model(opt: ConfigAssistant) -> None:
    if not opt.pre_classify_model:
        return
    model = opt.pre_classify_model.removeprefix("ollama/")
    model_names = {model}
    if ":" not in model.rsplit("/", 1)[-1]:
        model_names.add(f"{model}:latest")
    try:
        request = Request(f"http://{opt.ollama_host}:{opt.ollama_port}/api/ps")
        with urlopen(request, timeout=min(opt.ollama_timeout, 30)) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("Invalid Ollama running-models response")
        if not any(
            isinstance(entry, dict)
            and (entry.get("name") in model_names or entry.get("model") in model_names)
            for entry in payload["models"]
        ):
            return
        subprocess.run(
            ["ollama", "stop", model],
            env={**os.environ, "OLLAMA_HOST": f"http://{opt.ollama_host}:{opt.ollama_port}"},
            capture_output=True,
            text=True,
            check=True,
            timeout=min(opt.ollama_timeout, 30),
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        logger.warning("Unable to stop pre-classification Ollama model", model=model, error=str(error))
    else:
        logger.info("Stopped pre-classification Ollama model", model=model)


def _warmup_ollama_model(
    opt: ConfigAssistant,
    raw_model: str,
    *,
    configure_context_window: bool = False,
) -> str:
    """Warm a model, optionally creating a custom-context-window variant first."""
    if not configure_context_window or "-ctx" in raw_model or opt.context_window_size <= 4096:
        model_to_use = raw_model
    else:
        model_to_use = f"{raw_model}-ctx{opt.context_window_size}"
        create_req = Request(
            url=f"http://{opt.ollama_host}:{opt.ollama_port}/api/create",
            data=json.dumps(
                {
                    "model": model_to_use,
                    "from": raw_model,
                    "parameters": {
                        "num_ctx": opt.context_window_size,
                    },
                    "stream": False,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(create_req, timeout=opt.ollama_timeout):
                logger.info(
                    "Configured Ollama context window",
                    base_model=raw_model,
                    model=model_to_use,
                    num_ctx=opt.context_window_size,
                )
        except Exception as error:
            logger.warning(
                "Unable to create model with custom context window via Ollama API, using base model",
                base_model=raw_model,
                error=str(error),
            )
            model_to_use = raw_model

    candidates = [model_to_use]
    if model_to_use != raw_model:
        candidates.append(raw_model)
    for candidate in candidates:
        request = Request(
            url=f"http://{opt.ollama_host}:{opt.ollama_port}/api/generate",
            data=json.dumps(
                {
                    "model": candidate,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": "30m",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=opt.ollama_timeout):
                logger.info("Ollama model loaded", model=candidate)
            return candidate
        except (URLError, TimeoutError, OSError) as error:
            logger.warning("Unable to warm up Ollama model", model=candidate, error=str(error))

    return raw_model


def preload_ollama_models(opt: ConfigAssistant) -> None:
    """Warm the configured Ollama models needed by the selected backends."""
    opt.ollama_model = _warmup_ollama_model(
        opt,
        opt.ollama_model,
        configure_context_window=True,
    )

    if "llm" in {opt.pre_classify_backend, opt.routing_backend}:
        if opt.pre_classify_model:
            _warmup_ollama_model(opt, opt.pre_classify_model.removeprefix("ollama/"))
    else:
        _stop_ollama_model(opt)


async def main(opt: ConfigAssistant):

    logger.info("Starting Automotive AI Agent...")

    set_suppress_console_output(not opt.crewai_verbose)

    # initialize LLM (Ollama example)
    # ensure ollama is running: ollama serve
    # Resolve the actual model name (possibly a custom-context-window variant) once here,
    # and write it back so every LLM client built downstream (crewai and conversational) agrees.
    await asyncio.to_thread(preload_ollama_models, opt)

    data_event_queue: asyncio.Queue = asyncio.Queue()
    measurement_event_queue: asyncio.Queue = asyncio.Queue()

    # Event flow:
    # SourceManager -> data_event_queue -> DatabaseManager
    # DatabaseManager -> measurement_event_queue -> KnowledgeManager
    # KnowledgeManager -> knowledge_event_queue -> AutomotiveAgent

    # initialize database manager for data storage and retrieve
    database_manager = DatabaseManager(data_event_queue, opt, measurement_event_queue)
    database_manager.app_start_timestamp = datetime.now(timezone.utc).isoformat()  # type: ignore[assignment]

    agent = AutomotiveAgent(tts_manager=None, opt=opt)
    if agent.manual_manager is not None:
        await agent.manual_manager.prepare()

    # initialize knowledge manager to extract knowledge from data
    knowledge_manager = KnowledgeManager(
        database_manager=database_manager,
        data_event_queue=measurement_event_queue,
        knowledge_event_queue=agent.event_queue,
        opt=opt,
    )
    agent.set_knowledge_facts_provider(lambda: dict(knowledge_manager.context))
    # the message queue mirrors its unread count into the knowledge base
    agent.messages_task.update_knowledge = lambda name, values: (
        knowledge_manager.event_queue.put_nowait(
            {"type": "measurements_updated", "name": name, "values": values}
        )
    )
    # publish the known baseline (doors closed, lights off, ...) before any
    # real telemetry source starts feeding data
    await knowledge_manager.seed_default_knowledge()

    # initialize source manager
    source_manager = SourceManager(data_event_queue=data_event_queue, opt=opt)

    bridge = VehicleBridge(
        agent=agent,
        loop=asyncio.get_running_loop(),
        database_manager=database_manager,
        knowledge_manager=knowledge_manager,
        stt_manager=None,
        opt=opt,
    )
    web_ui = WebUIServer(
        bridge=bridge,
        host=opt.web_ui_host,
        port=opt.web_ui_port,
    )
    await web_ui.start()
    if opt.web_ui_open_browser:
        await asyncio.to_thread(open_browser_tab, web_ui.url)

    tasks = [
        asyncio.create_task(agent.run()),
        asyncio.create_task(knowledge_manager.run()),
        asyncio.create_task(database_manager.run()),
    ]
    if opt.is_socket_enabled:
        tasks.append(asyncio.create_task(source_manager.run_socket()))
    if opt.mqtt_enabled:
        if opt.mqtt_embedded_broker:
            tasks.append(asyncio.create_task(source_manager.run_embedded_broker()))
        tasks.append(asyncio.create_task(source_manager.run_mqtt()))
    if opt.data_replay_folder:
        tasks.append(asyncio.create_task(source_manager.run_replay()))
    try:
        if opt.tts_enabled:
            from voice.tts_manager import TTSManager

            agent.tts_manager = await asyncio.to_thread(
                TTSManager,
                enabled=opt.tts_enabled,
                hf_repo=opt.tts_hf_repo,
                device=opt.tts_device,
                n_q=opt.tts_n_q,
                cfg_coef=opt.tts_cfg_coef,
            )
        if opt.stt_enabled:
            from voice.stt_manager import STTManager

            stt_manager = await asyncio.to_thread(
                STTManager,
                enabled=opt.stt_enabled,
                hf_repo=opt.stt_hf_repo,
                device=opt.stt_device,
            )
            bridge.set_stt_manager(stt_manager)
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await web_ui.close()
        bridge.close()
        await asyncio.to_thread(database_manager.close)


if __name__ == "__main__":
    opt_cli = OmegaConf.from_cli()
    opt_default = OmegaConf.structured(ConfigAssistant())
    merged = OmegaConf.merge(opt_default, opt_cli)
    opt = OmegaConf.structured(merged)

    try:
        asyncio.run(main(opt))
    except KeyboardInterrupt:
        logger.info("Application interrupted.")

    logger.info("Application exited.")

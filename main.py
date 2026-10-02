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
from data.database_manager import DatabaseManager
from data.source_manager import SourceManager
from managers.knowledge_manager import KnowledgeManager
from omegaconf import OmegaConf
from ui.bridge import VehicleBridge
from ui.web_server import WebUIServer
from voice.stt_manager import STTManager
from voice.tts_manager import TTSManager

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


def _browser_already_running() -> bool:
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


def _open_browser_tab(url: str) -> None:
    if _browser_already_running():
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


def _load_ollama_model(opt: ConfigAssistant) -> str:
    """Resolve the model name to use (creating a custom-context-window variant on Ollama
    if needed) and warm it up so the first real request isn't slowed by a cold start."""
    raw_model = opt.ollama_model.removeprefix("ollama/")
    if "-ctx" in raw_model or opt.context_window_size <= 4096:
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

    request = Request(
        url=f"http://{opt.ollama_host}:{opt.ollama_port}/api/generate",
        data=json.dumps(
            {
                "model": model_to_use,
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
            logger.info("Ollama model loaded", model=model_to_use)
    except (URLError, TimeoutError, OSError) as error:
        logger.warning(
            "Unable to warm up Ollama model", model=model_to_use, error=str(error)
        )

    return f"ollama/{model_to_use}"


async def main(opt: ConfigAssistant):

    logger.info("Starting Automotive AI Agent...")

    set_suppress_console_output(not opt.crewai_verbose)

    # initialize LLM (Ollama example)
    # ensure ollama is running: ollama serve
    # Resolve the actual model name (possibly a custom-context-window variant) once here,
    # and write it back so every LLM client built downstream (crewai and conversational) agrees.
    opt.ollama_model = await asyncio.to_thread(_load_ollama_model, opt)

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

    # initialize knowledge manager to extract knowledge from data
    knowledge_manager = KnowledgeManager(
        database_manager=database_manager,
        data_event_queue=measurement_event_queue,
        knowledge_event_queue=agent.event_queue,
        opt=opt,
    )
    agent.set_knowledge_context_provider(
        lambda: list(knowledge_manager.context.values())
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
        await asyncio.to_thread(_open_browser_tab, web_ui.url)

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
            agent.tts_manager = await asyncio.to_thread(
                TTSManager,
                enabled=opt.tts_enabled,
                hf_repo=opt.tts_hf_repo,
                device=opt.tts_device,
                n_q=opt.tts_n_q,
                cfg_coef=opt.tts_cfg_coef,
            )
        if opt.stt_enabled:
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

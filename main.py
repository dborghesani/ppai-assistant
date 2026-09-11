import asyncio
from datetime import datetime, timezone
from logging import warning
import json
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen
from crewai import LLM
from config import ConfigAssistant
from automotive_agent import AutomotiveAgent
from data.database_manager import DatabaseManager
from data.source_manager import SourceManager
from knowledge_manager import KnowledgeManager
from ui.bridge import VehicleBridge
import structlog
from omegaconf import OmegaConf
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from qasync import QEventLoop
from crewai_core.printer import set_suppress_console_output
import logging

logging.basicConfig(level=logging.INFO)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

logger = structlog.get_logger()


def _load_ollama_model(opt: ConfigAssistant) -> None:
    model = opt.ollama_llm.removeprefix("ollama/")
    request = Request(
        url=f"http://{opt.ollama_host}:{opt.ollama_port}/api/generate",
        data=json.dumps({
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": "30m",
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=opt.ollama_timeout):
            logger.info("Ollama model loaded", model=model)
    except (URLError, TimeoutError, OSError) as error:
        logger.warning("Unable to warm up Ollama model", model=model, error=str(error))


async def main(opt: ConfigAssistant):

    logger.info("Starting Automotive AI Agent...")

    set_suppress_console_output(not opt.crewai_verbose)

    # initialize LLM (Ollama example)
    # ensure ollama is running: ollama serve
    await asyncio.to_thread(_load_ollama_model, opt)

    llm = LLM(
        model=opt.ollama_llm,
        base_url=f"http://{opt.ollama_host}:{opt.ollama_port}",
        timeout=opt.ollama_timeout,
    )

    data_event_queue: asyncio.Queue = asyncio.Queue()
    measurement_event_queue: asyncio.Queue = asyncio.Queue()

    # Event flow:
    # SourceManager -> data_event_queue -> DatabaseManager
    # DatabaseManager -> measurement_event_queue -> KnowledgeManager
    # KnowledgeManager -> knowledge_event_queue -> AutomotiveAgent

    # initialize database manager for data storage and retrieve
    database_manager = DatabaseManager(data_event_queue, opt, measurement_event_queue)
    database_manager.app_start_timestamp = datetime.now(timezone.utc).isoformat()  # type: ignore[assignment]

    # initialize Agent
    agent = AutomotiveAgent(llm=llm)

    # initialize knowledge manager to extract knowledge from data
    knowledge_manager = KnowledgeManager(
        database_manager=database_manager,
        data_event_queue=measurement_event_queue,
        knowledge_event_queue=agent.event_queue,
        opt=opt,
    )

    # initialize source manager
    source_manager = SourceManager(data_event_queue=data_event_queue, opt=opt)
    
    engine = QQmlApplicationEngine()
    bridge = VehicleBridge(
        agent=agent,
        loop=asyncio.get_running_loop(),
        database_manager=database_manager,
        opt=opt,
    )
    engine.rootContext().setContextProperty(
        "vehicleBridge",
        bridge,
    )
    engine.load("ui/main.qml")
    engine.warnings.connect(
        lambda warnings: [
            print(w.toString())
            for w in warnings
        ]
    )
    if not engine.rootObjects():
        raise RuntimeError(
            "Failed to load main.qml"
        )

    # run everything concurrently
    tasks = [
        asyncio.create_task(agent.run()),
        asyncio.create_task(knowledge_manager.run()),
        asyncio.create_task(database_manager.run()),
        asyncio.create_task(source_manager.run()),
    ]
    if opt.data_replay_folder:
        tasks.append(asyncio.create_task(source_manager.run_replay()))
    await asyncio.gather(*tasks)

if __name__ == "__main__":

    opt_cli = OmegaConf.from_cli()
    opt_default = OmegaConf.structured(ConfigAssistant())
    merged = OmegaConf.merge(opt_default, opt_cli)
    opt = OmegaConf.structured(merged)

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    with loop:
        loop.create_task(main(opt))
        loop.run_forever()

    logger.info("Application exited.")
import asyncio
from logging import warning
import sys
from crewai import LLM
from config import ConfigAssistant
from agents import AutomotiveAgent
from ui import bridge
from ui.bridge import VehicleBridge
import structlog
from omegaconf import OmegaConf
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from qasync import QEventLoop
from crewai_core.printer import set_suppress_console_output

logger = structlog.get_logger()

async def main(opt: ConfigAssistant):

    logger.info("Starting Automotive AI Agent...")

    set_suppress_console_output(not opt.crewai_verbose)
    
    # initialize LLM (Ollama example)
    # ensure ollama is running: ollama serve
    llm = LLM(
        model=opt.ollama_llm,
        base_url=f"http://{opt.ollama_host}:{opt.ollama_port}",
        timeout=opt.ollama_timeout,
    )
    
    # initialize Agent
    agent = AutomotiveAgent(llm)

    shutdown_event = asyncio.Event()
    engine = QQmlApplicationEngine()
    bridge = VehicleBridge(
        agent=agent,
        loop=asyncio.get_running_loop(),
        shutdown_event=shutdown_event
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
    ]

    await shutdown_event.wait()
    
    agent.is_active = False
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Shutting down Automotive AI Agent...")

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
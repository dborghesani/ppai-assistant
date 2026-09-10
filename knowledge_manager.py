from typing import Any, List
from dataclasses import dataclass
from omegaconf import OmegaConf
import structlog
import asyncio

from data.database_manager import DatabaseManager
from config import ConfigAssistant
from data.skill_map import SkillMap
from skill_manager import SkillType

class KnowledgeManager:
    event_skill_map: dict[str, List[SkillType]] = {}

    # context: dictionary containing relevant information for processing events
    
    def __init__(self, database_manager: DatabaseManager, event_queue: asyncio.Queue, opt: ConfigAssistant):
        self.logger = structlog.get_logger()
        self.database_manager = database_manager
        self.event_queue = event_queue
        self.opt = opt
        self.skill_map = SkillMap()

    async def run(self):
        while True:
            event = await self.event_queue.get()
            if event["event"] == "measure_updated":
                await self.process(event)
            await asyncio.sleep(0.1)

    async def process(self, event: dict):
        measure = event["measure"]
        value = event["value"]

        self.logger.info("processing event", measure=measure, value=value)

if __name__ == "__main__":
    logger = structlog.get_logger()
    opt_cli = OmegaConf.from_cli()
    opt_default = OmegaConf.structured(ConfigAssistant())
    merged = OmegaConf.merge(opt_default, opt_cli)
    opt = OmegaConf.structured(merged)

    database_manager = DatabaseManager(None, opt)
    logger.info("Database manager initialized", database_manager=database_manager)

    name = "DriverState"
    measure = "fatigue_level"
    range = "-1h"

    # extract knowledge from the database
    if database_manager:
        mean_value = database_manager.mean(name, measure, range)
        max_value = database_manager.max(name, measure, range)
        min_value = database_manager.min(name, measure, range)
        stddev_value = database_manager.stddev(name, measure, range)
        count_value = database_manager.count(name, measure, range)
        logger.info("Extracted knowledge from the database",
                    mean_value=mean_value,
                    max_value=max_value,
                    min_value=min_value,
                    stddev_value=stddev_value,
                    count_value=count_value)
        derivative_value = database_manager.derivative(name, measure, range)
        mean_derivative_value = database_manager.mean_derivative(name, measure, range)
        stddev_derivative_value = database_manager.stddev_derivative(name, measure, range)
        logger.info("Extracted trend knowledge from the database",
                    derivative_value=derivative_value,
                    mean_derivative_value=mean_derivative_value,
                    stddev_derivative_value=stddev_derivative_value)

        if derivative_value is not None:
            if derivative_value > 0:
                logger.info(f"{measure} has increasing trend")
            elif derivative_value < 0:
                logger.info(f"{measure} has decreasing trend")
            else:
                logger.info(f"{measure} has stable trend")

        if mean_derivative_value is not None:
            if mean_derivative_value > 0:
                logger.info(f"{measure} has increasing trend on average")
            elif mean_derivative_value < 0:
                logger.info(f"{measure} has decreasing trend on average")
            else:
                logger.info(f"{measure} has stable trend on average")

        if stddev_derivative_value is not None:
            if stddev_derivative_value > 0:
                logger.info(f"{measure} has high variability in the trend")
            elif stddev_derivative_value < 0:
                logger.info(f"{measure} has decreasing variability in the trend")
            else:
                logger.info(f"{measure} has stable trend with no variability")

    database_manager.close()

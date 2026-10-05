import asyncio

from omegaconf import OmegaConf

from config import ConfigAssistant
from managers.vehicle_manual_manager import VehicleManualManager


async def main(opt: ConfigAssistant) -> None:
    manager = VehicleManualManager(opt)
    await manager.prepare()
    print(f"Vehicle manual index ready: {manager.manual_path}")


if __name__ == "__main__":
    options = OmegaConf.merge(OmegaConf.structured(ConfigAssistant()), OmegaConf.from_cli())
    opt = OmegaConf.to_object(options)
    if not isinstance(opt, ConfigAssistant):
        raise TypeError("Expected ConfigAssistant options")
    asyncio.run(main(opt))
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from managers.vehicle_manual_manager import VehicleManualManager


async def response_context(
    manager: "VehicleManualManager | None", question: str, history: list[dict[str, str]],
) -> tuple[str, str]:
    from managers.vehicle_manual_manager import VehicleManualManager

    if manager is None:
        return await VehicleManualManager.unavailable_response_context(question, history)
    return await manager.response_context(question, history)
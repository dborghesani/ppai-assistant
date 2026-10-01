from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import structlog
from agents.agents_dataclasses import ActionType, SkillType
from data.assistant_dataclasses import DetectedObjects, VehicleState
from managers.action_manager import ActionManager
from managers.knowledge_manager import KnowledgeManager
from ui.bridge import VehicleBridge


class FakeDatabase:
    def __init__(self, state: VehicleState | None = None) -> None:
        self.current_state = {
            "VehicleState": state or VehicleState(),
            "DetectedObjects": DetectedObjects(),
        }
        self.writes: list[tuple[str, str, Any]] = []

    def write_measure(self, name: str, measure: str, value) -> None:
        self.writes.append((name, measure, value))
        setattr(self.current_state[name], measure, value)


def make_bridge(state: VehicleState | None = None):
    bridge = SimpleNamespace(database_manager=FakeDatabase(state), _emit=Mock())
    bridge.apply_vehicle_action = MethodType(VehicleBridge.apply_vehicle_action, bridge)
    bridge.dumpVehicleState = MethodType(VehicleBridge.dumpVehicleState, bridge)
    bridge.intChanged = MethodType(VehicleBridge.intChanged, bridge)
    bridge.dumpDetectedObjectsState = MethodType(
        VehicleBridge.dumpDetectedObjectsState, bridge
    )
    return bridge


def test_action_manager_updates_all_window_controls():
    bridge = make_bridge()
    agent = type("FakeAgent", (), {"set_assistant_status": Mock()})()
    manager = ActionManager(agent)
    manager.on_action = bridge.apply_vehicle_action

    manager.handle_decision(ActionType.OPEN_WINDOWS)

    state = bridge.database_manager.current_state["VehicleState"]
    assert state.window_open_front_left is True
    assert state.window_open_front_right is True
    assert state.window_open_rear_left is True
    assert state.window_open_rear_right is True
    bridge._emit.assert_called_once_with(
        "vehicleStateChanged",
        {
            "window_open_front_left": True,
            "window_open_front_right": True,
            "window_open_rear_left": True,
            "window_open_rear_right": True,
        },
    )


def test_temperature_fan_and_volume_actions_change_vehicle_state():
    bridge = make_bridge(VehicleState(internal_temperature=24.0, fan_speed=4, audio_volume=35))

    bridge.apply_vehicle_action(ActionType.INCREASE_TEMPERATURE, {})
    bridge.apply_vehicle_action(ActionType.DECREASE_FAN_SPEED, {})
    bridge.apply_vehicle_action(ActionType.INCREASE_AUDIO_VOLUME, {})

    state = bridge.database_manager.current_state["VehicleState"]
    assert state.internal_temperature == 25.0
    assert state.fan_speed == 3
    assert state.audio_volume == 40


def test_restrictive_adas_profile_updates_simulated_controls():
    bridge = make_bridge(
        VehicleState(adas_target_speed=70, following_distance_level=3)
    )

    bridge.apply_vehicle_action(ActionType.APPLY_RESTRICTIVE_ADAS_PROFILE, {})

    state = bridge.database_manager.current_state["VehicleState"]
    assert state.adas_target_speed == 60
    assert state.following_distance_level == 4
    assert state.adaptive_cruise_control_on is True
    assert state.lane_keep_assist_enabled is True
    assert state.blind_spot_monitor is True


def test_vehicle_state_snapshot_fills_unreported_fields_from_defaults():
    bridge = make_bridge(VehicleState(window_open_front_left=True, sunroof_open=None))

    snapshot = bridge.dumpVehicleState()

    assert snapshot["window_open_front_left"] is True
    assert snapshot["sunroof_open"] is False
    assert snapshot["fan_speed"] == 3
    assert snapshot["audio_volume"] == 20


def test_dangerous_object_slider_keeps_integer_in_detected_objects():
    bridge = make_bridge()

    bridge.intChanged("DetectedObjects", "dangerous_objects_around", 4)

    value = bridge.database_manager.current_state[
        "DetectedObjects"
    ].dangerous_objects_around
    assert value == 4
    assert isinstance(value, int)
    assert bridge.dumpDetectedObjectsState()["dangerous_objects_around"] == 4


def test_dangerous_object_count_becomes_boolean_knowledge():
    assert KnowledgeManager._field_skills(
        "DetectedObjects", "dangerous_objects_around"
    ) == (SkillType.DRIVING,)

    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.event_queue = asyncio.Queue()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=True)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager._last_speed_status = None
    manager.on_context_updated = None
    manager._is_first_batch = False

    async def process_counts():
        await manager.process_pending(
            {("DetectedObjects", "dangerous_objects_around"): 4}
        )
        assert manager.context["DetectedObjects.dangerous_objects_around"] == (
            "A dangerous object is detected around the vehicle."
        )
        event = await manager.knowledge_event_queue.get()
        assert event.event_value["DetectedObjects.dangerous_objects_around"] is True

        await manager.process_pending(
            {("DetectedObjects", "dangerous_objects_around"): 0}
        )
        assert manager.context["DetectedObjects.dangerous_objects_around"] == (
            "No dangerous objects are detected around the vehicle."
        )
        event = await manager.knowledge_event_queue.get()
        assert event.event_value["DetectedObjects.dangerous_objects_around"] is False

    asyncio.run(process_counts())


def test_knowledge_event_marks_changed_fact_ahead_of_supporting_facts():
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.context = {
        "DriverPhysicalState.fatigue_level": "Fatigue level is high.",
        "DetectedObjects.children_inside": "2 children are detected inside the vehicle.",
    }
    manager.opt = SimpleNamespace(use_laya=False)
    manager.knowledge_event_queue = asyncio.Queue()

    async def send_update():
        await manager._notify_agent({"DriverPhysicalState.fatigue_level": "high"})
        return await manager.knowledge_event_queue.get()

    event = asyncio.run(send_update())

    assert event.context == [
        "Changed just now: Fatigue level is high.",
        "2 children are detected inside the vehicle.",
    ]
    assert event.event_value == {"DriverPhysicalState.fatigue_level": "high"}
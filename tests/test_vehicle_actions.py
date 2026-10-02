from __future__ import annotations

import asyncio
import json
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import structlog
from agents.agents_dataclasses import ActionType, SkillType
from data.assistant_dataclasses import DetectedObjects, DriverPreferences, VehicleState
from data.database_manager import DatabaseManager
from managers.action_manager import ActionManager
from managers.knowledge_manager import KnowledgeManager
from data.source_manager import SourceManager
from ui.bridge import VehicleBridge


class FakeDatabase:
    def __init__(self, state: VehicleState | None = None) -> None:
        self.current_state = {
            "VehicleState": state or VehicleState(),
            "DetectedObjects": DetectedObjects(),
            "DriverPreferences": DriverPreferences(),
        }
        self.writes: list[tuple[str, str, Any]] = []

    def write_measure(self, name: str, measure: str, value) -> None:
        self.writes.append((name, measure, value))
        setattr(self.current_state[name], measure, value)


def make_bridge(state: VehicleState | None = None):
    agent = SimpleNamespace(
        driver_preferences=DriverPreferences(),
        event_queue=asyncio.Queue(),
        is_listening=True,
    )
    bridge = SimpleNamespace(
        agent=agent,
        database_manager=FakeDatabase(state),
        knowledge_manager=SimpleNamespace(context={}),
        _emit=Mock(),
    )
    bridge.apply_vehicle_action = MethodType(VehicleBridge.apply_vehicle_action, bridge)
    bridge.dumpVehicleState = MethodType(VehicleBridge.dumpVehicleState, bridge)
    bridge.dumpDriverPreferences = MethodType(
        VehicleBridge.dumpDriverPreferences, bridge
    )
    bridge.setDriverPreference = MethodType(
        VehicleBridge.setDriverPreference, bridge
    )
    bridge.userInput = MethodType(VehicleBridge.userInput, bridge)
    bridge.send_event = MethodType(VehicleBridge.send_event, bridge)
    bridge.get_dataclass_from_ui_event_type = MethodType(
        VehicleBridge.get_dataclass_from_ui_event_type, bridge
    )
    bridge.intChanged = MethodType(VehicleBridge.intChanged, bridge)
    bridge.boolChanged = MethodType(VehicleBridge.boolChanged, bridge)
    bridge.dumpDetectedObjectsState = MethodType(
        VehicleBridge.dumpDetectedObjectsState, bridge
    )
    return bridge


def test_driver_preferences_are_persisted_and_broadcast():
    bridge = make_bridge()

    bridge.setDriverPreference("preferred_cabin_temperature", 31)
    bridge.setDriverPreference("preferred_music", "  classic 1970s rock  ")

    assert bridge.dumpDriverPreferences() == {
        "preferred_cabin_temperature": 30.0,
        "preferred_music": "classic 1970s rock",
    }
    assert bridge.database_manager.writes == [
        ("DriverPreferences", "preferred_cabin_temperature", 30.0),
        ("DriverPreferences", "preferred_music", "classic 1970s rock"),
    ]
    assert bridge._emit.call_args_list[-1].args == (
        "driverPreferencesChanged",
        {
            "preferred_cabin_temperature": 30.0,
            "preferred_music": "classic 1970s rock",
        },
    )
    bridge.setDriverPreference("preferred_music", "   ")
    assert bridge.database_manager.writes[-1] == (
        "DriverPreferences",
        "preferred_music",
        "",
    )


def test_user_input_uses_driver_preferences_from_knowledge_context():
    bridge = make_bridge()
    bridge.setDriverPreference("preferred_cabin_temperature", 21.5)
    bridge.setDriverPreference("preferred_music", "classic 1970s rock")
    bridge.knowledge_manager.context = {
        "VehicleState.doors_locked": "The vehicle doors are locked.",
        "DriverPreferences.preferred_cabin_temperature": (
            "The driver's preferred cabin temperature is 21.5 C."
        ),
        "DriverPreferences.preferred_music": (
            "The driver's music preference is classic 1970s rock."
        ),
    }

    bridge.userInput("What music would you recommend?")

    event = bridge.agent.event_queue.get_nowait()
    assert event.context == [
        "The vehicle doors are locked.",
        "The driver's preferred cabin temperature is 21.5 C.",
        "The driver's music preference is classic 1970s rock.",
    ]


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


def test_privacy_mode_actions_update_simulated_state():
    bridge = make_bridge(VehicleState(privacy_mode=False))

    bridge.apply_vehicle_action(ActionType.ENABLE_PRIVACY_MODE, {})
    assert bridge.database_manager.current_state["VehicleState"].privacy_mode is True

    bridge.apply_vehicle_action(ActionType.DISABLE_PRIVACY_MODE, {})
    assert bridge.database_manager.current_state["VehicleState"].privacy_mode is False


def test_vehicle_state_snapshot_fills_unreported_fields_from_defaults():
    bridge = make_bridge(VehicleState(window_open_front_left=True, sunroof_open=None))

    snapshot = bridge.dumpVehicleState()

    assert snapshot["window_open_front_left"] is True
    assert snapshot["sunroof_open"] is False
    assert snapshot["privacy_mode"] is False
    assert snapshot["fan_speed"] == 3
    assert snapshot["audio_volume"] == 20
    assert bridge.dumpDetectedObjectsState()["people_inside"] == 0


def test_dangerous_object_slider_keeps_integer_in_detected_objects():
    bridge = make_bridge()

    bridge.intChanged("DetectedObjects", "dangerous_objects_around", 4)

    value = bridge.database_manager.current_state[
        "DetectedObjects"
    ].dangerous_objects_around
    assert value == 4
    assert isinstance(value, int)
    assert bridge.dumpDetectedObjectsState()["dangerous_objects_around"] == 4


def test_internal_object_payload_counts_people_and_children_separately():
    manager = SourceManager.__new__(SourceManager)
    manager.data_event_queue = asyncio.Queue()
    manager.logger = structlog.get_logger()

    asyncio.run(
        manager.process_message(
            json.dumps(
                {
                    "internal_objects": [
                        {"category": "adult"},
                        {"category": "child"},
                        {"category": "dog"},
                    ]
                }
            )
        )
    )

    detected = manager.data_event_queue.get_nowait()["data"]
    assert detected.people_inside == 2
    assert detected.children_inside == 1
    assert detected.animal_inside is True


def test_privacy_toggle_and_people_inside_control_update_dataclasses():
    bridge = make_bridge()

    bridge.boolChanged("VehicleState", "privacy_mode", True)
    bridge.intChanged("DetectedObjects", "people_inside", 3)

    assert bridge.database_manager.current_state["VehicleState"].privacy_mode is True
    detected = bridge.database_manager.current_state["DetectedObjects"]
    assert detected.people_inside == 3
    assert bridge.database_manager.writes == [
        ("VehicleState", "privacy_mode", True),
        ("DetectedObjects", "people_inside", 3),
    ]


def test_driver_preferences_become_non_notifying_knowledge():
    assert KnowledgeManager._field_skills("VehicleState", "privacy_mode") == (
        SkillType.DRIVING,
    )
    assert KnowledgeManager._field_skills("DetectedObjects", "people_inside") == (
        SkillType.WELLBEING,
        SkillType.DRIVING,
    )

    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.event_queue = asyncio.Queue()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=False)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager._last_speed_status = None
    manager.on_context_updated = None
    manager._is_first_batch = False

    async def update_knowledge():
        await manager.process_pending(
            {
                ("DriverPreferences", "preferred_cabin_temperature"): 21.5,
                ("DriverPreferences", "preferred_music"): "classic 1970s rock",
            }
        )
        assert manager.context["DriverPreferences.preferred_cabin_temperature"] == (
            "The driver's preferred cabin temperature is 21.5 C."
        )
        assert manager.context["DriverPreferences.preferred_music"] == (
            "The driver's music preference is classic 1970s rock."
        )
        assert manager.knowledge_event_queue.empty()

    asyncio.run(update_knowledge())


def test_privacy_mode_and_people_inside_are_readable_knowledge():
    assert KnowledgeManager._field_skills("VehicleState", "privacy_mode") == (
        SkillType.DRIVING,
    )
    assert KnowledgeManager._field_skills("DetectedObjects", "people_inside") == (
        SkillType.WELLBEING,
        SkillType.DRIVING,
    )

    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.event_queue = asyncio.Queue()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=False)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager._last_speed_status = None
    manager.on_context_updated = None
    manager._is_first_batch = False

    async def update_knowledge():
        await manager.process_pending(
            {
                ("VehicleState", "privacy_mode"): False,
                ("DetectedObjects", "people_inside"): 0,
            }
        )
        while not manager.knowledge_event_queue.empty():
            await manager.knowledge_event_queue.get()
        await manager.process_pending(
            {
                ("DetectedObjects", "people_inside"): 2,
            }
        )
        assert manager.context["DetectedObjects.people_inside"] == (
            "Multiple people are detected inside the vehicle."
        )
        return [
            await manager.knowledge_event_queue.get(),
            await manager.knowledge_event_queue.get(),
        ]

    events = asyncio.run(update_knowledge())
    assert {
        key: value
        for event in events
        for key, value in event.event_value.items()
    } == {
        "DetectedObjects.people_inside": 2,
    }
    assert {event.skill for event in events} == {SkillType.DRIVING, SkillType.WELLBEING}
    driving_event = next(event for event in events if event.skill is SkillType.DRIVING)
    assert driving_event.event_value == {
        "DetectedObjects.people_inside": 2,
    }
    assert "Privacy mode is off." in driving_event.context


def test_database_restores_driver_preferences_for_initial_knowledge():
    database = DatabaseManager.__new__(DatabaseManager)
    database.opt = SimpleNamespace(influxdb_bucket="test-bucket")
    database.logger = structlog.get_logger()
    database.current_state = {}
    database.run_records = Mock(
        return_value=[
            {"_field": "preferred_cabin_temperature", "_value": 21.5},
            {"_field": "preferred_music", "_value": "classic 1970s rock"},
        ]
    )

    database._load_driver_preferences()

    restored = database.current_state["DriverPreferences"]
    assert restored.preferred_cabin_temperature == 21.5
    assert restored.preferred_music == "classic 1970s rock"

    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.database_manager = database
    captured: dict[tuple[str, str], Any] = {}

    async def capture_seeded_knowledge(pending):
        captured.update(pending)

    manager.process_pending = capture_seeded_knowledge
    asyncio.run(manager.seed_default_knowledge())

    assert captured[("DriverPreferences", "preferred_cabin_temperature")] == 21.5
    assert captured[("DriverPreferences", "preferred_music")] == "classic 1970s rock"


def test_database_write_measure_coerces_int_for_float_field():
    database = DatabaseManager.__new__(DatabaseManager)
    database.current_state = {"VehicleState": VehicleState()}
    database.measurement_event_queue = None
    database.logger = Mock()
    database.write_point = Mock()

    database.write_measure("VehicleState", "internal_temperature", 25)

    assert database.current_state["VehicleState"].internal_temperature == 25.0
    assert type(database.current_state["VehicleState"].internal_temperature) is float
    line = database.write_point.call_args.args[0].to_line_protocol()
    assert "internal_temperature=25" in line
    assert "internal_temperature=25i" not in line


def test_database_write_dataclass_coerces_int_for_float_field():
    database = DatabaseManager.__new__(DatabaseManager)
    database.current_state = {}
    database.measurement_event_queue = None
    database.logger = Mock()
    database.write_point = Mock()
    state = VehicleState(internal_temperature=25)

    database.write_dataclass(state)

    assert type(state.internal_temperature) is float
    line = database.write_point.call_args.args[0].to_line_protocol()
    assert "internal_temperature=25" in line
    assert "internal_temperature=25i" not in line


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
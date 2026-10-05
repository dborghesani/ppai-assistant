from __future__ import annotations

import asyncio
import json
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import structlog
from data.agents_dataclasses import ActionType, SkillType
from data.assistant_dataclasses import DetectedObjects, DriverPreferences, VehicleState
from managers.database_manager import DatabaseManager
from managers.action_manager import ActionManager
from managers.knowledge_manager import KnowledgeManager
from managers.source_manager import SourceManager
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
        "driver_name": "David",
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
            "driver_name": "David",
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
    bridge.setDriverPreference("driver_name", "  Alex  ")
    assert bridge.dumpDriverPreferences()["driver_name"] == "Alex"
    assert bridge.database_manager.writes[-1] == ("DriverPreferences", "driver_name", "Alex")
    bridge.setDriverPreference("driver_name", "   ")
    assert bridge.dumpDriverPreferences()["driver_name"] == "David"


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


def test_heating_fan_and_volume_actions_change_vehicle_state():
    bridge = make_bridge(VehicleState(internal_temperature=24.0, fan_speed=4, audio_volume=35))

    bridge.apply_vehicle_action(ActionType.ENABLE_HEATING, {})
    bridge.apply_vehicle_action(ActionType.DECREASE_FAN_SPEED, {})
    bridge.apply_vehicle_action(ActionType.INCREASE_AUDIO_VOLUME, {})

    state = bridge.database_manager.current_state["VehicleState"]
    assert state.internal_temperature == 24.0
    assert state.heating_on is True
    assert state.fan_speed == 3
    assert state.audio_volume == 40


@pytest.mark.parametrize(
    ("action", "field", "enabled"),
    [
        (ActionType.ENABLE_HEATING, "heating_on", True),
        (ActionType.DISABLE_HEATING, "heating_on", False),
        (ActionType.ENABLE_AIR_CONDITIONING, "air_conditioning_on", True),
        (ActionType.DISABLE_AIR_CONDITIONING, "air_conditioning_on", False),
    ],
)
def test_climate_actions_toggle_controls_without_changing_temperature(
    action, field, enabled
):
    bridge = make_bridge(
        VehicleState(internal_temperature=24.0, heating_on=not enabled,
                     air_conditioning_on=not enabled)
    )
    preference = bridge.dumpDriverPreferences()["preferred_cabin_temperature"]
    bridge.apply_vehicle_action(action, {})
    state = bridge.database_manager.current_state["VehicleState"]
    assert getattr(state, field) is enabled
    assert state.internal_temperature == 24.0
    assert state.seat_heating_on is False
    assert bridge.dumpDriverPreferences()["preferred_cabin_temperature"] == preference
    bridge._emit.assert_called_once_with("vehicleStateChanged", {field: enabled})
    manager = KnowledgeManager.__new__(KnowledgeManager)
    knowledge = manager._boolean_knowledge("VehicleState", field, enabled)
    assert knowledge == (
        f"Cabin heating is {'on' if enabled else 'off'}."
        if field == "heating_on"
        else f"The air conditioning is {'on' if enabled else 'off'}."
    )


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


@pytest.mark.parametrize(
    ("temperature", "preferred", "expected_status"),
    [
        (25.0, 27.0, "low"),
        (23.0, 21.0, "high"),
        (25.0, 22.0, "high"),
        (19.0, 22.0, "low"),
        (26.0, 22.0, "too high"),
        (18.0, 22.0, "too low"),
        (23.0, 22.0, "optimal"),
        (21.0, 22.0, "optimal"),
        (27.0, 27.0, "optimal"),
        (17.0, 17.0, "optimal"),
        (28.0, None, None),
        (28.0, float("nan"), None),
    ],
)
def test_cabin_temperature_knowledge_is_relative_to_user_preference(
    temperature, preferred, expected_status
):
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=False)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager._last_speed_status = None
    manager.on_context_updated = None
    manager._is_first_batch = True
    changes = []
    manager.on_knowledge_changed = changes.append

    async def verify():
        await manager.process_pending(
            {("VehicleState", "internal_temperature"): 22.0}
        )
        assert manager.knowledge_event_queue.empty()
        await manager.process_pending(
            {
                ("VehicleState", "internal_temperature"): temperature,
                ("DriverPreferences", "preferred_cabin_temperature"): preferred,
            }
        )
        status_key = "VehicleState.internal_temperature"
        assert "VehicleState.temperature_status" not in manager.context
        if expected_status is None:
            assert status_key not in manager.context
            assert manager.knowledge_event_queue.empty()
            assert not changes
        else:
            fact = manager.context[status_key]
            assert fact == f"Cabin temperature is {expected_status}."
            event = await manager.knowledge_event_queue.get()
            assert event.skill is SkillType.WELLBEING
            assert event.event_value == {status_key: expected_status}
            assert event.context[0] == f"Changed just now: {fact}"
            assert changes == [[fact]]

    asyncio.run(verify())


def test_preference_changes_recalculate_cabin_temperature_without_new_reading():
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=False)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager._last_speed_status = None
    manager.on_context_updated = None
    manager._is_first_batch = True

    async def verify():
        await manager.process_pending(
            {
                ("VehicleState", "internal_temperature"): 25.0,
                ("DriverPreferences", "preferred_cabin_temperature"): 25.0,
            }
        )
        assert manager.knowledge_event_queue.empty()
        for preferred, expected_status in (
            (22.0, "high"),
            (29.0, "too low"),
            (27.0, "low"),
            (25.0, "optimal"),
        ):
            await manager.process_pending(
                {("DriverPreferences", "preferred_cabin_temperature"): preferred}
            )
            event = await manager.knowledge_event_queue.get()
            assert event.skill is SkillType.WELLBEING
            assert event.event_value == {"VehicleState.internal_temperature": expected_status}
            await manager.process_pending(
                {("VehicleState", "internal_temperature"): 25.0}
            )
            assert manager.knowledge_event_queue.empty()
        await manager.process_pending(
            {("DriverPreferences", "preferred_cabin_temperature"): None}
        )
        assert "VehicleState.internal_temperature" not in manager.context
        assert "DriverPreferences.preferred_cabin_temperature" not in manager.context
        assert "VehicleState.internal_temperature" not in manager._last_evaluated
        assert manager.knowledge_event_queue.empty()

    asyncio.run(verify())


def test_reference_knowledge_is_generic_and_independent_of_batch_order(monkeypatch):
    original_metadata = KnowledgeManager._knowledge_metadata
    fan_metadata = {
        **original_metadata("VehicleState", "fan_speed"),
        "reference": ("VehicleState", "following_distance_level"),
        "reference_tolerance": 0.5,
        "reference_extreme_threshold": 1.5,
        "reference_label": "Fan speed",
    }

    def reference_metadata(name, measure):
        if (name, measure) == ("VehicleState", "fan_speed"):
            return fan_metadata
        return original_metadata(name, measure)

    monkeypatch.setattr(
        KnowledgeManager, "_knowledge_metadata", staticmethod(reference_metadata)
    )
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=True)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager._last_speed_status = None
    manager.on_context_updated = None
    manager._is_first_batch = True

    async def verify():
        await manager.process_pending(
            {
                ("VehicleState", "fan_speed"): 0,
                ("VehicleState", "following_distance_level"): 0,
            }
        )
        assert manager.context["VehicleState.fan_speed"] == "Fan speed is optimal."
        await manager.process_pending({("VehicleState", "fan_speed"): 1})
        assert manager.context["VehicleState.fan_speed"] == "Fan speed is high."
        await manager.process_pending(
            {
                ("VehicleState", "following_distance_level"): 4,
                ("VehicleState", "fan_speed"): 2,
            }
        )
        assert manager.context["VehicleState.fan_speed"] == "Fan speed is too low."
        await manager.process_pending(
            {("VehicleState", "following_distance_level"): 2}
        )
        assert manager.context["VehicleState.fan_speed"] == "Fan speed is optimal."

    asyncio.run(verify())


@pytest.mark.parametrize(
    ("speed", "limit", "expected_level"),
    [
        (50.0, 50.0, "optimal"),
        (55.0, 50.0, "high"),
        (60.0, 50.0, "high"),
        (61.0, 50.0, "too high"),
        (45.0, 50.0, "low"),
        (40.0, 50.0, "low"),
        (39.0, 50.0, "too low"),
        (60.0, None, None),
        (60.0, float("nan"), None),
    ],
)
def test_speed_uses_shared_relative_knowledge(speed, limit, expected_level):
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=False)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager.on_context_updated = None
    manager._is_first_batch = True

    async def verify():
        await manager.process_pending(
            {
                ("VehicleMotion", "speed"): speed,
                ("DetectedObjects", "traffic_signs.speed_limit"): limit,
            }
        )
        assert "VehicleMotion.speed_status" not in manager.context
        assert manager.knowledge_event_queue.empty()
        if expected_level is None:
            assert "VehicleMotion.speed" not in manager.context
        else:
            assert manager.context["VehicleMotion.speed"] == (
                f"Vehicle speed is {expected_level}. Current value is {speed:g} km/h; "
                f"reference value is {limit:g} km/h."
            )

    asyncio.run(verify())


def test_changed_speed_limit_recalculates_relative_speed_without_new_measurement():
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=False)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager.on_context_updated = None
    manager._is_first_batch = True

    async def verify():
        await manager.process_pending(
            {
                ("VehicleMotion", "speed"): 60.0,
                ("DetectedObjects", "traffic_signs.speed_limit"): 60.0,
            }
        )
        await manager.process_pending(
            {("DetectedObjects", "traffic_signs.speed_limit"): 40.0}
        )
        events = []
        while not manager.knowledge_event_queue.empty():
            events.append(manager.knowledge_event_queue.get_nowait())
        driving = next(event for event in events if event.skill is SkillType.DRIVING)
        assert driving.event_value["VehicleMotion.speed"] == "too high"
        assert any(
            fact.startswith("Changed just now: Vehicle speed is too high.")
            for fact in driving.context
        )
        assert "VehicleMotion.speed_status" not in manager.context
        await manager.process_pending({("VehicleMotion", "speed"): 61.0})
        assert manager.knowledge_event_queue.empty()
        await manager.process_pending(
            {("DetectedObjects", "traffic_signs.speed_limit"): None}
        )
        assert "VehicleMotion.speed" not in manager.context
        assert "DetectedObjects.traffic_signs.speed_limit" not in manager.context
        assert manager.knowledge_event_queue.empty()

    asyncio.run(verify())


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


def test_attention_and_fatigue_persistence_knowledge_resets_when_levels_change():
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
    manager.persistence_since = {}
    manager.persistence_notified = set()
    timestamps = iter((100.0, 129.9, 130.1, 131.0))
    manager.monotonic_clock = lambda: next(timestamps)
    manager.on_context_updated = None
    manager._is_first_batch = False
    measurements = {
        ("DriverPhysicalState", "attention_level"): 0.2,
        ("DriverPhysicalState", "fatigue_level"): 0.7,
    }

    async def verify_persistence():
        await manager.process_pending(measurements)
        while not manager.knowledge_event_queue.empty():
            await manager.knowledge_event_queue.get()
        assert "persisted" not in manager.context[
            "DriverPhysicalState.attention_level"
        ]

        await manager.process_pending(measurements)
        assert manager.knowledge_event_queue.empty()

        await manager.process_pending(measurements)
        attention = manager.context["DriverPhysicalState.attention_level"]
        fatigue = manager.context["DriverPhysicalState.fatigue_level"]
        assert "persisted for a while" in attention
        assert "persisted for a while" in fatigue
        event = await manager.knowledge_event_queue.get()
        assert "persisted for a while" in " ".join(event.context)

        await manager.process_pending(
            {
                ("DriverPhysicalState", "attention_level"): 0.5,
                ("DriverPhysicalState", "fatigue_level"): 0.2,
            }
        )
        assert "persisted" not in manager.context[
            "DriverPhysicalState.attention_level"
        ]
        assert "persisted" not in manager.context["DriverPhysicalState.fatigue_level"]
        recovered_event = await manager.knowledge_event_queue.get()
        assert "persisted" not in " ".join(recovered_event.context)

    asyncio.run(verify_persistence())


@pytest.mark.parametrize("continuous_telemetry", [False, True])
def test_knowledge_manager_publishes_persistence_without_new_measurements(
    monkeypatch, continuous_telemetry
):
    assert KnowledgeManager.PERSISTENCE_CHECK_INTERVAL == 5.0
    monkeypatch.setattr(KnowledgeManager, "PERSISTENCE_CHECK_INTERVAL", 0.01)
    original_metadata = KnowledgeManager._knowledge_metadata

    def short_fatigue_threshold(name, measure):
        metadata = original_metadata(name, measure)
        if name == "DriverPhysicalState" and measure == "fatigue_level":
            return {**metadata, "persistence_seconds": 0.02}
        return metadata

    monkeypatch.setattr(
        KnowledgeManager,
        "_knowledge_metadata",
        staticmethod(short_fatigue_threshold),
    )
    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace()
    manager.event_queue = asyncio.Queue()
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=False, knowledge_update_interval=0.001)
    manager.context = {}
    manager._last_evaluated = {
        "DriverPhysicalState.fatigue_level": SimpleNamespace(
            value="medium", trend=None
        )
    }
    manager._raw_values = {}
    manager._last_speed_status = None
    manager.persistence_since = {}
    manager.persistence_notified = set()
    manager.on_context_updated = None
    manager._is_first_batch = False
    checks = Mock(wraps=manager._due_persistence_updates)
    monkeypatch.setattr(manager, "_due_persistence_updates", checks)

    async def produce_unrelated_telemetry():
        while True:
            manager.event_queue.put_nowait(
                {
                    "type": "measurements_updated",
                    "name": "DriverPreferences",
                    "values": {"preferred_music": "jazz"},
                }
            )
            await asyncio.sleep(0.001)

    async def verify_timer():
        await manager.process_pending(
            {("DriverPhysicalState", "fatigue_level"): 0.7}
        )
        await manager.knowledge_event_queue.get()
        task = asyncio.create_task(manager.run())
        producer = (
            asyncio.create_task(produce_unrelated_telemetry())
            if continuous_telemetry
            else None
        )
        try:
            event = await asyncio.wait_for(manager.knowledge_event_queue.get(), 0.5)
            assert any("persisted for a while" in fact for fact in event.context)
            assert checks.call_count >= 2
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(manager.knowledge_event_queue.get(), 0.03)
        finally:
            task.cancel()
            tasks = [task]
            if producer is not None:
                producer.cancel()
                tasks.append(producer)
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(verify_timer())


def test_knowledge_manager_emits_only_changed_facts_for_live_changes_panel():
    assert KnowledgeManager._knowledge_metadata(
        "DriverPhysicalState", "attention_level"
    )["persistence_seconds"] == 30.0
    assert KnowledgeManager._knowledge_metadata(
        "DriverPhysicalState", "fatigue_level"
    )["persistence_seconds"] == 30.0
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
    manager.persistence_since = {}
    manager.persistence_notified = set()
    timestamps = iter((10.0, 20.0, 40.1, 40.1, 41.0))
    manager.monotonic_clock = lambda: next(timestamps)
    manager.on_context_updated = None
    emitted_changes: list[list[str]] = []
    manager.on_knowledge_changed = emitted_changes.append
    manager._is_first_batch = False

    async def update_knowledge():
        attention = {("DriverPhysicalState", "attention_level"): 0.2}
        await manager.process_pending(attention)
        await manager.process_pending(
            {("DriverPreferences", "preferred_music"): "jazz"}
        )
        await manager.process_pending(
            {("DriverPreferences", "preferred_music"): "classical"}
        )
        await manager.process_pending(manager._due_persistence_updates(40.1))
        await manager.process_pending(
            {("DriverPhysicalState", "attention_level"): 0.5}
        )

    asyncio.run(update_knowledge())

    assert emitted_changes == [
        ["Attention level is low."],
        ["Attention level is low. This condition has persisted for a while."],
        ["Attention level is medium."],
    ]


def test_default_temperature_preference_seeds_reference_and_ui():
    assert DriverPreferences().preferred_cabin_temperature == 22.0
    assert make_bridge().dumpDriverPreferences()["preferred_cabin_temperature"] == 22.0
    metadata = KnowledgeManager._knowledge_metadata(
        "DriverPreferences", "preferred_cabin_temperature"
    )
    assert metadata["value_kind"] == "category"

    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager.logger = structlog.get_logger()
    manager.database_manager = SimpleNamespace(
        current_state={}, count=Mock(return_value=0)
    )
    manager.knowledge_event_queue = asyncio.Queue()
    manager.opt = SimpleNamespace(use_laya=False)
    manager.context = {}
    manager._last_evaluated = {}
    manager._raw_values = {}
    manager.on_context_updated = None
    manager._is_first_batch = True

    asyncio.run(manager.seed_default_knowledge())

    assert manager._raw_values["DriverPreferences.preferred_cabin_temperature"] == 22.0
    assert manager.context["DriverPreferences.preferred_cabin_temperature"] == (
        "The driver's preferred cabin temperature is 22 C."
    )
    assert manager.context["VehicleState.internal_temperature"] == (
        "Cabin temperature is optimal."
    )
    assert manager.knowledge_event_queue.empty()


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
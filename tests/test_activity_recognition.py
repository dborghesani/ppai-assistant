import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agents.tasks.activity_recognition import (
    ActivityDetectionExternal,
    ActivityDetectionInternal,
    ActivityRecognitionExternalTask,
    ActivityRecognitionInternalTask,
)
from config import ConfigAssistant
from data.agents_dataclasses import ActivityTypeExternal, ActivityTypeInternal, UrgencyType
from managers.knowledge_manager import KnowledgeManager
from tools.video_capture import VideoFrame, VideoFrames
from ui.bridge import VehicleBridge


@pytest.mark.parametrize("activity", list(ActivityTypeInternal))
def test_internal_activity_recognition_preserves_explicit_outcomes(activity):
    backend = SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"activity": "' + activity.value + '", "urgency": "none", "evidence": "Visible evidence"}'
        ))]
    )))
    task = ActivityRecognitionInternalTask(ConfigAssistant(), backend)
    frames = VideoFrames(tuple(
        VideoFrame("AA==", "test", seconds) for seconds in (0, 1)
    ), "test")
    result = asyncio.run(task.detect(frames))
    assert result.activity is activity
    assert result.urgency is UrgencyType.NONE
    assert result.model_dump(mode="json")["activity"] == activity.value
    content = backend.complete.call_args.kwargs["messages"][0]["content"]
    assert len([part for part in content if part["type"] == "image_url"]) == 2
    assert all(member.value in content[0]["text"] for member in ActivityTypeInternal)
    assert all(level.value in content[0]["text"] for level in UrgencyType)


@pytest.mark.parametrize("activity", list(ActivityTypeExternal))
def test_external_activity_task_returns_typed_activity_and_urgency(activity):
    backend = SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"activity": "' + activity.value + '", "urgency": "high", "evidence": "Visible evidence"}'
        ))]
    )))
    task = ActivityRecognitionExternalTask(ConfigAssistant(), backend)
    frames = VideoFrames(tuple(
        VideoFrame("AA==", "test", seconds) for seconds in (0, 1)
    ), "test")

    result = asyncio.run(task.detect(frames))

    assert result.activity is activity
    assert result.urgency is UrgencyType.HIGH
    prompt = backend.complete.call_args.kwargs["messages"][0]["content"][0]["text"]
    assert "security-related activity" in prompt
    assert all(member.value in prompt for member in ActivityTypeExternal)
    assert all(level.value in prompt for level in UrgencyType)
    assert "external activity recognition" == backend.complete.call_args.kwargs["call_name"]


def test_idle_is_not_a_detection_result():
    with pytest.raises(ValueError):
        ActivityDetectionInternal(activity="Idle", urgency=UrgencyType.NONE, evidence="")


@pytest.mark.parametrize("activities", [[], ["Talking"], ["Driving", "Talking"]])
def test_detection_rejects_activity_lists(activities):
    with pytest.raises(ValueError):
        ActivityDetectionInternal(activity=activities, urgency=UrgencyType.NONE, evidence="")


@pytest.mark.parametrize("activity", [ActivityTypeInternal.DRIVER_TALKING, ActivityTypeInternal.NONE])
def test_manual_activity_writes_one_value_and_updates_controls(activity):
    bridge = VehicleBridge.__new__(VehicleBridge)
    bridge.agent = SimpleNamespace(is_listening=True)
    bridge.database_manager = SimpleNamespace(write_measure=Mock())
    bridge._listeners = defaultdict(list)
    updates = []
    bridge.on("driverPhysicalStateChanged", updates.append)

    bridge.stringChanged("DriverPhysicalState", "activity", activity.value)

    bridge.database_manager.write_measure.assert_called_once_with(
        "DriverPhysicalState", "activity", activity.value,
    )
    assert updates == [{"activity": activity.value}]
    if activity is ActivityTypeInternal.NONE:
        knowledge = KnowledgeManager.__new__(KnowledgeManager)
        assert knowledge._categorical_knowledge("DriverPhysicalState", "activity", activity.value) == (
            "No monitored activity is currently detected inside the vehicle."
        )


@pytest.mark.parametrize("activity", [ActivityTypeInternal.DRIVER_EATING, ActivityTypeInternal.NONE, ActivityTypeInternal.UNKNOWN])
def test_detection_writes_single_activity_but_preserves_state_when_unknown(activity):
    async def check():
        bridge = VehicleBridge.__new__(VehicleBridge)
        bridge.agent = SimpleNamespace(
            activity_recognition_internal_task=SimpleNamespace(
                detect=AsyncMock(return_value=ActivityDetectionInternal(
                    activity=activity, urgency=UrgencyType.HIGH, evidence="Visible",
                )),
            ),
            capture_sources={"internal": "camera_stream"},
            video_capture=SimpleNamespace(
                buffer_source=Mock(return_value=None),
                start_buffer=Mock(return_value="/dev/video0"),
                get_buffer_frames=AsyncMock(return_value=VideoFrames(
                    (VideoFrame("AA==", "camera", 0.0),), "camera",
                )),
            ),
        )
        bridge.database_manager = SimpleNamespace(write_measure=Mock())
        bridge._listeners = defaultdict(list)
        detection_events = []
        bridge.on("activityDetectionChanged", detection_events.append)
        bridge.logger = Mock()

        await bridge._detect_activity()

        assert not detection_events[-1].get("error"), bridge.logger.warning.call_args
        if activity is ActivityTypeInternal.UNKNOWN:
            bridge.database_manager.write_measure.assert_not_called()
        else:
            bridge.database_manager.write_measure.assert_called_once_with(
                "DriverPhysicalState", "activity", activity.value,
            )

    asyncio.run(check())


@pytest.mark.parametrize(("classname", "activity", "expected"), [
    ("DriverPhysicalState", ActivityTypeInternal.DRIVER_EATING, "The driver is eating, which distracts from driving."),
    ("DriverPhysicalState", ActivityTypeInternal.CHILDREN_UNATTENDED, "Children are unattended inside the vehicle."),
    ("DriverPhysicalState", ActivityTypeInternal.ANIMAL_UNATTENDED, "An animal is unattended inside the vehicle."),
    ("VehicleSecurityState", ActivityTypeExternal.VEHICLE_IMPACT, "An impact against the vehicle has been detected."),
    ("VehicleSecurityState", ActivityTypeExternal.VANDALISM, "Vandalism against the vehicle has been detected."),
    ("VehicleSecurityState", ActivityTypeExternal.BREAK_IN_ATTEMPT, "An attempt to break into the vehicle has been detected."),
    ("VehicleSecurityState", ActivityTypeExternal.INTRUSION, "An intrusion into the vehicle has been detected."),
])
def test_recognized_activity_is_rendered_as_human_readable_knowledge(classname, activity, expected):
    knowledge = KnowledgeManager.__new__(KnowledgeManager)

    assert knowledge._categorical_knowledge(classname, "activity", activity.value) == expected


def test_capture_source_change_stops_the_generic_scope_buffer():
    bridge = VehicleBridge.__new__(VehicleBridge)
    bridge.agent = SimpleNamespace(
        capture_sources={
            "internal": "camera_stream", "external": "video_file", "scene": "video_file",
        },
        video_capture=SimpleNamespace(stop_buffer=Mock(), start_buffer=Mock()),
    )
    bridge.opt = SimpleNamespace(scene_understanding_source="video_file")
    bridge._listeners = defaultdict(list)

    assert bridge.setCaptureSource("scene", "camera_stream") == "camera_stream"
    assert bridge.agent.capture_sources["scene"] == "camera_stream"
    assert bridge.opt.scene_understanding_source == "camera_stream"
    bridge.agent.video_capture.stop_buffer.assert_called_once_with("scene")


@pytest.mark.parametrize(("scope", "expected_names"), [
    ("internal", ["activity_intl_driver_eating.mp4"]),
    ("external", ["activity_ext_vandalism.mp4"]),
])
def test_agent_lists_scope_specific_activity_recordings(scope, expected_names, tmp_path):
    from agents.automotive_agent import AutomotiveAgent
    from tools.video_capture import VideoCapture

    for name in (
        "activity_intl_driver_eating.mp4",
        "activity_ext_vandalism.mp4",
        "activity_scenario.mp4",
    ):
        (tmp_path / name).touch()
    agent = AutomotiveAgent.__new__(AutomotiveAgent)
    agent.video_capture = VideoCapture(tmp_path)

    assert [path.name for path in agent.activity_recordings(scope)] == expected_names


def test_selected_activity_video_starts_and_stops_its_generic_buffer(tmp_path):
    async def check():
        bridge = VehicleBridge.__new__(VehicleBridge)
        video_path = tmp_path / "activity_ext_vandalism.mp4"
        video_path.touch()
        video_capture = SimpleNamespace(
            start_buffer=Mock(), stop_buffer=Mock(), wait_buffer_ready=AsyncMock(),
        )
        bridge.agent = SimpleNamespace(
            capture_sources={"internal": "camera_stream", "external": "video_file"},
            activity_recordings=Mock(return_value=[video_path]),
            video_capture=video_capture,
        )
        bridge.dumpVehicleSecurityState = lambda: {"sentry_mode_enabled": False}
        bridge.boolChanged = Mock()
        bridge._listeners = defaultdict(list)
        updates = []
        bridge.on("activityVideoStateChanged", updates.append)

        assert await bridge.startActivityVideo("external", video_path.name, 4.5)
        bridge.boolChanged.assert_called_once_with(
            "VehicleSecurityState", "sentry_mode_enabled", True,
        )
        video_capture.start_buffer.assert_called_once_with("external", "video_file", video_path, 4.5)
        video_capture.wait_buffer_ready.assert_awaited_once_with("external")
        assert updates == [{"scope": "external", "active": True, "filename": video_path.name}]
        assert bridge.stopActivityVideo("external")
        video_capture.stop_buffer.assert_called_once_with("external")

    asyncio.run(check())
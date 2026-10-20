import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agents.tasks.activity_recognition import ActivityDetection, ActivityRecognitionTask
from config import ConfigAssistant
from data.agents_dataclasses import ActivityType
from managers.knowledge_manager import KnowledgeManager
from ui.bridge import VehicleBridge


@pytest.mark.parametrize("activity", list(ActivityType))
def test_activity_recognition_preserves_explicit_outcomes(activity):
    backend = SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"activity": "' + activity.value + '", "evidence": "Visible evidence"}'
        ))]
    )))
    task = ActivityRecognitionTask(ConfigAssistant(), backend)
    frames = [{"image": "data:image/jpeg;base64,AA==", "seconds": seconds} for seconds in (0, 1)]
    result = asyncio.run(task.detect(frames))
    assert result.activity is activity
    assert result.model_dump(mode="json")["activity"] == activity.value
    content = backend.complete.call_args.kwargs["messages"][0]["content"]
    assert len([part for part in content if part["type"] == "image_url"]) == 2
    assert all(member.value in content[0]["text"] for member in ActivityType)


def test_idle_is_not_a_detection_result():
    with pytest.raises(ValueError):
        ActivityDetection(activity="Idle", evidence="")


@pytest.mark.parametrize("activities", [[], ["Talking"], ["Driving", "Talking"]])
def test_detection_rejects_activity_lists(activities):
    with pytest.raises(ValueError):
        ActivityDetection(activity=activities, evidence="")


@pytest.mark.parametrize("activity", [ActivityType.TALKING, ActivityType.NONE])
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
    if activity is ActivityType.NONE:
        knowledge = KnowledgeManager.__new__(KnowledgeManager)
        assert knowledge._categorical_knowledge("DriverPhysicalState", "activity", activity.value) == (
            "No monitored activity is currently detected inside the vehicle."
        )


@pytest.mark.parametrize("activity", [ActivityType.EATING, ActivityType.NONE, ActivityType.UNKNOWN])
def test_detection_writes_single_activity_but_preserves_state_when_unknown(activity):
    async def check():
        bridge = VehicleBridge.__new__(VehicleBridge)
        bridge.agent = SimpleNamespace(activity_recognition_task=SimpleNamespace(
            detect=AsyncMock(return_value=ActivityDetection(activity=activity, evidence="Visible")),
        ))
        bridge.database_manager = SimpleNamespace(write_measure=Mock())
        bridge._listeners = defaultdict(list)
        bridge.logger = Mock()

        await bridge._detect_activity([])

        if activity is ActivityType.UNKNOWN:
            bridge.database_manager.write_measure.assert_not_called()
        else:
            bridge.database_manager.write_measure.assert_called_once_with(
                "DriverPhysicalState", "activity", activity.value,
            )

    asyncio.run(check())
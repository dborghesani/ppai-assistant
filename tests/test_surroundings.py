import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents.tasks.surroundings import SurroundingsTask
from tools.video_capture import VideoFrame, VideoFrames


def test_scene_analysis_disables_reasoning_and_sends_all_frames():
    backend = SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"description": "Traffic is moving steadily.", "notable": false, "subject": null}'
        ))]
    )))
    task = SurroundingsTask.__new__(SurroundingsTask)
    task.opt = SimpleNamespace(ollama_model="vision-model")
    task.backend = backend
    frames = VideoFrames(tuple(
        VideoFrame(image, "video", index) for index, image in enumerate(
            ("first-frame", "second-frame", "third-frame")
        )
    ), "video")

    scene = asyncio.run(task._run_analysis(frames, []))

    request = backend.complete.await_args.kwargs
    assert request["reasoning_effort"] == "none"
    image_parts = [part for part in request["messages"][0]["content"] if part["type"] == "image_url"]
    assert len(image_parts) == 3
    assert scene.description == "Traffic is moving steadily."
    assert scene.source == "video"
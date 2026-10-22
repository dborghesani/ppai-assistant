import asyncio
from collections import deque
import threading
import time

from tools.video_capture import VideoCapture, VideoFrame


def test_streaming_jpeg_parser_retains_partial_frame_between_reads():
    image = bytes([255, 216]) + b"jpeg-data" + bytes([255, 217])
    buffer = bytearray(b"noise" + image[:5])

    assert VideoCapture._extract_jpegs(buffer) == []

    buffer.extend(image[5:])
    assert VideoCapture._extract_jpegs(buffer) == [image]
    assert not buffer


def test_generic_buffer_returns_half_history_and_half_future(monkeypatch, tmp_path):
    capture = VideoCapture(
        tmp_path, camera_device="/dev/video-test",
        sample_rate_hz=10.0, buffer_capacity=6,
    )
    past_frames = [VideoFrame(f"past{index}", "camera", index / 10) for index in range(3)]
    future_frames = [VideoFrame(f"future{index}", "camera", (index + 3) / 10) for index in range(3)]

    def start_buffer(buffer_key, source, video_path=None):
        with capture._buffer_condition:
            capture._buffers[buffer_key] = deque(past_frames, maxlen=6)
            capture._buffer_settings[buffer_key] = (source, None)
        def produce_future_frames():
            time.sleep(0.02)
            with capture._buffer_condition:
                capture._buffers[buffer_key].extend(future_frames)
                capture._buffer_condition.notify_all()
        threading.Thread(target=produce_future_frames, daemon=True).start()
        return "/dev/video-test"

    monkeypatch.setattr(capture, "start_buffer", start_buffer)
    source = capture.start_buffer("test", "camera_stream")
    frames = asyncio.run(capture.get_buffer_frames("test"))

    assert len(frames.frames) == 6
    assert list(frames.frames[:3]) == past_frames
    assert list(frames.frames[3:]) == future_frames
    assert frames.source == "/dev/video-test"
    assert source == "/dev/video-test"
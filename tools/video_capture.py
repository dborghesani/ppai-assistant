"""Frame capture from local video recordings and Linux webcam devices."""
from __future__ import annotations

import asyncio
import base64
import os
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = structlog.get_logger()


@dataclass(frozen=True)
class VideoFrame:
    image_base64: str
    source: str = "video"
    seconds: float = 0.0


@dataclass(frozen=True)
class VideoFrames:
    """An ordered set of captured frames shared by visual recognition tasks."""

    frames: tuple[VideoFrame, ...]
    source: str


class VideoFrameSource(Protocol):
    def grab_frame(self, video_path: str | None, seconds: float | None) -> VideoFrame | None: ...


class VideoCapture:
    """Capture frames with ffmpeg from recordings or a V4L2 webcam."""

    VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})

    def __init__(
        self, recordings_dir: Path | None = None, camera_device: str | None = None,
        sample_rate_hz: float = 2.0, buffer_capacity: int = 6,
    ):
        self.recordings_dir = recordings_dir or PROJECT_ROOT / "recordings"
        self.camera_device: str = camera_device or os.environ.get("PPAI_CAMERA_DEVICE") or "/dev/video0"
        if sample_rate_hz <= 0:
            raise ValueError("Video sampling frequency must be positive")
        if not 2 <= buffer_capacity <= 8 or buffer_capacity % 2:
            raise ValueError("Video buffer capacity must be an even number between 2 and 8")
        self.sample_rate_hz = sample_rate_hz
        self.buffer_capacity = buffer_capacity
        self._buffer_condition = threading.Condition()
        self._buffers: dict[str, deque[VideoFrame]] = {}
        self._buffer_threads: dict[str, threading.Thread] = {}
        self._buffer_stops: dict[str, threading.Event] = {}
        self._buffer_settings: dict[str, tuple[str, Path | None]] = {}
        self._buffer_errors: dict[str, str] = {}
        self._buffer_processes: dict[str, subprocess.Popen] = {}

    @staticmethod
    def _ffmpeg() -> str:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for video capture")
        return ffmpeg

    @staticmethod
    def _split_jpegs(data: bytes) -> list[bytes]:
        frames: list[bytes] = []
        cursor = 0
        while True:
            start = data.find(b"\xff\xd8", cursor)
            if start < 0:
                return frames
            end = data.find(b"\xff\xd9", start + 2)
            if end < 0:
                return frames
            frames.append(data[start:end + 2])
            cursor = end + 2

    @staticmethod
    def _extract_jpegs(buffer: bytearray) -> list[bytes]:
        """Extract complete concatenated JPEGs while retaining a partial tail."""
        frames: list[bytes] = []
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                tail = buffer[-1:] if buffer.endswith(b"\xff") else b""
                buffer.clear()
                buffer.extend(tail)
                return frames
            if start:
                del buffer[:start]
            end = buffer.find(b"\xff\xd9", 2)
            if end < 0:
                return frames
            frames.append(bytes(buffer[:end + 2]))
            del buffer[:end + 2]

    def _run_capture(
        self, command: list[str], *, source: str, frame_count: int, minimum_frames: int,
    ) -> VideoFrames:
        result = subprocess.run(command, capture_output=True, timeout=30)
        if result.returncode:
            detail = result.stderr.decode(errors="replace")[-300:].strip()
            raise RuntimeError(detail or f"Could not capture frames from {source}")
        images = self._split_jpegs(result.stdout)[:frame_count]
        if len(images) < minimum_frames:
            raise RuntimeError(f"Not enough frames captured from {source}")
        frames = tuple(
            VideoFrame(
                image_base64=base64.b64encode(image).decode(),
                source=source,
                seconds=index / self.sample_rate_hz,
            )
            for index, image in enumerate(images)
        )
        return VideoFrames(frames, source)

    def _video_command(
        self, video_path: Path, frame_count: int, start_seconds: float = 0.0,
    ) -> list[str]:
        return [
            self._ffmpeg(), "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start_seconds):.3f}", "-i", str(video_path),
            "-vf", f"fps={self.sample_rate_hz},scale=512:-2", "-frames:v", str(frame_count),
            "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]

    def capture_recording_frames(
        self, video_path: Path, frame_count: int | None = None, start_seconds: float = 0.0,
    ) -> VideoFrames:
        frame_count = frame_count or self.buffer_capacity
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        return self._run_capture(
            self._video_command(video_path, frame_count, start_seconds), source=video_path.name,
            frame_count=frame_count, minimum_frames=2,
        )

    def capture_webcam_frames(self, frame_count: int | None = None) -> VideoFrames:
        frame_count = frame_count or self.buffer_capacity
        command = [
            self._ffmpeg(), "-hide_banner", "-loglevel", "error",
            "-f", "v4l2", "-framerate", str(self.sample_rate_hz),
            "-i", self.camera_device,
            "-vf", "scale=512:-2", "-frames:v", str(frame_count),
            "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]
        return self._run_capture(
            command, source=self.camera_device, frame_count=frame_count,
            minimum_frames=2,
        )

    def capture_webcam_frame(self) -> VideoFrame | None:
        command = [
            self._ffmpeg(), "-hide_banner", "-loglevel", "error",
            "-f", "v4l2", "-i", self.camera_device,
            "-frames:v", "1", "-vf", "scale=512:-2",
            "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True, timeout=30)
        images = self._split_jpegs(result.stdout) if result.returncode == 0 else []
        return VideoFrame(base64.b64encode(images[0]).decode(), "camera") if images else None

    def _stream_command(
        self, source: str, recording: Path | None, start_seconds: float,
    ) -> list[str]:
        command = [self._ffmpeg(), "-hide_banner", "-loglevel", "error"]
        if source == "camera_stream":
            command.extend([
                "-f", "v4l2", "-framerate", str(self.sample_rate_hz),
                "-i", self.camera_device,
            ])
        else:
            assert recording is not None
            command.extend([
                "-stream_loop", "-1", "-re", "-ss", f"{max(0.0, start_seconds):.3f}",
                "-i", str(recording),
            ])
        command.extend([
            "-vf", f"fps={self.sample_rate_hz},scale=512:-2",
            "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ])
        return command

    def _buffer_stream_loop(
        self, scope: str, source: str, recording: Path | None, start_seconds: float,
        stop: threading.Event,
    ) -> None:
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                self._stream_command(source, recording, start_seconds),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
            )
            with self._buffer_condition:
                self._buffer_processes[scope] = process
                self._buffer_condition.notify_all()
            if process.stdout is None:
                raise RuntimeError("ffmpeg did not open its frame stream")
            pending = bytearray()
            while not stop.is_set():
                chunk = process.stdout.read(65536)
                if not chunk:
                    if not stop.is_set():
                        raise RuntimeError(f"ffmpeg frame stream ended for {source}")
                    break
                pending.extend(chunk)
                for image in self._extract_jpegs(pending):
                    item = VideoFrame(
                        image_base64=base64.b64encode(image).decode(),
                        source=recording.name if recording else self.camera_device,
                        seconds=time.monotonic() - started,
                    )
                    with self._buffer_condition:
                        buffer = self._buffers.get(scope)
                        if buffer is not None:
                            buffer.append(item)
                            self._buffer_errors.pop(scope, None)
                            self._buffer_condition.notify_all()
        except Exception as error:
            with self._buffer_condition:
                self._buffer_errors[scope] = str(error)
                self._buffer_condition.notify_all()
        finally:
            with self._buffer_condition:
                active_process = self._buffer_processes.get(scope)
            if active_process is not None:
                if active_process.poll() is None:
                    active_process.terminate()
                try:
                    active_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    active_process.kill()
                    active_process.wait()
                if active_process.stdout is not None:
                    active_process.stdout.close()
                with self._buffer_condition:
                    if self._buffer_processes.get(scope) is active_process:
                        self._buffer_processes.pop(scope, None)
                    self._buffer_condition.notify_all()

    def start_buffer(
        self, buffer_key: str, source: str, video_path: Path | None = None,
        start_seconds: float = 0.0,
    ) -> str:
        if not buffer_key:
            raise ValueError("Buffer key is required")
        if source not in {"camera_stream", "video_file"}:
            raise ValueError("Capture source must be camera_stream or video_file")
        recording = video_path if source == "video_file" else None
        if source == "video_file" and (recording is None or not recording.is_file()):
            raise FileNotFoundError("A readable video file is required for video capture")
        settings = (source, recording)
        with self._buffer_condition:
            existing = self._buffer_threads.get(buffer_key)
            current_settings = self._buffer_settings.get(buffer_key)
            if existing is not None and existing.is_alive() and current_settings is not None:
                if current_settings == settings:
                    return recording.name if recording else self.camera_device
            previous_stop = self._buffer_stops.get(buffer_key)
            if previous_stop is not None:
                previous_stop.set()
            previous_thread = existing
        if previous_thread is not None and previous_thread.is_alive():
            previous_thread.join(timeout=2.0)
        stop = threading.Event()
        thread = threading.Thread(
            target=self._buffer_stream_loop,
            args=(buffer_key, source, recording, start_seconds, stop),
            name=f"video-frame-buffer-{buffer_key}", daemon=True,
        )
        with self._buffer_condition:
            self._buffers[buffer_key] = deque(maxlen=self.buffer_capacity)
            self._buffer_stops[buffer_key] = stop
            self._buffer_settings[buffer_key] = settings
            self._buffer_errors.pop(buffer_key, None)
            self._buffer_threads[buffer_key] = thread
        thread.start()
        return recording.name if recording else self.camera_device

    def _wait_for_buffer_sync(self, buffer_key: str, frame_count: int) -> None:
        timeout = max(2.0, frame_count / self.sample_rate_hz + 1.0)
        deadline = time.monotonic() + timeout
        with self._buffer_condition:
            while True:
                if buffer_key not in self._buffers:
                    raise RuntimeError("Frame buffer was stopped before it became ready")
                if len(self._buffers[buffer_key]) >= frame_count:
                    return
                if error := self._buffer_errors.get(buffer_key):
                    raise RuntimeError(error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Frame buffer did not fill in time")
                self._buffer_condition.wait(remaining)

    async def wait_buffer_ready(self, buffer_key: str, minimum_frames: int | None = None) -> None:
        minimum_frames = minimum_frames or self.buffer_capacity // 2
        if not 1 <= minimum_frames <= self.buffer_capacity:
            raise ValueError("Minimum buffered frame count is outside buffer capacity")
        await asyncio.to_thread(self._wait_for_buffer_sync, buffer_key, minimum_frames)

    async def get_buffer_frames(self, buffer_key: str, frame_count: int | None = None) -> VideoFrames:
        frame_count = frame_count or self.buffer_capacity
        if not 2 <= frame_count <= 8 or frame_count % 2:
            raise ValueError("Requested frame count must be even and between 2 and 8")
        past_count = future_count = frame_count // 2
        wait_timeout = max(2.0, (frame_count + 1) / self.sample_rate_hz + 2.0)
        deadline = time.monotonic() + wait_timeout
        await self.wait_buffer_ready(buffer_key, past_count)
        with self._buffer_condition:
            past = list(self._buffers[buffer_key])[-past_count:]
            after_seconds = past[-1].seconds
            future = [frame for frame in self._buffers[buffer_key] if frame.seconds > after_seconds]
            while len(future) < future_count:
                if buffer_key not in self._buffers:
                    raise RuntimeError("Frame buffer was stopped before future frames arrived")
                if error := self._buffer_errors.get(buffer_key):
                    raise RuntimeError(error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = self._buffer_errors.get(buffer_key, "Future frame buffer did not fill in time")
                    raise RuntimeError(detail)
                self._buffer_condition.wait(remaining)
                future = [frame for frame in self._buffers[buffer_key] if frame.seconds > after_seconds]
            frames = tuple(past + future[:future_count])
            source = self._buffer_settings[buffer_key][1]
            return VideoFrames(frames, source.name if source else self.camera_device)

    def stop_buffer(self, buffer_key: str) -> None:
        with self._buffer_condition:
            stop = self._buffer_stops.pop(buffer_key, None)
            thread = self._buffer_threads.pop(buffer_key, None)
            process = self._buffer_processes.get(buffer_key)
            self._buffer_settings.pop(buffer_key, None)
            self._buffers.pop(buffer_key, None)
            self._buffer_condition.notify_all()
        if stop is not None:
            stop.set()
        if process is not None and process.poll() is None:
            process.terminate()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def buffer_source(self, buffer_key: str) -> tuple[str, Path | None] | None:
        with self._buffer_condition:
            thread = self._buffer_threads.get(buffer_key)
            if thread is None or not thread.is_alive():
                return None
            return self._buffer_settings.get(buffer_key)

    def close(self) -> None:
        for scope in tuple(self._buffer_threads):
            self.stop_buffer(scope)

    def resolve_recording(self, requested: str | None, default: str | None = None) -> Path | None:
        """Map a bare name, file name or root-relative path to an existing file."""
        for name in ((requested or "").strip() or default or "",):
            if not name:
                continue
            raw = Path(name)
            if raw.is_absolute() and not raw.exists():
                raw = Path(*raw.parts[2:]) if len(raw.parts) > 2 and raw.parts[1] == "recordings" else Path(raw.name)
            choices = [raw]
            if not raw.suffix:
                choices.append(raw.with_suffix(".mp4"))
            for choice in choices:
                if choice.is_absolute() and choice.exists():
                    return choice
                if not choice.is_absolute():
                    for root in (PROJECT_ROOT, self.recordings_dir):
                        candidate = root / choice
                        if candidate.exists():
                            return candidate.resolve()
            logger.warning("Video recording not found", requested=name)
        return None

    def grab_frame(self, video_path: str | None, seconds: float | None) -> VideoFrame | None:
        if not video_path:
            return None
        path = Path(video_path)
        if not path.is_file():
            return None
        command = [
            self._ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss",
            f"{max(0.0, float(seconds or 0.0)):.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", "scale=512:-2", "-f", "image2pipe",
            "-vcodec", "mjpeg", "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True, timeout=30)
        images = self._split_jpegs(result.stdout) if result.returncode == 0 else []
        return VideoFrame(base64.b64encode(images[0]).decode()) if images else None


async def capture_frame(
    source: VideoFrameSource, video_path: str | None, seconds: float | None,
) -> VideoFrame | None:
    """Capture a frame without blocking the event loop."""
    return await asyncio.to_thread(source.grab_frame, video_path, seconds)
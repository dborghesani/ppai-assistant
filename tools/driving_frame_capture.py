"""Video frame capture tool.

A generic interface for grabbing single frames from a video source, decoupled
from how the video is reproduced: today the source is a local recording file
(ffmpeg seek); tomorrow a live camera or another pipeline can provide frames by
implementing the same ``grab_frame`` contract.
"""
from __future__ import annotations

import asyncio
import base64
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = structlog.get_logger()


@dataclass(frozen=True)
class VideoFrame:
    """One grabbed frame of a video source."""

    image_base64: str
    source: str = "video"


class VideoFrameSource(Protocol):
    """Anything that can capture single frames from a video source."""

    def grab_frame(self, video_path: str | None, seconds: float | None) -> VideoFrame | None: ...


class RecordingFrameCapture:
    """Grabs single frames from local video recordings via ffmpeg."""

    def __init__(self, recordings_dir: Path | None = None):
        self.recordings_dir = recordings_dir
        self.frame_dir = Path(tempfile.gettempdir()) / "ppai_driving_frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)

    def resolve_recording(self, requested: str | None, default: str | None = None) -> Path | None:
        """Map a bare name, file name or root-relative path to an existing file."""
        for name in ((requested or "").strip() or default or "",):
            if not name:
                continue
            raw = Path(name)
            if raw.is_absolute() and not raw.exists():
                # The UI may echo back the served URL ("/recordings/rome.mp4").
                raw = Path(*raw.parts[2:]) if len(raw.parts) > 2 and raw.parts[1] == "recordings" else Path(raw.name)
            choices = [raw]
            if not raw.suffix:
                choices.append(raw.with_suffix(".mp4"))
            for choice in choices:
                if choice.is_absolute():
                    if choice.exists():
                        return choice
                else:
                    relative = PROJECT_ROOT / choice
                    if relative.exists():
                        return relative.resolve()
                    if self.recordings_dir is not None:
                        local = self.recordings_dir / choice
                        if local.exists():
                            return local.resolve()
            logger.warning("Video recording not found", requested=name)
        return None

    def grab_jpeg_base64(self, video_path: Path, seconds: float) -> str | None:
        """One JPEG frame of a local recording, seeked straight to ``seconds``."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            logger.warning("ffmpeg is not installed; cannot grab a video frame")
            return None
        out_path = self.frame_dir / f"video_frame_{int(time.time() * 1000)}.jpg"
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-vf", "scale=512:-2", "-q:v", "5", "-y", str(out_path)],
            capture_output=True, timeout=30,
        )
        try:
            if not out_path.exists():
                logger.warning("ffmpeg produced no frame", stderr=result.stderr[-200:] if result.stderr else "")
                return None
            return base64.b64encode(out_path.read_bytes()).decode()
        finally:
            out_path.unlink(missing_ok=True)

    def grab_frame(self, video_path: str | None, seconds: float | None) -> VideoFrame | None:
        """Best-effort frame from a local recording at ``seconds`` (blocking)."""
        if not video_path:
            return None
        seconds = max(0.0, float(seconds or 0.0))
        path = Path(video_path)
        if not path.exists():
            logger.warning("Video recording file not found", path=str(path))
            return None
        try:
            image = self.grab_jpeg_base64(path, seconds)
        except Exception as error:
            logger.warning("Video frame extraction failed", error=str(error))
            return None
        return VideoFrame(image_base64=image) if image else None


async def capture_frame(
    source: VideoFrameSource, video_path: str | None, seconds: float | None,
) -> VideoFrame | None:
    """Await any :class:`VideoFrameSource` grab without blocking the event loop."""
    return await asyncio.to_thread(source.grab_frame, video_path, seconds)


# Backwards-compatible alias: the file-producing implementation used today.
DrivingFrameCapture = RecordingFrameCapture

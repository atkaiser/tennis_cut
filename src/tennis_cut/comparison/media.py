"""Exact media inspection for comparison workflows."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess

from .pro_selection import DecodedFrame, InspectedMedia


def inspect_media(video: Path) -> InspectedMedia:
    """Inspect ordered decoded video frames without FPS-based reconstruction."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=index,time_base:frame=stream_index,pts",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError("expected exactly one selected video stream")
    stream_index = int(streams[0]["index"])
    time_base = Fraction(streams[0]["time_base"])
    frames = tuple(
        DecodedFrame(
            stream_index=int(frame["stream_index"]),
            ordinal=ordinal,
            pts=int(frame["pts"]),
            time_base=time_base,
        )
        for ordinal, frame in enumerate(payload.get("frames", []))
    )
    if not frames:
        raise ValueError("selected video stream has no decoded frames")
    if any(frame.stream_index != stream_index for frame in frames):
        raise ValueError("decoded frame belongs to an unexpected stream")
    return InspectedMedia(frames=frames)

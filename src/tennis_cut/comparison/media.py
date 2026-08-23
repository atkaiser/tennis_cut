"""Exact media inspection for comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from math import lcm
from pathlib import Path
import subprocess
import tempfile
from typing import Protocol

from .planning import (
    MAX_EXACT_FILTER_TICK,
    MAX_OUTPUT_TIMESCALE,
    ComparisonRenderPlan,
    ComparisonSource,
    PlayerObservation,
    Rectangle,
    SelectedSourceWindow,
)
from .pro_selection import DecodedFrame, InspectedMedia


class MediaCommandFailed(RuntimeError):
    """Stable media-command failure with diagnostics retained out of band."""

    def __init__(self, executable: str, returncode: int, diagnostics: str) -> None:
        self.diagnostics = diagnostics
        super().__init__(f"{executable} exited with status {returncode}")


def _run_media_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MediaCommandFailed(
            command[0], completed.returncode, completed.stderr.strip()
        )
    return completed


@dataclass(frozen=True)
class _InspectedVideo:
    width: int
    height: int
    media: InspectedMedia


def _inspect_video(video: Path) -> _InspectedVideo:
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
            "stream=index,time_base,width,height:frame=stream_index,pts",
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
    stream = streams[0]
    stream_index = int(stream["index"])
    time_base = Fraction(stream["time_base"])
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
    return _InspectedVideo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        media=InspectedMedia(frames),
    )


def inspect_media(video: Path) -> InspectedMedia:
    """Inspect ordered decoded video frames without FPS-based reconstruction."""

    return _inspect_video(video).media


def inspect_comparison_source(video: Path) -> ComparisonSource:
    """Inspect exact frames and source geometry for comparison planning."""

    inspected = _inspect_video(video)
    return ComparisonSource(
        path=video,
        width=inspected.width,
        height=inspected.height,
        inspected_media=inspected.media,
    )


def has_audio_stream(video: Path) -> bool:
    """Return whether the source exposes at least one audio stream."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(json.loads(completed.stdout).get("streams", []))


class PlayerLocator(Protocol):
    """Locate one player in an exact decoded-frame image."""

    def find_box(self, image: Path, /) -> tuple[int, int, int, int] | None: ...


class FfmpegFrameImageReader:
    """Decode one exact source frame for the pro picker."""

    def read_frame(self, source: Path, frame: DecodedFrame) -> bytes:
        with tempfile.TemporaryDirectory() as directory:
            decoded = _decode_frames(source, (frame.ordinal,), Path(directory))
            return decoded[frame.ordinal].read_bytes()


def _decode_frames(
    source: Path,
    ordinals: tuple[int, ...],
    output_directory: Path,
) -> dict[int, Path]:
    unique_ordinals = tuple(sorted(set(ordinals)))
    if not unique_ordinals:
        return {}
    selection = "+".join(f"eq(n\\,{ordinal})" for ordinal in unique_ordinals)
    output_pattern = output_directory / "frame_%09d.png"
    _run_media_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-noautorotate",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"select={selection}",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "0",
            str(output_pattern),
            "-y",
        ]
    )
    paths = tuple(sorted(output_directory.glob("frame_*.png")))
    if len(paths) != len(unique_ordinals):
        raise ValueError("failed to decode every requested source frame")
    return dict(zip(unique_ordinals, paths))


def observe_players(
    window: SelectedSourceWindow,
    locator: PlayerLocator,
) -> tuple[PlayerObservation, ...]:
    """Sample player bounds across every unique frame in a selected window."""

    ordinals = tuple(item.frame.ordinal for item in window.normalized_frames)
    with tempfile.TemporaryDirectory() as directory:
        decoded = _decode_frames(window.source.path, ordinals, Path(directory))
        observations: list[PlayerObservation] = []
        for ordinal in sorted(decoded):
            box = locator.find_box(decoded[ordinal])
            if box is None:
                continue
            x1, y1, x2, y2 = box
            observations.append(
                PlayerObservation(ordinal, Rectangle(x1, y1, x2 - x1, y2 - y1))
            )
    return tuple(observations)


def render_comparison(plan: ComparisonRenderPlan) -> Path:
    """Render one planned silent, fixed-crop comparison clip."""

    plan.artifact.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        pro_panel = Path(directory) / "pro-panel.mkv"
        _prepare_pro_panel(plan, pro_panel)
        _render_plan_segment(
            plan,
            Path(directory) / "segment.mp4",
            plan.output_time_base.denominator,
            pro_panel,
        )
        Path(directory, "segment.mp4").replace(plan.artifact.path)
    return plan.artifact.path


def _render_plan_segment(
    plan: ComparisonRenderPlan,
    output: Path,
    timescale: int,
    pro_panel: Path,
) -> None:
    """Encode one comparison directly from its two source video streams."""

    if timescale % plan.output_time_base.denominator:
        raise ValueError("segment timescale must contain the plan time base")
    scale = timescale // plan.output_time_base.denominator
    branches: list[str] = []
    events: list[str] = []
    for index, event in enumerate(plan.events):
        branches.extend(
            (
                (
                    f"[0:v]select='eq(n\\,{event.user_frame.ordinal})',"
                    f"crop={plan.user.crop.width}:{plan.user.crop.height}:"
                    f"{plan.user.crop.x}:{plan.user.crop.y},"
                    f"scale={plan.layout.user_panel.width}:{plan.layout.user_panel.height},"
                    f"setsar=1[user{index}]"
                ),
                (
                    f"[1:v]select='eq(n\\,{event.pro_frame.ordinal})',"
                    f"scale={plan.layout.pro_panel.width}:{plan.layout.pro_panel.height},"
                    f"setsar=1[pro{index}]"
                ),
                f"[user{index}][pro{index}]hstack=inputs=2[event{index}]",
            )
        )
        events.append(f"[event{index}]")
    ticks = [event.output_tick * scale for event in plan.events]
    tick_expression = "+".join(
        f"{tick}*eq(N\\,{index})" for index, tick in enumerate(ticks)
    )
    branches.append(
        "".join(events)
        + f"concat=n={len(events)}:v=1:a=0,settb=expr=1/{timescale},"
        f"setpts={tick_expression},format=yuv420p[out]"
    )
    _run_media_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(plan.user.window.source.path),
            "-i",
            str(pro_panel),
            "-filter_complex",
            ";".join(branches),
            "-map",
            "[out]",
            "-an",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-enc_time_base",
            f"1:{timescale}",
            "-video_track_timescale",
            str(timescale),
            str(output),
            "-y",
        ]
    )


def _prepare_pro_panel(plan: ComparisonRenderPlan, output: Path) -> None:
    """Cache one lossless, cropped pro stream for all comparison segments."""

    _run_media_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(plan.pro.window.source.path),
            "-map",
            "0:v:0",
            "-vf",
            (
                f"crop={plan.pro.crop.width}:{plan.pro.crop.height}:"
                f"{plan.pro.crop.x}:{plan.pro.crop.y}"
            ),
            "-an",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            "-f",
            "matroska",
            str(output),
            "-y",
        ]
    )


def render_compilation(
    plans: tuple[ComparisonRenderPlan, ...], output: Path
) -> Path:
    """Encode ordered comparison plans once as one silent compilation."""

    if not plans:
        raise ValueError("at least one comparison plan is required")
    timescale = lcm(*(plan.output_time_base.denominator for plan in plans))
    if timescale > MAX_OUTPUT_TIMESCALE:
        raise ValueError("compilation requires an unsupported output timescale")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        pro_panel = temporary / "pro-panel.mkv"
        _prepare_pro_panel(plans[0], pro_panel)
        segments: list[Path] = []
        next_tick = 0
        for index, plan in enumerate(plans):
            scale = timescale // plan.output_time_base.denominator
            segment_end = plan.events[-1].output_tick * scale
            if next_tick + segment_end > MAX_EXACT_FILTER_TICK:
                raise ValueError("compilation ticks exceed the renderer's exact range")
            segment = temporary / f"comparison_{index:03d}.mp4"
            _render_plan_segment(plan, segment, timescale, pro_panel)
            segments.append(segment)
            next_tick += segment_end + 1

        concat_file = temporary / "segments.txt"
        concat_file.write_text(
            "".join(f"file '{segment}'\n" for segment in segments),
            encoding="utf-8",
        )
        _run_media_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-map",
                "0:v:0",
                "-an",
                "-c",
                "copy",
                "-video_track_timescale",
                str(timescale),
                str(output),
                "-y",
            ]
        )
    return output

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

from PIL import Image

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


def _render_panel(image_path: Path, crop: Rectangle, panel: Rectangle) -> Image.Image:
    with Image.open(image_path) as image:
        cropped = image.crop(
            (crop.x, crop.y, crop.x + crop.width, crop.y + crop.height)
        )
        return cropped.resize(
            (panel.width, panel.height), Image.Resampling.LANCZOS
        ).convert("RGB")


def _render_plan_event_images(
    plan: ComparisonRenderPlan,
    temporary: Path,
    event_directory: Path,
    *,
    plan_index: int,
    first_event_index: int,
) -> int:
    user_directory = temporary / f"user_{plan_index:03d}"
    pro_directory = temporary / f"pro_{plan_index:03d}"
    user_directory.mkdir()
    pro_directory.mkdir()
    user_frames = _decode_frames(
        plan.user.window.source.path,
        tuple(event.user_frame.ordinal for event in plan.events),
        user_directory,
    )
    pro_frames = _decode_frames(
        plan.pro.window.source.path,
        tuple(event.pro_frame.ordinal for event in plan.events),
        pro_directory,
    )
    for offset, event in enumerate(plan.events):
        canvas = Image.new(
            "RGB", (plan.layout.output.width, plan.layout.output.height), "black"
        )
        canvas.paste(
            _render_panel(
                user_frames[event.user_frame.ordinal],
                plan.user.crop,
                plan.layout.user_panel,
            ),
            plan.layout.user_panel.x_y,
        )
        canvas.paste(
            _render_panel(
                pro_frames[event.pro_frame.ordinal],
                plan.pro.crop,
                plan.layout.pro_panel,
            ),
            plan.layout.pro_panel.x_y,
        )
        canvas.save(event_directory / f"event_{first_event_index + offset:09d}.png")
    return first_event_index + len(plan.events)


def _encode_event_images(
    event_directory: Path,
    ticks: list[int],
    timescale: int,
    output: Path,
) -> None:
    tick_expression = "+".join(
        f"{tick}*eq(N\\,{index})" for index, tick in enumerate(ticks)
    )
    _run_media_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            "1",
            "-start_number",
            "0",
            "-i",
            str(event_directory / "event_%09d.png"),
            "-an",
            "-vf",
            f"settb=expr=1/{timescale},setpts={tick_expression}",
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


def render_comparison(plan: ComparisonRenderPlan) -> Path:
    """Render one planned silent, fixed-crop comparison clip."""

    plan.artifact.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        event_directory = temporary / "events"
        event_directory.mkdir()
        _render_plan_event_images(
            plan,
            temporary,
            event_directory,
            plan_index=0,
            first_event_index=0,
        )
        _encode_event_images(
            event_directory,
            [event.output_tick for event in plan.events],
            plan.output_time_base.denominator,
            plan.artifact.path,
        )
    return plan.artifact.path


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
        event_directory = temporary / "events"
        event_directory.mkdir()
        ticks: list[int] = []
        next_clip_tick = 0
        event_index = 0
        for plan_index, plan in enumerate(plans):
            event_index = _render_plan_event_images(
                plan,
                temporary,
                event_directory,
                plan_index=plan_index,
                first_event_index=event_index,
            )
            for event in plan.events:
                plan_tick_scale = timescale // plan.output_time_base.denominator
                ticks.append(next_clip_tick + event.output_tick * plan_tick_scale)
            next_clip_tick = ticks[-1] + 1

        if ticks[-1] > MAX_EXACT_FILTER_TICK:
            raise ValueError("compilation ticks exceed the renderer's exact range")

        _encode_event_images(event_directory, ticks, timescale, output)
    return output

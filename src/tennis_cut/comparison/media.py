"""Exact media inspection for comparison workflows."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Protocol

from PIL import Image

from .planning import (
    ComparisonRenderPlan,
    ComparisonSource,
    PlayerObservation,
    Rectangle,
    SelectedSourceWindow,
)
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


def inspect_comparison_source(video: Path) -> ComparisonSource:
    """Inspect exact frames and source geometry for comparison planning."""

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
    time_base = Fraction(stream["time_base"])
    stream_index = int(stream["index"])
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
    return ComparisonSource(
        path=video,
        width=int(stream["width"]),
        height=int(stream["height"]),
        inspected_media=InspectedMedia(frames),
    )


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
    subprocess.run(
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
        ],
        check=True,
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


def _decimal_seconds(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 40
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def render_comparison(plan: ComparisonRenderPlan) -> Path:
    """Render one planned silent, fixed-crop comparison clip."""

    plan.artifact.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        user_directory = temporary / "user"
        pro_directory = temporary / "pro"
        event_directory = temporary / "events"
        user_directory.mkdir()
        pro_directory.mkdir()
        event_directory.mkdir()
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

        event_paths: list[Path] = []
        for index, event in enumerate(plan.events):
            canvas = Image.new(
                "RGB",
                (plan.layout.output.width, plan.layout.output.height),
                "black",
            )
            user_panel = _render_panel(
                user_frames[event.user_frame.ordinal],
                plan.user.crop,
                plan.layout.user_panel,
            )
            pro_panel = _render_panel(
                pro_frames[event.pro_frame.ordinal],
                plan.pro.crop,
                plan.layout.pro_panel,
            )
            canvas.paste(user_panel, plan.layout.user_panel.x_y)
            canvas.paste(pro_panel, plan.layout.pro_panel.x_y)
            event_path = event_directory / f"event_{index:09d}.png"
            canvas.save(event_path)
            event_paths.append(event_path)

        concat_lines = ["ffconcat version 1.0"]
        for index, event_path in enumerate(event_paths):
            concat_lines.append(f"file '{event_path}'")
            if index + 1 < len(plan.events):
                tick_duration = (
                    plan.events[index + 1].output_tick - plan.events[index].output_tick
                )
                duration = tick_duration * plan.output_time_base
                concat_lines.append(f"duration {_decimal_seconds(duration)}")
        concat_path = temporary / "events.ffconcat"
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        timescale = plan.output_time_base.denominator
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-r",
                str(timescale),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
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
                str(plan.artifact.path),
                "-y",
            ],
            check=True,
        )
    return plan.artifact.path

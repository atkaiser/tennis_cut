"""Exact media inspection for comparison workflows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import json
import logging
from math import ceil, lcm
from pathlib import Path
import subprocess
import tempfile
from time import monotonic
from typing import Literal, Protocol

from PIL import Image

from tennis_cut.source_frames import (
    IndexedFrame,
    decode_exact_frame_windows,
    inspect_source_frame_index,
)

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

_LOG = logging.getLogger(__name__)
PLAYER_OBSERVATIONS_PER_NORMALIZED_SECOND = 10


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


def inspect_media(video: Path) -> InspectedMedia:
    """Inspect ordered decoded video frames without FPS-based reconstruction."""

    return _as_inspected_media(inspect_source_frame_index(video).frames)


def _as_inspected_media(frames: tuple[IndexedFrame, ...]) -> InspectedMedia:
    return InspectedMedia(
        tuple(
            DecodedFrame(
                frame.stream_index,
                frame.ordinal,
                frame.pts,
                frame.time_base,
            )
            for frame in frames
        )
    )


def inspect_comparison_source(video: Path) -> ComparisonSource:
    """Inspect exact frames and source geometry for comparison planning."""

    inspected = inspect_source_frame_index(video)
    return ComparisonSource(
        path=video,
        width=inspected.width,
        height=inspected.height,
        inspected_media=_as_inspected_media(inspected.frames),
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


def _balanced_addition(terms: list[str]) -> str:
    """Build a shallow FFmpeg expression tree for a non-empty sum."""

    if not terms:
        raise ValueError("at least one expression term is required")
    while len(terms) > 1:
        terms = [
            (
                f"({terms[index]}+{terms[index + 1]})"
                if index + 1 < len(terms)
                else terms[index]
            )
            for index in range(0, len(terms), 2)
        ]
    return terms[0]


def observe_players(
    window: SelectedSourceWindow,
    locator: PlayerLocator,
) -> tuple[PlayerObservation, ...]:
    """Locate a player from sparse, consumer-sized frames in one window."""

    workspace = DecodedFrameWorkspace()
    try:
        return workspace.observe_players_many((window,), locator)[0]
    finally:
        workspace.close()


def _sampled_player_ordinals(window: SelectedSourceWindow) -> tuple[int, ...]:
    """Select a stable, sparse set that includes both bounds and contact."""

    frames = window.normalized_frames
    if len(frames) <= 2:
        return tuple(frame.frame.ordinal for frame in frames)
    duration = frames[-1].offset - frames[0].offset
    sample_count = min(
        len(frames),
        max(2, ceil(duration * PLAYER_OBSERVATIONS_PER_NORMALIZED_SECOND) + 1),
    )
    indexes = {
        int(Fraction(index * (len(frames) - 1), sample_count - 1) + Fraction(1, 2))
        for index in range(sample_count)
    }
    indexes.add(
        next(
            index
            for index, frame in enumerate(frames)
            if frame.frame == window.contact_frame
        )
    )
    return tuple(frames[index].frame.ordinal for index in sorted(indexes))


def _find_player_boxes(
    locator: PlayerLocator, paths: tuple[Path, ...]
) -> tuple[tuple[int, int, int, int] | None, ...]:
    find_boxes = getattr(locator, "find_boxes", None)
    boxes = (
        tuple(find_boxes(paths))
        if callable(find_boxes)
        else tuple(locator.find_box(path) for path in paths)
    )
    if len(boxes) != len(paths):
        raise ValueError("player locator returned the wrong result count")
    return boxes


def _observations_from_decoded(
    sampled: tuple[tuple[int, ...], ...],
    locator: PlayerLocator,
    decoded: dict[int, Path],
    source_width: int,
    source_height: int,
) -> tuple[tuple[PlayerObservation, ...], ...]:
    ordinals = tuple(sorted({ordinal for group in sampled for ordinal in group}))
    with Image.open(decoded[ordinals[0]]) as sample_image:
        decoded_width, decoded_height = sample_image.size
    x_scale = Fraction(source_width, decoded_width)
    y_scale = Fraction(source_height, decoded_height)
    boxes = _find_player_boxes(
        locator,
        tuple(decoded[ordinal] for ordinal in ordinals),
    )
    by_ordinal = dict(zip(ordinals, boxes, strict=True))
    results = []
    for group in sampled:
        observations = []
        for ordinal in group:
            box = by_ordinal[ordinal]
            if box is None:
                continue
            x1, y1, x2, y2 = box
            x1 = round(x1 * x_scale)
            y1 = round(y1 * y_scale)
            x2 = round(x2 * x_scale)
            y2 = round(y2 * y_scale)
            observations.append(
                PlayerObservation(ordinal, Rectangle(x1, y1, x2 - x1, y2 - y1))
            )
        results.append(tuple(observations))
    return tuple(results)


class DecodedFrameWorkspace:
    """Retain consumer-sized player-observation frames for each source."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self._decoded: dict[Path, dict[int, Path]] = {}

    def observe_players_many(
        self,
        windows: tuple[SelectedSourceWindow, ...],
        locator: PlayerLocator,
    ) -> tuple[tuple[PlayerObservation, ...], ...]:
        if not windows:
            return ()
        source = windows[0].source.path
        if any(window.source.path != source for window in windows):
            raise ValueError("player-observation windows must share one source")
        sampled = tuple(_sampled_player_ordinals(window) for window in windows)
        sampled_ordinals = {ordinal for group in sampled for ordinal in group}
        all_frames_by_ordinal = {
            frame.frame.ordinal: frame.frame
            for window in windows
            for frame in window.normalized_frames
            if frame.frame.ordinal in sampled_ordinals
        }
        sampled_frames = tuple(
            all_frames_by_ordinal[ordinal]
            for ordinal in sorted(all_frames_by_ordinal)
        )
        decoded = self._frames(source, sampled_frames)
        _LOG.info(
            "Running player detection on %d consumer-sized sampled frames",
            len(sampled_frames),
        )
        source_geometry = windows[0].source
        return _observations_from_decoded(
            sampled,
            locator,
            decoded,
            source_geometry.width,
            source_geometry.height,
        )

    def _frames(
        self, source: Path, frames: tuple[DecodedFrame, ...]
    ) -> dict[int, Path]:
        decoded = self._decoded.get(source)
        if decoded is None:
            directory = Path(self._temporary.name) / f"source_{len(self._decoded):02d}"
            directory.mkdir()
            decoded = decode_exact_frame_windows(
                source,
                frames,
                directory,
                max_dimension=1280,
            )
            self._decoded[source] = decoded
        ordinals = {frame.ordinal for frame in frames}
        missing = set(ordinals) - decoded.keys()
        if missing:
            raise ValueError(
                "decoded frame workspace was not initialized with every required frame"
            )
        return decoded

    def close(self) -> None:
        self._temporary.cleanup()


def _panel_key(
    ordinal: int, crop: Rectangle, panel: Rectangle
) -> tuple[int, Rectangle, int, int]:
    return ordinal, crop, panel.width, panel.height


@dataclass(frozen=True)
class _PanelDecodeTask:
    role: Literal["user", "pro"]
    source: Path
    source_start_timestamp: Fraction
    crop: Rectangle
    output_size: tuple[int, int]
    frames: tuple[DecodedFrame, ...]


def _prepare_render_panels(
    plans: tuple[ComparisonRenderPlan, ...],
    temporary: Path,
) -> tuple[
    dict[tuple[int, Rectangle, int, int], Path],
    dict[tuple[int, Rectangle, int, int], Path],
]:
    user_source = plans[0].user.window.source.path
    pro_source = plans[0].pro.window.source.path
    if any(plan.user.window.source.path != user_source for plan in plans):
        raise ValueError("comparison plans must share one user source")
    if any(plan.pro.window.source.path != pro_source for plan in plans):
        raise ValueError("comparison plans must share one pro source")

    user_requests = {
        _panel_key(event.user_frame.ordinal, plan.user.crop, plan.layout.user_panel)
        for plan in plans
        for event in plan.events
    }
    pro_requests = {
        _panel_key(event.pro_frame.ordinal, plan.pro.crop, plan.layout.pro_panel)
        for plan in plans
        for event in plan.events
    }
    requested_user_frames = {
        event.user_frame.ordinal: event.user_frame
        for plan in plans
        for event in plan.events
    }
    requested_pro_frames = {
        event.pro_frame.ordinal: event.pro_frame
        for plan in plans
        for event in plan.events
    }

    def transform_tasks(
        role: Literal["user", "pro"],
        source: Path,
        source_start_timestamp: Fraction,
        requests: set[tuple[int, Rectangle, int, int]],
        frames_by_ordinal: dict[int, DecodedFrame],
    ) -> list[_PanelDecodeTask]:
        grouped: dict[tuple[Rectangle, int, int], set[int]] = {}
        for ordinal, crop, width, height in requests:
            grouped.setdefault((crop, width, height), set()).add(ordinal)
        return [
            _PanelDecodeTask(
                role,
                source,
                source_start_timestamp,
                crop,
                (width, height),
                tuple(frames_by_ordinal[ordinal] for ordinal in sorted(ordinals)),
            )
            for (crop, width, height), ordinals in sorted(
                grouped.items(),
                key=lambda item: (
                    item[0][0].x,
                    item[0][0].y,
                    item[0][0].width,
                    item[0][0].height,
                    item[0][1],
                    item[0][2],
                ),
            )
        ]

    tasks = transform_tasks(
        "user",
        user_source,
        plans[0].user.window.source.inspected_media.frames[0].timestamp,
        user_requests,
        requested_user_frames,
    ) + transform_tasks(
        "pro",
        pro_source,
        plans[0].pro.window.source.inspected_media.frames[0].timestamp,
        pro_requests,
        requested_pro_frames,
    )
    _LOG.info(
        "Decoding %d requested panels directly at output size in %d transform task(s)",
        len(user_requests) + len(pro_requests),
        len(tasks),
    )
    started = monotonic()

    def prepare_task(
        item: tuple[int, _PanelDecodeTask],
    ) -> tuple[Literal["user", "pro"], dict]:
        index, task = item
        decoded = decode_exact_frame_windows(
            task.source,
            task.frames,
            temporary / f"{task.role}_panels_{index:03d}",
            max_workers=1,
            source_start_timestamp=task.source_start_timestamp,
            crop=(task.crop.x, task.crop.y, task.crop.width, task.crop.height),
            output_size=task.output_size,
        )
        width, height = task.output_size
        return task.role, {
            _panel_key(ordinal, task.crop, Rectangle(0, 0, width, height)): path
            for ordinal, path in decoded.items()
        }

    workers = min(4, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(prepare_task, enumerate(tasks)))
    user_panels = {
        key: path for role, panels in results if role == "user" for key, path in panels.items()
    }
    pro_panels = {
        key: path for role, panels in results if role == "pro" for key, path in panels.items()
    }
    _LOG.info(
        "Finished direct panel decode: %d panels in %.1fs",
        len(user_panels) + len(pro_panels),
        monotonic() - started,
    )
    return user_panels, pro_panels


def _render_plan_event_images(
    plan: ComparisonRenderPlan,
    event_directory: Path,
    user_panels: dict[tuple[int, Rectangle, int, int], Path],
    pro_panels: dict[tuple[int, Rectangle, int, int], Path],
    *,
    plan_index: int,
    first_event_index: int,
) -> int:
    _LOG.info(
        "Materializing comparison %d: %d output events",
        plan_index + 1,
        len(plan.events),
    )
    started = monotonic()
    for offset, event in enumerate(plan.events):
        canvas = Image.new(
            "RGB", (plan.layout.output.width, plan.layout.output.height), "black"
        )
        user_path = user_panels[
            _panel_key(
                event.user_frame.ordinal,
                plan.user.crop,
                plan.layout.user_panel,
            )
        ]
        pro_path = pro_panels[
            _panel_key(
                event.pro_frame.ordinal,
                plan.pro.crop,
                plan.layout.pro_panel,
            )
        ]
        with Image.open(user_path) as user_panel:
            canvas.paste(user_panel, plan.layout.user_panel.x_y)
        with Image.open(pro_path) as pro_panel:
            canvas.paste(pro_panel, plan.layout.pro_panel.x_y)
        try:
            canvas.save(
                event_directory / f"event_{first_event_index + offset:09d}.png"
            )
        finally:
            canvas.close()
    _LOG.info(
        "Finished materializing comparison %d in %.1fs",
        plan_index + 1,
        monotonic() - started,
    )
    return first_event_index + len(plan.events)


def _encode_event_images(
    event_directory: Path,
    ticks: list[int],
    timescale: int,
    output: Path,
) -> None:
    _LOG.info("Encoding %d materialized events: %s", len(ticks), output)
    started = monotonic()
    tick_expression = _balanced_addition(
        [f"{tick}*eq(N\\,{index})" for index, tick in enumerate(ticks)]
    )
    _run_media_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            str(timescale),
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
    _LOG.info("Finished video encoding in %.1fs", monotonic() - started)


def render_comparison(plan: ComparisonRenderPlan) -> Path:
    """Render one planned silent, fixed-crop comparison clip."""

    plan.artifact.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        event_directory = temporary / "events"
        event_directory.mkdir()
        user_panels, pro_panels = _prepare_render_panels((plan,), temporary)
        _render_plan_event_images(
            plan,
            event_directory,
            user_panels,
            pro_panels,
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
    _LOG.info("Preparing materialized compilation for %d comparison(s)", len(plans))
    timescale = lcm(*(plan.output_time_base.denominator for plan in plans))
    if timescale > MAX_OUTPUT_TIMESCALE:
        raise ValueError("compilation requires an unsupported output timescale")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        event_directory = temporary / "events"
        event_directory.mkdir()
        user_panels, pro_panels = _prepare_render_panels(plans, temporary)
        ticks = _materialize_compilation_events(
            plans, event_directory, user_panels, pro_panels, timescale
        )
        _encode_event_images(event_directory, ticks, timescale, output)
    return output


def _materialize_compilation_events(
    plans: tuple[ComparisonRenderPlan, ...],
    event_directory: Path,
    user_panels: dict[tuple[int, Rectangle, int, int], Path],
    pro_panels: dict[tuple[int, Rectangle, int, int], Path],
    timescale: int,
) -> list[int]:
    ticks: list[int] = []
    next_clip_tick = 0
    event_index = 0
    for plan_index, plan in enumerate(plans):
        event_index = _render_plan_event_images(
            plan,
            event_directory,
            user_panels,
            pro_panels,
            plan_index=plan_index,
            first_event_index=event_index,
        )
        plan_tick_scale = timescale // plan.output_time_base.denominator
        ticks.extend(
            next_clip_tick + event.output_tick * plan_tick_scale
            for event in plan.events
        )
        next_clip_tick = ticks[-1] + 1
    if ticks[-1] > MAX_EXACT_FILTER_TICK:
        raise ValueError("compilation ticks exceed the renderer's exact range")
    return ticks


def render_comparison_artifacts(
    plans: tuple[ComparisonRenderPlan, ...],
    compilation: Path,
    clips: tuple[Path, ...],
) -> None:
    """Render all artifacts from one set of consumer-sized panels."""

    if not plans:
        raise ValueError("at least one comparison plan is required")
    if clips and len(clips) != len(plans):
        raise ValueError("clip count must match comparison plan count")
    if clips and any(
        plan.artifact.path != clip
        for plan, clip in zip(plans, clips, strict=True)
    ):
        raise ValueError("clip artifact does not match its render plan")
    _LOG.info("Preparing materialized compilation for %d comparison(s)", len(plans))
    timescale = lcm(*(plan.output_time_base.denominator for plan in plans))
    if timescale > MAX_OUTPUT_TIMESCALE:
        raise ValueError("compilation requires an unsupported output timescale")

    compilation.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        user_panels, pro_panels = _prepare_render_panels(plans, temporary)
        compilation_events = temporary / "compilation_events"
        compilation_events.mkdir()
        ticks = _materialize_compilation_events(
            plans,
            compilation_events,
            user_panels,
            pro_panels,
            timescale,
        )
        _encode_event_images(compilation_events, ticks, timescale, compilation)

        if clips:
            for plan_index, (plan, clip) in enumerate(
                zip(plans, clips, strict=True)
            ):
                clip.parent.mkdir(parents=True, exist_ok=True)
                clip_events = temporary / f"clip_events_{plan_index:03d}"
                clip_events.mkdir()
                _render_plan_event_images(
                    plan,
                    clip_events,
                    user_panels,
                    pro_panels,
                    plan_index=plan_index,
                    first_event_index=0,
                )
                _encode_event_images(
                    clip_events,
                    [event.output_tick for event in plan.events],
                    plan.output_time_base.denominator,
                    clip,
                )

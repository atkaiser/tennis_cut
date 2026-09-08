"""Exact media inspection for comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import logging
from math import ceil, lcm
from pathlib import Path
import subprocess
import tempfile
from time import monotonic
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
    selection = _balanced_addition(
        [f"eq(n\\,{ordinal})" for ordinal in unique_ordinals]
    )
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
) -> tuple[tuple[PlayerObservation, ...], ...]:
    ordinals = tuple(sorted({ordinal for group in sampled for ordinal in group}))
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
            observations.append(
                PlayerObservation(ordinal, Rectangle(x1, y1, x2 - x1, y2 - y1))
            )
        results.append(tuple(observations))
    return tuple(results)


class DecodedFrameWorkspace:
    """Share exact decoded frames across observation and artifact rendering."""

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
        all_ordinals = tuple(
            sorted(
                {
                    frame.frame.ordinal
                    for window in windows
                    for frame in window.normalized_frames
                }
            )
        )
        decoded = self._frames(source, all_ordinals)
        sampled = tuple(_sampled_player_ordinals(window) for window in windows)
        _LOG.info(
            "Running player detection on %d sampled frames from %d decoded frames",
            len({ordinal for group in sampled for ordinal in group}),
            len(all_ordinals),
        )
        return _observations_from_decoded(sampled, locator, decoded)

    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        compilation: Path,
        clips: tuple[Path, ...],
    ) -> None:
        render_comparison_artifacts(
            plans,
            compilation,
            clips,
            decoded_sources=self._decoded,
        )

    def _frames(
        self, source: Path, ordinals: tuple[int, ...]
    ) -> dict[int, Path]:
        decoded = self._decoded.get(source)
        if decoded is None:
            directory = Path(self._temporary.name) / f"source_{len(self._decoded):02d}"
            directory.mkdir()
            decoded = _decode_frames(source, ordinals, directory)
            self._decoded[source] = decoded
        missing = set(ordinals) - decoded.keys()
        if missing:
            raise ValueError(
                "decoded frame workspace was not initialized with every required frame"
            )
        return decoded

    def close(self) -> None:
        self._temporary.cleanup()


def _render_panel(image_path: Path, crop: Rectangle, panel: Rectangle) -> Image.Image:
    with Image.open(image_path) as image:
        cropped = image.crop(
            (crop.x, crop.y, crop.x + crop.width, crop.y + crop.height)
        )
        return cropped.resize(
            (panel.width, panel.height), Image.Resampling.LANCZOS
        ).convert("RGB")


def _panel_key(
    ordinal: int, crop: Rectangle, panel: Rectangle
) -> tuple[int, Rectangle, int, int]:
    return ordinal, crop, panel.width, panel.height


def _prepare_panels(
    decoded: dict[int, Path],
    requests: set[tuple[int, Rectangle, int, int]],
    output_directory: Path,
) -> dict[tuple[int, Rectangle, int, int], Path]:
    output_directory.mkdir()
    prepared = {}
    for index, key in enumerate(
        sorted(
            requests,
            key=lambda item: (
                item[0],
                item[1].x,
                item[1].y,
                item[1].width,
                item[1].height,
                item[2],
                item[3],
            ),
        )
    ):
        ordinal, crop, width, height = key
        destination = output_directory / f"panel_{index:09d}.png"
        panel = Rectangle(0, 0, width, height)
        image = _render_panel(decoded[ordinal], crop, panel)
        try:
            image.save(destination)
        finally:
            image.close()
        prepared[key] = destination
    return prepared


def _prepare_render_panels(
    plans: tuple[ComparisonRenderPlan, ...],
    temporary: Path,
    decoded_sources: dict[Path, dict[int, Path]] | None = None,
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
    if decoded_sources is None:
        user_directory = temporary / "user_frames"
        pro_directory = temporary / "pro_frames"
        user_directory.mkdir()
        pro_directory.mkdir()
        _LOG.info(
            "Decoding %d distinct user frames in one source traversal",
            len({request[0] for request in user_requests}),
        )
        user_frames = _decode_frames(
            user_source,
            tuple(request[0] for request in user_requests),
            user_directory,
        )
        _LOG.info(
            "Decoding %d distinct pro frames in one source traversal",
            len({request[0] for request in pro_requests}),
        )
        pro_frames = _decode_frames(
            pro_source,
            tuple(request[0] for request in pro_requests),
            pro_directory,
        )
    else:
        user_frames = decoded_sources[user_source]
        pro_frames = decoded_sources[pro_source]
    return (
        _prepare_panels(user_frames, user_requests, temporary / "user_panels"),
        _prepare_panels(pro_frames, pro_requests, temporary / "pro_panels"),
    )


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
    *,
    decoded_sources: dict[Path, dict[int, Path]] | None = None,
) -> None:
    """Render all artifacts from one decoded and prepared frame set."""

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
        user_panels, pro_panels = _prepare_render_panels(
            plans, temporary, decoded_sources
        )
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

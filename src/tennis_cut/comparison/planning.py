"""Pure exact-time planning for contact-aligned comparison clips."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Iterable

from tennis_cut.swing_detection import DetectedSwing
from tennis_cut.visual_contact import SourceFrameIdentity

from .policy import COMPARISON_POLICY, ComparisonPolicy
from .pro_selection import DecodedFrame, InspectedMedia, ProSelection

MAX_OUTPUT_TIMESCALE = 2_147_483_647
MAX_EXACT_FILTER_TICK = 2**53
COMPARISON_OUTPUT_FPS = 60


class UnrepresentableTimeline(ValueError):
    """The target container cannot assign a distinct tick to every event."""


@dataclass(frozen=True)
class ComparisonSource:
    """One inspected video and its pixel geometry."""

    path: Path
    width: int
    height: int
    inspected_media: InspectedMedia


@dataclass(frozen=True)
class NormalizedFrame:
    """A decoded source frame at an exact normalized frame-change event."""

    frame: DecodedFrame
    offset: Fraction


@dataclass(frozen=True)
class SelectedSourceWindow:
    """Complete normalized source window selected before player observation."""

    source: ComparisonSource
    swing_ordinal: int | None
    contact_timestamp: Fraction
    contact_frame: DecodedFrame
    normalized_frames: tuple[NormalizedFrame, ...]


@dataclass(frozen=True)
class SelectedComparisonWindows:
    """Reusable pro window and matching user windows in accepted order."""

    pro: SelectedSourceWindow
    user: tuple[SelectedSourceWindow, ...]


@dataclass(frozen=True)
class Rectangle:
    """Integer pixel rectangle in top-left origin coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x_y(self) -> tuple[int, int]:
        """Return the top-left coordinate for image composition."""

        return self.x, self.y


@dataclass(frozen=True)
class PlayerObservation:
    """Player bounds observed on one decoded frame."""

    frame_ordinal: int
    bounds: Rectangle


@dataclass(frozen=True)
class PreparedSourceWindow:
    """Selected schedule with reusable observations and one fixed crop."""

    window: SelectedSourceWindow
    observations: tuple[PlayerObservation, ...]
    crop: Rectangle


@dataclass(frozen=True)
class PanelLayout:
    """Stable full-bleed user-left/pro-right output geometry."""

    output: Rectangle
    user_panel: Rectangle
    pro_panel: Rectangle


@dataclass(frozen=True)
class ClipBounds:
    """Exact normalized and output bounds for one comparison clip."""

    normalized_start: Fraction
    normalized_end: Fraction
    output_start_tick: int
    output_end_tick: int


@dataclass(frozen=True)
class RenderEvent:
    """One output frame event and the source frames active there."""

    normalized_time: Fraction
    output_tick: int
    user_frame: DecodedFrame
    pro_frame: DecodedFrame


@dataclass(frozen=True)
class ArtifactRequest:
    """Requested comparison artifact, independent of encoder details."""

    path: Path


@dataclass(frozen=True)
class ComparisonRenderPlan:
    """Complete immutable description consumed by a media renderer."""

    user: PreparedSourceWindow
    pro: PreparedSourceWindow
    events: tuple[RenderEvent, ...]
    layout: PanelLayout
    clip_bounds: ClipBounds
    output_time_base: Fraction
    slow_motion: Fraction
    artifact: ArtifactRequest


def _active_frame(frames: Iterable[DecodedFrame], timestamp: Fraction) -> DecodedFrame:
    active: DecodedFrame | None = None
    for frame in frames:
        if frame.timestamp > timestamp:
            break
        active = frame
    if active is None:
        raise ValueError("no source frame is active at the requested timestamp")
    return active


def _select_source_window(
    *,
    source: ComparisonSource,
    contact_timestamp: Fraction,
    speed: Fraction,
    swing_ordinal: int | None,
    policy: ComparisonPolicy,
    required_contact_frame: DecodedFrame | None = None,
) -> SelectedSourceWindow | None:
    if speed <= 0:
        raise ValueError("source speed must be positive")
    frames = source.inspected_media.frames
    normalized_start = -policy.pre_contact
    normalized_end = policy.post_contact
    source_start = contact_timestamp + normalized_start / speed
    source_end = contact_timestamp + normalized_end / speed
    if not frames or frames[0].timestamp > source_start or frames[-1].timestamp < source_end:
        return None

    boundary_frame = _active_frame(frames, source_start)
    contact_frame = required_contact_frame or _active_frame(frames, contact_timestamp)
    if required_contact_frame is not None and required_contact_frame not in frames:
        raise ValueError("selected contact frame does not belong to the source")
    if required_contact_frame is not None and required_contact_frame.timestamp != contact_timestamp:
        raise ValueError("selected contact frame timestamp does not match the contact timestamp")

    normalized_frames = [NormalizedFrame(boundary_frame, normalized_start)]
    normalized_frames.extend(
        NormalizedFrame(frame, (frame.timestamp - contact_timestamp) * speed)
        for frame in frames
        if source_start < frame.timestamp <= source_end
    )
    return SelectedSourceWindow(
        source=source,
        swing_ordinal=swing_ordinal,
        contact_timestamp=contact_timestamp,
        contact_frame=contact_frame,
        normalized_frames=tuple(normalized_frames),
    )


def _frame_for_identity(
    source: ComparisonSource, identity: SourceFrameIdentity
) -> DecodedFrame:
    matches = tuple(
        frame
        for frame in source.inspected_media.frames
        if (
            frame.stream_index == identity.stream_index
            and frame.pts == identity.pts
            and frame.time_base == identity.time_base
        )
    )
    if len(matches) != 1:
        raise ValueError("selected user contact frame does not belong to the source")
    return matches[0]


def select_comparison_windows(
    *,
    user_source: ComparisonSource,
    user_swings: tuple[DetectedSwing, ...],
    pro_source: ComparisonSource,
    pro_selection: ProSelection,
    pro_speed: Fraction,
    policy: ComparisonPolicy = COMPARISON_POLICY,
) -> SelectedComparisonWindows:
    """Select complete matching windows without player-location effects."""

    if pro_selection.source != pro_source.path:
        raise ValueError("pro selection does not belong to the pro source")
    pro_window = _select_source_window(
        source=pro_source,
        contact_timestamp=pro_selection.frame.timestamp,
        speed=pro_speed,
        swing_ordinal=None,
        policy=policy,
        required_contact_frame=pro_selection.frame,
    )
    if pro_window is None:
        raise ValueError("pro selection lacks the complete comparison window")

    user_windows: list[SelectedSourceWindow] = []
    for swing in user_swings:
        if (
            swing.shot_type not in policy.supported_shot_types
            or swing.shot_type != pro_selection.shot_type
        ):
            continue
        window = _select_source_window(
            source=user_source,
            contact_timestamp=swing.contact_timestamp,
            speed=Fraction(1),
            swing_ordinal=swing.ordinal,
            policy=policy,
            required_contact_frame=(
                _frame_for_identity(user_source, swing.contact_frame)
                if swing.contact_frame is not None
                else None
            ),
        )
        if window is not None:
            user_windows.append(window)
    return SelectedComparisonWindows(pro=pro_window, user=tuple(user_windows))


def _validate_rectangle(rectangle: Rectangle, source: ComparisonSource) -> None:
    if rectangle.width <= 0 or rectangle.height <= 0:
        raise ValueError("player observation must have positive dimensions")
    if (
        rectangle.x < 0
        or rectangle.y < 0
        or rectangle.x + rectangle.width > source.width
        or rectangle.y + rectangle.height > source.height
    ):
        raise ValueError("player observation lies outside the source frame")


def _fixed_crop(
    source: ComparisonSource,
    observations: tuple[PlayerObservation, ...],
    policy: ComparisonPolicy,
) -> Rectangle:
    if not observations:
        raise ValueError("at least one player observation is required")
    for observation in observations:
        _validate_rectangle(observation.bounds, source)

    left = min(observation.bounds.x for observation in observations)
    top = min(observation.bounds.y for observation in observations)
    right = max(
        observation.bounds.x + observation.bounds.width
        for observation in observations
    )
    bottom = max(
        observation.bounds.y + observation.bounds.height
        for observation in observations
    )
    envelope_width = right - left
    envelope_height = bottom - top
    expanded_width = ceil(envelope_width * (1 + policy.crop_margin))
    expanded_height = ceil(envelope_height * (1 + policy.crop_margin))
    aspect_width, aspect_height = policy.panel_aspect
    aspect_scale = max(
        ceil(Fraction(expanded_width, aspect_width)),
        ceil(Fraction(expanded_height, aspect_height)),
    )
    crop_width = aspect_scale * aspect_width
    crop_height = aspect_scale * aspect_height
    if crop_width > source.width or crop_height > source.height:
        raise ValueError("8:9 crop with 25% margin cannot fit inside the source")

    center_x = Fraction(left + right, 2)
    center_y = Fraction(top + bottom, 2)
    crop_x = int(center_x - Fraction(crop_width, 2))
    crop_y = int(center_y - Fraction(crop_height, 2))
    crop_x = min(max(0, crop_x), source.width - crop_width)
    crop_y = min(max(0, crop_y), source.height - crop_height)
    return Rectangle(crop_x, crop_y, crop_width, crop_height)


def prepare_source_window(
    window: SelectedSourceWindow,
    observations: tuple[PlayerObservation, ...],
    policy: ComparisonPolicy = COMPARISON_POLICY,
) -> PreparedSourceWindow:
    """Union complete-window observations into one reusable fixed crop."""

    return PreparedSourceWindow(
        window=window,
        observations=observations,
        crop=_fixed_crop(window.source, observations, policy),
    )


def _frame_active_at(
    normalized_frames: tuple[NormalizedFrame, ...], offset: Fraction
) -> DecodedFrame:
    active = normalized_frames[0].frame
    for normalized_frame in normalized_frames[1:]:
        if normalized_frame.offset > offset:
            break
        active = normalized_frame.frame
    return active


def _output_time_base(
    event_times: tuple[Fraction, ...],
    *,
    normalized_start: Fraction,
    slow_motion: Fraction,
) -> Fraction:
    for event_time in event_times:
        output_time = (event_time - normalized_start) / slow_motion
        if (output_time * COMPARISON_OUTPUT_FPS).denominator != 1:
            raise UnrepresentableTimeline(
                "comparison events cannot be represented at constant 60 fps"
            )
    return Fraction(1, COMPARISON_OUTPUT_FPS)


def build_render_plan(
    *,
    user: PreparedSourceWindow,
    pro: PreparedSourceWindow,
    slow_motion: Fraction,
    artifact: ArtifactRequest,
    policy: ComparisonPolicy = COMPARISON_POLICY,
) -> ComparisonRenderPlan:
    """Build the exact event union and stable equal-panel render plan."""

    if not 0 < slow_motion <= 1:
        raise ValueError("slow motion must be greater than zero and at most one")
    for prepared in (user, pro):
        offsets: dict[Fraction, DecodedFrame] = {}
        for normalized_frame in prepared.window.normalized_frames:
            previous = offsets.get(normalized_frame.offset)
            if previous is not None and previous != normalized_frame.frame:
                raise UnrepresentableTimeline(
                    "distinct source frames occupy the same normalized time"
                )
            offsets[normalized_frame.offset] = normalized_frame.frame
    normalized_start = -policy.pre_contact
    normalized_end = policy.post_contact
    event_times = {
        normalized_start,
        Fraction(0),
        normalized_end,
        *(
            item.offset
            for item in user.window.normalized_frames
            if normalized_start <= item.offset <= normalized_end
        ),
        *(
            item.offset
            for item in pro.window.normalized_frames
            if normalized_start <= item.offset <= normalized_end
        ),
    }
    ordered_times = tuple(sorted(event_times))
    time_base = _output_time_base(
        ordered_times,
        normalized_start=normalized_start,
        slow_motion=slow_motion,
    )
    events = tuple(
        RenderEvent(
            normalized_time=event_time,
            output_tick=int(
                ((event_time - normalized_start) / slow_motion) / time_base
            ),
            user_frame=(
                user.window.contact_frame
                if event_time == 0
                else _frame_active_at(user.window.normalized_frames, event_time)
            ),
            pro_frame=(
                pro.window.contact_frame
                if event_time == 0
                else _frame_active_at(pro.window.normalized_frames, event_time)
            ),
        )
        for event_time in ordered_times
    )
    if len({event.output_tick for event in events}) != len(events):
        raise UnrepresentableTimeline("distinct comparison events merged into one tick")
    if events[-1].output_tick > MAX_EXACT_FILTER_TICK:
        raise UnrepresentableTimeline(
            "comparison event ticks exceed the renderer's exact integer range"
        )

    output_width, output_height = policy.output_size
    panel_width = output_width // 2
    layout = PanelLayout(
        output=Rectangle(0, 0, output_width, output_height),
        user_panel=Rectangle(*policy.user_panel_origin, panel_width, output_height),
        pro_panel=Rectangle(*policy.pro_panel_origin, panel_width, output_height),
    )
    return ComparisonRenderPlan(
        user=user,
        pro=pro,
        events=events,
        layout=layout,
        clip_bounds=ClipBounds(
            normalized_start=normalized_start,
            normalized_end=normalized_end,
            output_start_tick=events[0].output_tick,
            output_end_tick=events[-1].output_tick,
        ),
        output_time_base=time_base,
        slow_motion=slow_motion,
        artifact=artifact,
    )

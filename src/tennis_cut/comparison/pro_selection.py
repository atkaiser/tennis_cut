"""Find the exact visual contact frame in a professional reference clip."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from tennis_cut.visual_contact import ContactSelection

from .policy import COMPARISON_POLICY


@dataclass(frozen=True)
class DecodedFrame:
    """Exact identity of one decoded video frame."""

    stream_index: int
    ordinal: int
    pts: int
    time_base: Fraction

    @property
    def timestamp(self) -> Fraction:
        return self.pts * self.time_base


@dataclass(frozen=True)
class InspectedMedia:
    """Ordered decoded video frames exposed by media inspection."""

    frames: tuple[DecodedFrame, ...]

    def frame_at(self, ordinal: int) -> DecodedFrame:
        """Return the decoded frame with the requested media ordinal."""

        return next(frame for frame in self.frames if frame.ordinal == ordinal)


@dataclass(frozen=True)
class ProSelection:
    """Automatically detected exact contact frame for one pro video."""

    source: Path
    frame: DecodedFrame
    shot_type: str = "forehand"


@dataclass(frozen=True)
class SelectionProcessingFailure:
    """Typed contact-finding failure that should stop comparison processing."""

    stage: str
    message: str


class VisualContactFinder(Protocol):
    """Find visual contact within a source-time radius of a timestamp."""

    def select(
        self,
        source: Path,
        candidate_timestamp: Fraction,
        *,
        radius: Fraction = Fraction(2, 5),
    ) -> ContactSelection: ...


def _middle_timestamp(inspected_media: InspectedMedia) -> Fraction:
    first_frame = inspected_media.frames[0]
    last_frame = inspected_media.frames[-1]
    return first_frame.timestamp + (last_frame.timestamp - first_frame.timestamp) / 2


def _middle_75_percent_radius(inspected_media: InspectedMedia) -> Fraction:
    first_frame = inspected_media.frames[0]
    last_frame = inspected_media.frames[-1]
    return (last_frame.timestamp - first_frame.timestamp) * Fraction(3, 8)


def _exact_decoded_frame(
    inspected_media: InspectedMedia, selection: ContactSelection
) -> DecodedFrame | None:
    if selection.frame is None:
        return None
    evidence = selection.frame.evidence
    matches = tuple(
        frame
        for frame in inspected_media.frames
        if (
            frame.ordinal == evidence.ordinal
            and frame.stream_index == evidence.stream_index
            and frame.pts == evidence.pts
            and frame.time_base == evidence.time_base
        )
    )
    return matches[0] if len(matches) == 1 else None


def find_pro_contact(
    *,
    pro_video: Path,
    pro_speed: Fraction,
    inspected_media: InspectedMedia,
    finder: VisualContactFinder,
) -> ProSelection | SelectionProcessingFailure:
    """Find and validate the pro clip's contact frame using visual evidence."""

    if not inspected_media.frames:
        return SelectionProcessingFailure(
            stage="pro contact detection",
            message="pro video has no decoded frames",
        )

    selection = finder.select(
        pro_video,
        _middle_timestamp(inspected_media),
        radius=_middle_75_percent_radius(inspected_media),
    )
    if selection.frame is None:
        return SelectionProcessingFailure(
            stage="pro contact detection",
            message=selection.omission_reason or "visual contact unavailable",
        )

    frame = _exact_decoded_frame(inspected_media, selection)
    if frame is None:
        return SelectionProcessingFailure(
            stage="pro contact detection",
            message="visual contact finder returned a frame outside the pro source",
        )

    first_frame = inspected_media.frames[0]
    last_frame = inspected_media.frames[-1]
    available_before = (frame.timestamp - first_frame.timestamp) * pro_speed
    available_after = (last_frame.timestamp - frame.timestamp) * pro_speed
    if (
        available_before < COMPARISON_POLICY.pre_contact
        or available_after < COMPARISON_POLICY.post_contact
    ):
        return SelectionProcessingFailure(
            stage="pro contact detection",
            message="detected contact lacks the complete comparison window",
        )

    return ProSelection(pro_video, frame)


__all__ = [
    "DecodedFrame",
    "InspectedMedia",
    "ProSelection",
    "SelectionProcessingFailure",
    "VisualContactFinder",
    "find_pro_contact",
]

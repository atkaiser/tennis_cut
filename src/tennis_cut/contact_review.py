"""Validated contact-review dataset used by offline detector experiments."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewedFrame:
    stream_index: int
    ordinal: int
    pts: int
    time_base: Fraction


@dataclass(frozen=True)
class ReviewedVisualState:
    frames: tuple[ReviewedFrame, ...]


@dataclass(frozen=True)
class ReviewedContact:
    video: Path
    annotation: Path
    family: str
    split: str
    suitability: str
    shot_index: int
    labeled_timestamp: Fraction
    shot_type: str
    status: str
    canonical_frame: ReviewedFrame | None
    visual_states: tuple[ReviewedVisualState, ...]

    @property
    def acceptable_ordinals(self) -> tuple[int, ...]:
        return tuple(frame.ordinal for state in self.visual_states for frame in state.frames)


@dataclass(frozen=True)
class ReviewDataset:
    contacts: tuple[ReviewedContact, ...]
    invalid: tuple[str, ...]

    def primary(self, split: str) -> tuple[ReviewedContact, ...]:
        return tuple(
            contact
            for contact in self.contacts
            if contact.suitability == "in_scope" and contact.split == split
        )

    @property
    def out_of_scope(self) -> tuple[ReviewedContact, ...]:
        return tuple(
            contact
            for contact in self.contacts
            if contact.suitability == "out_of_scope_view"
        )


def _frame(payload: Any) -> ReviewedFrame:
    if not isinstance(payload, dict):
        raise ValueError("frame must be an object")
    time_base = payload.get("time_base")
    if not isinstance(time_base, dict):
        raise ValueError("frame time_base must be an object")
    return ReviewedFrame(
        stream_index=int(payload["stream_index"]),
        ordinal=int(payload["ordinal"]),
        pts=int(payload["pts"]),
        time_base=Fraction(
            int(time_base["numerator"]), int(time_base["denominator"])
        ),
    )


def _reviewed_contact(
    video_entry: dict[str, Any],
    annotation_path: Path,
    annotation: dict[str, Any],
    selected_shot: dict[str, Any],
    suitability: str,
) -> ReviewedContact:
    shot_index = int(selected_shot["shot_index"])
    shot = annotation["shots"][shot_index]
    if shot.get("time") != selected_shot.get("time"):
        raise ValueError("shot time differs from manifest")
    if shot.get("type") != selected_shot.get("type"):
        raise ValueError("shot type differs from manifest")
    review = shot.get("contact_review")
    if not isinstance(review, dict):
        raise ValueError("missing contact_review")
    status = review.get("status")
    expected_states = {
        "single_state": 1,
        "between_state": 2,
        "unresolved": 0,
    }.get(status)
    if expected_states is None:
        raise ValueError(f"unsupported review status {status!r}")
    state_payloads = review.get("visual_states", [])
    if not isinstance(state_payloads, list) or len(state_payloads) != expected_states:
        requirement = {0: "no visual states", 1: "one visual state", 2: "two visual states"}
        raise ValueError(f"{status} requires {requirement[expected_states]}")
    states = []
    for state_payload in state_payloads:
        if not isinstance(state_payload, dict):
            raise ValueError("visual state must be an object")
        frames_payload = state_payload.get("frames")
        if not isinstance(frames_payload, list) or not frames_payload:
            raise ValueError("visual state requires at least one frame")
        frames = tuple(_frame(item) for item in frames_payload)
        if tuple(frame.ordinal for frame in frames) != tuple(
            sorted({frame.ordinal for frame in frames})
        ):
            raise ValueError("visual state frames must have unique increasing ordinals")
        states.append(ReviewedVisualState(frames))
    canonical = None if status == "unresolved" else _frame(review.get("frame"))
    if canonical is not None and canonical.ordinal not in {
        frame.ordinal for frame in states[0].frames
    }:
        raise ValueError("canonical frame must belong to the first visual state")
    return ReviewedContact(
        video=Path(video_entry["video"]),
        annotation=annotation_path,
        family=str(video_entry["family"]),
        split=str(video_entry["split"]),
        suitability=suitability,
        shot_index=shot_index,
        labeled_timestamp=Fraction(str(shot["time"])),
        shot_type=str(shot["type"]),
        status=str(status),
        canonical_frame=canonical,
        visual_states=tuple(states),
    )


def load_review_dataset(manifest_path: Path) -> ReviewDataset:
    """Load valid reviewed contacts and report excluded records with reasons."""
    manifest = json.loads(manifest_path.read_text())
    contacts: list[ReviewedContact] = []
    invalid: list[str] = []
    for video_entry in manifest.get("videos", []):
        annotation_path = Path(video_entry["annotation"])
        annotation = json.loads(annotation_path.read_text())
        metadata = annotation.get("contact_review_metadata")
        prefix = f"{annotation_path}:"
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
            invalid.extend(
                f"{prefix} shot {shot.get('shot_index')}: incompatible review schema"
                for shot in video_entry.get("shots", [])
            )
            continue
        if metadata.get("source_fingerprint") != video_entry.get("source_fingerprint"):
            invalid.extend(
                f"{prefix} shot {shot.get('shot_index')}: source fingerprint differs"
                for shot in video_entry.get("shots", [])
            )
            continue
        suitability = metadata.get("side_view_suitability")
        if suitability not in {"in_scope", "out_of_scope_view"}:
            invalid.extend(
                f"{prefix} shot {shot.get('shot_index')}: missing side-view suitability"
                for shot in video_entry.get("shots", [])
            )
            continue
        for selected_shot in video_entry.get("shots", []):
            shot_index = selected_shot.get("shot_index")
            try:
                contacts.append(
                    _reviewed_contact(
                        video_entry,
                        annotation_path,
                        annotation,
                        selected_shot,
                        suitability,
                    )
                )
            except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                invalid.append(f"{prefix} shot {shot_index}: {error}")
    return ReviewDataset(tuple(contacts), tuple(invalid))


__all__ = [
    "ReviewDataset",
    "ReviewedContact",
    "ReviewedFrame",
    "ReviewedVisualState",
    "load_review_dataset",
]

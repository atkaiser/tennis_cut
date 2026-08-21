"""Resolve an exact, reusable contact frame in a pro video."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
from typing import Protocol


SCHEMA_VERSION = 1
SUPPORTED_SHOT_TYPES = frozenset({"forehand", "backhand", "volley", "serve"})
PRE_CONTACT = Fraction(6, 5)
POST_CONTACT = Fraction(7, 10)


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
class PickerSelection:
    """Selection returned by an interactive picker adapter."""

    ordinal: int
    shot_type: str


@dataclass(frozen=True)
class ProSelection:
    """Confirmed exact contact frame and shot type for one pro video."""

    source: Path
    frame: DecodedFrame
    shot_type: str


@dataclass(frozen=True)
class SelectionProcessingFailure:
    """Typed resolver failure that should stop comparison processing."""

    stage: str
    message: str


@dataclass(frozen=True)
class SelectionCancelled:
    """Typed result for picker cancellation or closure."""

    message: str = "pro selection cancelled"


@dataclass(frozen=True)
class ConfirmationStatus:
    """Exact footage availability for a proposed picker selection."""

    can_confirm: bool
    available_before: Fraction
    available_after: Fraction
    missing_before: Fraction
    missing_after: Fraction


class SidecarStore(Protocol):
    def read(self, path: Path) -> str | None: ...

    def write(self, path: Path, contents: str) -> None: ...


class FileSidecarStore:
    """Read and atomically replace sidecars beside their source videos."""

    def read(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def write(self, path: Path, contents: str) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(contents)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, path)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise


class PickerAdapter(Protocol):
    def pick(self, session: "PickerSession") -> PickerSelection | None: ...


@dataclass(frozen=True)
class PickerSession:
    """Exact media and playback context presented to a picker."""

    inspected_media: InspectedMedia
    pro_speed: Fraction

    def confirmation_status(self, ordinal: int, shot_type: str) -> ConfirmationStatus:
        frame = self.inspected_media.frame_at(ordinal)
        first_frame = self.inspected_media.frames[0]
        last_frame = self.inspected_media.frames[-1]
        available_before = (frame.timestamp - first_frame.timestamp) * self.pro_speed
        available_after = (last_frame.timestamp - frame.timestamp) * self.pro_speed
        missing_before = max(Fraction(0), PRE_CONTACT - available_before)
        missing_after = max(Fraction(0), POST_CONTACT - available_after)
        return ConfirmationStatus(
            can_confirm=(
                shot_type in SUPPORTED_SHOT_TYPES
                and missing_before == 0
                and missing_after == 0
            ),
            available_before=available_before,
            available_after=available_after,
            missing_before=missing_before,
            missing_after=missing_after,
        )


def _sidecar_path(pro_video: Path) -> Path:
    return pro_video.with_name(f"{pro_video.name}.tennis-compare.json")


def _read_reusable_selection(
    pro_video: Path,
    inspected_media: InspectedMedia,
    sidecar_store: SidecarStore,
) -> ProSelection | None:
    contents = sidecar_store.read(_sidecar_path(pro_video))
    if contents is None:
        return None
    try:
        payload = json.loads(contents)
        source = payload["source"]
        stream = payload["video_stream"]
        time_base_payload = stream["time_base"]
        contact_frame = payload["contact_frame"]
        shot_type = payload["shot_type"]
        source_stat = pro_video.stat()
        integer_fields = (
            payload["schema_version"],
            source["size_bytes"],
            source["mtime_ns"],
            stream["index"],
            time_base_payload["numerator"],
            time_base_payload["denominator"],
            contact_frame["ordinal"],
            contact_frame["pts"],
        )
        if (
            not all(type(value) is int for value in integer_fields)
            or payload["schema_version"] != SCHEMA_VERSION
            or type(source["name"]) is not str
            or source["name"] != pro_video.name
            or source["size_bytes"] != source_stat.st_size
            or source["mtime_ns"] != source_stat.st_mtime_ns
            or type(shot_type) is not str
            or shot_type not in SUPPORTED_SHOT_TYPES
            or time_base_payload["numerator"] <= 0
            or time_base_payload["denominator"] <= 0
        ):
            return None
        time_base = Fraction(
            time_base_payload["numerator"], time_base_payload["denominator"]
        )
        frame = inspected_media.frame_at(contact_frame["ordinal"])
        if (
            frame.pts != contact_frame["pts"]
            or frame.stream_index != stream["index"]
            or frame.time_base != time_base
        ):
            return None
    except (
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ):
        return None
    return ProSelection(pro_video, frame, shot_type)


def resolve_pro_selection(
    *,
    pro_video: Path,
    pro_speed: Fraction,
    inspected_media: InspectedMedia,
    sidecar_store: SidecarStore,
    picker: PickerAdapter,
) -> ProSelection | SelectionCancelled | SelectionProcessingFailure:
    """Resolve and persist a pro contact-frame selection."""

    session = PickerSession(inspected_media, pro_speed)
    reusable = _read_reusable_selection(pro_video, inspected_media, sidecar_store)
    if reusable is not None:
        reuse_status = session.confirmation_status(
            reusable.frame.ordinal, reusable.shot_type
        )
        if reuse_status.can_confirm:
            return reusable
        return SelectionProcessingFailure(
            stage="pro selection",
            message="saved selection lacks required footage at current pro speed",
        )
    picker_selection = picker.pick(session)
    if picker_selection is None:
        return SelectionCancelled()
    try:
        status = session.confirmation_status(
            picker_selection.ordinal, picker_selection.shot_type
        )
        frame = inspected_media.frame_at(picker_selection.ordinal)
    except StopIteration:
        return SelectionProcessingFailure(
            stage="pro selection", message="picker returned an unknown frame ordinal"
        )
    if not status.can_confirm:
        return SelectionProcessingFailure(
            stage="pro selection",
            message="picker returned a selection that cannot be confirmed",
        )
    selection = ProSelection(pro_video, frame, picker_selection.shot_type)
    source_stat = pro_video.stat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "name": pro_video.name,
            "size_bytes": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        },
        "video_stream": {
            "index": frame.stream_index,
            "time_base": {
                "numerator": frame.time_base.numerator,
                "denominator": frame.time_base.denominator,
            },
        },
        "contact_frame": {"ordinal": frame.ordinal, "pts": frame.pts},
        "shot_type": selection.shot_type,
    }
    try:
        sidecar_store.write(
            _sidecar_path(pro_video), json.dumps(payload, indent=2) + "\n"
        )
    except OSError as error:
        return SelectionProcessingFailure(
            stage="persist pro selection", message=str(error)
        )
    return selection

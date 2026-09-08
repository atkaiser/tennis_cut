"""Versioned persistence for expensive visual contact decisions."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

from .annotation_json import load_annotation, save_annotation
from .visual_contact import (
    ContactDecisionDiagnostics,
    ContactSelection,
    Detection,
    DeterministicSelection,
    FrameEvidence,
    RankedFrame,
    TemporalPrediction,
    VisualFrame,
)

CACHE_FIELD = "contact_detection_cache"
CACHE_SCHEMA_VERSION = 2


def detector_identity(ranker: object | None) -> str:
    """Return a stable identity that invalidates cache entries on model change."""

    if ranker is None:
        return "deterministic-v1"
    artifact = getattr(ranker, "artifact", None)
    to_dict = getattr(artifact, "to_dict", None)
    if callable(to_dict):
        encoded = json.dumps(
            to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return f"{type(ranker).__name__}:{hashlib.sha256(encoded).hexdigest()[:16]}"
    return (
        f"{type(ranker).__name__}:"
        f"feature-{getattr(ranker, 'feature_version', 'unknown')}"
    )


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _parse_fraction(value: Any) -> Fraction:
    if not isinstance(value, dict):
        raise ValueError("cached fraction must be an object")
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _fingerprint(source: Path) -> dict[str, int]:
    stat = source.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _dedicated_path(source: Path) -> Path:
    return source.with_suffix(source.suffix + ".contact.json")


def _cache_path(source: Path) -> Path:
    annotation = source.with_suffix(".json")
    if annotation.is_file():
        try:
            load_annotation(annotation)
        except (OSError, ValueError, json.JSONDecodeError):
            return _dedicated_path(source)
        return annotation
    return _dedicated_path(source)


class ContactDecisionCache:
    """Store decisions in an existing annotation JSON or a dedicated sidecar."""

    def load(
        self,
        source: Path,
        candidate_timestamp: Fraction,
        radius: Fraction,
        detector_id: str,
    ) -> ContactSelection | None:
        if not source.is_file():
            return None
        path = _cache_path(source)
        if not path.is_file():
            return None
        try:
            payload = load_annotation(path)
            cache = payload.get(CACHE_FIELD)
            if (
                not isinstance(cache, dict)
                or cache.get("schema_version") != CACHE_SCHEMA_VERSION
                or cache.get("source_fingerprint") != _fingerprint(source)
            ):
                return None
            for entry in cache.get("entries", []):
                if (
                    entry.get("detector_id") == detector_id
                    and _parse_fraction(entry.get("candidate_timestamp"))
                    == candidate_timestamp
                    and _parse_fraction(entry.get("radius")) == radius
                ):
                    return self._selection(entry)
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            ZeroDivisionError,
        ):
            return None
        return None

    def store(
        self,
        source: Path,
        candidate_timestamp: Fraction,
        radius: Fraction,
        detector_id: str,
        selection: ContactSelection,
    ) -> None:
        if not source.is_file():
            return
        path = _cache_path(source)
        payload = load_annotation(path) if path.is_file() else {}
        fingerprint = _fingerprint(source)
        cache = payload.get(CACHE_FIELD)
        if (
            not isinstance(cache, dict)
            or cache.get("schema_version") != CACHE_SCHEMA_VERSION
            or cache.get("source_fingerprint") != fingerprint
        ):
            cache = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "source_fingerprint": fingerprint,
                "entries": [],
            }
        entries = [
            entry
            for entry in cache.get("entries", [])
            if not (
                entry.get("detector_id") == detector_id
                and _parse_fraction(entry.get("candidate_timestamp"))
                == candidate_timestamp
                and _parse_fraction(entry.get("radius")) == radius
            )
        ]
        entries.append(
            {
                "detector_id": detector_id,
                "candidate_timestamp": _fraction(candidate_timestamp),
                "radius": _fraction(radius),
                "confidence": selection.contact_confidence,
                "plausible_frames": list(selection.plausible_frames),
                "omission_reason": selection.omission_reason,
                "frame": self._frame(selection),
                "diagnostics": self._diagnostics(selection.diagnostics),
            }
        )
        cache["entries"] = entries
        payload[CACHE_FIELD] = cache
        save_annotation(path, payload)

    @staticmethod
    def _frame(selection: ContactSelection) -> dict[str, Any] | None:
        if selection.frame is None:
            return None
        evidence = selection.frame.evidence
        identity = evidence.identity
        return {
            "ordinal": evidence.ordinal,
            "stream_index": identity.stream_index,
            "pts": identity.pts,
            "time_base": _fraction(identity.time_base),
        }

    @staticmethod
    def _diagnostics(
        diagnostics: ContactDecisionDiagnostics | None,
    ) -> dict[str, Any] | None:
        if diagnostics is None:
            return None
        deterministic = diagnostics.deterministic
        prediction = diagnostics.temporal_prediction
        return {
            "frames": [
                {
                    "ordinal": frame.evidence.ordinal,
                    "timestamp": _fraction(frame.evidence.timestamp),
                    "stream_index": frame.evidence.stream_index,
                    "pts": frame.evidence.pts,
                    "time_base": (
                        None
                        if frame.evidence.time_base is None
                        else _fraction(frame.evidence.time_base)
                    ),
                    "detections": [
                        {
                            "kind": detection.kind,
                            "box": list(detection.box),
                            "confidence": detection.confidence,
                        }
                        for detection in frame.evidence.detections
                    ],
                }
                for frame in diagnostics.frames
            ],
            "deterministic": {
                "selected_frame": deterministic.selected_frame,
                "plausible_frames": list(deterministic.plausible_frames),
                "ranked_frames": [
                    {
                        "ordinal": ranked.ordinal,
                        "score": ranked.score,
                        "direct_proximity": ranked.direct_proximity,
                        "disappearance": ranked.disappearance,
                        "trajectory": ranked.trajectory,
                    }
                    for ranked in deterministic.ranked_frames
                ],
                "racket_frames": list(deterministic.racket_frames),
                "omission_reason": deterministic.omission_reason,
                "feature_version": deterministic.feature_version,
            },
            "temporal_prediction": (
                None
                if prediction is None
                else {
                    "frame_ordinal": prediction.frame_ordinal,
                    "confidence": prediction.confidence,
                }
            ),
            "confidence_threshold": diagnostics.confidence_threshold,
        }

    @staticmethod
    def _selection(entry: dict[str, Any]) -> ContactSelection:
        frame_payload = entry.get("frame")
        frame = None
        if frame_payload is not None:
            time_base = _parse_fraction(frame_payload["time_base"])
            pts = int(frame_payload["pts"])
            frame = VisualFrame(
                FrameEvidence(
                    ordinal=int(frame_payload["ordinal"]),
                    timestamp=pts * time_base,
                    detections=(),
                    stream_index=int(frame_payload["stream_index"]),
                    pts=pts,
                    time_base=time_base,
                )
            )
        return ContactSelection(
            frame=frame,
            contact_confidence=float(entry["confidence"]),
            plausible_frames=tuple(int(value) for value in entry["plausible_frames"]),
            omission_reason=entry.get("omission_reason"),
            diagnostics=ContactDecisionCache._parse_diagnostics(
                entry.get("diagnostics")
            ),
        )

    @staticmethod
    def _parse_diagnostics(
        payload: Any,
    ) -> ContactDecisionDiagnostics | None:
        if payload is None:
            return None
        frames = tuple(
            VisualFrame(
                FrameEvidence(
                    ordinal=int(frame["ordinal"]),
                    timestamp=_parse_fraction(frame["timestamp"]),
                    detections=tuple(
                        Detection(
                            detection["kind"],
                            tuple(float(value) for value in detection["box"]),
                            float(detection["confidence"]),
                        )
                        for detection in frame["detections"]
                    ),
                    stream_index=int(frame["stream_index"]),
                    pts=None if frame["pts"] is None else int(frame["pts"]),
                    time_base=(
                        None
                        if frame["time_base"] is None
                        else _parse_fraction(frame["time_base"])
                    ),
                )
            )
            for frame in payload["frames"]
        )
        deterministic_payload = payload["deterministic"]
        deterministic = DeterministicSelection(
            selected_frame=(
                None
                if deterministic_payload["selected_frame"] is None
                else int(deterministic_payload["selected_frame"])
            ),
            plausible_frames=tuple(
                int(value) for value in deterministic_payload["plausible_frames"]
            ),
            ranked_frames=tuple(
                RankedFrame(
                    ordinal=int(ranked["ordinal"]),
                    score=float(ranked["score"]),
                    direct_proximity=float(ranked["direct_proximity"]),
                    disappearance=float(ranked["disappearance"]),
                    trajectory=float(ranked["trajectory"]),
                )
                for ranked in deterministic_payload["ranked_frames"]
            ),
            racket_frames=tuple(
                int(value) for value in deterministic_payload["racket_frames"]
            ),
            omission_reason=deterministic_payload["omission_reason"],
            feature_version=int(deterministic_payload["feature_version"]),
        )
        prediction_payload = payload["temporal_prediction"]
        prediction = (
            None
            if prediction_payload is None
            else TemporalPrediction(
                frame_ordinal=int(prediction_payload["frame_ordinal"]),
                confidence=float(prediction_payload["confidence"]),
            )
        )
        return ContactDecisionDiagnostics(
            frames=frames,
            deterministic=deterministic,
            temporal_prediction=prediction,
            confidence_threshold=float(payload["confidence_threshold"]),
        )


__all__ = ["CACHE_FIELD", "ContactDecisionCache", "detector_identity"]

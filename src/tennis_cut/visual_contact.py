"""Visual evidence primitives for comparison contact-frame selection.

The selector is deliberately independent of media decoding and model loading.
Adapters provide decoded frames and object detections; the comparison workflow
can therefore test the complete contact decision without downloading weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Literal, Protocol

ObjectKind = Literal["person", "ball", "racket"]
FEATURE_VERSION = 1
TIE_SCORE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SourceFrameIdentity:
    """Exact identity of one decoded source frame."""

    stream_index: int
    pts: int
    time_base: Fraction

    @property
    def timestamp(self) -> Fraction:
        return self.pts * self.time_base


@dataclass(frozen=True)
class Detection:
    """One stock detector observation in source pixel coordinates."""

    kind: ObjectKind
    box: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class FrameEvidence:
    """A decoded source frame and its exact presentation timestamp."""

    ordinal: int
    timestamp: Fraction
    detections: tuple[Detection, ...]
    stream_index: int = 0
    pts: int | None = None
    time_base: Fraction | None = None

    @property
    def identity(self) -> SourceFrameIdentity:
        return SourceFrameIdentity(
            self.stream_index,
            self.pts if self.pts is not None else 0,
            self.time_base if self.time_base is not None else Fraction(1),
        )


@dataclass(frozen=True)
class VisualFrame:
    """Frame wrapper used by selector adapters and tests."""

    evidence: FrameEvidence

    @property
    def ordinal(self) -> int:
        return self.evidence.ordinal

    @property
    def timestamp(self) -> Fraction:
        return self.evidence.timestamp


@dataclass(frozen=True)
class TemporalFeatures:
    """Versioned per-frame features consumed by a temporal ranker."""

    frame_ordinal: int
    direct_proximity: float
    racket_ball_gap: float
    disappearance: float
    trajectory: float
    racket_quality: float
    ball_quality: float
    ball_missing: float

    @property
    def values(self) -> tuple[float, ...]:
        return (
            self.direct_proximity,
            self.racket_ball_gap,
            self.disappearance,
            self.trajectory,
            self.racket_quality,
            self.ball_quality,
            self.ball_missing,
        )


@dataclass(frozen=True)
class TemporalPrediction:
    """A temporal ranker's strongest existing source frame and confidence."""

    frame_ordinal: int
    confidence: float


class TemporalRanker(Protocol):
    """Replaceable temporal corroboration seam."""

    feature_version: int
    confidence_threshold: float

    def predict(self, features: tuple[TemporalFeatures, ...]) -> TemporalPrediction:
        """Return the strongest frame from the supplied ordered feature window."""


class VisualContactSelector(Protocol):
    """Source-level adapter used by comparison swing detection."""

    def select(
        self,
        source: Path,
        candidate_timestamp: Fraction,
    ) -> ContactSelection:
        """Select visual contact evidence near one audio candidate."""


@dataclass(frozen=True)
class RankedFrame:
    ordinal: int
    score: float
    direct_proximity: float
    disappearance: float
    trajectory: float


@dataclass(frozen=True)
class DeterministicSelection:
    """Result of the visual scorer before temporal corroboration."""

    selected_frame: int | None
    plausible_frames: tuple[int, ...]
    ranked_frames: tuple[RankedFrame, ...]
    racket_frames: tuple[int, ...]
    omission_reason: str | None
    feature_version: int = FEATURE_VERSION


@dataclass(frozen=True)
class ContactSelection:
    """Accepted comparison contact frame or a typed omission."""

    frame: VisualFrame | None
    contact_confidence: float
    plausible_frames: tuple[int, ...]
    omission_reason: str | None
    feature_version: int = FEATURE_VERSION

    @property
    def selected_frame(self) -> int | None:
        return None if self.frame is None else self.frame.ordinal


@dataclass
class _PreparedFrame:
    frame: VisualFrame
    diagonal: float
    player: Detection | None
    rackets: list[dict[str, float | tuple[float, float, float, float]]]
    balls: list[dict[str, float | tuple[float, float, float, float] | bool]]


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_box_distance(
    point: tuple[float, float], box: tuple[float, float, float, float]
) -> float:
    x = max(box[0], min(point[0], box[2]))
    y = max(box[1], min(point[1], box[3]))
    return _distance(point, (x, y))


def _credible_confidence(confidence: float) -> float:
    return max(0.0, min(1.0, (confidence - 0.05) / 0.35))


def _prepare_frames(frames: tuple[VisualFrame, ...]) -> list[_PreparedFrame]:
    prepared: list[_PreparedFrame] = []
    for frame in frames:
        detections = frame.evidence.detections
        people = [d for d in detections if d.kind == "person"]
        player = max(people, key=lambda item: _area(item.box), default=None)
        if player is None:
            prepared.append(_PreparedFrame(frame, 1.0, None, [], []))
            continue
        player_center = _center(player.box)
        diagonal = max(
            1.0,
            math.hypot(player.box[2] - player.box[0], player.box[3] - player.box[1]),
        )
        rackets: list[dict[str, float | tuple[float, float, float, float]]] = []
        for racket in (d for d in detections if d.kind == "racket"):
            center = _center(racket.box)
            relation = _point_box_distance(center, player.box) / diagonal
            main_ownership = relation + 0.15 * _distance(center, player_center) / diagonal
            other_ownership = []
            for person in people:
                if person is player:
                    continue
                person_diagonal = max(
                    1.0,
                    math.hypot(
                        person.box[2] - person.box[0], person.box[3] - person.box[1]
                    ),
                )
                other_ownership.append(
                    _point_box_distance(center, person.box) / person_diagonal
                    + 0.15
                    * _distance(center, _center(person.box))
                    / person_diagonal
                )
            belongs_to_player = not other_ownership or main_ownership <= min(
                other_ownership
            ) + 0.03
            if relation <= 0.45 and belongs_to_player:
                quality = _credible_confidence(racket.confidence) * (
                    1 - relation / 0.6
                )
                rackets.append({"box": racket.box, "quality": quality})
        rackets.sort(key=lambda item: float(item["quality"]), reverse=True)

        balls: list[dict[str, float | tuple[float, float, float, float] | bool]] = []
        for ball in (d for d in detections if d.kind == "ball"):
            if _distance(_center(ball.box), player_center) / diagonal <= 1.35:
                balls.append(
                    {
                        "box": ball.box,
                        "confidence": ball.confidence,
                        "diagonal": diagonal,
                    }
                )
        prepared.append(_PreparedFrame(frame, diagonal, player, rackets[:1], balls))

    stationary_limit = max(3, min(5, len(prepared) - 1))
    for index, frame in enumerate(prepared):
        for ball in frame.balls:
            center = _center(ball["box"])  # type: ignore[arg-type]
            stationary_hits = 0
            motion_sides: set[int] = set()
            for offset in range(-6, 7):
                if offset == 0 or not 0 <= index + offset < len(prepared):
                    continue
                candidates = prepared[index + offset].balls
                if not candidates:
                    continue
                nearest = min(
                    _distance(center, _center(other["box"]))  # type: ignore[arg-type]
                    for other in candidates
                )
                diagonal = float(ball["diagonal"])
                if nearest <= 0.008 * diagonal:
                    stationary_hits += 1
                if 0.003 * diagonal <= nearest <= 0.16 * diagonal * abs(offset):
                    motion_sides.add(-1 if offset < 0 else 1)
            support = len(motion_sides) / 2
            stationary = stationary_hits >= stationary_limit
            ball["stationary"] = stationary
            ball["quality"] = (
                0.0
                if stationary
                else _credible_confidence(float(ball["confidence"]))
                * (0.45 + 0.55 * support)
            )
    return prepared


def _best_direct_pair(
    frame: _PreparedFrame,
) -> tuple[float, float]:
    best = (0.0, 1.0)
    for racket in frame.rackets:
        for ball in frame.balls:
            if float(ball.get("quality", 0.0)) < 0.05:
                continue
            gap = _point_box_distance(
                _center(ball["box"]),  # type: ignore[arg-type]
                racket["box"],  # type: ignore[arg-type]
            ) / frame.diagonal
            proximity = math.exp(-gap / 0.10)
            evidence = math.sqrt(
                max(0.0, float(ball["quality"]) * float(racket["quality"]))
            ) * proximity
            if evidence > best[0]:
                best = (evidence, gap)
    return best


def _moving_ball_center(frame: _PreparedFrame) -> tuple[float, float] | None:
    usable = [ball for ball in frame.balls if float(ball.get("quality", 0.0)) >= 0.05]
    if not usable:
        return None
    ball = max(usable, key=lambda item: float(item["quality"]))
    return _center(ball["box"])  # type: ignore[arg-type]


def _disappearance(index: int, frames: list[_PreparedFrame]) -> float:
    frame = frames[index]
    if not frame.rackets:
        return 0.0

    racket = frame.rackets[0]
    before: tuple[int, dict[str, object]] | None = None
    after: tuple[int, dict[str, object]] | None = None
    for gap in range(1, 5):
        if before is None and index - gap >= 0:
            candidates = [
                ball
                for ball in frames[index - gap].balls
                if float(ball.get("quality", 0.0)) >= 0.05
            ]
            if candidates:
                before = (index - gap, max(candidates, key=lambda item: float(item["quality"])))  # type: ignore[assignment]
        if after is None and index + gap < len(frames):
            candidates = [
                ball
                for ball in frames[index + gap].balls
                if float(ball.get("quality", 0.0)) >= 0.05
            ]
            if candidates:
                after = (index + gap, max(candidates, key=lambda item: float(item["quality"])))  # type: ignore[assignment]
    if before is None or after is None:
        return 0.0
    before_index, before_ball = before
    after_index, after_ball = after
    before_center = _center(before_ball["box"])  # type: ignore[arg-type]
    after_center = _center(after_ball["box"])  # type: ignore[arg-type]
    travel = _distance(before_center, after_center) / frame.diagonal
    if not 0.008 <= travel <= 0.55:
        return 0.0
    fraction = (index - before_index) / (after_index - before_index)
    interpolated = (
        before_center[0] + (after_center[0] - before_center[0]) * fraction,
        before_center[1] + (after_center[1] - before_center[1]) * fraction,
    )
    gap = _point_box_distance(interpolated, racket["box"]) / frame.diagonal  # type: ignore[arg-type]
    quality = (
        float(before_ball["quality"])
        * float(after_ball["quality"])
        * float(racket["quality"])
    ) ** (1 / 3)
    return max(0.0, quality * math.exp(-gap / 0.12) * (1 - 0.08 * (after_index - before_index - 2)))


def _trajectory(index: int, frames: list[_PreparedFrame]) -> float:
    centers = [_moving_ball_center(frame) for frame in frames]
    current = centers[index]
    if current is not None:
        before = next((centers[i] for i in range(index - 1, max(-1, index - 3), -1) if centers[i] is not None), None)
        after = next((centers[i] for i in range(index + 1, min(len(centers), index + 3)) if centers[i] is not None), None)
        if before is None or after is None:
            return 0.0
        incoming = (current[0] - before[0], current[1] - before[1])
        outgoing = (after[0] - current[0], after[1] - current[1])
    else:
        before_points = [centers[i] for i in range(max(0, index - 3), index) if centers[i] is not None]
        after_points = [centers[i] for i in range(index + 1, min(len(centers), index + 4)) if centers[i] is not None]
        if len(before_points) < 2 or len(after_points) < 2:
            return 0.0
        incoming = (before_points[-1][0] - before_points[-2][0], before_points[-1][1] - before_points[-2][1])
        outgoing = (after_points[1][0] - after_points[0][0], after_points[1][1] - after_points[0][1])
    incoming_length = math.hypot(*incoming)
    outgoing_length = math.hypot(*outgoing)
    if incoming_length < 2 or outgoing_length < 2:
        return 0.0
    cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (incoming_length * outgoing_length)
    return max(0.0, min(1.0, (0.5 - cosine) / 1.5))


def _feature_rows(frames: list[_PreparedFrame]) -> tuple[TemporalFeatures, ...]:
    base: list[TemporalFeatures] = []
    direct = [_best_direct_pair(frame) for frame in frames]
    for index, frame in enumerate(frames):
        base.append(
            TemporalFeatures(
                frame_ordinal=frame.frame.ordinal,
                direct_proximity=direct[index][0],
                racket_ball_gap=min(direct[index][1], 1.0),
                disappearance=_disappearance(index, frames),
                trajectory=_trajectory(index, frames),
                racket_quality=max((float(item["quality"]) for item in frame.rackets), default=0.0),
                ball_quality=max((float(item.get("quality", 0.0)) for item in frame.balls), default=0.0),
                ball_missing=float(_moving_ball_center(frame) is None),
            )
        )
    return tuple(base)


def _rank_prepared_frames(
    prepared: list[_PreparedFrame],
) -> DeterministicSelection:
    if not prepared:
        return DeterministicSelection(None, (), (), (), "no decoded frames")
    racket_frames = tuple(frame.frame.ordinal for frame in prepared if frame.rackets)
    if not racket_frames:
        return DeterministicSelection(None, (), (), (), "no player-related racket")
    if not any(_moving_ball_center(frame) is not None for frame in prepared):
        return DeterministicSelection(None, (), (), racket_frames, "no moving ball evidence")

    feature_rows = _feature_rows(prepared)
    direct = [_best_direct_pair(frame) for frame in prepared]
    ranking = tuple(
        sorted(
            (
                RankedFrame(
                    frame.frame.ordinal,
                    min(
                        1.0,
                        0.55 * max(direct[index][0], min(1.0, 1.15 * feature.disappearance))
                        + 0.30 * feature.trajectory * max(direct[index][0], min(1.0, 1.15 * feature.disappearance))
                        + 0.15 * feature.racket_quality
                        + (0.16 * min(1.0, feature.disappearance / 0.35) if feature.ball_missing else 0.0),
                    ),
                    direct[index][0],
                    feature.disappearance,
                    feature.trajectory,
                )
                for index, (frame, feature) in enumerate(zip(prepared, feature_rows, strict=True))
            ),
            key=lambda item: (-item.score, item.ordinal),
        )
    )
    highest_score = ranking[0].score
    strongest = min(
        (
            item
            for item in ranking
            if item.score >= highest_score - TIE_SCORE_TOLERANCE
        ),
        key=lambda item: item.ordinal,
    )
    if strongest.score < 0.12:
        return DeterministicSelection(None, (), ranking, racket_frames, "weak visual evidence")
    plausible = tuple(
        sorted(
            item.ordinal
            for item in ranking
            if item.score >= max(strongest.score * 0.90, strongest.score - 0.055)
        )
    )
    if len(plausible) > 2 or (len(plausible) == 2 and plausible[1] != plausible[0] + 1):
        return DeterministicSelection(strongest.ordinal, plausible, ranking, racket_frames, "broad or separated ambiguity")
    return DeterministicSelection(strongest.ordinal, plausible, ranking, racket_frames, None)


def rank_contact_frames(frames: tuple[VisualFrame, ...]) -> DeterministicSelection:
    """Select the strongest existing frame from visual evidence."""

    return _rank_prepared_frames(_prepare_frames(frames))


def extract_temporal_features(frames: tuple[VisualFrame, ...]) -> tuple[TemporalFeatures, ...]:
    """Build the seven versioned evidence values for each decoded frame."""

    return _feature_rows(_prepare_frames(frames))


class DeterministicTemporalRanker:
    """Temporary corroborator used until a trained artifact is supplied."""

    feature_version = FEATURE_VERSION
    confidence_threshold = 0.12

    def predict(self, features: tuple[TemporalFeatures, ...]) -> TemporalPrediction:
        strongest = max(features, key=lambda item: (item.direct_proximity, -item.frame_ordinal))
        return TemporalPrediction(strongest.frame_ordinal, strongest.direct_proximity)


def select_contact_frame(
    frames: tuple[VisualFrame, ...],
    ranker: TemporalRanker | None = None,
) -> ContactSelection:
    """Apply deterministic selection and replaceable temporal corroboration."""

    prepared = _prepare_frames(frames)
    deterministic = _rank_prepared_frames(prepared)
    if deterministic.omission_reason is not None:
        return ContactSelection(
            None,
            0.0,
            deterministic.plausible_frames,
            deterministic.omission_reason,
        )
    assert deterministic.selected_frame is not None
    if ranker is None:
        ranker = DeterministicTemporalRanker()
    if ranker.feature_version != deterministic.feature_version:
        return ContactSelection(None, 0.0, (), "incompatible temporal feature version")
    prediction = ranker.predict(_feature_rows(prepared))
    if all(frame.ordinal != prediction.frame_ordinal for frame in frames):
        return ContactSelection(None, 0.0, (), "temporal ranker selected unknown frame")
    if abs(prediction.frame_ordinal - deterministic.selected_frame) > 1:
        return ContactSelection(None, 0.0, deterministic.plausible_frames, "temporal ranker disagrees")
    confidence = prediction.confidence
    if prediction.frame_ordinal == deterministic.selected_frame:
        confidence = min(1.0, confidence + getattr(ranker, "exact_agreement_bonus", 0.0))
    if confidence < getattr(ranker, "confidence_threshold", 0.0):
        return ContactSelection(
            None,
            0.0,
            deterministic.plausible_frames,
            "below contact confidence threshold",
        )
    selected = next(frame for frame in frames if frame.ordinal == deterministic.selected_frame)
    plausible = tuple(sorted(set(deterministic.plausible_frames) | {prediction.frame_ordinal}))
    if len(plausible) > 2 or (len(plausible) == 2 and plausible[1] != plausible[0] + 1):
        return ContactSelection(None, 0.0, plausible, "broad or separated ambiguity")
    return ContactSelection(selected, max(0.0, min(1.0, confidence)), plausible, None)


class _StockVisualEvidence:
    """Decode exact source frames and run the stock YOLO detector lazily."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO("yolov8n.pt")
        return self._model

    def _frame_timestamps(
        self, source: Path
    ) -> tuple[tuple[int, int, int, Fraction], ...]:
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
                "stream=index,time_base:frame=pts,stream_index",
                "-of",
                "json",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        stream_index = int(stream["index"])
        time_base = Fraction(stream["time_base"])
        return tuple(
            (ordinal, stream_index, int(frame["pts"]), time_base)
            for ordinal, frame in enumerate(payload.get("frames", []))
            if int(frame["stream_index"]) == stream_index
        )

    def frames(
        self,
        source: Path,
        candidate_timestamp: Fraction,
        radius: Fraction = Fraction(2, 5),
    ) -> tuple[VisualFrame, ...]:
        metadata = self._frame_timestamps(source)
        selected = tuple(
            item
            for item in metadata
            if candidate_timestamp - radius
            <= item[2] * item[3]
            <= candidate_timestamp + radius
        )
        if not selected:
            return ()
        import cv2

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {source}")
        images: list[object] = []
        visual_frames: list[VisualFrame] = []
        first = selected[0][0]
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for ordinal, stream_index, pts, time_base in selected:
            if ordinal != first + len(images):
                capture.set(cv2.CAP_PROP_POS_FRAMES, ordinal)
            ok, image = capture.read()
            if not ok:
                break
            images.append(image)
            visual_frames.append(
                VisualFrame(
                    FrameEvidence(
                        ordinal,
                        pts * time_base,
                        (),
                        stream_index,
                        pts,
                        time_base,
                    )
                )
            )
        capture.release()
        if not visual_frames:
            return ()
        model = self._load_model()
        results = model.predict(
            images,
            classes=[0, 32, 38],
            conf=0.05,
            iou=0.5,
            imgsz=1280,
            device=self.device,
            verbose=False,
        )
        populated: list[VisualFrame] = []
        for frame, result in zip(visual_frames, results, strict=True):
            detections: list[Detection] = []
            boxes = result.boxes
            if boxes is not None:
                for box, kind, confidence in zip(
                    boxes.xyxy.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    strict=True,
                ):
                    names = {0: "person", 32: "ball", 38: "racket"}
                    detected_kind = names.get(int(kind))
                    if detected_kind is not None:
                        detections.append(
                            Detection(detected_kind, tuple(float(v) for v in box), float(confidence))
                        )
            populated.append(
                VisualFrame(
                    FrameEvidence(
                        frame.ordinal,
                        frame.timestamp,
                        tuple(detections),
                        frame.evidence.stream_index,
                        frame.evidence.pts,
                        frame.evidence.time_base,
                    )
                )
            )
        return tuple(populated)


class StockVisualContactSelector:
    """Production selector using stock YOLO evidence and a replaceable ranker."""

    def __init__(
        self,
        *,
        ranker: TemporalRanker | None = None,
        device: str | None = None,
        evidence_provider: _StockVisualEvidence | None = None,
    ) -> None:
        self.ranker = ranker
        self.device = device
        self.evidence_provider = evidence_provider

    def select(
        self,
        source: Path,
        candidate_timestamp: Fraction,
    ) -> ContactSelection:
        if self.evidence_provider is None:
            self.evidence_provider = _StockVisualEvidence(self.device)
        provider = self.evidence_provider
        frames = provider.frames(source, candidate_timestamp)
        return select_contact_frame(frames, self.ranker)


__all__ = [
    "ContactSelection",
    "Detection",
    "DeterministicSelection",
    "FEATURE_VERSION",
    "FrameEvidence",
    "ObjectKind",
    "SourceFrameIdentity",
    "TemporalFeatures",
    "TemporalPrediction",
    "TemporalRanker",
    "VisualFrame",
    "VisualContactSelector",
    "StockVisualContactSelector",
    "extract_temporal_features",
    "rank_contact_frames",
    "select_contact_frame",
]

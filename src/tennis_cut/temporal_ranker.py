"""Versioned temporal ranker artifacts for visual contact selection.

The implementation is intentionally small and JSON based.  Comparison runs
load a frozen artifact; fitting and threshold calibration belong to the
offline evaluator in :mod:`tennis_cut.evaluate_temporal_ranker`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .visual_contact import FEATURE_VERSION, TemporalFeatures, TemporalPrediction, VisualFrame, extract_temporal_features

SCORER_VERSION = 3
ARTIFACT_VERSION = 1
CONTEXT_RADIUS = 4
FEATURES_PER_FRAME = 7
TEMPORAL_VECTOR_SIZE = (CONTEXT_RADIUS * 2 + 1) * FEATURES_PER_FRAME
EXACT_AGREEMENT_BONUS = 0.12
PROTOTYPE_EXACT_AGREEMENT_BONUS = 0.25


class TemporalRankerArtifactError(ValueError):
    """Stable error raised for an unusable ranker artifact."""


def temporal_feature_vector(features: Sequence[TemporalFeatures], index: int) -> tuple[float, ...]:
    """Return the documented 9-frame, 63-value clamped context vector."""

    if not features:
        raise ValueError("cannot extract temporal features from an empty window")
    if not 0 <= index < len(features):
        raise IndexError("temporal feature index is outside the frame window")
    values: list[float] = []
    for offset in range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1):
        neighbor = features[min(len(features) - 1, max(0, index + offset))]
        values.extend(neighbor.values)
    return tuple(values)


def all_temporal_feature_vectors(
    features: Sequence[TemporalFeatures],
) -> tuple[tuple[float, ...], ...]:
    """Extract one clamped context vector for every existing source frame."""

    return tuple(temporal_feature_vector(features, index) for index in range(len(features)))


def feature_vectors_from_frames(frames: tuple[VisualFrame, ...]) -> tuple[tuple[float, ...], ...]:
    """Extract production evidence and the documented clamped contexts."""

    return all_temporal_feature_vectors(extract_temporal_features(frames))


@dataclass(frozen=True)
class TemporalRankerArtifact:
    """Serialized metadata and parameters for one production ranker."""

    weights: tuple[float, ...]
    bias: float
    supported_shot_type: str
    feature_version: int = FEATURE_VERSION
    scorer_version: int = SCORER_VERSION
    operating_threshold: float = 0.0
    artifact_version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if len(self.weights) != TEMPORAL_VECTOR_SIZE:
            raise TemporalRankerArtifactError(
                f"invalid weights: expected {TEMPORAL_VECTOR_SIZE} values"
            )
        if not self.supported_shot_type:
            raise TemporalRankerArtifactError("invalid supported shot type")
        if not math.isfinite(self.bias) or any(not math.isfinite(weight) for weight in self.weights):
            raise TemporalRankerArtifactError("invalid model parameters")
        if not math.isfinite(self.operating_threshold):
            raise TemporalRankerArtifactError("invalid operating threshold")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "feature_version": self.feature_version,
            "scorer_version": self.scorer_version,
            "supported_shot_type": self.supported_shot_type,
            "operating_threshold": self.operating_threshold,
            "model": {"type": "linear", "weights": list(self.weights), "bias": self.bias},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def load_temporal_ranker(
    path: Path,
    *,
    expected_feature_version: int = FEATURE_VERSION,
    expected_scorer_version: int = SCORER_VERSION,
    expected_shot_type: str = "forehand",
) -> "LinearTemporalRanker":
    """Load and semantically validate a production artifact."""

    if not path.is_file():
        raise TemporalRankerArtifactError(f"ranker artifact missing: {path}")
    try:
        payload = json.loads(path.read_text())
        model = payload["model"]
        artifact = TemporalRankerArtifact(
            tuple(float(value) for value in model["weights"]),
            float(model["bias"]),
            str(payload["supported_shot_type"]),
            int(payload["feature_version"]),
            int(payload["scorer_version"]),
            float(payload["operating_threshold"]),
            int(payload["artifact_version"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, OverflowError) as error:
        raise TemporalRankerArtifactError(f"malformed ranker artifact: {path}") from error
    if artifact.artifact_version != ARTIFACT_VERSION:
        raise TemporalRankerArtifactError("unsupported ranker artifact version")
    if artifact.feature_version != expected_feature_version:
        raise TemporalRankerArtifactError("incompatible temporal feature version")
    if artifact.scorer_version != expected_scorer_version:
        raise TemporalRankerArtifactError("incompatible visual scorer version")
    if artifact.supported_shot_type != expected_shot_type:
        raise TemporalRankerArtifactError("unsupported temporal ranker shot type")
    return LinearTemporalRanker(artifact)


@dataclass(frozen=True)
class LinearTemporalRanker:
    """Frozen ranker implementing the production ``TemporalRanker`` seam."""

    artifact: TemporalRankerArtifact

    @property
    def feature_version(self) -> int:
        return self.artifact.feature_version

    @property
    def confidence_threshold(self) -> float:
        return self.artifact.operating_threshold

    @property
    def exact_agreement_bonus(self) -> float:
        return EXACT_AGREEMENT_BONUS

    def predict(self, features: tuple[TemporalFeatures, ...]) -> TemporalPrediction:
        vectors = all_temporal_feature_vectors(features)
        scores = [sum(weight * value for weight, value in zip(self.artifact.weights, vector)) + self.artifact.bias for vector in vectors]
        best = max(range(len(scores)), key=lambda index: (scores[index], -features[index].frame_ordinal))
        local = max(scores[max(0, best - 1): min(len(scores), best + 2)])
        outside = scores[:max(0, best - 1)] + scores[min(len(scores), best + 2):]
        margin = local - max(outside, default=0.0)
        confidence = max(0.0, min(1.0, margin))
        return TemporalPrediction(features[best].frame_ordinal, confidence)


@dataclass
class PrototypeTemporalRanker:
    """The throwaway prototype's HGB temporal corroborator.

    The estimator is intentionally kept in memory: prototype records are
    labeled pilot data, not a portable production artifact.
    """

    model: object
    operating_threshold: float
    feature_version: int = FEATURE_VERSION

    @property
    def confidence_threshold(self) -> float:
        return self.operating_threshold

    @property
    def exact_agreement_bonus(self) -> float:
        return PROTOTYPE_EXACT_AGREEMENT_BONUS

    def predict(self, features: tuple[TemporalFeatures, ...]) -> TemporalPrediction:
        import numpy as np

        vectors = all_temporal_feature_vectors(features)
        scores = np.asarray(self.model.predict(np.asarray(vectors)), dtype=float)
        best = int(np.argmax(scores))
        local = float(scores[max(0, best - 1): min(len(scores), best + 2)].max())
        outside = np.concatenate((scores[: max(0, best - 1)], scores[min(len(scores), best + 2):]))
        margin = local - float(outside.max()) if len(outside) else local
        confidence = max(0.0, min(1.0, margin))
        return TemporalPrediction(features[best].frame_ordinal, confidence)


def fit_prototype_temporal_ranker(records: Sequence[object]) -> PrototypeTemporalRanker:
    """Train the original prototype model and calibrate its confidence gate."""

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    usable = tuple(records)
    groups = sorted({str(record.group) for record in usable})
    if len(groups) < 2:
        raise ValueError("prototype ranker calibration requires at least two camera-roll groups")

    def train(training: Sequence[object]) -> object:
        rows: list[tuple[tuple[float, ...], float]] = []
        for record in training:
            vectors = all_temporal_feature_vectors(record.features)
            label_index = next(
                (index for index, feature in enumerate(record.features)
                 if feature.frame_ordinal == record.label_frame),
                None,
            )
            if label_index is None:
                raise ValueError(f"prototype label frame is outside its feature window: {record.label_frame}")
            for index in range(max(0, label_index - 12), min(len(vectors), label_index + 13)):
                rows.append((vectors[index], math.exp(-abs(index - label_index) / 0.8)))
        if not rows:
            raise ValueError("cannot train prototype ranker without labeled windows")
        model = HistGradientBoostingRegressor(
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=2,
            random_state=7,
        )
        return model.fit(np.asarray([row for row, _ in rows]), np.asarray([target for _, target in rows]))

    held_out_predictions: list[tuple[object, TemporalPrediction]] = []
    for group in groups:
        training = tuple(record for record in usable if str(record.group) != group)
        model = train(training)
        ranker = PrototypeTemporalRanker(model, 0.0)
        held_out_predictions.extend(
            (record, ranker.predict(record.features))
            for record in usable
            if str(record.group) == group
        )

    confidences: list[float] = []
    correctness: list[bool] = []
    total = sum(getattr(record, "total_swings", 1) for record in usable)
    for record, prediction in held_out_predictions:
        if getattr(record, "omission_reason", None) is None and (
            record.deterministic_frame is None or abs(prediction.frame_ordinal - record.deterministic_frame) <= 1
        ):
            confidence = prediction.confidence
            if record.deterministic_frame == prediction.frame_ordinal:
                confidence = min(1.0, confidence + PROTOTYPE_EXACT_AGREEMENT_BONUS)
            confidences.append(confidence)
            correctness.append(abs(prediction.frame_ordinal - record.label_frame) <= 1)
    from .evaluate_temporal_ranker import choose_operating_threshold

    threshold, _, _ = choose_operating_threshold(
        confidences,
        lambda _: False,
        total_swings=total,
        minimum_precision=0.95,
        correctness=correctness,
    )
    return PrototypeTemporalRanker(train(usable), threshold)


def fit_temporal_ranker(
    samples: Iterable[tuple[Sequence[TemporalFeatures], int]],
    *,
    supported_shot_type: str = "forehand",
    epochs: int = 120,
    learning_rate: float = 0.03,
    regularization: float = 0.001,
) -> TemporalRankerArtifact:
    """Fit a deterministic linear scorer from labeled frame windows.

    Targets decay exponentially away from the labeled frame.  This is enough
    capacity for the lightweight pilot and keeps exported artifacts portable.
    """

    rows: list[tuple[tuple[float, ...], float]] = []
    for features, label_frame in samples:
        vectors = all_temporal_feature_vectors(features)
        for feature, vector in zip(features, vectors, strict=True):
            rows.append((vector, math.exp(-abs(feature.frame_ordinal - label_frame) / 0.8)))
    if not rows:
        raise ValueError("cannot train temporal ranker without labeled windows")
    weights = [0.0] * TEMPORAL_VECTOR_SIZE
    bias = 0.0
    for _ in range(epochs):
        for vector, target in rows:
            prediction = sum(weight * value for weight, value in zip(weights, vector)) + bias
            error = prediction - target
            for index, value in enumerate(vector):
                weights[index] -= learning_rate * (error * value + regularization * weights[index])
            bias -= learning_rate * error
    return TemporalRankerArtifact(tuple(weights), bias, supported_shot_type)


__all__ = [
    "ARTIFACT_VERSION",
    "CONTEXT_RADIUS",
    "EXACT_AGREEMENT_BONUS",
    "PROTOTYPE_EXACT_AGREEMENT_BONUS",
    "FEATURES_PER_FRAME",
    "SCORER_VERSION",
    "TEMPORAL_VECTOR_SIZE",
    "LinearTemporalRanker",
    "PrototypeTemporalRanker",
    "TemporalRankerArtifact",
    "TemporalRankerArtifactError",
    "all_temporal_feature_vectors",
    "fit_temporal_ranker",
    "fit_prototype_temporal_ranker",
    "feature_vectors_from_frames",
    "load_temporal_ranker",
    "temporal_feature_vector",
]

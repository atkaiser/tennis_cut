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
    "FEATURES_PER_FRAME",
    "SCORER_VERSION",
    "TEMPORAL_VECTOR_SIZE",
    "LinearTemporalRanker",
    "TemporalRankerArtifact",
    "TemporalRankerArtifactError",
    "all_temporal_feature_vectors",
    "fit_temporal_ranker",
    "feature_vectors_from_frames",
    "load_temporal_ranker",
    "temporal_feature_vector",
]

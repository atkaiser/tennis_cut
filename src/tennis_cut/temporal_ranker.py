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

from .visual_contact import (
    FEATURE_VERSION,
    KINEMATIC_FEATURES_PER_FRAME,
    KINEMATIC_FEATURE_VERSION,
    TemporalFeatures,
    TemporalPrediction,
    VisualFrame,
    extract_kinematic_features,
    extract_temporal_features,
)

SCORER_VERSION = 3
ARTIFACT_VERSION = 1
CONTEXT_RADIUS = 4
FEATURES_PER_FRAME = 7
TEMPORAL_VECTOR_SIZE = (CONTEXT_RADIUS * 2 + 1) * FEATURES_PER_FRAME
KINEMATIC_VECTOR_SIZE = (
    (CONTEXT_RADIUS * 2 + 1) * KINEMATIC_FEATURES_PER_FRAME
)
EXACT_AGREEMENT_BONUS = 0.12
HGB_EXACT_AGREEMENT_BONUS = 0.25


class TemporalRankerArtifactError(ValueError):
    """Stable error raised for an unusable ranker artifact."""


def _validate_trees(
    trees: tuple[tuple["HistGradientBoostingNode", ...], ...], vector_size: int
) -> None:
    if not trees or any(not tree for tree in trees):
        raise TemporalRankerArtifactError("invalid histogram gradient boosting trees")
    for tree in trees:
        if any(
            not math.isfinite(node.value) or not math.isfinite(node.threshold)
            for node in tree
        ):
            raise TemporalRankerArtifactError(
                "invalid histogram gradient boosting tree values"
            )
        visited: set[int] = set()
        active: set[int] = set()

        def visit(node_index: int) -> None:
            if not 0 <= node_index < len(tree) or node_index in active:
                raise TemporalRankerArtifactError(
                    "invalid histogram gradient boosting tree indexes"
                )
            if node_index in visited:
                return
            node = tree[node_index]
            visited.add(node_index)
            if node.is_leaf:
                return
            if not 0 <= node.feature_index < vector_size:
                raise TemporalRankerArtifactError(
                    "invalid histogram gradient boosting tree feature index"
                )
            active.add(node_index)
            visit(node.left)
            visit(node.right)
            active.remove(node_index)

        visit(0)


@dataclass(frozen=True)
class HistGradientBoostingNode:
    """One portable regression-tree node from the accepted prototype ranker."""

    value: float
    feature_index: int
    threshold: float
    missing_go_to_left: bool
    left: int
    right: int
    is_leaf: bool


@dataclass(frozen=True)
class HistGradientBoostingArtifact:
    """Portable trees and compatibility metadata for production inference."""

    baseline: float
    trees: tuple[tuple[HistGradientBoostingNode, ...], ...]
    supported_shot_type: str
    operating_threshold: float
    feature_version: int = FEATURE_VERSION
    scorer_version: int = SCORER_VERSION
    exact_agreement_bonus: float = HGB_EXACT_AGREEMENT_BONUS
    artifact_version: int = ARTIFACT_VERSION
    vector_size: int = TEMPORAL_VECTOR_SIZE

    def __post_init__(self) -> None:
        _validate_trees(self.trees, self.vector_size)
        if not self.supported_shot_type:
            raise TemporalRankerArtifactError("invalid supported shot type")
        values = (self.baseline, self.operating_threshold, self.exact_agreement_bonus)
        if any(not math.isfinite(value) for value in values):
            raise TemporalRankerArtifactError("invalid histogram gradient boosting parameters")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "feature_version": self.feature_version,
            "scorer_version": self.scorer_version,
            "supported_shot_type": self.supported_shot_type,
            "operating_threshold": self.operating_threshold,
            "exact_agreement_bonus": self.exact_agreement_bonus,
            "model": {
                "type": "hist-gradient-boosting",
                "baseline": self.baseline,
                "trees": [
                    [
                        [
                            node.value,
                            node.feature_index,
                            node.threshold,
                            node.missing_go_to_left,
                            node.left,
                            node.right,
                            node.is_leaf,
                        ]
                        for node in tree
                    ]
                    for tree in self.trees
                ],
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class KinematicEnsembleArtifact:
    """One portable package containing both validated kinematic rankers."""

    ordinary: HistGradientBoostingArtifact
    family_balanced: HistGradientBoostingArtifact
    ordinary_weight: float
    supported_shot_type: str
    operating_threshold: float = 0.0
    feature_version: int = FEATURE_VERSION
    kinematic_feature_version: int = KINEMATIC_FEATURE_VERSION
    scorer_version: int = SCORER_VERSION
    artifact_version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.ordinary_weight <= 1.0:
            raise TemporalRankerArtifactError("invalid ensemble weight")
        if not self.supported_shot_type:
            raise TemporalRankerArtifactError("invalid supported shot type")
        if not math.isfinite(self.operating_threshold):
            raise TemporalRankerArtifactError("invalid operating threshold")
        _validate_trees(self.ordinary.trees, KINEMATIC_VECTOR_SIZE)
        _validate_trees(self.family_balanced.trees, KINEMATIC_VECTOR_SIZE)

    @staticmethod
    def _component(artifact: HistGradientBoostingArtifact) -> dict[str, object]:
        return {
            "baseline": artifact.baseline,
            "trees": artifact.to_dict()["model"]["trees"],  # type: ignore[index]
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "feature_version": self.feature_version,
            "kinematic_feature_version": self.kinematic_feature_version,
            "scorer_version": self.scorer_version,
            "supported_shot_type": self.supported_shot_type,
            "operating_threshold": self.operating_threshold,
            "model": {
                "type": "kinematic-ensemble",
                "ordinary_weight": self.ordinary_weight,
                "ordinary": self._component(self.ordinary),
                "family_balanced": self._component(self.family_balanced),
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n")

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


def all_kinematic_feature_vectors(
    frames: tuple[VisualFrame, ...],
) -> tuple[tuple[float, ...], ...]:
    """Extract one clamped 234-value kinematic context per source frame."""

    rows = extract_kinematic_features(frames)
    vectors: list[tuple[float, ...]] = []
    for index in range(len(rows)):
        values: list[float] = []
        for offset in range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1):
            values.extend(rows[min(len(rows) - 1, max(0, index + offset))])
        vectors.append(tuple(values))
    return tuple(vectors)


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
) -> "LinearTemporalRanker | HistGradientBoostingTemporalRanker | KinematicEnsembleRanker":
    """Load and semantically validate a production artifact."""

    if not path.is_file():
        raise TemporalRankerArtifactError(f"ranker artifact missing: {path}")
    try:
        payload = json.loads(path.read_text())
        model = payload["model"]
        model_type = str(model["type"])
        artifact: (
            TemporalRankerArtifact
            | HistGradientBoostingArtifact
            | KinematicEnsembleArtifact
        )

        def trees_from(value: object) -> tuple[tuple[HistGradientBoostingNode, ...], ...]:
            return tuple(
                tuple(
                    HistGradientBoostingNode(
                        float(node[0]),
                        int(node[1]),
                        float(node[2]),
                        bool(node[3]),
                        int(node[4]),
                        int(node[5]),
                        bool(node[6]),
                    )
                    for node in tree
                )
                for tree in value  # type: ignore[union-attr]
            )

        if model_type == "linear":
            artifact = TemporalRankerArtifact(
                tuple(float(value) for value in model["weights"]),
                float(model["bias"]),
                str(payload["supported_shot_type"]),
                int(payload["feature_version"]),
                int(payload["scorer_version"]),
                float(payload["operating_threshold"]),
                int(payload["artifact_version"]),
            )
            ranker: (
                LinearTemporalRanker
                | HistGradientBoostingTemporalRanker
                | KinematicEnsembleRanker
            ) = LinearTemporalRanker(artifact)
        elif model_type == "hist-gradient-boosting":
            artifact = HistGradientBoostingArtifact(
                float(model["baseline"]),
                trees_from(model["trees"]),
                str(payload["supported_shot_type"]),
                float(payload["operating_threshold"]),
                int(payload["feature_version"]),
                int(payload["scorer_version"]),
                float(payload["exact_agreement_bonus"]),
                int(payload["artifact_version"]),
            )
            ranker = HistGradientBoostingTemporalRanker(artifact)
        elif model_type == "kinematic-ensemble":
            shot_type = str(payload["supported_shot_type"])

            def component(name: str) -> HistGradientBoostingArtifact:
                values = model[name]
                return HistGradientBoostingArtifact(
                    float(values["baseline"]),
                    trees_from(values["trees"]),
                    shot_type,
                    0.0,
                    vector_size=KINEMATIC_VECTOR_SIZE,
                )

            artifact = KinematicEnsembleArtifact(
                component("ordinary"),
                component("family_balanced"),
                float(model["ordinary_weight"]),
                shot_type,
                float(payload["operating_threshold"]),
                int(payload["feature_version"]),
                int(payload["kinematic_feature_version"]),
                int(payload["scorer_version"]),
                int(payload["artifact_version"]),
            )
            ranker = KinematicEnsembleRanker(artifact)
        else:
            raise TemporalRankerArtifactError("unsupported temporal ranker model type")
    except TemporalRankerArtifactError:
        raise
    except (OSError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError, OverflowError) as error:
        raise TemporalRankerArtifactError(f"malformed ranker artifact: {path}") from error
    if artifact.artifact_version != ARTIFACT_VERSION:
        raise TemporalRankerArtifactError("unsupported ranker artifact version")
    if artifact.feature_version != expected_feature_version:
        raise TemporalRankerArtifactError("incompatible temporal feature version")
    if artifact.scorer_version != expected_scorer_version:
        raise TemporalRankerArtifactError("incompatible visual scorer version")
    if artifact.supported_shot_type != expected_shot_type:
        raise TemporalRankerArtifactError("unsupported temporal ranker shot type")
    if (
        isinstance(artifact, KinematicEnsembleArtifact)
        and artifact.kinematic_feature_version != KINEMATIC_FEATURE_VERSION
    ):
        raise TemporalRankerArtifactError("incompatible kinematic feature version")
    return ranker


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


@dataclass(frozen=True)
class HistGradientBoostingTemporalRanker:
    """Dependency-free inference for the accepted prototype's exported trees."""

    artifact: HistGradientBoostingArtifact

    @property
    def feature_version(self) -> int:
        return self.artifact.feature_version

    @property
    def confidence_threshold(self) -> float:
        return self.artifact.operating_threshold

    @property
    def exact_agreement_bonus(self) -> float:
        return self.artifact.exact_agreement_bonus

    def _score(self, vector: tuple[float, ...]) -> float:
        score = self.artifact.baseline
        for tree in self.artifact.trees:
            node_index = 0
            while not tree[node_index].is_leaf:
                node = tree[node_index]
                value = vector[node.feature_index]
                if math.isnan(value):
                    go_left = node.missing_go_to_left
                else:
                    go_left = value <= node.threshold
                node_index = node.left if go_left else node.right
            score += tree[node_index].value
        return score

    def predict(self, features: tuple[TemporalFeatures, ...]) -> TemporalPrediction:
        vectors = all_temporal_feature_vectors(features)
        scores = tuple(self._score(vector) for vector in vectors)
        best = max(
            range(len(scores)),
            key=lambda index: (scores[index], -features[index].frame_ordinal),
        )
        local = max(scores[max(0, best - 1) : min(len(scores), best + 2)])
        outside = scores[: max(0, best - 1)] + scores[min(len(scores), best + 2) :]
        margin = local - max(outside, default=0.0)
        confidence = max(0.0, min(1.0, margin))
        return TemporalPrediction(features[best].frame_ordinal, confidence)


@dataclass(frozen=True)
class KinematicEnsembleRanker:
    """Dependency-free direct frame ranker selected by the bake-off."""

    artifact: KinematicEnsembleArtifact

    @property
    def feature_version(self) -> int:
        return self.artifact.feature_version

    @property
    def confidence_threshold(self) -> float:
        return self.artifact.operating_threshold

    @staticmethod
    def _scores(
        artifact: HistGradientBoostingArtifact,
        vectors: tuple[tuple[float, ...], ...],
    ) -> tuple[float, ...]:
        ranker = HistGradientBoostingTemporalRanker(artifact)
        return tuple(ranker._score(vector) for vector in vectors)

    @staticmethod
    def _standardize(values: tuple[float, ...]) -> tuple[float, ...]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = max(1e-8, math.sqrt(variance))
        return tuple((value - mean) / scale for value in values)

    def predict_frames(self, frames: tuple[VisualFrame, ...]) -> TemporalPrediction:
        if not frames:
            raise ValueError("cannot rank an empty frame window")
        vectors = all_kinematic_feature_vectors(frames)
        ordinary = self._standardize(self._scores(self.artifact.ordinary, vectors))
        balanced = self._standardize(
            self._scores(self.artifact.family_balanced, vectors)
        )
        weight = self.artifact.ordinary_weight
        scores = tuple(
            weight * first + (1 - weight) * second
            for first, second in zip(ordinary, balanced, strict=True)
        )
        best = max(
            range(len(scores)),
            key=lambda index: (scores[index], -frames[index].ordinal),
        )
        outside = [
            score for index, score in enumerate(scores) if abs(index - best) > 1
        ]
        margin = scores[best] - max(outside, default=scores[best])
        confidence = max(0.0, min(1.0, margin))
        return TemporalPrediction(frames[best].ordinal, confidence)


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
    "HGB_EXACT_AGREEMENT_BONUS",
    "FEATURES_PER_FRAME",
    "HistGradientBoostingArtifact",
    "HistGradientBoostingNode",
    "HistGradientBoostingTemporalRanker",
    "KINEMATIC_VECTOR_SIZE",
    "KinematicEnsembleArtifact",
    "KinematicEnsembleRanker",
    "SCORER_VERSION",
    "TEMPORAL_VECTOR_SIZE",
    "LinearTemporalRanker",
    "TemporalRankerArtifact",
    "TemporalRankerArtifactError",
    "all_temporal_feature_vectors",
    "all_kinematic_feature_vectors",
    "fit_temporal_ranker",
    "feature_vectors_from_frames",
    "load_temporal_ranker",
    "temporal_feature_vector",
]

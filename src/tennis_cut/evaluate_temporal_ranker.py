"""Offline training, grouped evaluation, and threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .temporal_ranker import (
    HGB_EXACT_AGREEMENT_BONUS,
    HistGradientBoostingArtifact,
    HistGradientBoostingNode,
    LinearTemporalRanker,
    all_temporal_feature_vectors,
    fit_temporal_ranker,
)
from .visual_contact import TemporalFeatures, TemporalPrediction


@dataclass(frozen=True)
class LabeledWindow:
    """One labeled candidate window used by the offline evaluator."""

    group: str
    label_frame: int
    features: tuple[TemporalFeatures, ...]
    deterministic_frame: int | None = None
    omission_reason: str | None = None
    total_swings: int = 1


@dataclass(frozen=True)
class EvaluationMetrics:
    total_swings: int
    included_swings: int
    coverage: float
    exact_frame_precision: float
    within_one_frame_precision: float
    precision_coverage_points: tuple[dict[str, float | int], ...]
    omission_reasons: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_swings": self.total_swings,
            "included_swings": self.included_swings,
            "coverage": self.coverage,
            "exact_frame_precision": self.exact_frame_precision,
            "within_one_frame_precision": self.within_one_frame_precision,
            "precision_coverage_points": list(self.precision_coverage_points),
            "omission_reasons": self.omission_reasons,
        }


def grouped_holdout_split(
    records: Sequence[LabeledWindow], held_out_group: str
) -> tuple[tuple[LabeledWindow, ...], tuple[LabeledWindow, ...]]:
    """Split by complete camera-roll family, never by individual windows."""

    train = tuple(record for record in records if record.group != held_out_group)
    test = tuple(record for record in records if record.group == held_out_group)
    return train, test


def grouped_holdout_folds(records: Sequence[LabeledWindow]) -> tuple[tuple[str, tuple[LabeledWindow, ...], tuple[LabeledWindow, ...]], ...]:
    """Return deterministic leave-one-family-out train/test folds."""

    groups = sorted({record.group for record in records})
    return tuple(
        (group, *grouped_holdout_split(records, group))
        for group in groups
    )


def choose_operating_threshold(
    confidences: Iterable[float],
    correct_within_one: Callable[[float], bool],
    *,
    total_swings: int,
    minimum_precision: float = 0.95,
    correctness: Sequence[bool] | None = None,
) -> tuple[float, tuple[dict[str, float | int], ...], bool]:
    """Maximize included coverage subject to a held-out precision floor."""

    confidence_list = [float(value) for value in confidences]
    values = sorted(set(confidence_list), reverse=True)
    points: list[dict[str, float | int]] = []
    for threshold in values + [0.0]:
        included_indexes = [index for index, value in enumerate(confidence_list) if value >= threshold]
        correct = sum(
            (correctness[index] if correctness is not None else correct_within_one(confidence_list[index]))
            for index in included_indexes
        )
        points.append({
            "threshold": threshold,
            "included": len(included_indexes),
            "coverage": len(included_indexes) / total_swings if total_swings else 0.0,
            "within_one_frame_precision": correct / len(included_indexes) if included_indexes else 1.0,
        })
    feasible = [point for point in points if point["included"] > 0 and point["within_one_frame_precision"] >= minimum_precision]
    if not feasible:
        maximum = max(values, default=0.0)
        return (math.nextafter(maximum, math.inf), tuple(points), False)
    winner = max(feasible, key=lambda point: (int(point["included"]), -float(point["threshold"])))
    return float(winner["threshold"]), tuple(points), True


def _prediction(ranker: LinearTemporalRanker, record: LabeledWindow) -> TemporalPrediction:
    return ranker.predict(record.features)


def evaluate_predictions(
    records: Sequence[LabeledWindow],
    predictions: Sequence[TemporalPrediction | None],
    threshold: float,
    *,
    exact_agreement_bonus: float = 0.12,
) -> EvaluationMetrics:
    """Calculate release-gate metrics and stable omission accounting."""

    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")
    included: list[tuple[LabeledWindow, TemporalPrediction]] = []
    omissions: dict[str, int] = {}
    for record, prediction in zip(records, predictions, strict=True):
        reason = record.omission_reason
        if prediction is None:
            reason = reason or "no prediction"
        elif record.deterministic_frame is not None and abs(prediction.frame_ordinal - record.deterministic_frame) > 1:
            reason = reason or "temporal ranker disagrees"
        elif _confidence_for(record, prediction, exact_agreement_bonus) < threshold:
            reason = reason or "below operating threshold"
        if reason is None:
            included.append((record, prediction))
        else:
            omissions[reason] = omissions.get(reason, 0) + 1
    selected_frames = tuple(
        (
            record.label_frame,
            prediction.frame_ordinal
            if record.deterministic_frame is None
            else record.deterministic_frame,
        )
        for record, prediction in included
    )
    exact = sum(selected == label for label, selected in selected_frames)
    within = sum(abs(selected - label) <= 1 for label, selected in selected_frames)
    total = sum(record.total_swings for record in records)
    count = len(included)
    return EvaluationMetrics(
        total,
        count,
        count / total if total else 0.0,
        exact / count if count else 0.0,
        within / count if count else 0.0,
        (),
        omissions,
    )


def _confidence_for(
    record: LabeledWindow,
    prediction: TemporalPrediction,
    exact_agreement_bonus: float = 0.12,
) -> float:
    confidence = prediction.confidence
    if record.deterministic_frame == prediction.frame_ordinal:
        confidence = min(1.0, confidence + exact_agreement_bonus)
    return confidence


def calibrate_threshold(
    records: Sequence[LabeledWindow],
    predictions: Sequence[TemporalPrediction | None],
    *,
    minimum_precision: float = 0.95,
    exact_agreement_bonus: float = 0.12,
) -> tuple[float, EvaluationMetrics]:
    """Choose a threshold from held-out predictions and return its metrics."""

    candidates = [
        (record, prediction)
        for record, prediction in zip(records, predictions, strict=True)
        if prediction is not None
        and record.omission_reason is None
        and (
            record.deterministic_frame is None
            or abs(prediction.frame_ordinal - record.deterministic_frame) <= 1
        )
    ]
    confidences = [
        _confidence_for(record, prediction, exact_agreement_bonus)
        for record, prediction in candidates
    ]

    threshold, points, _ = choose_operating_threshold(
        confidences,
        lambda _confidence: False,
        total_swings=sum(record.total_swings for record in records),
        minimum_precision=minimum_precision,
        correctness=tuple(
            abs(
                (
                    prediction.frame_ordinal
                    if record.deterministic_frame is None
                    else record.deterministic_frame
                )
                - record.label_frame
            )
            <= 1
            for record, prediction in candidates
        ),
    )
    metrics = evaluate_predictions(
        records,
        predictions,
        threshold,
        exact_agreement_bonus=exact_agreement_bonus,
    )
    return threshold, EvaluationMetrics(
        metrics.total_swings,
        metrics.included_swings,
        metrics.coverage,
        metrics.exact_frame_precision,
        metrics.within_one_frame_precision,
        points,
        metrics.omission_reasons,
    )


def evaluate_ranker(records: Sequence[LabeledWindow], ranker: LinearTemporalRanker, threshold: float | None = None) -> EvaluationMetrics:
    predictions = tuple(_prediction(ranker, record) if record.omission_reason is None else None for record in records)
    if threshold is None:
        threshold, metrics = calibrate_threshold(records, predictions)
        return metrics
    return evaluate_predictions(records, predictions, threshold)


def grouped_cross_validate(
    records: Sequence[LabeledWindow],
) -> tuple[TemporalPrediction | None, ...]:
    """Predict every record with a ranker trained without its camera-roll group."""

    predictions: list[TemporalPrediction | None] = [None] * len(records)
    positions = {id(record): index for index, record in enumerate(records)}
    for _, training, held_out in grouped_holdout_folds(records):
        if not training:
            continue
        artifact = fit_temporal_ranker((record.features, record.label_frame) for record in training)
        ranker = LinearTemporalRanker(artifact)
        for record in held_out:
            predictions[positions[id(record)]] = (
                None if record.omission_reason is not None else ranker.predict(record.features)
            )
    return tuple(predictions)


def _fit_hist_gradient_boosting(records: Sequence[LabeledWindow]) -> Any:
    """Fit the accepted prototype estimator to labeled temporal windows."""

    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    rows: list[tuple[tuple[float, ...], float]] = []
    for record in records:
        vectors = all_temporal_feature_vectors(record.features)
        label_index = next(
            (
                index
                for index, feature in enumerate(record.features)
                if feature.frame_ordinal == record.label_frame
            ),
            None,
        )
        if label_index is None:
            raise ValueError(
                f"label frame is outside its feature window: {record.label_frame}"
            )
        for index in range(
            max(0, label_index - 12), min(len(vectors), label_index + 13)
        ):
            rows.append(
                (vectors[index], math.exp(-abs(index - label_index) / 0.8))
            )
    if not rows:
        raise ValueError("cannot train temporal ranker without labeled windows")
    return HistGradientBoostingRegressor(
        max_iter=160,
        max_leaf_nodes=15,
        l2_regularization=2,
        random_state=7,
    ).fit(
        np.asarray([values for values, _ in rows]),
        np.asarray([target for _, target in rows]),
    )


def _hist_gradient_boosting_prediction(
    model: Any,
    features: tuple[TemporalFeatures, ...],
) -> TemporalPrediction:
    import numpy as np

    vectors = all_temporal_feature_vectors(features)
    scores = np.asarray(model.predict(np.asarray(vectors)), dtype=float)
    best = int(np.argmax(scores))
    local = float(scores[max(0, best - 1) : min(len(scores), best + 2)].max())
    outside = np.concatenate(
        (scores[: max(0, best - 1)], scores[min(len(scores), best + 2) :])
    )
    margin = local - float(outside.max()) if len(outside) else local
    confidence = max(0.0, min(1.0, margin))
    return TemporalPrediction(features[best].frame_ordinal, confidence)


def grouped_cross_validate_hist_gradient_boosting(
    records: Sequence[LabeledWindow],
) -> tuple[TemporalPrediction | None, ...]:
    """Reproduce the prototype's camera-family-held-out predictions."""

    predictions: list[TemporalPrediction | None] = [None] * len(records)
    positions = {id(record): index for index, record in enumerate(records)}
    for _, training, held_out in grouped_holdout_folds(records):
        if not training:
            continue
        model = _fit_hist_gradient_boosting(training)
        for record in held_out:
            predictions[positions[id(record)]] = (
                None
                if record.omission_reason is not None
                else _hist_gradient_boosting_prediction(model, record.features)
            )
    return tuple(predictions)


def _portable_hist_gradient_boosting_artifact(
    model: Any,
    *,
    threshold: float,
    supported_shot_type: str,
) -> HistGradientBoostingArtifact:
    trees = tuple(
        tuple(
            HistGradientBoostingNode(
                float(node["value"]),
                int(node["feature_idx"]),
                float(node["num_threshold"]),
                bool(node["missing_go_to_left"]),
                int(node["left"]),
                int(node["right"]),
                bool(node["is_leaf"]),
            )
            for node in predictors[0].nodes
        )
        for predictors in model._predictors
    )
    return HistGradientBoostingArtifact(
        baseline=float(model._baseline_prediction[0, 0]),
        trees=trees,
        supported_shot_type=supported_shot_type,
        operating_threshold=threshold,
    )


def train_and_export(
    records: Sequence[LabeledWindow],
    output: Path,
    *,
    supported_shot_type: str = "forehand",
) -> tuple[HistGradientBoostingArtifact, EvaluationMetrics]:
    """Evaluate, fit, and export the accepted prototype ranker without pickle."""

    if len({record.group for record in records}) < 2:
        raise ValueError("temporal ranker calibration requires at least two camera-roll groups")
    predictions = grouped_cross_validate_hist_gradient_boosting(records)
    threshold, metrics = calibrate_threshold(
        records,
        predictions,
        exact_agreement_bonus=HGB_EXACT_AGREEMENT_BONUS,
    )
    model = _fit_hist_gradient_boosting(records)
    artifact = _portable_hist_gradient_boosting_artifact(
        model,
        threshold=threshold,
        supported_shot_type=supported_shot_type,
    )
    artifact.save(output)
    return artifact, metrics


def load_json_records(path: Path) -> tuple[LabeledWindow, ...]:
    """Read the small, detection-free JSON format used by the offline CLI."""

    payload = json.loads(path.read_text())
    records: list[LabeledWindow] = []
    for item in payload:
        features = tuple(
            TemporalFeatures(
                int(frame["frame_ordinal"]),
                *(float(value) for value in frame["values"]),
            )
            for frame in item["features"]
        )
        records.append(LabeledWindow(
            str(item["group"]),
            int(item["label_frame"]),
            features,
            None if item.get("deterministic_frame") is None else int(item["deterministic_frame"]),
            item.get("omission_reason"),
            int(item.get("total_swings", 1)),
        ))
    return tuple(records)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    _, metrics = train_and_export(load_json_records(args.records), args.artifact)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "EvaluationMetrics",
    "LabeledWindow",
    "calibrate_threshold",
    "choose_operating_threshold",
    "evaluate_predictions",
    "evaluate_ranker",
    "grouped_holdout_folds",
    "grouped_holdout_split",
    "grouped_cross_validate",
    "load_json_records",
    "train_and_export",
]

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json

from tennis_cut.evaluate_temporal_ranker import (
    LabeledWindow,
    calibrate_threshold,
    evaluate_predictions,
    grouped_holdout_folds,
    train_and_export,
)
from tennis_cut.temporal_ranker import (
    TEMPORAL_VECTOR_SIZE,
    TemporalRankerArtifactError,
    all_temporal_feature_vectors,
    load_temporal_ranker,
    TemporalRankerArtifact,
    PrototypeTemporalRanker,
    fit_prototype_temporal_ranker,
)
from tennis_cut.visual_contact import TemporalFeatures, TemporalPrediction


def feature_rows(count: int = 3) -> tuple[TemporalFeatures, ...]:
    return tuple(TemporalFeatures(index, 0.1 * index, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0) for index in range(count))


class TemporalRankerTests(unittest.TestCase):
    def test_extracts_nine_frames_and_clamps_at_boundaries(self) -> None:
        vectors = all_temporal_feature_vectors(feature_rows())

        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(vectors[0]), TEMPORAL_VECTOR_SIZE)
        self.assertEqual(vectors[0][:7], vectors[0][7:14])
        self.assertEqual(vectors[-1][-7:], vectors[-1][-14:-7])

    def test_loader_rejects_missing_and_incompatible_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ranker.json"
            with self.assertRaisesRegex(TemporalRankerArtifactError, "missing"):
                load_temporal_ranker(path)
            artifact = TemporalRankerArtifact((0.0,) * TEMPORAL_VECTOR_SIZE, 0.0, "backhand")
            artifact.save(path)
            with self.assertRaisesRegex(TemporalRankerArtifactError, "shot type"):
                load_temporal_ranker(path)
            payload = artifact.to_dict()
            payload["supported_shot_type"] = "forehand"
            payload["feature_version"] = 99
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(TemporalRankerArtifactError, "feature version"):
                load_temporal_ranker(path)

    def test_grouped_holdouts_keep_each_camera_roll_family_whole(self) -> None:
        records = tuple(
            LabeledWindow(group, index, feature_rows())
            for group, index in (("a", 0), ("a", 1), ("b", 2))
        )

        folds = grouped_holdout_folds(records)

        self.assertEqual([fold[0] for fold in folds], ["a", "b"])
        self.assertEqual(len(folds[0][1]), 1)
        self.assertEqual(len(folds[0][2]), 2)

    def test_metrics_report_precision_coverage_and_omissions(self) -> None:
        records = (
            LabeledWindow("a", 1, feature_rows(), deterministic_frame=1),
            LabeledWindow("b", 4, feature_rows(), deterministic_frame=4),
            LabeledWindow("c", 2, feature_rows(), deterministic_frame=2),
        )
        predictions = (
            TemporalPrediction(1, 0.8),
            TemporalPrediction(7, 0.9),
            None,
        )

        metrics = evaluate_predictions(records, predictions, 0.5)

        self.assertEqual(metrics.total_swings, 3)
        self.assertEqual(metrics.included_swings, 1)
        self.assertEqual(metrics.exact_frame_precision, 1.0)
        self.assertEqual(metrics.omission_reasons, {"temporal ranker disagrees": 1, "no prediction": 1})

    def test_threshold_uses_maximum_coverage_at_precision_floor(self) -> None:
        records = tuple(LabeledWindow("a", index, feature_rows(), deterministic_frame=index) for index in range(4))
        predictions = tuple(
            TemporalPrediction(index if index < 3 else index + 3, confidence)
            for index, confidence in enumerate((0.9, 0.8, 0.7, 0.6))
        )

        threshold, metrics = calibrate_threshold(records, predictions)

        self.assertEqual(threshold, 0.0)
        self.assertEqual(metrics.included_swings, 3)
        self.assertEqual(metrics.within_one_frame_precision, 1.0)

    def test_training_exports_ranker_and_compatibility_metadata(self) -> None:
        records = tuple(
            LabeledWindow(group, 1, feature_rows(), deterministic_frame=1)
            for group in ("camera-a", "camera-b")
        )
        with TemporaryDirectory() as directory:
            artifact, metrics = train_and_export(records, Path(directory) / "ranker.json")

            self.assertEqual(artifact.supported_shot_type, "forehand")
            self.assertEqual(artifact.feature_version, 1)
            self.assertEqual(artifact.scorer_version, 3)
            self.assertTrue((Path(directory) / "ranker.json").is_file())
            self.assertEqual(metrics.total_swings, 2)

    def test_prototype_ranker_uses_grouped_hgb_and_original_bonus(self) -> None:
        records = tuple(
            LabeledWindow(group, 2, feature_rows(5), deterministic_frame=2)
            for group in ("camera-a", "camera-b")
        )

        ranker = fit_prototype_temporal_ranker(records)

        self.assertIsInstance(ranker, PrototypeTemporalRanker)
        self.assertEqual(ranker.exact_agreement_bonus, 0.25)
        self.assertIsInstance(ranker.predict(feature_rows(5)), TemporalPrediction)


if __name__ == "__main__":
    unittest.main()

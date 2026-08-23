from __future__ import annotations

from fractions import Fraction
import unittest

from tennis_cut.visual_contact import (
    Detection,
    FEATURE_VERSION,
    FrameEvidence,
    ObjectKind,
    TemporalPrediction,
    VisualFrame,
    rank_contact_frames,
    select_contact_frame,
)


def detection(
    kind: ObjectKind,
    box: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> Detection:
    return Detection(kind, box, confidence)


def frame(ordinal: int, detections: tuple[Detection, ...]) -> VisualFrame:
    return VisualFrame(
        FrameEvidence(
            ordinal=ordinal,
            timestamp=Fraction(ordinal, 120),
            detections=detections,
        )
    )


class VisualContactSelectionTests(unittest.TestCase):
    def test_uses_largest_person_and_rejects_other_player_racket(self) -> None:
        frames = tuple(
            frame(
                ordinal,
                (
                    detection("person", (0, 0, 100, 300)),
                    detection("person", (500, 0, 570, 180)),
                    *(
                        ()
                        if ordinal == 1
                        else (detection("racket", (35, 140, 90, 230)),)
                    ),
                    detection("racket", (515, 130, 640, 230)),
                    detection(
                        "ball",
                        (
                            (10, 30, 65, 100, 130)[ordinal],
                            150,
                            (20, 40, 75, 110, 140)[ordinal],
                            160,
                        ),
                    ),
                ),
            )
            for ordinal in range(5)
        )

        result = rank_contact_frames(frames)

        self.assertIsNotNone(result.selected_frame)
        self.assertIn(result.selected_frame, range(5))
        self.assertIn(0, result.racket_frames)
        self.assertIsNone(result.omission_reason)

    def test_rejects_stationary_ball_and_reports_missing_moving_ball(self) -> None:
        frames = tuple(
            frame(
                ordinal,
                (
                    detection("person", (0, 0, 100, 300)),
                    detection("racket", (35, 140, 90, 230)),
                    detection("ball", (70, 155, 80, 165)),
                ),
            )
            for ordinal in range(5)
        )

        result = rank_contact_frames(frames)

        self.assertIsNone(result.selected_frame)
        self.assertEqual(result.omission_reason, "no moving ball evidence")

    def test_tie_resolves_to_earlier_existing_frame_and_ranker_is_replaceable(self) -> None:
        frames = tuple(
            frame(
                ordinal,
                (
                    detection("person", (0, 0, 100, 300)),
                    detection("racket", (35, 140, 90, 230)),
                    detection("ball", (72 + abs(2 - ordinal) * 8, 155, 82 + abs(2 - ordinal) * 8, 165)),
                ),
            )
            for ordinal in range(5)
        )
        ranking = rank_contact_frames(frames)

        class Ranker:
            feature_version = ranking.feature_version

            def predict(self, features):
                return TemporalPrediction(frame_ordinal=1, confidence=0.8)

        result = select_contact_frame(frames, Ranker())

        self.assertEqual(ranking.selected_frame, 2)
        self.assertEqual(result.selected_frame, 2)
        self.assertEqual(result.frame.timestamp, Fraction(2, 120))
        self.assertEqual(result.contact_confidence, 0.8)

    def test_omits_temporal_prediction_below_ranker_confidence_threshold(self) -> None:
        frames = tuple(
            frame(
                ordinal,
                (
                    detection("person", (0, 0, 100, 300)),
                    detection("racket", (35, 140, 90, 230)),
                    detection("ball", (70 + ordinal * 8, 155, 80 + ordinal * 8, 165)),
                ),
            )
            for ordinal in range(5)
        )

        class LowConfidenceRanker:
            feature_version = FEATURE_VERSION
            confidence_threshold = 0.5

            def predict(self, features):
                return TemporalPrediction(features[2].frame_ordinal, 0.1)

        result = select_contact_frame(frames, LowConfidenceRanker())

        self.assertIsNone(result.frame)
        self.assertEqual(result.omission_reason, "below contact confidence threshold")


if __name__ == "__main__":
    unittest.main()

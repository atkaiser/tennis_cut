from dataclasses import FrozenInstanceError
from fractions import Fraction
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from tennis_cut.swing_detection import (
    AudioCandidate,
    DetectedSwing,
    DetectionConfig,
    LegacySwingDetails,
    _normalize_bounce_candidates,
    _suppress_audio_candidates,
    detect_comparison_user_swings,
    detect_user_swings,
)
from tennis_cut.visual_contact import (
    ContactSelection,
    FrameEvidence,
    SourceFrameIdentity,
    VisualFrame,
)


class DetectedSwingTests(unittest.TestCase):
    def test_exposes_only_accepted_swing_identity_as_immutable_data(self) -> None:
        swing = DetectedSwing(
            ordinal=2,
            contact_timestamp=Fraction(41, 8),
            shot_type="forehand",
        )

        self.assertEqual(
            (swing.ordinal, swing.contact_timestamp, swing.shot_type),
            (2, Fraction(41, 8), "forehand"),
        )
        with self.assertRaises(FrozenInstanceError):
            swing.ordinal = 3  # type: ignore[misc]


class AudioCandidateTests(unittest.TestCase):
    def test_suppression_prefers_score_then_earlier_ties_and_preserves_boundaries(self) -> None:
        candidates = (
            AudioCandidate(2.0, 0.70, 4),
            AudioCandidate(1.0, 0.80, 1),
            AudioCandidate(1.10, 0.90, 2),
            AudioCandidate(1.35, 0.90, 3),
            AudioCandidate(2.10, 0.70, 5),
        )

        kept, omitted = _suppress_audio_candidates(candidates, 0.25)

        self.assertEqual(
            kept,
            (
                AudioCandidate(1.10, 0.90, 2),
                AudioCandidate(1.35, 0.90, 3),
                AudioCandidate(2.0, 0.70, 4),
            ),
        )
        self.assertEqual(
            omitted,
            (
                AudioCandidate(1.0, 0.80, 1),
                AudioCandidate(2.10, 0.70, 5),
            ),
        )

    def test_bounce_normalization_collapses_measured_pairs_to_later_events(self) -> None:
        candidates = tuple(
            AudioCandidate(timestamp, score, index)
            for index, (timestamp, score) in enumerate(
                (
                    (2.575, 0.99),
                    (3.075, 0.60),
                    (5.675, 0.55),
                    (6.275, 0.95),
                    (8.775, 0.97),
                    (9.225, 0.65),
                )
            )
        )

        kept, omitted = _normalize_bounce_candidates(candidates)

        self.assertEqual(
            tuple(candidate.timestamp for candidate in kept),
            (3.075, 6.275, 9.225),
        )
        self.assertEqual(
            tuple(candidate.source_index for candidate in omitted),
            (0, 2, 4),
        )

    def test_bounce_normalization_keeps_isolated_and_out_of_range_events(self) -> None:
        candidates = tuple(
            AudioCandidate(timestamp, 0.8, index)
            for index, timestamp in enumerate((1.0, 1.34, 2.10, 3.0))
        )

        kept, omitted = _normalize_bounce_candidates(candidates)

        self.assertEqual(kept, candidates)
        self.assertEqual(omitted, ())

    def test_bounce_normalization_uses_inclusive_boundaries_and_collapses_chains(self) -> None:
        candidates = tuple(
            AudioCandidate(timestamp, score, index)
            for index, (timestamp, score) in enumerate(
                ((1.0, 0.99), (1.35, 0.80), (2.10, 0.70), (4.0, 0.60))
            )
        )

        kept, omitted = _normalize_bounce_candidates(candidates)

        self.assertEqual(
            tuple(candidate.timestamp for candidate in kept),
            (2.10, 4.0),
        )
        self.assertEqual(
            tuple(candidate.timestamp for candidate in omitted),
            (1.0, 1.35),
        )


class DetectUserSwingsTests(unittest.TestCase):
    def _detect_audio_candidates(
        self,
        candidates: list[AudioCandidate],
        *,
        shot_results: list[bool] | None,
    ) -> tuple[DetectedSwing, ...]:
        pop_detector = Mock()
        pop_detector.find_candidates.return_value = candidates
        person_detector = Mock()
        person_detector.find_box.return_value = (0, 0, 32, 32)
        shot_detector = Mock()
        if shot_results is not None:
            shot_detector.is_shot.side_effect = shot_results

        def create_frame(_video: Path, _time: float, output: Path) -> None:
            Image.new("RGB", (32, 32), "white").save(output)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "user.mp4"
            source.touch()
            config = DetectionConfig(
                audio_model=Path("audio.pth"),
                shot_model=None if shot_results is None else Path("shot.pkl"),
                shot_type_model=None,
                device="cpu",
            )
            with (
                patch(
                    "tennis_cut.swing_detection.probe_video",
                    return_value={"resolution": (32, 32)},
                ),
                patch("tennis_cut.swing_detection.extract_audio"),
                patch(
                    "tennis_cut.swing_detection.extract_frame",
                    side_effect=create_frame,
                ),
                patch(
                    "tennis_cut.swing_detection.PopDetector",
                    return_value=pop_detector,
                ),
                patch(
                    "tennis_cut.swing_detection.PersonDetector",
                    return_value=person_detector,
                ),
                patch(
                    "tennis_cut.swing_detection.ShotDetector",
                    return_value=shot_detector,
                ),
            ):
                return detect_user_swings(source, config)

    def test_rejected_high_score_event_cannot_suppress_accepted_swing(self) -> None:
        swings = self._detect_audio_candidates(
            [
                AudioCandidate(1.0, 0.99, 0),
                AudioCandidate(2.0, 0.60, 1),
            ],
            shot_results=[False, True],
        )

        self.assertEqual(
            tuple(swing.contact_timestamp for swing in swings),
            (Fraction(2),),
        )

    def test_final_suppression_prefers_higher_score_accepted_swing(self) -> None:
        swings = self._detect_audio_candidates(
            [
                AudioCandidate(1.0, 0.60, 0),
                AudioCandidate(2.0, 0.90, 1),
            ],
            shot_results=[True, True],
        )

        self.assertEqual(
            tuple(swing.contact_timestamp for swing in swings),
            (Fraction(2),),
        )
        self.assertEqual(swings[0].ordinal, 0)

    def test_final_suppression_retains_exact_boundary_events(self) -> None:
        swings = self._detect_audio_candidates(
            [
                AudioCandidate(1.0, 0.90, 0),
                AudioCandidate(2.25, 0.60, 1),
            ],
            shot_results=[True, True],
        )

        self.assertEqual(
            tuple(swing.contact_timestamp for swing in swings),
            (Fraction(1), Fraction(9, 4)),
        )
        self.assertEqual(tuple(swing.ordinal for swing in swings), (0, 1))

    def test_disabled_shot_model_still_applies_final_suppression(self) -> None:
        swings = self._detect_audio_candidates(
            [
                AudioCandidate(1.0, 0.60, 0),
                AudioCandidate(2.0, 0.90, 1),
            ],
            shot_results=None,
        )

        self.assertEqual(
            tuple(swing.contact_timestamp for swing in swings),
            (Fraction(2),),
        )

    def test_automatic_device_selection_applies_to_every_detection_model(self) -> None:
        pop_detector = Mock()
        pop_detector.find_candidates.return_value = []

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "user.mp4"
            source.touch()
            config = DetectionConfig(
                audio_model=Path("audio.pth"),
                shot_model=Path("shot.pkl"),
                shot_type_model=Path("type.pkl"),
            )

            with (
                patch(
                    "tennis_cut.swing_detection.probe_video",
                    return_value={"resolution": (640, 360)},
                ),
                patch("tennis_cut.swing_detection.extract_audio"),
                patch(
                    "tennis_cut.swing_detection.PopDetector",
                    return_value=pop_detector,
                ) as pop_type,
                patch("tennis_cut.swing_detection.PersonDetector") as person_type,
                patch("tennis_cut.swing_detection.ShotDetector") as shot_type,
                patch(
                    "tennis_cut.swing_detection.ShotTypeClassifier"
                ) as classifier_type,
                patch("torch.backends.mps.is_available", return_value=True),
                patch("torch.cuda.is_available", return_value=True),
            ):
                swings = detect_user_swings(source, config)

        self.assertEqual(swings, ())
        pop_type.assert_called_once_with(Path("audio.pth"), device="mps")
        person_type.assert_called_once_with("mps")
        shot_type.assert_called_once_with(Path("shot.pkl"), device="mps")
        classifier_type.assert_called_once_with(Path("type.pkl"), device="mps")

    def test_returns_only_accepted_swings_in_source_order_with_exact_time(self) -> None:
        pop_detector = Mock()
        pop_detector.find_candidates.return_value = [
            AudioCandidate(2.125, 0.9, 0),
            AudioCandidate(4.25, 0.8, 1),
            AudioCandidate(6.5, 0.7, 2),
        ]

        person_detector = Mock()
        person_detector.find_box.return_value = (100, 100, 300, 500)

        shot_detector = Mock()
        shot_detector.is_shot.side_effect = [False, True, True]

        shot_type_classifier = Mock()
        shot_type_classifier.predict_label.side_effect = ["forehand", "serve"]

        def create_frame(_video: Path, _time: float, output: Path) -> None:
            Image.new("RGB", (640, 360), "white").save(output)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "user.mp4"
            source.touch()
            config = DetectionConfig(
                audio_model=Path("audio.pth"),
                shot_model=Path("shot.pkl"),
                shot_type_model=Path("type.pkl"),
                device="cpu",
            )

            with (
                patch(
                    "tennis_cut.swing_detection.probe_video",
                    return_value={"resolution": (640, 360)},
                ),
                patch("tennis_cut.swing_detection.extract_audio"),
                patch(
                    "tennis_cut.swing_detection.extract_frame",
                    side_effect=create_frame,
                ),
                patch(
                    "tennis_cut.swing_detection.PopDetector",
                    return_value=pop_detector,
                ),
                patch(
                    "tennis_cut.swing_detection.PersonDetector",
                    return_value=person_detector,
                ),
                patch(
                    "tennis_cut.swing_detection.ShotDetector",
                    return_value=shot_detector,
                ),
                patch(
                    "tennis_cut.swing_detection.ShotTypeClassifier",
                    return_value=shot_type_classifier,
                ),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                swings = detect_user_swings(source, config)

        self.assertEqual(
            swings,
            (
                DetectedSwing(0, Fraction(17, 4), "forehand"),
                DetectedSwing(1, Fraction(13, 2), "serve"),
            ),
        )
        self.assertEqual(stdout.getvalue(), "")

    def test_recovers_exact_audio_grid_time_from_float_artifact(self) -> None:
        pop_detector = Mock()
        pop_detector.find_candidates.return_value = [
            AudioCandidate(0.42500000000000004, 0.9, 0)
        ]

        person_detector = Mock()
        person_detector.find_box.return_value = (100, 100, 300, 300)

        def create_frame(_video: Path, _time: float, output: Path) -> None:
            Image.new("RGB", (640, 360), "white").save(output)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "user.mp4"
            source.touch()
            config = DetectionConfig(
                audio_model=Path("audio.pth"),
                shot_model=None,
                shot_type_model=None,
                device="cpu",
            )

            with (
                patch(
                    "tennis_cut.swing_detection.probe_video",
                    return_value={"resolution": (640, 360)},
                ),
                patch("tennis_cut.swing_detection.extract_audio"),
                patch(
                    "tennis_cut.swing_detection.extract_frame",
                    side_effect=create_frame,
                ),
                patch(
                    "tennis_cut.swing_detection.PopDetector",
                    return_value=pop_detector,
                ),
                patch(
                    "tennis_cut.swing_detection.PersonDetector",
                    return_value=person_detector,
                ),
            ):
                swings = detect_user_swings(source, config)

        self.assertEqual(swings[0].contact_timestamp, Fraction(17, 40))

    def test_comparison_detection_replaces_audio_time_and_omits_visual_failures(self) -> None:
        source = Path("user.mp4")
        details = (
            LegacySwingDetails(
                DetectedSwing(0, Fraction(1), "forehand"),
                1.0,
                0.0,
                2.0,
                (0, 0, 1, 1),
            ),
            LegacySwingDetails(
                DetectedSwing(1, Fraction(3), "forehand"),
                3.0,
                2.0,
                4.0,
                (0, 0, 1, 1),
            ),
        )

        class Selector:
            def __init__(self) -> None:
                self.candidates: list[Fraction] = []

            def select(self, _source: Path, candidate: Fraction) -> ContactSelection:
                self.candidates.append(candidate)
                if candidate == Fraction(3):
                    return ContactSelection(None, 0.0, (), "weak visual evidence")
                return ContactSelection(
                    VisualFrame(
                        FrameEvidence(
                            17,
                            Fraction(17, 120),
                            (),
                            3,
                            17,
                            Fraction(1, 120),
                        )
                    ),
                    0.9,
                    (17,),
                    None,
                )

        selector = Selector()
        with patch(
            "tennis_cut.swing_detection._detect_user_swings_with_details",
            return_value=details,
        ), patch(
            "tennis_cut.swing_detection.probe_video",
            return_value={"resolution": (1920, 1080)},
        ):
            swings = detect_comparison_user_swings(
                source,
                DetectionConfig(),
                contact_selector=selector,
            )

        self.assertEqual(selector.candidates, [Fraction(1), Fraction(3)])
        self.assertEqual(
            swings,
            (
                DetectedSwing(
                    0,
                    Fraction(17, 120),
                    "forehand",
                    SourceFrameIdentity(3, 17, Fraction(1, 120)),
                ),
            ),
        )

    def test_comparison_batches_visual_contact_selection_for_all_candidates(self) -> None:
        source = Path("user.mp4")
        details = tuple(
            LegacySwingDetails(
                DetectedSwing(index, timestamp, "forehand"),
                float(timestamp),
                float(timestamp - 1),
                float(timestamp + 1),
                (0, 0, 1, 1),
            )
            for index, timestamp in enumerate((Fraction(1), Fraction(3)))
        )

        class BatchSelector:
            def __init__(self) -> None:
                self.candidates: tuple[Fraction, ...] | None = None

            def select_many(
                self,
                selected_source: Path,
                candidates: tuple[Fraction, ...],
            ) -> tuple[ContactSelection, ...]:
                self.assert_source(selected_source)
                self.candidates = candidates
                return tuple(
                    ContactSelection(
                        VisualFrame(
                            FrameEvidence(
                                120 * index,
                                candidate,
                                (),
                                0,
                                120 * index,
                                Fraction(1, 120),
                            )
                        ),
                        0.9,
                        (120 * index,),
                        None,
                    )
                    for index, candidate in enumerate(candidates, start=1)
                )

            def select(self, _source: Path, _candidate: Fraction) -> ContactSelection:
                raise AssertionError("candidate selection must be batched")

            @staticmethod
            def assert_source(selected_source: Path) -> None:
                if selected_source != source:
                    raise AssertionError("unexpected source")

        selector = BatchSelector()
        with (
            patch(
                "tennis_cut.swing_detection._detect_user_swings_with_details",
                return_value=details,
            ),
            patch(
                "tennis_cut.swing_detection.probe_video",
                return_value={"resolution": (1920, 1080)},
            ),
        ):
            swings = detect_comparison_user_swings(
                source,
                DetectionConfig(),
                contact_selector=selector,
            )

        self.assertEqual(selector.candidates, (Fraction(1), Fraction(3)))
        self.assertEqual(
            tuple(swing.contact_timestamp for swing in swings),
            (Fraction(1), Fraction(3)),
        )

    def test_comparison_loads_configured_ranker_into_stock_selector(self) -> None:
        source = Path("user.mp4")
        details = (
            LegacySwingDetails(
                DetectedSwing(0, Fraction(1), "forehand"),
                1.0,
                0.0,
                2.0,
                (0, 0, 1, 1),
            ),
        )
        ranker = Mock()
        frame_timeline = Mock()
        selector = Mock()
        selector.select.return_value = ContactSelection(
            VisualFrame(FrameEvidence(17, Fraction(17, 120), ())),
            0.9,
            (17,),
            None,
        )
        ranker_path = Path("models/temporal_ranker.json")

        with (
            patch(
                "tennis_cut.swing_detection._detect_user_swings_with_details",
                return_value=details,
            ),
            patch("tennis_cut.swing_detection.probe_video", return_value={"resolution": (1920, 1080)}),
            patch("tennis_cut.swing_detection.StockVisualContactSelector", return_value=selector) as selector_type,
            patch("tennis_cut.temporal_ranker.load_temporal_ranker", return_value=ranker) as load_ranker,
        ):
            swings = detect_comparison_user_swings(
                source,
                DetectionConfig(temporal_ranker_model=ranker_path),
                frame_timeline=frame_timeline,
            )

        load_ranker.assert_called_once_with(ranker_path)
        selector_type.assert_called_once_with(
            device=None,
            ranker=ranker,
            frame_timeline=frame_timeline,
        )
        self.assertEqual(swings[0].contact_timestamp, Fraction(17, 120))

    def test_comparison_logs_aggregate_selector_diagnostics(self) -> None:
        source = Path("user.mp4")
        details = (
            LegacySwingDetails(
                DetectedSwing(0, Fraction(1), "forehand"),
                1.0,
                0.0,
                2.0,
                (0, 0, 1, 1),
            ),
            LegacySwingDetails(
                DetectedSwing(1, Fraction(3), "forehand"),
                3.0,
                2.0,
                4.0,
                (0, 0, 1, 1),
            ),
        )
        selector = Mock()
        selector.select.side_effect = [
            ContactSelection(
                VisualFrame(FrameEvidence(17, Fraction(17, 120), ())),
                0.9,
                (17,),
                None,
            ),
            ContactSelection(None, 0.0, (), "below contact confidence threshold"),
        ]

        with (
            patch(
                "tennis_cut.swing_detection._detect_user_swings_with_details",
                return_value=details,
            ),
            patch(
                "tennis_cut.swing_detection.probe_video",
                return_value={"resolution": (1920, 1080)},
            ),
            self.assertLogs("tennis_cut.swing_detection", level="INFO") as logs,
        ):
            detect_comparison_user_swings(
                source, DetectionConfig(), contact_selector=selector
            )

        self.assertTrue(
            any(
                "visual contact selection: candidates=2 accepted=1 omitted=1 "
                "(below contact confidence threshold=1)" in message
                for message in logs.output
            )
        )


if __name__ == "__main__":
    unittest.main()

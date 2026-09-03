from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from tennis_cut.comparison.diagnostics import (
    CandidateDiagnostic,
    SwingDiagnostics,
    SwingDiagnosticsRecorder,
    write_swing_diagnostics_report,
)
from tennis_cut.swing_detection import (
    AudioCandidate,
    DetectionConfig,
    detect_comparison_user_swings,
)
from tennis_cut.visual_contact import (
    ContactSelection,
    FrameEvidence,
    SourceFrameIdentity,
    VisualFrame,
)


class SwingDiagnosticsReportTests(unittest.TestCase):
    def test_planning_explains_shot_type_and_incomplete_window_omissions(self) -> None:
        recorder = SwingDiagnosticsRecorder(Path("user.mov"))
        recorder.record_audio_candidates([1.0, 2.0, 3.0])
        selection = ContactSelection(
            VisualFrame(FrameEvidence(60, Fraction(1), ())),
            0.8,
            (60,),
            None,
        )
        for index, shot_type in enumerate(("forehand", "backhand", "forehand")):
            recorder.accept_swing_candidate(index, shot_type)
            recorder.record_visual_selection(index, selection, index)

        recorder.record_planning({0}, "forehand")

        self.assertEqual(
            [(item.disposition, item.reason) for item in recorder.snapshot().candidates],
            [
                ("accepted", "selected for comparison rendering"),
                ("omitted", "shot type backhand does not match pro shot type forehand"),
                ("omitted", "incomplete pre-contact or post-contact window"),
            ],
        )

    def test_pipeline_preserves_every_audio_candidate_through_its_disposition(self) -> None:
        pop_detector = Mock()
        pop_detector.find_candidates.return_value = [
            AudioCandidate(1.0, 0.9, 0),
            AudioCandidate(2.0, 0.8, 1),
            AudioCandidate(3.0, 0.7, 2),
        ]
        person_detector = Mock()
        person_detector.find_box.side_effect = [None, (0, 0, 10, 10), (0, 0, 10, 10)]
        shot_detector = Mock()
        shot_detector.is_shot.side_effect = [False, True]
        shot_classifier = Mock()
        shot_classifier.predict_label.return_value = "forehand"
        selector = Mock()
        selector.select.return_value = ContactSelection(
            VisualFrame(
                FrameEvidence(181, Fraction(181, 60), (), 0, 181, Fraction(1, 60))
            ),
            0.72,
            (181,),
            None,
        )

        def create_frame(_video: Path, _time: float, output: Path) -> None:
            Image.new("RGB", (32, 32), "white").save(output)

        with tempfile.TemporaryDirectory() as directory_name:
            source = Path(directory_name) / "user.mov"
            source.touch()
            recorder = SwingDiagnosticsRecorder(source)
            with (
                patch(
                    "tennis_cut.swing_detection.probe_video",
                    return_value={"resolution": (32, 32)},
                ),
                patch("tennis_cut.swing_detection.extract_audio"),
                patch("tennis_cut.swing_detection.extract_frame", side_effect=create_frame),
                patch("tennis_cut.swing_detection.PopDetector", return_value=pop_detector),
                patch(
                    "tennis_cut.swing_detection.PersonDetector",
                    return_value=person_detector,
                ),
                patch("tennis_cut.swing_detection.ShotDetector", return_value=shot_detector),
                patch(
                    "tennis_cut.swing_detection.ShotTypeClassifier",
                    return_value=shot_classifier,
                ),
            ):
                detect_comparison_user_swings(
                    source,
                    DetectionConfig(device="cpu"),
                    contact_selector=selector,
                    diagnostics=recorder,
                )

        snapshot = recorder.snapshot()
        self.assertEqual(
            [(item.disposition, item.reason) for item in snapshot.candidates],
            [
                ("omitted", "no person found"),
                ("omitted", "swing classifier: not a swing"),
                ("accepted", "visual contact accepted"),
            ],
        )
        self.assertEqual(snapshot.candidates[2].contact_frame_ordinal, 181)

    def test_diagnostics_distinguish_each_audio_pipeline_omission_stage(self) -> None:
        pop_detector = Mock()
        pop_detector.find_candidates.return_value = [
            AudioCandidate(1.0, 0.99, 0),
            AudioCandidate(1.5, 0.60, 1),
            AudioCandidate(2.5, 0.95, 2),
            AudioCandidate(4.0, 0.90, 3),
            AudioCandidate(5.0, 0.80, 4),
            AudioCandidate(6.25, 0.70, 5),
        ]
        person_detector = Mock()
        person_detector.find_box.side_effect = [
            None,
            (0, 0, 10, 10),
            (0, 0, 10, 10),
            (0, 0, 10, 10),
            (0, 0, 10, 10),
        ]
        shot_detector = Mock()
        shot_detector.is_shot.side_effect = [False, True, True, True]
        selector = Mock()
        selector.select.return_value = ContactSelection(
            VisualFrame(FrameEvidence(181, Fraction(181, 60), ())),
            0.72,
            (181,),
            None,
        )

        def create_frame(_video: Path, _time: float, output: Path) -> None:
            Image.new("RGB", (32, 32), "white").save(output)

        with tempfile.TemporaryDirectory() as directory_name:
            source = Path(directory_name) / "user.mov"
            source.touch()
            recorder = SwingDiagnosticsRecorder(source)
            with (
                patch(
                    "tennis_cut.swing_detection.probe_video",
                    return_value={"resolution": (32, 32)},
                ),
                patch("tennis_cut.swing_detection.extract_audio"),
                patch("tennis_cut.swing_detection.extract_frame", side_effect=create_frame),
                patch("tennis_cut.swing_detection.PopDetector", return_value=pop_detector),
                patch(
                    "tennis_cut.swing_detection.PersonDetector",
                    return_value=person_detector,
                ),
                patch("tennis_cut.swing_detection.ShotDetector", return_value=shot_detector),
            ):
                detect_comparison_user_swings(
                    source,
                    DetectionConfig(shot_type_model=None, device="cpu"),
                    contact_selector=selector,
                    diagnostics=recorder,
                )

        snapshot = recorder.snapshot()
        self.assertEqual(
            [(item.audio_candidate_index, item.disposition, item.reason) for item in snapshot.candidates],
            [
                (0, "omitted", "bounce normalization: earlier short-gap precursor"),
                (1, "omitted", "no person found"),
                (2, "omitted", "swing classifier: not a swing"),
                (3, "accepted", "visual contact accepted"),
                (4, "omitted", "final audio suppression: lower-scoring nearby candidate"),
                (5, "accepted", "visual contact accepted"),
            ],
        )
        self.assertEqual(
            [item.audio_score for item in snapshot.candidates],
            [0.99, 0.60, 0.95, 0.90, 0.80, 0.70],
        )

    def test_reports_every_audio_candidate_disposition_and_selected_contact(self) -> None:
        diagnostics = SwingDiagnostics(
            source=Path("user.mov"),
            candidates=(
                CandidateDiagnostic(
                    audio_candidate_index=0,
                    audio_timestamp=Fraction(1),
                    disposition="omitted",
                    reason="no person found",
                ),
                CandidateDiagnostic(
                    audio_candidate_index=1,
                    audio_timestamp=Fraction(2),
                    disposition="omitted",
                    reason="swing classifier: not a swing",
                ),
                CandidateDiagnostic(
                    audio_candidate_index=2,
                    audio_timestamp=Fraction(3),
                    disposition="accepted",
                    reason="visual contact accepted",
                    shot_type="forehand",
                    contact_frame_ordinal=181,
                    contact_frame=SourceFrameIdentity(0, 543, Fraction(1, 180)),
                    contact_confidence=0.72,
                    plausible_frames=(181, 182),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as directory_name:
            report = Path(directory_name) / "report.html"
            write_swing_diagnostics_report(report, diagnostics)
            document = report.read_text()

        self.assertIn("3 audio candidates", document)
        self.assertIn("Candidate 0", document)
        self.assertIn("no person found", document)
        self.assertIn("Candidate 1", document)
        self.assertIn("swing classifier: not a swing", document)
        self.assertIn("Candidate 2", document)
        self.assertIn("frame 181", document)
        self.assertIn("3.016667s", document)
        self.assertIn("confidence 0.720", document)
        self.assertIn("plausible frames 181, 182", document)


if __name__ == "__main__":
    unittest.main()

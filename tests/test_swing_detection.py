from dataclasses import FrozenInstanceError
from fractions import Fraction
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from tennis_cut.swing_detection import (
    DetectedSwing,
    DetectionConfig,
    detect_user_swings,
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


class DetectUserSwingsTests(unittest.TestCase):
    def test_returns_only_accepted_swings_in_source_order_with_exact_time(self) -> None:
        pop_detector = Mock()
        pop_detector.find_impacts.return_value = [2.125, 4.25, 6.5]

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
        pop_detector.find_impacts.return_value = [0.42500000000000004]

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


if __name__ == "__main__":
    unittest.main()

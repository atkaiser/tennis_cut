import argparse
import io
import json
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tennis_cut.swing_detection import DetectedSwing, LegacySwingDetails
from tennis_cut.tennis_cut import process_video


class LegacyCommandCharacterizationTests(unittest.TestCase):
    def test_preserves_outputs_metadata_and_processed_input_movement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incoming" / "user.mp4"
            source.parent.mkdir()
            source.write_bytes(b"source video")
            output = root / "processed"
            args = argparse.Namespace(
                output_dir=str(output),
                audio_model="audio.pth",
                shot_model="shot.pkl",
                shot_type_model="type.pkl",
                clips=False,
                slowmo=0.5,
                no_metadata=False,
                no_stitch=False,
                device="cpu",
            )
            details = LegacySwingDetails(
                swing=DetectedSwing(
                    ordinal=0,
                    contact_timestamp=Fraction(17, 40),
                    shot_type="forehand",
                ),
                legacy_contact=0.42500000000000004,
                start=-0.7749999999999999,
                end=1.125,
                crop=(10, 20, 100, 200),
            )

            def create_ffmpeg_output(command: list[str]) -> None:
                Path(command[-2]).touch()

            with (
                patch(
                    "tennis_cut.tennis_cut.probe",
                    return_value={
                        "fps": 120.0,
                        "resolution": (1920, 1080),
                        "audio_codec": "aac",
                    },
                ),
                patch(
                    "tennis_cut.tennis_cut.detect_user_swings_for_legacy",
                    return_value=(details,),
                ),
                patch(
                    "tennis_cut.tennis_cut.run_cmd",
                    side_effect=create_ffmpeg_output,
                ),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                return_code = process_video(source, args)

            self.assertEqual(return_code, 0)
            self.assertFalse(source.exists())
            self.assertEqual((output / "user.mp4").read_bytes(), b"source video")
            self.assertTrue((output / "user_forehand_slow0.5x.mp4").exists())
            self.assertIn("Processing video: user.mp4", stdout.getvalue())
            self.assertIn("Swing 0 extracted", stdout.getvalue())
            self.assertIn("Average extraction time per swing", stdout.getvalue())
            self.assertEqual(
                json.loads((output / "user_swings.json").read_text()),
                {
                    "video": "user.mp4",
                    "sample_rate": 48_000,
                    "swings": [
                        {
                            "index": 0,
                            "start": -0.7749999999999999,
                            "end": 1.125,
                            "contact": 0.42500000000000004,
                            "crop": [10, 20, 100, 200],
                            "label": "forehand",
                        }
                    ],
                },
            )


if __name__ == "__main__":
    unittest.main()

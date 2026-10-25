import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tennis_cut.tennis_cut import cut_swing


class SwingExtractionTests(unittest.TestCase):
    def test_legacy_cut_does_not_force_cadence_or_encoder_threads(self) -> None:
        with patch("tennis_cut.tennis_cut.run_cmd") as run_cmd:
            cut_swing(
                Path("source.mov"),
                0.5,
                1.0,
                Path("swing.mp4"),
                (0, 0, 160, 90),
                0.0625,
            )

        command = run_cmd.call_args.args[0]
        self.assertEqual(command[0], "ffmpeg")
        self.assertNotIn("-filter_threads", command)
        self.assertNotIn("-threads", command)
        self.assertNotIn("-threads:v", command)
        self.assertNotIn("-x264-params", command)
        self.assertNotIn("-r", command)
        self.assertNotIn("-fps_mode", command)
        video_filter = command[command.index("-filter:v") + 1]
        self.assertEqual(
            video_filter,
            "crop=160:90:0:0,setpts=16.000000*(PTS-STARTPTS)",
        )
        self.assertNotIn("fps=", video_filter)

    def test_high_frame_rate_slow_motion_preserves_source_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mov"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=160x90:rate=120:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
                    "-video_track_timescale", "2400", "-c:a", "aac",
                    str(source),
                ],
                check=True, capture_output=True,
            )
            output = root / "swing.mp4"
            cut_swing(source, 0.5, 1.0, output, (0, 0, 160, 90), 0.0625)
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_streams",
                    "-of", "json", str(output),
                ],
                check=True, capture_output=True, text=True,
            )
            streams = json.loads(result.stdout)["streams"]
            video = next(s for s in streams if s["codec_type"] == "video")
            audio = next(s for s in streams if s["codec_type"] == "audio")
            self.assertAlmostEqual(float(video["duration"]), 8.0, delta=0.2)
            self.assertAlmostEqual(float(audio["duration"]), 8.0, delta=0.25)
            self.assertLessEqual(int(video["nb_frames"]), 62)
            self.assertEqual(video["pix_fmt"], "yuv420p10le")


if __name__ == "__main__":
    unittest.main()

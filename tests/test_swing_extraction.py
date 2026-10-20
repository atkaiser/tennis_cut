import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tennis_cut.tennis_cut import cut_swing


class SwingExtractionTests(unittest.TestCase):
    def test_high_frame_rate_slow_motion_has_explicit_output_cadence(self) -> None:
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
            for speed in (None, 0.25, 0.0625):
                with self.subTest(speed=speed):
                    output = root / f"swing-{speed}.mp4"
                    cut_swing(source, 0.5, 1.0, output, (0, 0, 160, 90), speed)
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
                    expected = 0.5 / (speed or 1)
                    self.assertEqual(video["avg_frame_rate"], "60/1")
                    self.assertAlmostEqual(float(video["duration"]), expected, delta=0.05)
                    self.assertAlmostEqual(float(audio["duration"]), expected, delta=0.25)
                    self.assertEqual(int(video["nb_frames"]), round(expected * 60))
                    self.assertEqual(video["pix_fmt"], "yuv420p10le")


if __name__ == "__main__":
    unittest.main()

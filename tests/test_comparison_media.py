from fractions import Fraction
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from tennis_cut.comparison.media import inspect_media
from tennis_cut.comparison.pro_selection import DecodedFrame, InspectedMedia


class InspectMediaTests(unittest.TestCase):
    def test_exposes_ordered_frames_with_exact_stream_pts_and_time_base(self) -> None:
        ffprobe_output = {
            "streams": [{"index": 2, "time_base": "1/90000"}],
            "frames": [
                {"stream_index": 2, "pts": 9000},
                {"stream_index": 2, "pts": 12753},
                {"stream_index": 2, "pts": 16501},
            ],
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(ffprobe_output),
            stderr="",
        )

        with patch("subprocess.run", return_value=completed):
            result = inspect_media(Path("pro.mov"))

        self.assertEqual(
            result,
            InspectedMedia(
                frames=(
                    DecodedFrame(2, 0, 9000, Fraction(1, 90000)),
                    DecodedFrame(2, 1, 12753, Fraction(1, 90000)),
                    DecodedFrame(2, 2, 16501, Fraction(1, 90000)),
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()

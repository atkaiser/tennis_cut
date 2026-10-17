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
            "streams": [
                {
                    "index": 2,
                    "time_base": "1/90000",
                    "width": 1920,
                    "height": 1080,
                    "start_pts": 9000,
                    "duration_ts": 10000,
                    "nb_frames": "3",
                }
            ],
            "packets": [
                {"stream_index": 2, "pts": 8000, "flags": "K_"},
                {"stream_index": 2, "pts": 9000, "flags": "K_"},
                {"stream_index": 2, "pts": 16501, "flags": "__"},
                {"stream_index": 2, "pts": 12753, "flags": "__"},
            ],
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(ffprobe_output),
            stderr="",
        )

        with patch("subprocess.run", return_value=completed) as run:
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
        self.assertIn("-show_packets", run.call_args.args[0])

    def test_falls_back_when_packets_do_not_have_unique_presentation_times(self) -> None:
        stream = {
            "index": 0,
            "time_base": "1/100",
            "width": 320,
            "height": 180,
            "nb_frames": "2",
        }
        packet_result = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [stream],
                    "packets": [
                        {"stream_index": 0, "pts": 10},
                        {"stream_index": 0, "pts": 10},
                    ],
                }
            ),
            stderr="",
        )
        decoded_result = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [stream],
                    "frames": [
                        {"stream_index": 0, "pts": 10},
                        {"stream_index": 0, "pts": 11},
                    ],
                }
            ),
            stderr="",
        )

        with patch("subprocess.run", side_effect=(packet_result, decoded_result)) as run:
            result = inspect_media(Path("unusual.mov"))

        self.assertEqual(tuple(frame.pts for frame in result.frames), (10, 11))
        self.assertEqual(run.call_count, 2)
        self.assertIn("-show_frames", run.call_args_list[1].args[0])

    def test_falls_back_when_a_packet_lacks_a_presentation_timestamp(self) -> None:
        stream = {
            "index": 0,
            "time_base": "1/100",
            "width": 320,
            "height": 180,
            "nb_frames": "2",
        }
        packet_result = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [stream],
                    "packets": [
                        {"stream_index": 0, "pts": 10},
                        {"stream_index": 0},
                    ],
                }
            ),
            stderr="",
        )
        decoded_result = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [stream],
                    "frames": [
                        {"stream_index": 0, "pts": 10},
                        {"stream_index": 0, "pts": 11},
                    ],
                }
            ),
            stderr="",
        )

        with patch("subprocess.run", side_effect=(packet_result, decoded_result)) as run:
            result = inspect_media(Path("missing-pts.mov"))

        self.assertEqual(tuple(frame.pts for frame in result.frames), (10, 11))
        self.assertEqual(run.call_count, 2)

    def test_falls_back_when_packet_count_does_not_match_frame_count(self) -> None:
        stream = {
            "index": 0,
            "time_base": "1/100",
            "width": 320,
            "height": 180,
            "nb_frames": "2",
        }
        packet_result = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [stream],
                    "packets": [{"stream_index": 0, "pts": 10}],
                }
            ),
            stderr="",
        )
        decoded_result = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [stream],
                    "frames": [
                        {"stream_index": 0, "pts": 10},
                        {"stream_index": 0, "pts": 11},
                    ],
                }
            ),
            stderr="",
        )

        with patch("subprocess.run", side_effect=(packet_result, decoded_result)) as run:
            result = inspect_media(Path("multi-frame-packet.mov"))

        self.assertEqual(tuple(frame.pts for frame in result.frames), (10, 11))
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from label_videos.prototype_contact_review import _cache_directory, prepare_video


class ContactReviewPreparationTests(unittest.TestCase):
    def test_rebuilds_cache_when_requested_shots_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_entry = {
                "video": str(root / "rally.mp4"),
                "source_fingerprint": {"size": 10, "mtime_ns": 20},
                "shots": [
                    {"shot_index": 0, "time": 0.0, "type": "forehand"},
                    {"shot_index": 1, "time": 2.0, "type": "forehand"},
                ],
            }
            cache_root = root / "cache"
            cache_directory = _cache_directory(video_entry, cache_root)
            cache_directory.mkdir(parents=True)
            cached_frame = cache_directory / "cached.jpg"
            cached_frame.touch()
            (cache_directory / "frames.json").write_text(
                json.dumps(
                    {
                        "video": video_entry["video"],
                        "windows": [
                            {
                                **video_entry["shots"][0],
                                "center_ordinal": 0,
                                "ordinals": [0],
                            }
                        ],
                        "frames": [{"ordinal": 0, "path": str(cached_frame)}],
                    }
                )
            )
            media = SimpleNamespace(
                frames=[
                    SimpleNamespace(
                        stream_index=0,
                        ordinal=ordinal,
                        pts=ordinal,
                        time_base=Fraction(1, 1),
                        timestamp=Fraction(ordinal, 1),
                    )
                    for ordinal in range(3)
                ]
            )

            def write_decoded_frames(command: list[str], **_: object) -> None:
                pattern = Path(command[-2])
                for ordinal in range(3):
                    Path(str(pattern).replace("%06d", f"{ordinal:06d}")).touch()

            with (
                patch(
                    "label_videos.prototype_contact_review.inspect_media",
                    return_value=media,
                ),
                patch(
                    "label_videos.prototype_contact_review.subprocess.run",
                    side_effect=write_decoded_frames,
                ),
            ):
                prepared = prepare_video(video_entry, cache_root)

            self.assertEqual(len(prepared["windows"]), len(video_entry["shots"]))


if __name__ == "__main__":
    unittest.main()

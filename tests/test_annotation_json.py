import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from label_videos.tennis_annotate import annotation_payload, load_annotation_state
from tennis_cut.annotation_json import load_annotation, save_annotation


class AnnotationJsonTests(unittest.TestCase):
    def test_round_trip_preserves_unknown_document_and_shot_fields(self) -> None:
        original = {
            "video": "rally.MOV",
            "impacts": [1.25],
            "shots": [
                {
                    "time": 1.25,
                    "type": "forehand",
                    "contact_review": {
                        "status": "single_frame",
                        "frame": {"ordinal": 149},
                    },
                }
            ],
            "done": True,
            "contact_review_metadata": {"schema_version": 1},
            "future_field": {"keep": "me"},
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rally.json"
            path.write_text(json.dumps(original))

            loaded = load_annotation(path)
            save_annotation(path, loaded)

            self.assertEqual(json.loads(path.read_text()), original)

    def test_failed_replacement_leaves_previous_document_unchanged(self) -> None:
        original = {"video": "rally.MOV", "shots": []}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rally.json"
            path.write_text(json.dumps(original) + "\n")
            before = path.read_bytes()

            with (
                patch(
                    "tennis_cut.annotation_json.os.replace",
                    side_effect=OSError("simulated replacement failure"),
                ),
                self.assertRaisesRegex(OSError, "simulated replacement failure"),
            ):
                save_annotation(path, {"video": "changed.MOV", "shots": []})

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_annotator_state_preserves_extension_fields_when_core_fields_change(
        self,
    ) -> None:
        original = {
            "video": "rally.MOV",
            "impacts": [1.25],
            "shots": [
                {
                    "time": 1.25,
                    "type": "forehand",
                    "contact_review": {"status": "single_frame"},
                }
            ],
            "done": False,
            "contact_review_metadata": {"schema_version": 1},
        }

        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "rally.mp4"
            annotation_path = video_path.with_suffix(".json")
            annotation_path.write_text(json.dumps(original))

            state = load_annotation_state(video_path, fps=120.0)
            state.done = True
            state.shots[0].type = "backhand"
            result = annotation_payload(state)

            self.assertTrue(result["done"])
            self.assertEqual(result["shots"][0]["type"], "backhand")
            self.assertEqual(
                result["shots"][0]["contact_review"],
                {"status": "single_frame"},
            )
            self.assertEqual(
                result["contact_review_metadata"], {"schema_version": 1}
            )


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from tennis_cut.contact_review import load_review_dataset


def frame(ordinal: int) -> dict:
    return {
        "stream_index": 0,
        "ordinal": ordinal,
        "pts": ordinal * 128,
        "time_base": {"numerator": 1, "denominator": 15360},
    }


class ContactReviewDatasetTests(unittest.TestCase):
    def test_loads_visual_states_and_separates_primary_from_abstention_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            in_scope_path = root / "in-scope.json"
            out_of_scope_path = root / "out-of-scope.json"
            in_scope_path.write_text(
                json.dumps(
                    {
                        "contact_review_metadata": {
                            "schema_version": 2,
                            "source_fingerprint": {"size": 10, "mtime_ns": 20},
                            "side_view_suitability": "in_scope",
                        },
                        "shots": [
                            {
                                "time": 1.0,
                                "type": "forehand",
                                "contact_review": {
                                    "status": "single_state",
                                    "frame": frame(10),
                                    "visual_states": [
                                        {"frames": [frame(10), frame(11)]}
                                    ],
                                },
                            },
                            {
                                "time": 2.0,
                                "type": "forehand",
                                "contact_review": {
                                    "status": "between_state",
                                    "frame": frame(20),
                                    "visual_states": [
                                        {"frames": [frame(20)]},
                                        {"frames": [frame(21)]},
                                    ],
                                },
                            },
                            {
                                "time": 3.0,
                                "type": "forehand",
                                "contact_review": {"status": "unresolved"},
                            },
                        ],
                    }
                )
            )
            out_of_scope_path.write_text(
                json.dumps(
                    {
                        "contact_review_metadata": {
                            "schema_version": 2,
                            "source_fingerprint": {"size": 30, "mtime_ns": 40},
                            "side_view_suitability": "out_of_scope_view",
                        },
                        "shots": [
                            {
                                "time": 4.0,
                                "type": "forehand",
                                "contact_review": {
                                    "status": "single_state",
                                    "frame": frame(40),
                                    "visual_states": [{"frames": [frame(40)]}],
                                },
                            }
                        ],
                    }
                )
            )
            manifest = {
                "schema_version": 1,
                "videos": [
                    {
                        "video": str(root / "in-scope.mp4"),
                        "annotation": str(in_scope_path),
                        "family": "family-a",
                        "split": "test",
                        "source_fingerprint": {"size": 10, "mtime_ns": 20},
                        "shots": [
                            {"shot_index": index, "time": float(index + 1), "type": "forehand"}
                            for index in range(3)
                        ],
                    },
                    {
                        "video": str(root / "out-of-scope.mp4"),
                        "annotation": str(out_of_scope_path),
                        "family": "family-b",
                        "split": "out_of_scope",
                        "source_fingerprint": {"size": 30, "mtime_ns": 40},
                        "shots": [
                            {"shot_index": 0, "time": 4.0, "type": "forehand"}
                        ],
                    },
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            dataset = load_review_dataset(manifest_path)

            self.assertEqual(len(dataset.contacts), 4)
            self.assertEqual(dataset.invalid, ())
            self.assertEqual(dataset.contacts[0].acceptable_ordinals, (10, 11))
            self.assertEqual(dataset.contacts[1].acceptable_ordinals, (20, 21))
            self.assertEqual(
                [contact.status for contact in dataset.primary("test")],
                ["single_state", "between_state", "unresolved"],
            )
            self.assertEqual(
                [contact.status for contact in dataset.out_of_scope],
                ["single_state"],
            )

    def test_excludes_invalid_reviews_with_an_auditable_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path = root / "bad.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "contact_review_metadata": {
                            "schema_version": 2,
                            "source_fingerprint": {"size": 10, "mtime_ns": 20},
                            "side_view_suitability": "in_scope",
                        },
                        "shots": [
                            {
                                "time": 1.0,
                                "type": "forehand",
                                "contact_review": {
                                    "status": "single_state",
                                    "frame": frame(10),
                                    "visual_states": [],
                                },
                            }
                        ],
                    }
                )
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "video": str(root / "bad.mp4"),
                                "annotation": str(annotation_path),
                                "family": "family-a",
                                "split": "train",
                                "source_fingerprint": {"size": 10, "mtime_ns": 20},
                                "shots": [
                                    {
                                        "shot_index": 0,
                                        "time": 1.0,
                                        "type": "forehand",
                                    }
                                ],
                            }
                        ]
                    }
                )
            )

            dataset = load_review_dataset(manifest_path)

            self.assertEqual(dataset.contacts, ())
            self.assertEqual(len(dataset.invalid), 1)
            self.assertIn("single_state requires one visual state", dataset.invalid[0])


if __name__ == "__main__":
    unittest.main()

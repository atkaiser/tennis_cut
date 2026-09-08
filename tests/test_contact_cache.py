from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from tennis_cut.contact_cache import CACHE_FIELD, ContactDecisionCache
from tennis_cut.visual_contact import (
    ContactDecisionDiagnostics,
    ContactSelection,
    Detection,
    DeterministicSelection,
    FrameEvidence,
    RankedFrame,
    TemporalPrediction,
    VisualFrame,
)


class ContactDecisionCacheTests(unittest.TestCase):
    def test_round_trips_exact_frame_in_existing_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rally.mp4"
            source.write_bytes(b"video")
            annotation = source.with_suffix(".json")
            annotation.write_text(
                json.dumps(
                    {
                        "video": source.name,
                        "shots": [{"time": 1.0, "type": "forehand"}],
                        "future_field": {"keep": True},
                    }
                )
            )
            selection = ContactSelection(
                VisualFrame(
                    FrameEvidence(
                        ordinal=120,
                        timestamp=Fraction(2),
                        detections=(),
                        stream_index=0,
                        pts=30720,
                        time_base=Fraction(1, 15360),
                    )
                ),
                0.8,
                (120,),
                None,
                diagnostics=ContactDecisionDiagnostics(
                    frames=(
                        VisualFrame(
                            FrameEvidence(
                                ordinal=120,
                                timestamp=Fraction(2),
                                detections=(
                                    Detection("racket", (1.0, 2.0, 3.0, 4.0), 0.7),
                                ),
                                stream_index=0,
                                pts=30720,
                                time_base=Fraction(1, 15360),
                            )
                        ),
                    ),
                    deterministic=DeterministicSelection(
                        selected_frame=120,
                        plausible_frames=(120,),
                        ranked_frames=(RankedFrame(120, 0.8, 0.7, 0.2, 0.1),),
                        racket_frames=(120,),
                        omission_reason=None,
                    ),
                    temporal_prediction=TemporalPrediction(120, 0.8),
                    confidence_threshold=0.2,
                ),
            )

            cache = ContactDecisionCache()
            cache.store(source, Fraction(2), Fraction(2, 5), "model-a", selection)
            loaded = cache.load(
                source, Fraction(2), Fraction(2, 5), "model-a"
            )

            self.assertEqual(loaded.selected_frame, 120)
            self.assertEqual(loaded.frame.evidence.pts, 30720)
            self.assertEqual(loaded.diagnostics, selection.diagnostics)
            payload = json.loads(annotation.read_text())
            self.assertEqual(payload["future_field"], {"keep": True})
            self.assertEqual(payload["shots"], [{"time": 1.0, "type": "forehand"}])
            self.assertIn(CACHE_FIELD, payload)

    def test_source_or_model_change_invalidates_a_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rally.mp4"
            source.write_bytes(b"first")
            cache = ContactDecisionCache()
            omitted = ContactSelection(None, 0.0, (), "unlocalizable")
            cache.store(source, Fraction(1), Fraction(2, 5), "model-a", omitted)

            self.assertIsNone(
                cache.load(source, Fraction(1), Fraction(2, 5), "model-b")
            )
            source.write_bytes(b"changed-size")
            self.assertIsNone(
                cache.load(source, Fraction(1), Fraction(2, 5), "model-a")
            )

    def test_uses_a_nonconflicting_sidecar_without_an_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pro.mp4"
            source.write_bytes(b"video")

            ContactDecisionCache().store(
                source,
                Fraction(5),
                Fraction(3),
                "model-a",
                ContactSelection(None, 0.0, (), "unsupported_view"),
            )

            self.assertFalse(source.with_suffix(".json").exists())
            self.assertTrue(
                source.with_suffix(".mp4.contact.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()

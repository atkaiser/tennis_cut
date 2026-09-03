from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tennis_cut.visual_contact import (
    ContactSelection,
    Detection,
    FEATURE_VERSION,
    FrameEvidence,
    ObjectKind,
    StockVisualContactSelector,
    _StockVisualEvidence,
    TemporalPrediction,
    VisualFrame,
    decode_exact_frame_images,
    rank_contact_frames,
    select_contact_frame,
)
from tennis_cut.comparison.pro_selection import DecodedFrame, InspectedMedia


def detection(
    kind: ObjectKind,
    box: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> Detection:
    return Detection(kind, box, confidence)


def frame(ordinal: int, detections: tuple[Detection, ...]) -> VisualFrame:
    return VisualFrame(
        FrameEvidence(
            ordinal=ordinal,
            timestamp=Fraction(ordinal, 120),
            detections=detections,
        )
    )


class VisualContactSelectionTests(unittest.TestCase):
    def test_uses_largest_person_and_rejects_other_player_racket(self) -> None:
        frames = tuple(
            frame(
                ordinal,
                (
                    detection("person", (0, 0, 100, 300)),
                    detection("person", (500, 0, 570, 180)),
                    *(
                        ()
                        if ordinal == 1
                        else (detection("racket", (35, 140, 90, 230)),)
                    ),
                    detection("racket", (515, 130, 640, 230)),
                    detection(
                        "ball",
                        (
                            (10, 30, 65, 100, 130)[ordinal],
                            150,
                            (20, 40, 75, 110, 140)[ordinal],
                            160,
                        ),
                    ),
                ),
            )
            for ordinal in range(5)
        )

        result = rank_contact_frames(frames)

        self.assertIsNotNone(result.selected_frame)
        self.assertIn(result.selected_frame, range(5))
        self.assertIn(0, result.racket_frames)
        self.assertIsNone(result.omission_reason)

    def test_rejects_stationary_ball_and_reports_missing_moving_ball(self) -> None:
        frames = tuple(
            frame(
                ordinal,
                (
                    detection("person", (0, 0, 100, 300)),
                    detection("racket", (35, 140, 90, 230)),
                    detection("ball", (70, 155, 80, 165)),
                ),
            )
            for ordinal in range(5)
        )

        result = rank_contact_frames(frames)

        self.assertIsNone(result.selected_frame)
        self.assertEqual(result.omission_reason, "no moving ball evidence")

    def test_tie_resolves_to_earlier_existing_frame_and_ranker_is_replaceable(self) -> None:
        frames = tuple(
            frame(
                ordinal,
                (
                    detection("person", (0, 0, 100, 300)),
                    detection("racket", (35, 140, 90, 230)),
                    detection("ball", (72 + abs(2 - ordinal) * 8, 155, 82 + abs(2 - ordinal) * 8, 165)),
                ),
            )
            for ordinal in range(5)
        )
        ranking = rank_contact_frames(frames)

        class Ranker:
            feature_version = ranking.feature_version

            def predict(self, features):
                return TemporalPrediction(frame_ordinal=1, confidence=0.8)

        result = select_contact_frame(frames, Ranker())

        self.assertEqual(ranking.selected_frame, 2)
        self.assertEqual(result.selected_frame, 2)
        self.assertEqual(result.frame.timestamp, Fraction(2, 120))
        self.assertEqual(result.contact_confidence, 0.8)

    def test_omits_temporal_prediction_below_ranker_confidence_threshold(self) -> None:
        frames = tuple(
            frame(
                ordinal,
                (
                    detection("person", (0, 0, 100, 300)),
                    detection("racket", (35, 140, 90, 230)),
                    detection("ball", (70 + ordinal * 8, 155, 80 + ordinal * 8, 165)),
                ),
            )
            for ordinal in range(5)
        )

        class LowConfidenceRanker:
            feature_version = FEATURE_VERSION
            confidence_threshold = 0.5

            def predict(self, features):
                return TemporalPrediction(features[2].frame_ordinal, 0.1)

        result = select_contact_frame(frames, LowConfidenceRanker())

        self.assertIsNone(result.frame)
        self.assertEqual(result.omission_reason, "below contact confidence threshold")

    def test_wide_search_uses_local_windows_and_selects_highest_confidence(self) -> None:
        class RecordingProvider:
            calls = []

            def frames_many(self, source, candidate_timestamps, radius):
                self.calls.append((source, candidate_timestamps, radius))
                return tuple((frame(index, ()),) for index in range(8))

        provider = RecordingProvider()
        rejected = ContactSelection(None, 0.0, (), "no moving ball evidence")
        lower = ContactSelection(frame(100, ()), 0.7, (100,), None)
        higher = ContactSelection(frame(200, ()), 0.9, (200,), None)
        selections = (rejected, lower, rejected, higher, rejected, rejected, rejected, rejected)
        selector = StockVisualContactSelector(evidence_provider=provider)

        with patch(
            "tennis_cut.visual_contact.select_contact_frame",
            side_effect=selections,
        ):
            result = selector.select(
                Path("pro.mov"),
                Fraction(10),
                radius=Fraction(3),
            )

        self.assertEqual(result, higher)
        self.assertEqual(
            provider.calls,
            [
                (
                    Path("pro.mov"),
                    (
                        Fraction(37, 5),
                        Fraction(41, 5),
                        Fraction(9),
                        Fraction(49, 5),
                        Fraction(53, 5),
                        Fraction(57, 5),
                        Fraction(61, 5),
                        Fraction(63, 5),
                    ),
                    Fraction(2, 5),
                )
            ],
        )


class ExactVisualFrameDecodingTests(unittest.TestCase):
    def test_reuses_preinspected_timeline_without_running_ffprobe(self) -> None:
        timeline = InspectedMedia(
            (
                DecodedFrame(0, 0, 100, Fraction(1, 100)),
                DecodedFrame(0, 1, 110, Fraction(1, 100)),
            )
        )
        provider = _StockVisualEvidence(device="cpu", frame_timeline=timeline)

        with patch("tennis_cut.visual_contact.subprocess.run") as run:
            first = provider._frame_timestamps(Path("user.mov"))
            second = provider._frame_timestamps(Path("user.mov"))

        run.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            (
                (0, 0, 100, Fraction(1, 100)),
                (1, 0, 110, Fraction(1, 100)),
            ),
        )

    def test_decodes_requested_ffmpeg_ordinals_without_frame_seeking(self) -> None:
        colors = ((20, 40, 60), (70, 90, 110), (120, 140, 160), (170, 190, 210))
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for ordinal, color in enumerate(colors):
                Image.new("RGB", (64, 36), color).save(
                    directory / f"frame_{ordinal:02d}.png"
                )
            video = directory / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-framerate",
                    "5",
                    "-start_number",
                    "0",
                    "-i",
                    str(directory / "frame_%02d.png"),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "0",
                    "-pix_fmt",
                    "yuv444p",
                    str(video),
                    "-y",
                ],
                check=True,
            )

            decoded = decode_exact_frame_images(video, (3, 1, 3))

        self.assertEqual(tuple(decoded), (1, 3))
        for ordinal in (1, 3):
            # OpenCV images are BGR; lossless H.264 can still differ by one level.
            actual = tuple(float(decoded[ordinal][:, :, channel].mean()) for channel in range(3))
            expected = tuple(reversed(colors[ordinal]))
            self.assertTrue(
                all(abs(channel - target) <= 2 for channel, target in zip(actual, expected))
            )

    def test_evidence_images_share_the_ffprobe_ordinals_attached_to_them(self) -> None:
        provider = _StockVisualEvidence(device="cpu")
        metadata = (
            (10, 0, 100, Fraction(1, 100)),
            (11, 0, 110, Fraction(1, 100)),
            (12, 0, 120, Fraction(1, 100)),
        )
        exact_images = {10: object(), 11: object(), 12: object()}

        class EmptyValues:
            def cpu(self):
                return self

            def tolist(self):
                return []

        class EmptyBoxes:
            xyxy = EmptyValues()
            cls = EmptyValues()
            conf = EmptyValues()

        class Result:
            boxes = EmptyBoxes()

        class Model:
            observed_images = None

            def predict(self, images, **_options):
                self.observed_images = images
                return [Result() for _ in images]

        model = Model()
        with (
            patch.object(provider, "_frame_timestamps", return_value=metadata),
            patch.object(provider, "_load_model", return_value=model),
            patch(
                "tennis_cut.visual_contact._materialize_exact_frame_paths",
                return_value={ordinal: Path(str(ordinal)) for ordinal in exact_images},
            ) as materialize,
            patch(
                "cv2.imread",
                side_effect=lambda path, _mode: exact_images[int(Path(path).name)],
            ),
        ):
            frames = provider.frames(
                Path("variable-frame-rate.mov"),
                Fraction(11, 10),
                radius=Fraction(1, 10),
            )

        self.assertEqual(materialize.call_count, 1)
        self.assertEqual(
            materialize.call_args.args[:2],
            (Path("variable-frame-rate.mov"), (10, 11, 12)),
        )
        self.assertEqual(model.observed_images, [exact_images[index] for index in (10, 11, 12)])
        self.assertEqual([frame.ordinal for frame in frames], [10, 11, 12])

    def test_batches_candidate_windows_into_one_ffmpeg_decode(self) -> None:
        class EmptyValues:
            def cpu(self):
                return self

            def tolist(self):
                return []

        class EmptyBoxes:
            xyxy = EmptyValues()
            cls = EmptyValues()
            conf = EmptyValues()

        class Result:
            boxes = EmptyBoxes()

        class Model:
            def predict(self, images, **_options):
                return [Result() for _ in images]

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            for ordinal in range(4):
                Image.new("RGB", (64, 36), (ordinal * 40, 20, 10)).save(
                    directory / f"frame_{ordinal:02d}.png"
                )
            video = directory / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-framerate",
                    "5",
                    "-start_number",
                    "0",
                    "-i",
                    str(directory / "frame_%02d.png"),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "0",
                    "-pix_fmt",
                    "yuv444p",
                    str(video),
                    "-y",
                ],
                check=True,
            )
            timeline = InspectedMedia(
                tuple(
                    DecodedFrame(0, ordinal, ordinal, Fraction(1, 5))
                    for ordinal in range(4)
                )
            )
            provider = _StockVisualEvidence(device="cpu", frame_timeline=timeline)

            with (
                patch.object(provider, "_load_model", return_value=Model()),
                patch(
                    "tennis_cut.visual_contact.subprocess.run",
                    wraps=subprocess.run,
                ) as run,
            ):
                windows = provider.frames_many(
                    video,
                    (Fraction(1, 5), Fraction(3, 5)),
                    radius=Fraction(1, 5),
                )

        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            tuple(tuple(item.ordinal for item in window) for window in windows),
            ((0, 1, 2), (2, 3)),
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from tennis_cut.comparison import ComparisonRequest, ComparisonResult, compare_videos
from tennis_cut.comparison.planning import ComparisonSource
from tennis_cut.comparison.planning import PlayerObservation, Rectangle
from tennis_cut.comparison.pro_selection import (
    DecodedFrame,
    InspectedMedia,
    ProSelection,
    SelectionCancelled,
)
from tennis_cut.comparison.workflow import (
    ComparisonProcessingFailed,
    ComparisonSelectionCancelled,
    OutputCollision,
)


def comparison_source(path: Path) -> ComparisonSource:
    frames = tuple(
        DecodedFrame(0, ordinal, ordinal, Fraction(1, 10))
        for ordinal in range(31)
    )
    return ComparisonSource(path, 1920, 1080, InspectedMedia(frames))


class ZeroComparisonDependencies:
    def __init__(self, user_video: Path, pro_video: Path) -> None:
        self.user_source = comparison_source(user_video)
        self.pro_source = comparison_source(pro_video)
        self.events: list[str] = []

    def executable_exists(self, name: str) -> bool:
        return True

    def inspect_source(self, path: Path) -> ComparisonSource:
        self.events.append(f"inspect:{path.name}")
        return self.user_source if path == self.user_source.path else self.pro_source

    def user_has_audio(self, path: Path) -> bool:
        return True

    def resolve_selection(
        self,
        pro_video: Path,
        pro_speed: Fraction,
        inspected_media: InspectedMedia,
    ) -> ProSelection:
        self.events.append("select")
        return ProSelection(pro_video, inspected_media.frames[15], "forehand")

    def detect_swings(self, request: ComparisonRequest):
        self.events.append("detect")
        return ()

    def create_player_locator(self, device: str | None):
        raise AssertionError("player locator must stay lazy when nothing is emitted")

    def observe_players(self, window, locator):
        raise AssertionError("zero comparisons must not observe players")

    def render_artifacts(self, plans, primary, clips):
        raise AssertionError("zero comparisons must not render artifacts")


class MatchingDependencies(ZeroComparisonDependencies):
    def __init__(self, user_video: Path, pro_video: Path) -> None:
        super().__init__(user_video, pro_video)
        self.observed_ordinals: list[int | None] = []
        self.rendered_ordinals: list[int | None] = []
        self.locator_creations = 0

    def detect_swings(self, request: ComparisonRequest):
        from tennis_cut.swing_detection import DetectedSwing

        self.events.append("detect")
        return (
            DetectedSwing(4, Fraction(3, 2), "forehand"),
            DetectedSwing(5, Fraction(2), "backhand"),
            DetectedSwing(9, Fraction(3, 2), "forehand"),
        )

    def create_player_locator(self, device: str | None):
        self.locator_creations += 1
        return object()

    def observe_players(self, window, locator):
        self.observed_ordinals.append(window.swing_ordinal)
        return (PlayerObservation(0, Rectangle(400, 100, 160, 360)),)

    def render_artifacts(self, plans, primary, clips):
        self.rendered_ordinals = [plan.user.window.swing_ordinal for plan in plans]
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_bytes(b"silent compilation")
        for clip in clips:
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"clip")


class FailingRenderDependencies(MatchingDependencies):
    def render_artifacts(self, plans, primary, clips):
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_bytes(b"partial")
        raise OSError("encoder stopped")


class CancelledSelectionDependencies(ZeroComparisonDependencies):
    def resolve_selection(self, pro_video, pro_speed, inspected_media):
        self.events.append("select")
        return SelectionCancelled()


class CompareVideosTests(unittest.TestCase):
    def test_zero_comparisons_succeeds_without_output_and_selects_before_detection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.write_bytes(b"user source")
            pro_video.write_bytes(b"pro source")
            audio_model = directory / "audio.pth"
            shot_model = directory / "shot.pkl"
            shot_type_model = directory / "type.pkl"
            for model in (audio_model, shot_model, shot_type_model):
                model.touch()
            output_directory = directory / "not-created"
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                output_directory=output_directory,
                audio_model=audio_model,
                shot_model=shot_model,
                shot_type_model=shot_type_model,
            )
            dependencies = ZeroComparisonDependencies(user_video, pro_video)

            result = compare_videos(request, dependencies)

            self.assertEqual(result, ComparisonResult((), 0))
            self.assertEqual(
                dependencies.events,
                ["inspect:user.mov", "inspect:pro.mov", "select", "detect"],
            )
            self.assertFalse(output_directory.exists())
            self.assertEqual(user_video.read_bytes(), b"user source")
            self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_matching_swings_reuse_pro_work_and_publish_in_accepted_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "lesson.mov"
            pro_video = directory / "pro.mov"
            user_video.write_bytes(b"user source")
            pro_video.write_bytes(b"pro source")
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                slow_motion=Fraction(1, 2),
                output_directory=directory / "outputs",
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )
            dependencies = MatchingDependencies(user_video, pro_video)

            result = compare_videos(request, dependencies)

            primary = request.output_directory / "lesson_vs_pro_slow0.5x.mp4"
            self.assertEqual(result, ComparisonResult((primary,), 2))
            self.assertEqual(primary.read_bytes(), b"silent compilation")
            self.assertEqual(dependencies.locator_creations, 1)
            self.assertEqual(dependencies.observed_ordinals, [None, 4, 9])
            self.assertEqual(dependencies.rendered_ordinals, [4, 9])
            self.assertEqual(user_video.read_bytes(), b"user source")
            self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_requested_clips_are_numbered_without_gaps_and_published_together(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.touch()
            pro_video.touch()
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                slow_motion=Fraction(1),
                output_directory=directory / "outputs",
                clips=True,
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )

            result = compare_videos(
                request, MatchingDependencies(user_video, pro_video)
            )

            primary = request.output_directory / "user_vs_pro_slow1x.mp4"
            clips_directory = request.output_directory / "user_vs_pro_slow1x_clips"
            self.assertEqual(
                result.published_paths,
                (
                    primary,
                    clips_directory / "comparison_001.mp4",
                    clips_directory / "comparison_002.mp4",
                ),
            )
            self.assertEqual(
                tuple(sorted(path.name for path in clips_directory.iterdir())),
                ("comparison_001.mp4", "comparison_002.mp4"),
            )

    def test_render_failure_leaves_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.write_bytes(b"user")
            pro_video.write_bytes(b"pro")
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            output_directory = directory / "outputs"
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                output_directory=output_directory,
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )

            with self.assertRaisesRegex(
                ComparisonProcessingFailed, "comparison rendering.*encoder stopped"
            ):
                compare_videos(
                    request, FailingRenderDependencies(user_video, pro_video)
                )

            self.assertFalse(output_directory.exists())
            self.assertEqual(user_video.read_bytes(), b"user")
            self.assertEqual(pro_video.read_bytes(), b"pro")

    def test_cancellation_stops_before_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.touch()
            pro_video.touch()
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )
            dependencies = CancelledSelectionDependencies(user_video, pro_video)

            with self.assertRaises(ComparisonSelectionCancelled):
                compare_videos(request, dependencies)

            self.assertNotIn("detect", dependencies.events)

    def test_preflight_reports_primary_and_clip_directory_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.touch()
            pro_video.touch()
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            output_directory = directory / "outputs"
            output_directory.mkdir()
            primary = output_directory / "user_vs_pro_slow0.0625x.mp4"
            clips_directory = output_directory / "user_vs_pro_slow0.0625x_clips"
            primary.touch()
            clips_directory.mkdir()
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                output_directory=output_directory,
                clips=True,
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )

            with self.assertRaises(OutputCollision) as raised:
                compare_videos(
                    request, ZeroComparisonDependencies(user_video, pro_video)
                )

            self.assertEqual(raised.exception.paths, (primary, clips_directory))


if __name__ == "__main__":
    unittest.main()

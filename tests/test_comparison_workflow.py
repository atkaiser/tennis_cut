from __future__ import annotations

from fractions import Fraction
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tennis_cut.comparison.workflow as comparison_workflow
from tennis_cut.comparison import ComparisonRequest, ComparisonResult, compare_videos
from tennis_cut.comparison.planning import (
    ComparisonRenderPlan,
    ComparisonSource,
    PlayerObservation,
    Rectangle,
    SelectedSourceWindow,
)
from tennis_cut.comparison.pro_selection import (
    DecodedFrame,
    InspectedMedia,
    ProSelection,
    SelectionProcessingFailure,
)
from tennis_cut.comparison.workflow import (
    ComparisonProcessingFailed,
    InvalidComparisonRequest,
    OutputCollision,
    SystemComparisonDependencies,
)
from tennis_cut.swing_detection import DetectedSwing
from tennis_cut.temporal_ranker import TEMPORAL_VECTOR_SIZE, TemporalRankerArtifact


def comparison_source(path: Path) -> ComparisonSource:
    frames = tuple(
        DecodedFrame(0, ordinal, ordinal, Fraction(1, 10))
        for ordinal in range(31)
    )
    return ComparisonSource(path, 1920, 1080, InspectedMedia(frames))


def comparison_files(
    directory: Path, *, user_name: str = "user.mov"
) -> tuple[Path, Path, tuple[Path, Path, Path]]:
    user_video = directory / user_name
    pro_video = directory / "pro.mov"
    user_video.write_bytes(b"user source")
    pro_video.write_bytes(b"pro source")
    models = tuple(directory / name for name in ("audio", "shot", "type"))
    for model in models:
        model.touch()
    return user_video, pro_video, models


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
        detection_config: object,
    ) -> ProSelection:
        self.events.append("select")
        return ProSelection(pro_video, inspected_media.frames[15], "forehand")

    def detect_swings(
        self, request: ComparisonRequest, user_source: ComparisonSource
    ) -> tuple[DetectedSwing, ...]:
        self.events.append("detect")
        return ()

    def create_player_locator(self, device: str | None) -> object:
        raise AssertionError("player locator must stay lazy when nothing is emitted")

    def observe_players(
        self, window: SelectedSourceWindow, locator: object
    ) -> tuple[PlayerObservation, ...]:
        raise AssertionError("zero comparisons must not observe players")

    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        raise AssertionError("zero comparisons must not render artifacts")


class SystemDependencyProgressTests(unittest.TestCase):
    def test_pro_contact_uses_configured_visual_finder(self) -> None:
        dependencies = SystemComparisonDependencies()
        source = comparison_source(Path("pro.mov"))
        request = ComparisonRequest(
            user_video=Path("user.mov"),
            pro_video=source.path,
            pro_speed=Fraction(1),
            device="cpu",
        )
        ranker = object()
        finder = object()
        expected = ProSelection(source.path, source.inspected_media.frames[15])

        with (
            patch(
                "tennis_cut.comparison.workflow.load_temporal_ranker",
                return_value=ranker,
            ) as load_ranker,
            patch(
                "tennis_cut.visual_contact.StockVisualContactSelector",
                return_value=finder,
            ) as finder_type,
            patch(
                "tennis_cut.comparison.pro_selection.find_pro_contact",
                return_value=expected,
            ) as find_contact,
        ):
            result = dependencies.resolve_selection(
                source.path,
                request.pro_speed,
                source.inspected_media,
                request.detection_config,
            )

        self.assertEqual(result, expected)
        load_ranker.assert_called_once_with(request.temporal_ranker_model)
        finder_type.assert_called_once_with(
            device="cpu",
            ranker=ranker,
            frame_timeline=source.inspected_media,
        )
        find_contact.assert_called_once_with(
            pro_video=source.path,
            pro_speed=Fraction(1),
            inspected_media=source.inspected_media,
            finder=finder,
        )

    def test_video_inspection_logs_start_and_completion(self) -> None:
        dependencies = SystemComparisonDependencies()
        source = comparison_source(Path("user.mov"))

        with (
            patch(
                "tennis_cut.comparison.media.inspect_comparison_source",
                return_value=source,
            ),
            self.assertLogs("tennis_cut.comparison.workflow", level="INFO") as logs,
        ):
            result = dependencies.inspect_source(source.path)

        self.assertEqual(result, source)
        self.assertTrue(
            any("Inspecting video metadata: user.mov" in entry for entry in logs.output)
        )
        self.assertTrue(
            any(
                "Finished video metadata: user.mov (31 frames, 1920x1080)" in entry
                for entry in logs.output
            )
        )

    def test_swing_detection_receives_the_inspected_user_timeline(self) -> None:
        dependencies = SystemComparisonDependencies()
        source = comparison_source(Path("user.mov"))
        request = ComparisonRequest(
            user_video=source.path,
            pro_video=Path("pro.mov"),
            pro_speed=Fraction(1),
        )

        with patch(
            "tennis_cut.comparison.workflow.detect_comparison_user_swings",
            return_value=(),
        ) as detect:
            result = dependencies.detect_swings(request, source)

        self.assertEqual(result, ())
        self.assertIs(
            detect.call_args.kwargs["frame_timeline"], source.inspected_media
        )


class MatchingDependencies(ZeroComparisonDependencies):
    def __init__(self, user_video: Path, pro_video: Path) -> None:
        super().__init__(user_video, pro_video)
        self.observed_ordinals: list[int | None] = []
        self.rendered_ordinals: list[int | None] = []
        self.locator_creations = 0

    def detect_swings(
        self, request: ComparisonRequest, user_source: ComparisonSource
    ) -> tuple[DetectedSwing, ...]:
        self.events.append("detect")
        return (
            DetectedSwing(4, Fraction(3, 2), "forehand"),
            DetectedSwing(5, Fraction(2), "backhand"),
            DetectedSwing(6, Fraction(2), "overhead"),
            DetectedSwing(7, Fraction(1, 2), "forehand"),
            DetectedSwing(9, Fraction(3, 2), "forehand"),
        )

    def create_player_locator(self, device: str | None) -> object:
        self.locator_creations += 1
        return object()

    def observe_players(
        self, window: SelectedSourceWindow, locator: object
    ) -> tuple[PlayerObservation, ...]:
        self.observed_ordinals.append(window.swing_ordinal)
        return (PlayerObservation(0, Rectangle(400, 100, 160, 360)),)

    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        self.rendered_ordinals = [plan.user.window.swing_ordinal for plan in plans]
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_bytes(b"silent compilation")
        for clip in clips:
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"clip")


class FailingRenderDependencies(MatchingDependencies):
    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_bytes(b"partial")
        raise OSError("encoder stopped")


class FailedSelectionDependencies(ZeroComparisonDependencies):
    def resolve_selection(
        self,
        pro_video: Path,
        pro_speed: Fraction,
        inspected_media: InspectedMedia,
        detection_config: object,
    ) -> SelectionProcessingFailure:
        return SelectionProcessingFailure(
            "pro contact detection", "visual contact unavailable"
        )


class FailedDetectionDependencies(ZeroComparisonDependencies):
    def resolve_selection(
        self,
        pro_video: Path,
        pro_speed: Fraction,
        inspected_media: InspectedMedia,
        detection_config: object,
    ) -> ProSelection:
        return ProSelection(pro_video, inspected_media.frames[15], "forehand")

    def detect_swings(
        self, request: ComparisonRequest, user_source: ComparisonSource
    ) -> tuple[DetectedSwing, ...]:
        raise OSError("model could not load")


class FailedInspectionDependencies(ZeroComparisonDependencies):
    def inspect_source(self, path: Path) -> ComparisonSource:
        raise OSError("probe failed")


class FailedAudioInspectionDependencies(ZeroComparisonDependencies):
    def user_has_audio(self, path: Path) -> bool:
        raise OSError("audio probe failed")


class MissingAudioDependencies(ZeroComparisonDependencies):
    def user_has_audio(self, path: Path) -> bool:
        return False


class FailedLocatorDependencies(MatchingDependencies):
    def create_player_locator(self, device: str | None) -> object:
        raise OSError("locator failed")


class FailedUserObservationDependencies(MatchingDependencies):
    def observe_players(
        self, window: SelectedSourceWindow, locator: object
    ) -> tuple[PlayerObservation, ...]:
        if window.swing_ordinal is not None:
            raise OSError("user observation failed")
        return super().observe_players(window, locator)


class DirectRenderDependencies(MatchingDependencies):
    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        primary.write_bytes(b"silent compilation")


class CompareVideosTests(unittest.TestCase):
    def test_diagnostics_only_writes_report_and_skips_player_work_and_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
            report = directory / "diagnostics.html"

            class DiagnosticDependencies(MatchingDependencies):
                def write_swing_diagnostics(
                    self,
                    windows: tuple[SelectedSourceWindow, ...],
                    pro_shot_type: str,
                    diagnostic_report: Path | None,
                ) -> None:
                    self.events.append(
                        f"diagnose:{pro_shot_type}:{[window.swing_ordinal for window in windows]}"
                    )
                    assert diagnostic_report is not None
                    diagnostic_report.write_text("diagnostics")

            dependencies = DiagnosticDependencies(user_video, pro_video)
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
                diagnostic_report=report,
                diagnostics_only=True,
            )

            result = compare_videos(request, dependencies)

            self.assertEqual(result, ComparisonResult((report,), 2))
            self.assertEqual(report.read_text(), "diagnostics")
            self.assertEqual(dependencies.locator_creations, 0)
            self.assertEqual(dependencies.rendered_ordinals, [])
            self.assertIn("diagnose:forehand:[4, 9]", dependencies.events)

    def test_unsupported_pro_shot_type_fails_before_user_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)

            class UnsupportedSelectionDependencies(ZeroComparisonDependencies):
                def resolve_selection(
                    self,
                    pro_video: Path,
                    pro_speed: Fraction,
                    inspected_media: InspectedMedia,
                    detection_config: object,
                ) -> ProSelection:
                    self.events.append("select")
                    return ProSelection(pro_video, inspected_media.frames[15], "backhand")

                def detect_swings(
                    self,
                    request: ComparisonRequest,
                    user_source: ComparisonSource,
                ) -> tuple[DetectedSwing, ...]:
                    raise AssertionError("unsupported pro shot must stop before detection")

            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )

            with self.assertRaisesRegex(InvalidComparisonRequest, "unsupported pro shot type"):
                compare_videos(request, UnsupportedSelectionDependencies(user_video, pro_video))

    def test_invalid_temporal_ranker_is_rejected_before_video_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
            ranker_path = directory / "ranker.json"
            ranker_path.write_text("not json")
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
                temporal_ranker_model=ranker_path,
            )
            dependencies = ZeroComparisonDependencies(user_video, pro_video)

            with self.assertRaisesRegex(InvalidComparisonRequest, "invalid temporal ranker artifact"):
                compare_videos(request, dependencies)
            self.assertEqual(dependencies.events, [])

    def test_valid_temporal_ranker_path_reaches_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
            ranker_path = directory / "ranker.json"
            TemporalRankerArtifact((0.0,) * TEMPORAL_VECTOR_SIZE, 0.0, "forehand").save(ranker_path)
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
                temporal_ranker_model=ranker_path,
            )
            dependencies = ZeroComparisonDependencies(user_video, pro_video)

            result = compare_videos(request, dependencies)

            self.assertEqual(result, ComparisonResult((), 0))
            self.assertEqual(dependencies.events, ["inspect:user.mov", "inspect:pro.mov", "select", "detect"])

    def test_automatic_device_selection_prefers_mps_then_cuda_then_cpu(self) -> None:
        cases = (
            (None, True, True, "mps"),
            (None, False, True, "cuda"),
            (None, False, False, "cpu"),
            ("cpu", True, True, "cpu"),
        )
        for requested, mps_available, cuda_available, expected in cases:
            with (
                self.subTest(requested=requested, expected=expected),
                patch(
                    "torch.backends.mps.is_available", return_value=mps_available
                ),
                patch("torch.cuda.is_available", return_value=cuda_available),
                patch("utilities.PersonDetector", side_effect=lambda device: device),
            ):
                selected = SystemComparisonDependencies().create_player_locator(
                    requested
                )

            self.assertEqual(selected, expected)

    def test_preflight_rejects_two_paths_to_the_same_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, _, models = comparison_files(directory)
            pro_video = directory / "pro-hardlink.mov"
            pro_video.hardlink_to(user_video)
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )

            with self.assertRaisesRegex(
                InvalidComparisonRequest,
                "user and pro videos must be distinct files",
            ):
                compare_videos(
                    request, ZeroComparisonDependencies(user_video, pro_video)
                )

    def test_preflight_rejects_destination_without_search_permission(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                output_directory=directory / "outputs",
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )

            with (
                patch.object(
                    comparison_workflow.os,
                    "access",
                    side_effect=lambda path, mode: mode == os.W_OK,
                ),
                self.assertRaisesRegex(
                    InvalidComparisonRequest,
                    "output destination is not writable",
                ),
            ):
                compare_videos(
                    request, ZeroComparisonDependencies(user_video, pro_video)
                )

    def test_zero_comparisons_succeeds_without_output_and_selects_before_detection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
            output_directory = directory / "not-created"
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                output_directory=output_directory,
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
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
            user_video, pro_video, models = comparison_files(
                directory, user_name="lesson.mov"
            )
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

    def test_requested_comparison_clips_are_numbered_and_published_together(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
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
            user_video, pro_video, models = comparison_files(directory)
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
            self.assertEqual(user_video.read_bytes(), b"user source")
            self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_preflight_reports_primary_and_clip_directory_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
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
                dependencies = ZeroComparisonDependencies(user_video, pro_video)
                compare_videos(request, dependencies)

            self.assertEqual(raised.exception.paths, (primary, clips_directory))
            self.assertEqual(dependencies.events, [])
            self.assertEqual(user_video.read_bytes(), b"user source")
            self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_input_effect_failures_are_invalid_before_selection(self) -> None:
        cases = (
            (
                FailedInspectionDependencies,
                "invalid video input: probe failed",
            ),
            (
                FailedAudioInspectionDependencies,
                "cannot inspect user audio: audio probe failed",
            ),
            (MissingAudioDependencies, "user video has no audio"),
        )
        for dependency_type, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory_name,
            ):
                directory = Path(directory_name)
                user_video, pro_video, models = comparison_files(directory)
                request = ComparisonRequest(
                    user_video=user_video,
                    pro_video=pro_video,
                    pro_speed=Fraction(1),
                    audio_model=models[0],
                    shot_model=models[1],
                    shot_type_model=models[2],
                )
                dependencies = dependency_type(user_video, pro_video)

                with self.assertRaisesRegex(InvalidComparisonRequest, message):
                    compare_videos(request, dependencies)

                self.assertNotIn("select", dependencies.events)
                self.assertNotIn("detect", dependencies.events)
                self.assertEqual(user_video.read_bytes(), b"user source")
                self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_preparation_effect_failures_are_typed_and_leave_no_output(self) -> None:
        cases = (
            (FailedLocatorDependencies, "prepare pro window.*locator failed"),
            (
                FailedUserObservationDependencies,
                "prepare user swing 4.*user observation failed",
            ),
        )
        for dependency_type, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory_name,
            ):
                directory = Path(directory_name)
                user_video, pro_video, models = comparison_files(directory)
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

                with self.assertRaisesRegex(ComparisonProcessingFailed, message):
                    compare_videos(
                        request, dependency_type(user_video, pro_video)
                    )

                self.assertFalse(output_directory.exists())
                self.assertEqual(user_video.read_bytes(), b"user source")
                self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_publication_failure_rolls_back_clips_primary_and_output_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
            output_directory = directory / "outputs"
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
            primary = output_directory / "user_vs_pro_slow0.0625x.mp4"
            replace = os.replace

            def fail_primary_publication(source: Path, destination: Path) -> None:
                if Path(destination) == primary:
                    raise OSError("publication stopped")
                replace(source, destination)

            with (
                patch.object(
                    comparison_workflow.os,
                    "replace",
                    side_effect=fail_primary_publication,
                ),
                self.assertRaisesRegex(
                    ComparisonProcessingFailed,
                    "artifact publication.*publication stopped",
                ),
            ):
                compare_videos(
                    request, MatchingDependencies(user_video, pro_video)
                )

            self.assertFalse(output_directory.exists())
            self.assertEqual(tuple(directory.glob(".tennis-compare-*")), ())
            self.assertEqual(user_video.read_bytes(), b"user source")
            self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_staging_and_output_directory_failures_are_publication_failures(
        self,
    ) -> None:
        failure_cases = ("staging", "output directory")
        for failure in failure_cases:
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory_name,
            ):
                directory = Path(directory_name)
                user_video, pro_video, models = comparison_files(directory)
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
                dependencies = DirectRenderDependencies(user_video, pro_video)
                original_mkdir = Path.mkdir

                def fail_output_mkdir(path: Path, *args, **kwargs) -> None:
                    if path == output_directory:
                        raise OSError("cannot create output directory")
                    original_mkdir(path, *args, **kwargs)

                effect = (
                    patch.object(
                        comparison_workflow.tempfile,
                        "TemporaryDirectory",
                        side_effect=OSError("cannot create staging directory"),
                    )
                    if failure == "staging"
                    else patch.object(Path, "mkdir", new=fail_output_mkdir)
                )
                expected = (
                    "artifact publication failed: cannot create staging directory"
                    if failure == "staging"
                    else "artifact publication failed: cannot create output directory"
                )

                with effect, self.assertRaisesRegex(
                    ComparisonProcessingFailed, expected
                ):
                    compare_videos(request, dependencies)

                self.assertFalse(output_directory.exists())
                self.assertEqual(user_video.read_bytes(), b"user source")
                self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_staging_cleanup_failure_rolls_back_already_published_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
            output_directory = directory / "outputs"
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
            staging_paths: list[Path] = []

            class CleanupFailsOnce:
                def __init__(self, *, prefix: str, dir: Path) -> None:
                    self.name = comparison_workflow.tempfile.mkdtemp(
                        prefix=prefix, dir=dir
                    )
                    self.cleanup_calls = 0
                    staging_paths.append(Path(self.name))

                def cleanup(self) -> None:
                    self.cleanup_calls += 1
                    if self.cleanup_calls == 1:
                        if Path(self.name).exists():
                            comparison_workflow.shutil.rmtree(self.name)
                        raise OSError("staging cleanup failed")

            with (
                patch.object(
                    comparison_workflow.tempfile,
                    "TemporaryDirectory",
                    CleanupFailsOnce,
                ),
                self.assertRaisesRegex(
                    ComparisonProcessingFailed,
                    "artifact publication failed: staging cleanup failed",
                ),
            ):
                compare_videos(
                    request, MatchingDependencies(user_video, pro_video)
                )

            self.assertFalse(output_directory.exists())
            self.assertTrue(all(not path.exists() for path in staging_paths))
            self.assertEqual(user_video.read_bytes(), b"user source")
            self.assertEqual(pro_video.read_bytes(), b"pro source")

    def test_selection_and_detection_failures_are_typed_by_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video, pro_video, models = comparison_files(directory)
            request = ComparisonRequest(
                user_video=user_video,
                pro_video=pro_video,
                pro_speed=Fraction(1),
                audio_model=models[0],
                shot_model=models[1],
                shot_type_model=models[2],
            )

            cases = (
                (
                    FailedSelectionDependencies(user_video, pro_video),
                    "pro contact detection",
                ),
                (
                    FailedDetectionDependencies(user_video, pro_video),
                    "swing detection",
                ),
            )
            for dependencies, expected_stage in cases:
                with self.subTest(stage=expected_stage):
                    with self.assertRaises(ComparisonProcessingFailed) as raised:
                        compare_videos(request, dependencies)
                    self.assertEqual(raised.exception.stage, expected_stage)


if __name__ == "__main__":
    unittest.main()

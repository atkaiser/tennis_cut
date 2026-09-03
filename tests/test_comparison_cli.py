from __future__ import annotations

from fractions import Fraction
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tennis_cut.comparison.cli import _configure_logging, build_parser, main
from tennis_cut.comparison.media import MediaCommandFailed
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
from tennis_cut.comparison.workflow import ComparisonRequest
from tennis_cut.swing_detection import DEFAULT_TEMPORAL_RANKER_MODEL, DetectedSwing
from tennis_cut.temporal_ranker import load_temporal_ranker


class ComparisonCliParserTests(unittest.TestCase):
    def test_normal_logging_is_timestamped_info_and_quiet_suppresses_it(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            _configure_logging(verbose=False, quiet=False)
            logging.getLogger("tennis_cut.progress-test").info(
                "Inspecting video metadata"
            )

        self.assertRegex(
            stderr.getvalue(),
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO: "
            r"Inspecting video metadata\n$",
        )

        with patch("sys.stderr", new_callable=io.StringIO) as quiet_stderr:
            _configure_logging(verbose=False, quiet=True)
            logging.getLogger("tennis_cut.progress-test").info("hidden progress")

        self.assertEqual(quiet_stderr.getvalue(), "")

    def test_debug_report_only_selects_report_without_video_rendering(self) -> None:
        args = build_parser().parse_args(
            [
                "user.mov",
                "pro.mov",
                "--pro-speed",
                "1",
                "--debug-report-only",
                "diagnostics.html",
            ]
        )

        self.assertEqual(args.debug_report_only, Path("diagnostics.html"))
        self.assertIsNone(args.debug_report)

    def test_defaults_to_bundled_visual_contact_ranker(self) -> None:
        build_parser().parse_args(["user.mov", "pro.mov", "--pro-speed", "1"])

        self.assertTrue(DEFAULT_TEMPORAL_RANKER_MODEL.is_absolute())
        self.assertTrue(DEFAULT_TEMPORAL_RANKER_MODEL.is_file())
        self.assertEqual(
            load_temporal_ranker(DEFAULT_TEMPORAL_RANKER_MODEL).feature_version,
            1,
        )

    def test_keeps_the_bundled_ranker_out_of_the_public_cli(self) -> None:
        with (
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as exit_context,
        ):
            build_parser().parse_args(
                [
                    "user.mov",
                    "pro.mov",
                    "--pro-speed",
                    "1",
                    "--visual-contact-ranker-model",
                    "ranker.json",
                ]
            )

        self.assertEqual(exit_context.exception.code, 2)
        self.assertIn(
            "unrecognized arguments: --visual-contact-ranker-model",
            stderr.getvalue(),
        )
        self.assertNotIn("ranker", build_parser().format_help())

    def test_rejects_runtime_prototype_training_options(self) -> None:
        with (
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as exit_context,
        ):
            build_parser().parse_args(
                [
                    "user.mov",
                    "pro.mov",
                    "--pro-speed",
                    "1",
                    "--prototype",
                    "--prototype-records",
                    "records.json",
                ]
            )

        self.assertEqual(exit_context.exception.code, 2)
        self.assertIn("unrecognized arguments: --prototype", stderr.getvalue())


class ValidDependencies:
    def __init__(self, user_video: Path, pro_video: Path) -> None:
        frames = tuple(
            DecodedFrame(0, ordinal, ordinal, Fraction(1, 10))
            for ordinal in range(31)
        )
        media = InspectedMedia(frames)
        self.sources = {
            user_video: ComparisonSource(user_video, 1920, 1080, media),
            pro_video: ComparisonSource(pro_video, 1920, 1080, media),
        }
        self.selection_resolutions = 0

    def executable_exists(self, name: str) -> bool:
        return True

    def inspect_source(self, path: Path) -> ComparisonSource:
        return self.sources[path]

    def user_has_audio(self, path: Path) -> bool:
        return True

    def resolve_selection(
        self,
        pro_video: Path,
        pro_speed: Fraction,
        inspected_media: InspectedMedia,
        detection_config: object,
    ) -> ProSelection | SelectionProcessingFailure:
        self.selection_resolutions += 1
        return ProSelection(pro_video, inspected_media.frames[15])

    def detect_swings(
        self, request: ComparisonRequest, user_source: ComparisonSource
    ) -> tuple[DetectedSwing, ...]:
        return (DetectedSwing(0, Fraction(3, 2), "forehand"),)

    def create_player_locator(self, device: str | None) -> object:
        return object()

    def observe_players(
        self, window: SelectedSourceWindow, locator: object
    ) -> tuple[PlayerObservation, ...]:
        return (PlayerObservation(0, Rectangle(400, 100, 160, 360)),)

    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_bytes(b"compilation")


class FailedEncoderDependencies(ValidDependencies):
    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        raise MediaCommandFailed("ffmpeg", 1, "raw encoder details")


class StoppedSelectionDependencies(ValidDependencies):
    def __init__(
        self,
        user_video: Path,
        pro_video: Path,
        result: SelectionProcessingFailure,
    ) -> None:
        super().__init__(user_video, pro_video)
        self.result = result

    def resolve_selection(
        self,
        pro_video: Path,
        pro_speed: Fraction,
        inspected_media: InspectedMedia,
        detection_config: object,
    ) -> SelectionProcessingFailure:
        self.selection_resolutions += 1
        return self.result

    def detect_swings(
        self, request: ComparisonRequest, user_source: ComparisonSource
    ) -> tuple[DetectedSwing, ...]:
        raise AssertionError("models must remain lazy when selection does not complete")


class MissingAudioDependencies(ValidDependencies):
    def user_has_audio(self, path: Path) -> bool:
        return False


class ComparisonCliTests(unittest.TestCase):
    def test_parser_has_comparison_defaults(self) -> None:
        args = build_parser().parse_args(
            ["user.mov", "pro.mov", "--pro-speed", "0.25"]
        )

        self.assertEqual(args.pro_speed, Fraction(1, 4))
        self.assertEqual(args.slowmo, Fraction(1, 16))
        self.assertEqual(args.output_dir, Path("processed_vids"))
        self.assertIsNone(args.device)
        self.assertFalse(args.clips)

    def test_parser_rejects_invalid_decimal_playback_factors(self) -> None:
        for option, value in (
            ("--pro-speed", "1/4"),
            ("--slowmo", "1/4"),
            ("--slowmo", "NaN"),
            ("--slowmo", "Infinity"),
        ):
            arguments = ["user.mov", "pro.mov", "--pro-speed", "1"]
            if option == "--pro-speed":
                arguments[-1] = value
            else:
                arguments.extend((option, value))
            with (
                self.subTest(option=option, value=value),
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
                self.assertRaises(SystemExit) as raised,
            ):
                build_parser().parse_args(arguments)

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(
                stderr.getvalue(),
                f"tennis-compare: argument {option}: invalid decimal value: {value}\n",
            )

    def test_verbose_and_quiet_are_mutually_exclusive(self) -> None:
        with (
            patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            build_parser().parse_args(
                [
                    "user.mov",
                    "pro.mov",
                    "--pro-speed",
                    "1",
                    "--verbose",
                    "--quiet",
                ]
            )

        self.assertEqual(raised.exception.code, 2)

    def test_invalid_request_has_stable_stderr_and_status_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            missing_user = directory / "missing.mov"
            pro_video = directory / "pro.mov"
            pro_video.touch()
            with (
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                dependencies = ValidDependencies(missing_user, pro_video)
                status = main(
                    [
                        str(missing_user),
                        str(pro_video),
                        "--pro-speed",
                        "1",
                    ],
                    dependencies=dependencies,
                )

        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            f"tennis-compare: missing user video: {missing_user}\n",
        )
        self.assertEqual(dependencies.selection_resolutions, 0)

    def test_semantic_preflight_errors_have_stable_status_and_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.write_bytes(b"user")
            pro_video.write_bytes(b"pro")
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            base_argv = [
                str(user_video),
                str(pro_video),
                "--pro-speed",
                "1",
                "--audio-model",
                str(models[0]),
                "--shot-model",
                str(models[1]),
                "--shot-type-model",
                str(models[2]),
            ]
            cases = (
                (
                    [*base_argv[:3], "0", *base_argv[4:]],
                    ValidDependencies(user_video, pro_video),
                    "pro speed must be greater than zero",
                ),
                (
                    [*base_argv, "--slowmo", "2"],
                    ValidDependencies(user_video, pro_video),
                    "slow motion must be in (0, 1]",
                ),
                (
                    base_argv,
                    MissingAudioDependencies(user_video, pro_video),
                    f"user video has no audio: {user_video}",
                ),
            )
            for argv, dependencies, message in cases:
                with (
                    self.subTest(message=message),
                    patch("sys.stdout", new_callable=io.StringIO) as stdout,
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    status = main(argv, dependencies=dependencies)

                self.assertEqual(status, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(stderr.getvalue(), f"tennis-compare: {message}\n")
                self.assertEqual(dependencies.selection_resolutions, 0)
                self.assertEqual(user_video.read_bytes(), b"user")
                self.assertEqual(pro_video.read_bytes(), b"pro")

    def test_all_output_collisions_are_reported_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.write_bytes(b"user")
            pro_video.write_bytes(b"pro")
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            output_directory = directory / "output"
            output_directory.mkdir()
            primary = output_directory / "user_vs_pro_slow0.0625x.mp4"
            clips_directory = output_directory / "user_vs_pro_slow0.0625x_clips"
            primary.touch()
            clips_directory.mkdir()
            dependencies = ValidDependencies(user_video, pro_video)

            with (
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                status = main(
                    [
                        str(user_video),
                        str(pro_video),
                        "--pro-speed",
                        "1",
                        "--clips",
                        "--output-dir",
                        str(output_directory),
                        "--audio-model",
                        str(models[0]),
                        "--shot-model",
                        str(models[1]),
                        "--shot-type-model",
                        str(models[2]),
                    ],
                    dependencies=dependencies,
                )

            self.assertEqual(status, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue(),
                f"tennis-compare: output already exists: {primary}, {clips_directory}\n",
            )
            self.assertEqual(dependencies.selection_resolutions, 0)
            self.assertEqual(user_video.read_bytes(), b"user")
            self.assertEqual(pro_video.read_bytes(), b"pro")

    def test_selection_stop_precedes_models_and_writes_no_comparison_output(
        self,
    ) -> None:
        cases = (
            (
                SelectionProcessingFailure(
                    "pro contact detection", "visual contact unavailable"
                ),
                "pro contact detection failed: visual contact unavailable",
            ),
        )
        for selection_result, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory_name,
            ):
                directory = Path(directory_name)
                user_video = directory / "user.mov"
                pro_video = directory / "pro.mov"
                user_video.touch()
                pro_video.touch()
                models = tuple(directory / name for name in ("audio", "shot", "type"))
                for model in models:
                    model.touch()
                output_directory = directory / "output"
                dependencies = StoppedSelectionDependencies(
                    user_video, pro_video, selection_result
                )
                with (
                    patch("sys.stdout", new_callable=io.StringIO) as stdout,
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    status = main(
                        [
                            str(user_video),
                            str(pro_video),
                            "--pro-speed",
                            "1",
                            "--output-dir",
                            str(output_directory),
                            "--audio-model",
                            str(models[0]),
                            "--shot-model",
                            str(models[1]),
                            "--shot-type-model",
                            str(models[2]),
                        ],
                        dependencies=dependencies,
                    )

                self.assertEqual(status, 1)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(stderr.getvalue(), f"tennis-compare: {message}\n")
                self.assertFalse(output_directory.exists())

    def test_automatic_contact_runs_complete_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.write_bytes(b"user")
            pro_video.write_bytes(b"pro")
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            output_directory = directory / "output"
            argv = [
                str(user_video),
                str(pro_video),
                "--pro-speed",
                "1",
                "--slowmo",
                "0.5",
                "--output-dir",
                str(output_directory),
                "--audio-model",
                str(models[0]),
                "--shot-model",
                str(models[1]),
                "--shot-type-model",
                str(models[2]),
            ]
            dependencies = ValidDependencies(user_video, pro_video)

            with (
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                status = main(argv, dependencies=dependencies)

            primary = output_directory / "user_vs_pro_slow0.5x.mp4"
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue(), f"{primary}\n")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(primary.read_bytes(), b"compilation")
            self.assertEqual(user_video.read_bytes(), b"user")
            self.assertEqual(pro_video.read_bytes(), b"pro")

    def test_encoder_diagnostics_are_verbose_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.touch()
            pro_video.touch()
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            base_argv = [
                str(user_video),
                str(pro_video),
                "--pro-speed",
                "1",
                "--audio-model",
                str(models[0]),
                "--shot-model",
                str(models[1]),
                "--shot-type-model",
                str(models[2]),
                "--output-dir",
                str(directory / "output"),
            ]
            dependencies = FailedEncoderDependencies(user_video, pro_video)

            with patch("sys.stderr", new_callable=io.StringIO) as normal_stderr:
                normal_status = main(base_argv, dependencies=dependencies)
            with patch("sys.stderr", new_callable=io.StringIO) as verbose_stderr:
                verbose_status = main(
                    [*base_argv, "--verbose"], dependencies=dependencies
                )

        stable_error = (
            "tennis-compare: comparison rendering failed: "
            "ffmpeg exited with status 1\n"
        )
        self.assertEqual(normal_status, 1)
        self.assertEqual(normal_stderr.getvalue(), stable_error)
        self.assertEqual(verbose_status, 1)
        self.assertIn("ERROR: raw encoder details\n", verbose_stderr.getvalue())
        self.assertTrue(verbose_stderr.getvalue().endswith(stable_error))


if __name__ == "__main__":
    unittest.main()

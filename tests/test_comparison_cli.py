from __future__ import annotations

from fractions import Fraction
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tennis_cut.comparison.cli import build_parser, main
from tennis_cut.comparison.planning import (
    ComparisonSource,
    PlayerObservation,
    Rectangle,
)
from tennis_cut.comparison.pro_selection import (
    DecodedFrame,
    FileSidecarStore,
    InspectedMedia,
    resolve_pro_selection,
)
from tennis_cut.swing_detection import DetectedSwing


class NoPicker:
    def pick(self, session):
        raise AssertionError("a valid saved selection must remain noninteractive")


class ValidSidecarDependencies:
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

    def executable_exists(self, name: str) -> bool:
        return True

    def inspect_source(self, path: Path) -> ComparisonSource:
        return self.sources[path]

    def user_has_audio(self, path: Path) -> bool:
        return True

    def resolve_selection(self, pro_video, pro_speed, inspected_media):
        return resolve_pro_selection(
            pro_video=pro_video,
            pro_speed=pro_speed,
            inspected_media=inspected_media,
            sidecar_store=FileSidecarStore(),
            picker=NoPicker(),
        )

    def detect_swings(self, request):
        return (DetectedSwing(0, Fraction(3, 2), "forehand"),)

    def create_player_locator(self, device):
        return object()

    def observe_players(self, window, locator):
        return (PlayerObservation(0, Rectangle(400, 100, 160, 360)),)

    def render_artifacts(self, plans, primary, clips):
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_bytes(b"compilation")


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
                status = main(
                    [
                        str(missing_user),
                        str(pro_video),
                        "--pro-speed",
                        "1",
                    ],
                    dependencies=ValidSidecarDependencies(missing_user, pro_video),
                )

        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            f"tennis-compare: missing user video: {missing_user}\n",
        )

    def test_valid_saved_selection_runs_complete_noninteractive_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_video = directory / "user.mov"
            pro_video = directory / "pro.mov"
            user_video.write_bytes(b"user")
            pro_video.write_bytes(b"pro")
            models = tuple(directory / name for name in ("audio", "shot", "type"))
            for model in models:
                model.touch()
            pro_stat = pro_video.stat()
            sidecar = pro_video.with_name(f"{pro_video.name}.tennis-compare.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": {
                            "name": pro_video.name,
                            "size_bytes": pro_stat.st_size,
                            "mtime_ns": pro_stat.st_mtime_ns,
                        },
                        "video_stream": {
                            "index": 0,
                            "time_base": {"numerator": 1, "denominator": 10},
                        },
                        "contact_frame": {"ordinal": 15, "pts": 15},
                        "shot_type": "forehand",
                    }
                ),
                encoding="utf-8",
            )
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
            dependencies = ValidSidecarDependencies(user_video, pro_video)

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


if __name__ == "__main__":
    unittest.main()

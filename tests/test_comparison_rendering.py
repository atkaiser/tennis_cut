from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Literal
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageStat

from tennis_cut.comparison.cli import main
from tennis_cut.comparison.media import (
    inspect_comparison_source,
    observe_players,
    render_comparison,
    render_compilation,
)
from tennis_cut.comparison.planning import (
    ComparisonRenderPlan,
    ComparisonSource,
    PlayerObservation,
    SelectedSourceWindow,
)
from tennis_cut.comparison.pro_selection import InspectedMedia, ProSelection
from tennis_cut.comparison.workflow import ComparisonRequest
from tennis_cut.swing_detection import DetectedSwing


class WhitePlayerLocator:
    def find_box(self, image_path: Path) -> tuple[int, int, int, int] | None:
        with Image.open(image_path) as image:
            pixels = image.convert("RGB")
            matching = [
                (x, y)
                for y in range(pixels.height)
                for x in range(pixels.width)
                if min(pixels.getpixel((x, y))) > 220
            ]
        if not matching:
            return None
        xs, ys = zip(*matching)
        return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def frame_color(ordinal: int, role: Literal["user", "pro"]) -> tuple[int, int, int]:
    levels = (25, 85, 145, 205)
    digits = (ordinal % 4, (ordinal // 4) % 4, (ordinal // 16) % 4)
    if role == "pro":
        digits = digits[1], digits[2], digits[0]
    return tuple(levels[digit] for digit in digits)


def decoded_ordinal(
    median: list[float], role: Literal["user", "pro"], frame_count: int
) -> int:
    matches = [
        ordinal
        for ordinal in range(frame_count)
        if all(
            abs(actual - expected) <= 8
            for actual, expected in zip(median, frame_color(ordinal, role))
        )
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"decoded {role} color {median} did not identify exactly one source frame"
        )
    return matches[0]


def create_source(
    directory: Path,
    *,
    name: str,
    frame_count: int,
    rate: int,
    timescale: int,
    role: Literal["user", "pro"],
) -> Path:
    frames = directory / f"{name}_frames"
    frames.mkdir()
    for ordinal in range(frame_count):
        image = Image.new("RGB", (320, 180), frame_color(ordinal, role))
        player_x = 4 + (ordinal % 6) * 4 if role == "user" else 284 - (ordinal % 6) * 4
        player_y = 45 + (ordinal % 5) * 2
        ImageDraw.Draw(image).rectangle(
            (player_x, player_y, player_x + 31, player_y + 79), fill="white"
        )
        image.save(frames / f"frame_{ordinal:04d}.png")
    output = directory / f"{name}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            str(rate),
            "-start_number",
            "0",
            "-i",
            str(frames / "frame_%04d.png"),
            "-vf",
            "setpts=PTS+10",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-video_track_timescale",
            str(timescale),
            str(output),
            "-y",
        ],
        check=True,
    )
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_output(video: Path, directory: Path) -> tuple[Path, ...]:
    directory.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            str(directory / "output_%04d.png"),
        ],
        check=True,
    )
    return tuple(sorted(directory.glob("output_*.png")))


class GeneratedWorkflowDependencies:
    def __init__(self, user_path: Path, pro_path: Path) -> None:
        self.sources = {
            user_path: inspect_comparison_source(user_path),
            pro_path: inspect_comparison_source(pro_path),
        }
        self.pro_path = pro_path
        self.rendered_plans: list[tuple[ComparisonRenderPlan, ...]] = []

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
    ) -> ProSelection:
        self.pro_speed = pro_speed
        return ProSelection(pro_video, inspected_media.frames[25], "forehand")

    def detect_swings(self, request: ComparisonRequest) -> tuple[DetectedSwing, ...]:
        return (
            DetectedSwing(2, Fraction(5, 2), "forehand"),
            DetectedSwing(7, Fraction(3), "forehand"),
        )

    def create_player_locator(self, device: str | None) -> WhitePlayerLocator:
        self.device = device
        return WhitePlayerLocator()

    def observe_players(
        self, window: SelectedSourceWindow, locator: WhitePlayerLocator
    ) -> tuple[PlayerObservation, ...]:
        return observe_players(window, locator)

    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        self.rendered_plans.append(plans)
        render_compilation(plans, primary)
        for plan, clip in zip(plans, clips):
            if plan.artifact.path != clip:
                raise AssertionError("workflow changed the requested clip path")
            render_comparison(plan)


def probe_output(video: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=codec_type,width,height,time_base,duration_ts:frame=pts",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


class GeneratedMediaRenderingTests(unittest.TestCase):
    def test_complete_cli_workflow_with_generated_media(self) -> None:
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            user_path = create_source(
                directory,
                name="user",
                frame_count=31,
                rate=10,
                timescale=1000,
                role="user",
            )
            pro_path = create_source(
                directory,
                name="pro",
                frame_count=41,
                rate=5,
                timescale=90000,
                role="pro",
            )
            original_hashes = (sha256(user_path), sha256(pro_path))
            user_source = inspect_comparison_source(user_path)
            pro_source = inspect_comparison_source(pro_path)
            self.assertNotEqual(
                user_source.inspected_media.frames[0].time_base,
                pro_source.inspected_media.frames[0].time_base,
            )
            self.assertGreater(user_source.inspected_media.frames[0].pts, 0)
            self.assertGreater(pro_source.inspected_media.frames[0].pts, 0)
            model_paths = tuple(directory / name for name in ("audio", "shot", "type"))
            for model_path in model_paths:
                model_path.touch()

            for clips in (False, True):
                with self.subTest(clips=clips):
                    output_directory = directory / (
                        "with-clips" if clips else "default"
                    )
                    dependencies = GeneratedWorkflowDependencies(user_path, pro_path)
                    arguments = [
                        str(user_path),
                        str(pro_path),
                        "--pro-speed",
                        "0.25",
                        "--slowmo",
                        "1",
                        "--output-dir",
                        str(output_directory),
                        "--audio-model",
                        str(model_paths[0]),
                        "--shot-model",
                        str(model_paths[1]),
                        "--shot-type-model",
                        str(model_paths[2]),
                        "--device",
                        "cpu",
                    ]
                    if clips:
                        arguments.append("--clips")
                    with (
                        patch("sys.stdout", new_callable=io.StringIO) as stdout,
                        patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    ):
                        status = main(arguments, dependencies=dependencies)

                    primary = output_directory / "user_vs_pro_slow1x.mp4"
                    clips_directory = output_directory / "user_vs_pro_slow1x_clips"
                    self.assertEqual(status, 0)
                    self.assertEqual(stdout.getvalue(), f"{primary}\n")
                    self.assertEqual(stderr.getvalue(), "")
                    self.assertEqual(dependencies.pro_speed, Fraction(1, 4))
                    self.assertEqual(dependencies.device, "cpu")
                    self.assertEqual(len(dependencies.rendered_plans), 1)
                    plans = dependencies.rendered_plans[0]
                    self.assertEqual(
                        [plan.user.window.swing_ordinal for plan in plans], [2, 7]
                    )
                    self.assertLess(
                        min(
                            observation.bounds.x
                            for plan in plans
                            for observation in plan.user.observations
                        ),
                        10,
                    )
                    self.assertGreater(
                        max(
                            observation.bounds.x + observation.bounds.width
                            for observation in plans[0].pro.observations
                        ),
                        310,
                    )
                    for plan in plans:
                        self.assertEqual(
                            (
                                plan.layout.user_panel.width,
                                plan.layout.user_panel.height,
                            ),
                            (640, 720),
                        )
                        self.assertEqual(
                            (plan.layout.pro_panel.width, plan.layout.pro_panel.height),
                            (640, 720),
                        )
                        for prepared in (plan.user, plan.pro):
                            self.assertEqual(
                                prepared.crop.width * 9, prepared.crop.height * 8
                            )
                    expected_paths = [primary]
                    if clips:
                        expected_paths.extend(
                            clips_directory / f"comparison_{index:03d}.mp4"
                            for index in (1, 2)
                        )
                    else:
                        self.assertFalse(clips_directory.exists())
                    self.assertEqual(
                        sorted(
                            path
                            for path in output_directory.rglob("*")
                            if path.is_file()
                        ),
                        sorted(expected_paths),
                    )
                    for clip_path, plan in zip(expected_paths[1:], plans):
                        clip_payload = probe_output(clip_path)
                        self.assertEqual(
                            [
                                stream["codec_type"]
                                for stream in clip_payload["streams"]
                            ],
                            ["video"],
                        )
                        self.assertEqual(
                            [int(frame["pts"]) for frame in clip_payload["frames"]],
                            [event.output_tick for event in plan.events],
                        )
                        self.assertEqual(
                            int(clip_payload["streams"][0]["duration_ts"]),
                            plan.events[-1].output_tick + 1,
                        )

                    payload = probe_output(primary)
                    streams = payload["streams"]
                    self.assertEqual(
                        [stream["codec_type"] for stream in streams], ["video"]
                    )
                    self.assertEqual(
                        (streams[0]["width"], streams[0]["height"]), (1280, 720)
                    )
                    self.assertEqual(
                        Fraction(streams[0]["time_base"]), plans[0].output_time_base
                    )
                    expected_ticks: list[int] = []
                    expected_events = []
                    next_tick = 0
                    for plan in plans:
                        expected_ticks.extend(
                            next_tick + event.output_tick for event in plan.events
                        )
                        expected_events.extend(plan.events)
                        next_tick = expected_ticks[-1] + 1
                    self.assertEqual(
                        [int(frame["pts"]) for frame in payload["frames"]],
                        expected_ticks,
                    )
                    self.assertEqual(
                        int(streams[0]["duration_ts"]), expected_ticks[-1] + 1
                    )

                    decoded = decode_output(
                        primary,
                        directory / f"decoded-{'clips' if clips else 'default'}",
                    )
                    self.assertEqual(len(decoded), len(expected_events))
                    decoded_user_ordinals: list[int] = []
                    decoded_pro_ordinals: list[int] = []
                    for image_path in decoded:
                        with Image.open(image_path) as image:
                            user_panel = image.crop((0, 0, 640, 720))
                            pro_panel = image.crop((640, 0, 1280, 720))
                            user_statistics = ImageStat.Stat(user_panel)
                            pro_statistics = ImageStat.Stat(pro_panel)
                            user_median = user_statistics.median
                            pro_median = pro_statistics.median
                            self.assertTrue(
                                all(high > 220 for _, high in user_statistics.extrema),
                                "the cropped user player must remain visible",
                            )
                            self.assertTrue(
                                all(high > 220 for _, high in pro_statistics.extrema),
                                "the cropped pro player must remain visible",
                            )
                        decoded_user_ordinals.append(
                            decoded_ordinal(user_median, "user", 31)
                        )
                        decoded_pro_ordinals.append(
                            decoded_ordinal(pro_median, "pro", 41)
                        )
                    self.assertEqual(
                        decoded_user_ordinals,
                        [event.user_frame.ordinal for event in expected_events],
                    )
                    self.assertEqual(
                        decoded_pro_ordinals,
                        [event.pro_frame.ordinal for event in expected_events],
                    )

                    clip_offset = 0
                    for plan in plans:
                        contact_index = next(
                            index
                            for index, event in enumerate(plan.events)
                            if event.normalized_time == 0
                        )
                        contact = plan.events[contact_index]
                        self.assertEqual(
                            contact.user_frame, plan.user.window.contact_frame
                        )
                        self.assertEqual(
                            contact.pro_frame, plan.pro.window.contact_frame
                        )
                        self.assertEqual(
                            decoded_user_ordinals[clip_offset + contact_index],
                            plan.user.window.contact_frame.ordinal,
                        )
                        self.assertEqual(
                            decoded_pro_ordinals[clip_offset + contact_index],
                            plan.pro.window.contact_frame.ordinal,
                        )
                        user_slice = decoded_user_ordinals[
                            clip_offset : clip_offset + len(plan.events)
                        ]
                        pro_slice = decoded_pro_ordinals[
                            clip_offset : clip_offset + len(plan.events)
                        ]
                        self.assertEqual(
                            set(user_slice),
                            {
                                item.frame.ordinal
                                for item in plan.user.window.normalized_frames
                            },
                        )
                        self.assertEqual(
                            set(pro_slice),
                            {
                                item.frame.ordinal
                                for item in plan.pro.window.normalized_frames
                            },
                        )
                        self.assertGreater(len(set(pro_slice)), len(set(user_slice)))
                        clip_offset += len(plan.events)
                    cut_index = len(plans[0].events)
                    self.assertNotEqual(
                        expected_events[cut_index - 1].user_frame,
                        expected_events[cut_index].user_frame,
                    )
                    self.assertEqual(
                        (sha256(user_path), sha256(pro_path)), original_hashes
                    )
                    self.assertFalse(
                        pro_path.with_name(
                            f"{pro_path.name}.tennis-compare.json"
                        ).exists()
                    )

        self.assertLess(time.monotonic() - started, 120)


if __name__ == "__main__":
    unittest.main()

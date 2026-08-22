from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Literal
import unittest

from PIL import Image, ImageDraw, ImageStat

from tennis_cut.comparison.media import (
    inspect_comparison_source,
    observe_players,
    render_comparison,
    render_compilation,
)
from tennis_cut.comparison.planning import (
    ArtifactRequest,
    build_render_plan,
    prepare_source_window,
    select_comparison_windows,
)
from tennis_cut.comparison.pro_selection import ProSelection
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
        color = (
            (30 + ordinal * 5, 20, 20)
            if role == "user"
            else (20, 30 + ordinal * 4, 20)
        )
        image = Image.new("RGB", (320, 180), color)
        player_x = 90 + (ordinal % 8) * 4
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


class GeneratedMediaRenderingTests(unittest.TestCase):
    def test_decoded_clip_preserves_exact_unequal_source_event_union(self) -> None:
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
            windows = select_comparison_windows(
                user_source=user_source,
                user_swings=(DetectedSwing(0, Fraction(5, 2), "forehand"),),
                pro_source=pro_source,
                pro_selection=ProSelection(
                    pro_path, pro_source.inspected_media.frames[25], "forehand"
                ),
                pro_speed=Fraction(1, 4),
            )
            locator = WhitePlayerLocator()
            user = prepare_source_window(
                windows.user[0], observe_players(windows.user[0], locator)
            )
            pro = prepare_source_window(
                windows.pro, observe_players(windows.pro, locator)
            )
            output = directory / "comparison.mp4"
            plan = build_render_plan(
                user=user,
                pro=pro,
                slow_motion=Fraction(1),
                artifact=ArtifactRequest(output),
            )

            self.assertEqual(render_comparison(plan), output)

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_frames",
                    "-show_entries",
                    "stream=codec_type,width,height,time_base:frame=pts",
                    "-of",
                    "json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(probe.stdout)
            streams = payload["streams"]
            self.assertEqual([stream["codec_type"] for stream in streams], ["video"])
            self.assertEqual((streams[0]["width"], streams[0]["height"]), (1280, 720))
            self.assertEqual(Fraction(streams[0]["time_base"]), plan.output_time_base)
            self.assertEqual(
                [int(frame["pts"]) for frame in payload["frames"]],
                [event.output_tick for event in plan.events],
            )

            decoded = decode_output(output, directory / "decoded")
            self.assertEqual(len(decoded), len(plan.events))
            for image_path, event in zip(decoded, plan.events):
                with Image.open(image_path) as image:
                    user_median = ImageStat.Stat(image.crop((0, 0, 640, 720))).median
                    pro_median = ImageStat.Stat(image.crop((640, 0, 1280, 720))).median
                self.assertAlmostEqual(
                    user_median[0], 30 + event.user_frame.ordinal * 5, delta=8
                )
                self.assertAlmostEqual(
                    pro_median[1], 30 + event.pro_frame.ordinal * 4, delta=8
                )

            contact_index = next(
                index
                for index, event in enumerate(plan.events)
                if event.normalized_time == 0
            )
            self.assertEqual(plan.events[contact_index].user_frame.ordinal, 15)
            self.assertEqual(plan.events[contact_index].pro_frame.ordinal, 25)
            self.assertEqual(
                {event.pro_frame.ordinal for event in plan.events},
                {item.frame.ordinal for item in pro.window.normalized_frames},
            )
            self.assertGreater(
                len({event.pro_frame.ordinal for event in plan.events}),
                len({event.user_frame.ordinal for event in plan.events}),
            )
            self.assertEqual((sha256(user_path), sha256(pro_path)), original_hashes)

    def test_compilation_encodes_ordered_clips_once_with_hard_gapless_cut(self) -> None:
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
            user_source = inspect_comparison_source(user_path)
            pro_source = inspect_comparison_source(pro_path)
            windows = select_comparison_windows(
                user_source=user_source,
                user_swings=(
                    DetectedSwing(2, Fraction(5, 2), "forehand"),
                    DetectedSwing(7, Fraction(3), "forehand"),
                ),
                pro_source=pro_source,
                pro_selection=ProSelection(
                    pro_path, pro_source.inspected_media.frames[25], "forehand"
                ),
                pro_speed=Fraction(1, 4),
            )
            locator = WhitePlayerLocator()
            pro = prepare_source_window(
                windows.pro, observe_players(windows.pro, locator)
            )
            plans = tuple(
                build_render_plan(
                    user=prepare_source_window(
                        window, observe_players(window, locator)
                    ),
                    pro=pro,
                    slow_motion=Fraction(1),
                    artifact=ArtifactRequest(directory / f"clip_{index}.mp4"),
                )
                for index, window in enumerate(windows.user)
            )
            output = directory / "compilation.mp4"

            self.assertEqual(render_compilation(plans, output), output)

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_frames",
                    "-show_entries",
                    "stream=codec_type,time_base:frame=pts",
                    "-of",
                    "json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(probe.stdout)
            self.assertEqual(
                [stream["codec_type"] for stream in payload["streams"]], ["video"]
            )
            first_ticks = [event.output_tick for event in plans[0].events]
            second_start = first_ticks[-1] + 1
            expected_ticks = first_ticks + [
                second_start + event.output_tick for event in plans[1].events
            ]
            self.assertEqual(
                [int(frame["pts"]) for frame in payload["frames"]], expected_ticks
            )

            decoded = decode_output(output, directory / "decoded-compilation")
            cut_index = len(plans[0].events)
            with Image.open(decoded[cut_index - 1]) as before_cut:
                before_red = ImageStat.Stat(before_cut.crop((0, 0, 640, 720))).median[0]
            with Image.open(decoded[cut_index]) as after_cut:
                after_red = ImageStat.Stat(after_cut.crop((0, 0, 640, 720))).median[0]
            expected_second_red = 30 + plans[1].events[0].user_frame.ordinal * 5
            self.assertNotEqual(before_red, after_red)
            self.assertAlmostEqual(after_red, expected_second_red, delta=8)


if __name__ == "__main__":
    unittest.main()

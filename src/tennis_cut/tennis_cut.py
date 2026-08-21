#!/usr/bin/env python3
"""
CLI tool for extracting tennis swings from a video.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
if __package__:
    from .swing_detection import (
        DEFAULT_AUDIO_MODEL,
        DEFAULT_SHOT_MODEL,
        DEFAULT_SHOT_TYPE_MODEL,
        SAMPLE_RATE,
        DetectionConfig,
        detect_user_swings_for_legacy,
        probe_video,
    )
    from .subprocess_utils import run_command as run_cmd
else:
    from swing_detection import (
        DEFAULT_AUDIO_MODEL,
        DEFAULT_SHOT_MODEL,
        DEFAULT_SHOT_TYPE_MODEL,
        SAMPLE_RATE,
        DetectionConfig,
        detect_user_swings_for_legacy,
        probe_video,
    )
    from subprocess_utils import run_command as run_cmd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


_LOG = logging.getLogger(__name__)


def move_input_video_to_output(input_path: Path, output_dir: Path) -> None:
    """Move a processed input video into *output_dir* without clobbering files."""
    if input_path.parent == output_dir:
        return

    dest = output_dir / input_path.name
    if dest.exists():
        stem, suffix = input_path.stem, input_path.suffix
        idx = 1
        while True:
            candidate = output_dir / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            idx += 1

    shutil.move(str(input_path), str(dest))
    _LOG.info("Moved processed video to %s", dest)


@dataclass
class Swing:
    index: int
    start: float
    end: float
    contact: float
    crop: Sequence[int] | None = None
    label: str | None = None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract tennis swings from video")
    p.add_argument("input", help="Input video file")
    p.add_argument(
        "-o", "--output-dir", default="./processed_vids/", help="Output directory"
    )
    p.add_argument(
        "--audio_model",
        help="Path to trained audio model",
        default=str(DEFAULT_AUDIO_MODEL),
    )
    p.add_argument(
        "--shot-model",
        help="Path to trained binary shot detector",
        default=str(DEFAULT_SHOT_MODEL),
    )
    p.add_argument(
        "--shot-type-model",
        help="Path to trained shot-type classifier",
        default=str(DEFAULT_SHOT_TYPE_MODEL),
    )
    p.add_argument("--clips", action="store_true", help="Export each swing separately")
    p.add_argument(
        "--slowmo",
        type=float,
        help="Generate a slow-motion version; e.g. 0.5 for half speed",
        default=0.0625,
    )
    p.add_argument(
        "--no_metadata", action="store_true", help="Skip writing JSON manifest"
    )
    p.add_argument("--no-stitch", action="store_true", help="Skip the merged video")
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="mps",
        help="PyTorch device to run the models on",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p.add_argument("-q", "--quiet", action="store_true", help="Errors only")
    return p.parse_args(argv)


def setup_logging(args: argparse.Namespace) -> None:
    level = logging.WARNING
    if args.verbose:
        level = logging.INFO
    if args.quiet:
        level = logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.shot_type_model and not args.shot_model:
        raise SystemExit("--shot-type-model requires --shot-model")


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print(
            "ffmpeg not found. Install it first (e.g. `brew install ffmpeg`).",
            file=sys.stderr,
        )
        sys.exit(1)


probe = probe_video


def cut_swing(
    video: Path,
    start: float,
    end: float,
    out_path: Path,
    crop: Sequence[int] | None,
    slowmo: float | None = None,
) -> None:
    """Extract *video* segment and optionally crop and slow down."""

    cmd = [
        "ffmpeg",
        "-ss",
        str(start),
        "-t",
        str(end - start),
        "-i",
        str(video),
    ]

    v_filters = []
    if crop is not None:
        x, y, w, h = crop
        v_filters.append(f"crop={w}:{h}:{x}:{y}")
    if slowmo is not None:
        if not 0 < slowmo <= 1:
            raise ValueError("slowmo must be in (0, 1]")
        v_filters.append(f"setpts={1 / slowmo:.6f}*PTS")

    if v_filters:
        cmd += ["-filter:v", ",".join(v_filters)]

    if slowmo is not None:
        ATEMPO_LIMIT = 0.5
        remaining = slowmo
        a_filters = []
        while remaining < ATEMPO_LIMIT:
            a_filters.append("atempo=0.5")
            remaining /= ATEMPO_LIMIT
        a_filters.append(f"atempo={remaining:.3f}")
        a_filter = ",".join(a_filters)
        cmd += ["-filter:a", a_filter, "-c:a", "aac"]
    else:
        cmd += ["-c:a", "copy"]

    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        "18",
        str(out_path),
        "-y",
    ]
    run_cmd(cmd)


def process_video(input_path: Path, args: argparse.Namespace) -> int:
    """Process a single video according to *args*."""

    if input_path.suffix.lower() not in VIDEO_EXTS:
        _LOG.info("Skipping %s (not a video file)", input_path.name)
        return 0

    if not input_path.exists():
        _LOG.error("Input file not found: %s", input_path)
        return 1

    print(f"Processing video: {input_path.name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = input_path.stem
    stitched_path = output_dir / f"{base}_swings.mp4"
    meta_path = output_dir / f"{base}_swings.json"

    if stitched_path.exists() or meta_path.exists():
        _LOG.info("Skipping %s (already processed)", input_path.name)
        return 0

    meta = probe(input_path)
    _LOG.info(
        "Video fps=%.2f res=%s audio=%s",
        meta["fps"],
        meta["resolution"],
        meta["audio_codec"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        detection_config = DetectionConfig(
            audio_model=Path(args.audio_model),
            shot_model=Path(args.shot_model) if args.shot_model else None,
            shot_type_model=(
                Path(args.shot_type_model) if args.shot_type_model else None
            ),
            device=args.device,
        )
        detections = detect_user_swings_for_legacy(
            input_path, detection_config, meta
        )
        swings = [
            Swing(
                index=details.swing.ordinal,
                start=details.start,
                end=details.end,
                contact=details.legacy_contact,
                crop=details.crop,
                label=details.swing.shot_type,
            )
            for details in detections
        ]

        if not swings:
            _LOG.warning("No swings detected")
            move_input_video_to_output(input_path, output_dir)
            return 0

        clip_paths: List[Path] = []
        swing_extraction_times: List[float] = []
        for swing in swings:
            _LOG.info(
                "Extracting swing %d: %.2f - %.2f (contact %.2f)",
                swing.index,
                swing.start,
                swing.end,
                swing.contact,
            )
            extraction_started_at = time.perf_counter()
            out_tmp = tmpdir_path / f"swing_{swing.index}.mp4"
            cut_swing(
                input_path,
                swing.start,
                swing.end,
                out_tmp,
                swing.crop,
                slowmo=args.slowmo,
            )
            extraction_elapsed = time.perf_counter() - extraction_started_at
            swing_extraction_times.append(extraction_elapsed)
            print(
                f"Swing {swing.index} extracted in {extraction_elapsed:.3f}s "
                f"({swing.start:.3f}s-{swing.end:.3f}s)"
            )
            clip_paths.append(out_tmp)

        if swing_extraction_times:
            avg_extraction_time = sum(swing_extraction_times) / len(
                swing_extraction_times
            )
            print(
                f"Average extraction time per swing: {avg_extraction_time:.3f}s "
                f"({len(swing_extraction_times)} swings)"
            )

        if args.clips:
            for i, src_path in enumerate(clip_paths):
                dest = output_dir / f"{base}_swing{i}.mp4"
                shutil.move(src_path, dest)
                clip_paths[i] = dest

        label_groups: dict[str, List[Path]] = {}
        for swing, path in zip(swings, clip_paths):
            key = swing.label or "shot"
            label_groups.setdefault(key, []).append(path)

        if not args.no_stitch:
            if args.shot_type_model and args.slowmo:
                for label, paths in label_groups.items():
                    _LOG.info("Stitching %s swings", label)
                    concat_file = tmpdir_path / f"concat_{label}.txt"
                    with open(concat_file, "w") as fh:
                        for p in paths:
                            fh.write(f"file '{p.resolve()}'\n")
                    out_file = output_dir / f"{base}_{label}_slow{args.slowmo}x.mp4"
                    run_cmd(
                        [
                            "ffmpeg",
                            "-f",
                            "concat",
                            "-safe",
                            "0",
                            "-i",
                            str(concat_file),
                            "-c",
                            "copy",
                            str(out_file),
                            "-y",
                        ]
                    )
            else:
                _LOG.info("Stitching swings")
                concat_file = tmpdir_path / "concat.txt"
                with open(concat_file, "w") as fh:
                    for p in clip_paths:
                        fh.write(f"file '{p.resolve()}'\n")
                run_cmd(
                    [
                        "ffmpeg",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_file),
                        "-c",
                        "copy",
                        str(stitched_path),
                        "-y",
                    ]
                )

        if not args.no_metadata:
            records = [
                {
                    "index": sw.index,
                    "start": sw.start,
                    "end": sw.end,
                    "contact": sw.contact,
                    "crop": sw.crop,
                    "label": sw.label,
                }
                for sw in swings
            ]
            with open(meta_path, "w") as fh:
                json.dump(
                    {
                        "video": str(input_path.name),
                        "sample_rate": SAMPLE_RATE,
                        "swings": records,
                    },
                    fh,
                    indent=2,
                )

    move_input_video_to_output(input_path, output_dir)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    setup_logging(args)
    validate_args(args)
    check_ffmpeg()

    input_path = Path(args.input)

    if input_path.is_dir():
        rc = 0
        for path in sorted(input_path.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                result = process_video(path, args)
                rc = rc or result
        return rc

    return process_video(input_path, args)


if __name__ == "__main__":
    main()

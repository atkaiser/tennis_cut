#!/usr/bin/env python3
"""CLI tool for extracting tennis swings from a video.

This first version implements the pipeline from ``spec.md`` section 4 but skips
vision-based verification and cropping. Candidate swing windows are generated
purely from audio peaks.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import ffmpeg
import torchaudio


_LOG = logging.getLogger(__name__)


@dataclass
class Swing:
    index: int
    start: float
    end: float
    contact: float
    crop: Sequence[int] | None = None


class PopDetector:
    """Very small audio peak detector placeholder.

    The real implementation would load a trained PyTorch model. For now we simply
    look for RMS peaks in 0.25 s windows using a stride of 0.05 s.
    """

    def __init__(self, stride_s: float = 0.05) -> None:
        self.stride_s = stride_s

    def find_impacts(self, wav_path: Path) -> List[float]:
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != 48_000:
            waveform = torchaudio.functional.resample(waveform, sr, 48_000)
            sr = 48_000
        samples = waveform.shape[1]
        window = int(sr * 0.25)
        stride = int(sr * self.stride_s)
        if samples < window:
            return []
        unfolded = waveform.squeeze(0).unfold(0, window, stride)
        rms = (unfolded ** 2).mean(dim=1).sqrt()
        thresh = rms.mean() + 2 * rms.std()
        peaks: List[float] = []
        for i in range(len(rms)):
            if rms[i] < thresh:
                continue
            prev = rms[i - 1] if i > 0 else 0.0
            next_ = rms[i + 1] if i + 1 < len(rms) else 0.0
            if rms[i] >= prev and rms[i] >= next_:
                center = i * self.stride_s + 0.125
                peaks.append(float(center))
        _LOG.info("Detected %d audio peaks", len(peaks))
        return peaks


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract tennis swings from video")
    p.add_argument("input", help="Input video file")
    p.add_argument("-o", "--output-dir", default="./out/", help="Output directory")
    p.add_argument("--clips", action="store_true", help="Export each swing separately")
    p.add_argument("--slowmo", choices=["0.5", "0.25"], nargs="*", help="Generate slow-motion version(s)")
    p.add_argument("--metadata", action="store_true", help="Write JSON manifest")
    p.add_argument("--no-stitch", action="store_true", help="Skip the merged video")
    p.add_argument("--tracker", action="store_true", help="Use SORT tracker (unused)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p.add_argument("-q", "--quiet", action="store_true", help="Errors only")
    return p.parse_args(argv)


def setup_logging(args: argparse.Namespace) -> None:
    level = logging.WARNING
    if args.verbose:
        level = logging.INFO
    if args.quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Install it first (e.g. `brew install ffmpeg`).", file=sys.stderr)
        sys.exit(1)


def probe(video: Path) -> dict:
    try:
        meta = ffmpeg.probe(str(video))
    except ffmpeg.Error as e:
        print(e.stderr.decode(), file=sys.stderr)
        raise
    v_stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    a_stream = next(s for s in meta["streams"] if s["codec_type"] == "audio")
    fps_parts = v_stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])
    return {
        "fps": fps,
        "resolution": (v_stream["width"], v_stream["height"]),
        "audio_codec": a_stream.get("codec_name"),
    }


def extract_audio(video: Path, wav_path: Path) -> None:
    (
        ffmpeg.input(str(video))
        .output(str(wav_path), ac=1, ar=48_000)
        .overwrite_output()
        .run(quiet=True)
    )


def merge_windows(windows: List[tuple[float, float]]) -> List[tuple[float, float]]:
    if not windows:
        return []
    windows.sort(key=lambda w: w[0])
    merged = [list(windows[0])]
    for start, end in windows[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def cut_swing(video: Path, start: float, end: float, out_path: Path) -> None:
    (
        ffmpeg.input(str(video))
        .output(str(out_path), ss=start, to=end, c="copy")
        .overwrite_output()
        .run(quiet=True)
    )


def slowmo_video(src: Path, dst: Path, factor: float) -> None:
    if factor == 0.5:
        v_filter = "setpts=2.0*PTS"
        a_filter = "atempo=0.5"
    else:  # 0.25
        v_filter = "setpts=4.0*PTS"
        a_filter = "atempo=0.5,atempo=0.5"
    (
        ffmpeg.input(str(src))
        .filter_("fps", fps=30)  # ensure compatibility
        .output(str(dst), **{"filter:v": v_filter, "filter:a": a_filter})
        .overwrite_output()
        .run(quiet=True)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    setup_logging(args)
    check_ffmpeg()

    input_path = Path(args.input)
    if not input_path.exists():
        _LOG.error("Input file not found: %s", input_path)
        return 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = input_path.stem
    stitched_path = output_dir / f"{base}_swings.mp4"
    meta_path = output_dir / f"{base}_swings.json"
    slow_paths = {
        float(f): output_dir / f"{base}_swings_slow{f}x.mp4" for f in (args.slowmo or [])
    }

    candidates = []
    if not args.no_stitch:
        candidates.append(stitched_path)
    if args.metadata:
        candidates.append(meta_path)
    for f in slow_paths.values():
        candidates.append(f)
    if args.clips:
        # Only check enough swing filenames lazily after we know swing count
        pass
    for path in candidates:
        if path.exists():
            _LOG.error("Output file %s already exists", path)
            return 2

    meta = probe(input_path)
    _LOG.info("Video fps=%.2f res=%s audio=%s", meta["fps"], meta["resolution"], meta["audio_codec"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        wav_path = tmpdir_path / "audio.wav"
        extract_audio(input_path, wav_path)
        detector = PopDetector()
        impact_times = detector.find_impacts(wav_path)
        candidate_windows = [(t - 1.20, t + 0.70) for t in impact_times]
        merged = merge_windows(candidate_windows)
        swings: List[Swing] = []
        for i, (start, end) in enumerate(merged):
            swings.append(Swing(index=i, start=start, end=end, contact=(start + 1.20)))

        if not swings:
            _LOG.warning("No swings detected")
            if args.metadata:
                with open(meta_path, "w") as fh:
                    json.dump({"video": str(input_path.name), "sample_rate": 48_000, "swings": []}, fh, indent=2)
            return 0

        clip_paths: List[Path] = []
        for swing in swings:
            out_tmp = tmpdir_path / f"swing_{swing.index}.mp4"
            cut_swing(input_path, swing.start, swing.end, out_tmp)
            clip_paths.append(out_tmp)

        if args.clips:
            for i, src_path in enumerate(clip_paths):
                dest = output_dir / f"{base}_swing{i}.mp4"
                if dest.exists():
                    _LOG.error("Output file %s already exists", dest)
                    return 2
                shutil.move(src_path, dest)
                clip_paths[i] = dest

        if not args.no_stitch:
            concat_file = tmpdir_path / "concat.txt"
            with open(concat_file, "w") as fh:
                for p in clip_paths:
                    fh.write(f"file '{p}'\n")
            (
                ffmpeg.input(str(concat_file), format="concat", safe=0)
                .output(str(stitched_path), c="copy")
                .overwrite_output()
                .run(quiet=True)
            )

        if args.slowmo:
            base_input = stitched_path if not args.no_stitch else clip_paths[0]
            for factor_str in args.slowmo:
                factor = float(factor_str)
                dst = slow_paths[factor]
                slowmo_video(base_input, dst, factor)

        if args.metadata:
            records = [
                {
                    "index": sw.index,
                    "start": sw.start,
                    "end": sw.end,
                    "contact": sw.contact,
                    "crop": sw.crop,
                }
                for sw in swings
            ]
            with open(meta_path, "w") as fh:
                json.dump({"video": str(input_path.name), "sample_rate": 48_000, "swings": records}, fh, indent=2)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # pragma: no cover - emergency guard
        with open("tennis_cut_error.log", "w") as fh:
            logging.exception("Unhandled error", exc_info=True, file=fh)
        sys.exit(99)


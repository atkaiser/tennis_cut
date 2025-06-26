#!/usr/bin/env python3
"""
CLI tool for extracting tennis swings from a video.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import subprocess
import torchaudio


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_STRIDE_S = 0.05
SAMPLE_RATE = 48_000
WINDOW_DURATION = 0.25
PEAK_THRESHOLD = 0.5
PEAK_MIN_SEPARATION = 2.0
BATCH_SIZE = 128
ATEMPO_HALF = 0.5
PRE_CONTACT_BUFFER = 1.20
POST_CONTACT_BUFFER = 0.70


_LOG = logging.getLogger(__name__)


def run_cmd(cmd: Sequence[str]) -> None:
    """Run a subprocess and raise with logged output on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        joined = " ".join(cmd)
        _LOG.error("Command failed (%s): %s", result.returncode, joined)
        if result.stdout:
            _LOG.error(result.stdout.strip())
        if result.stderr:
            _LOG.error(result.stderr.strip())
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


@dataclass
class Swing:
    index: int
    start: float
    end: float
    contact: float
    crop: Sequence[int] | None = None


class PopDetector:
    """Audio impact detector using the trained CNN."""

    def __init__(self, model_path: Path, stride_s: float = DEFAULT_STRIDE_S, device: str | None = None) -> None:
        import torch
        from fastai.learner import load_learner

        self.stride_s = stride_s
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message="load_learner` uses Python's insecure pickle")
            learner = load_learner(model_path, cpu=self.device.type == "cpu")
        learner.to(self.device)
        learner.model.eval()
        self.learner = learner


    def find_impacts(self, wav_path: Path) -> List[float]:
        import torch
        import pandas as pd

        waveform, sr = torchaudio.load(str(wav_path))
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE

        window = int(sr * WINDOW_DURATION)
        stride = int(sr * self.stride_s)
        if waveform.shape[1] < window:
            return []

        starts = [sample_start / sr for sample_start in range(0, waveform.shape[1] - window + 1, stride)]
        df = pd.DataFrame({"wav_path": str(wav_path), "start": starts})
        dl = self.learner.dls.test_dl(df, bs=BATCH_SIZE)

        with torch.no_grad():
            preds, _ = self.learner.get_preds(dl=dl, reorder=False)
            probs = preds[:, 1]

        candidates: List[tuple[float, float]] = []
        for i, p in enumerate(probs):
            score = float(p)
            if score > PEAK_THRESHOLD:
                center = i * self.stride_s + (WINDOW_DURATION/2)
                candidates.append((center, score))

        # Non-max suppression: only keep the highest-scoring peak in any
        # two-second window so that all swings have equal duration.
        candidates.sort(key=lambda c: c[1], reverse=True)
        kept: List[tuple[float, float]] = []
        for timestamp, score in candidates:
            if all(abs(timestamp - kept_timestamp) >= PEAK_MIN_SEPARATION for kept_timestamp, _ in kept):
                kept.append((timestamp, score))
        kept.sort(key=lambda c: c[0])

        peaks = [timestamp for timestamp, _ in kept]
        _LOG.info("Detected %d audio peaks", len(peaks))
        _LOG.info("Detected peaks: " + ", ".join(f"{p:.3f}" for p in peaks))
        return peaks


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract tennis swings from video")
    p.add_argument("input", help="Input video file")
    p.add_argument("-o", "--output-dir", default="./out/", help="Output directory")
    p.add_argument("--model", required=True, help="Path to trained audio model")
    p.add_argument("--clips", action="store_true", help="Export each swing separately")
    p.add_argument(
        "--slowmo",
        type=float,
        nargs="*",
        help="Generate slow-motion version(s); e.g. 0.5 for half speed",
    )
    p.add_argument("--metadata", action="store_true", help="Write JSON manifest")
    p.add_argument("--no-stitch", action="store_true", help="Skip the merged video")
    p.add_argument("--tracker", action="store_true", help="Use SORT tracker (unused)")
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


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Install it first (e.g. `brew install ffmpeg`).", file=sys.stderr)
        sys.exit(1)


def probe(video: Path) -> dict:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        raise
    meta = json.loads(result.stdout)
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
    run_cmd(
        [
            "ffmpeg",
            "-i",
            str(video),
            "-ac",
            "1",
            "-ar",
            "48000",
            str(wav_path),
            "-y",
        ]
    )



def cut_swing(video: Path, start: float, end: float, out_path: Path) -> None:
    run_cmd(
        [
            "ffmpeg",
            "-i",
            str(video),
            "-ss",
            str(start),
            "-to",
            str(end),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-c:a",
            "copy",
            str(out_path),
            "-y",
        ]
    )


def slowmo_video(src: Path, dst: Path, factor: float) -> None:
    """
    Re-encode *src* at 1/factor speed (factor ∈ (0, 1]).
    factor = 0.5  →  half-speed
    factor = 0.25 →  quarter-speed
    """
    ATEMPO_LIMIT = 0.5

    if not 0 < factor <= 1:
        raise ValueError("factor must be in (0, 1]")

    # -------- audio tempo chain (0.5–2.0 per stage) ----------
    remaining = factor
    a_filters = []
    while remaining < ATEMPO_LIMIT:       # split into 0.5× pieces
        a_filters.append("atempo=0.5")
        remaining /= ATEMPO_LIMIT
    a_filters.append(f"atempo={remaining:.3f}")
    a_filter = ",".join(a_filters)

    # -------- video filter – stretch presentation timestamps ----
    v_filter = f"setpts={1/factor:.6f}*PTS"   # e.g. factor 0.5 ⇒ 2.0*PTS

    # -------- build the ffmpeg command -------------------------
    run_cmd([
        "ffmpeg", "-i", str(src),
        "-vf", v_filter,            # slow the video
        "-af", a_filter,            # slow the audio
        "-c:v", "libx264", "-crf", "20",
        "-c:a", "aac",
        "-movflags", "+faststart",  # Puts metadata at the start of the video
        str(dst), "-y",
    ])


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
        detector = PopDetector(Path(args.model), device=args.device)
        impact_times = detector.find_impacts(wav_path)
        candidate_windows = [
            (t - PRE_CONTACT_BUFFER, t + POST_CONTACT_BUFFER) for t in impact_times
        ]
        swings: List[Swing] = []
        for i, (start, end) in enumerate(candidate_windows):
            swings.append(
                Swing(index=i, start=start, end=end, contact=(start + PRE_CONTACT_BUFFER))
            )

        if not swings:
            _LOG.warning("No swings detected")
            return 0

        clip_paths: List[Path] = []
        for swing in swings:
            _LOG.info(
                "Extracting swing %d: %.2f - %.2f (contact %.2f)",
                swing.index,
                swing.start,
                swing.end,
                swing.contact,
            )
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

        if args.slowmo:
            base_input = stitched_path if not args.no_stitch else clip_paths[0]
            for factor_str in args.slowmo:
                _LOG.info("Generating slowmo %s", factor_str)
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
                json.dump({"video": str(input_path.name), "sample_rate": SAMPLE_RATE, "swings": records}, fh, indent=2)

    return 0


if __name__ == "__main__":
    main()


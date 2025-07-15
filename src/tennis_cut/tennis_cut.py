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
from typing import List, Sequence, Tuple

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


class PersonDetector:
    """Wrapper around YOLOv8-n person detector."""

    def __init__(self, device: str) -> None:
        from ultralytics import YOLO

        self.device = device
        self.model = YOLO("yolov8n.pt")

    def find_box(self, img_path: Path) -> Tuple[int, int, int, int] | None:
        """Return the largest person box as (x1, y1, x2, y2) integers."""

        results = self.model.predict(
            str(img_path), classes=0, device=self.device, verbose=False
        )
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy.numel() == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = areas.argmax()
        x1, y1, x2, y2 = xyxy[idx]
        return int(x1), int(y1), int(x2), int(y2)


def extract_frame(video: Path, time: float, out_path: Path) -> None:
    """Extract a single frame from *video* at *time* seconds."""

    run_cmd([
        "ffmpeg",
        "-ss",
        str(time),
        "-i",
        str(video),
        "-frames:v",
        "1",
        str(out_path),
        "-y",
    ])


def expand_box(box: Tuple[int, int, int, int], res: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Pad box by 20% and fit to the video aspect ratio."""

    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    w *= 1.2
    h *= 1.2
    frame_w, frame_h = res
    aspect = frame_w / frame_h
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    w = int(round(w))
    h = int(round(h))
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x1 + w > frame_w:
        x1 = max(0, frame_w - w)
    if y1 + h > frame_h:
        y1 = max(0, frame_h - h)
    return x1, y1, w, h


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract tennis swings from video")
    p.add_argument("input", help="Input video file")
    p.add_argument("-o", "--output-dir", default="./out/", help="Output directory")
    p.add_argument("--model", required=True, help="Path to trained audio model")
    p.add_argument("--clips", action="store_true", help="Export each swing separately")
    p.add_argument(
        "--slowmo",
        type=float,
        help="Generate a slow-motion version; e.g. 0.5 for half speed",
    )
    p.add_argument("--metadata", action="store_true", help="Write JSON manifest")
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
        v_filters.append(f"setpts={1/slowmo:.6f}*PTS")

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

    if not input_path.exists():
        _LOG.error("Input file not found: %s", input_path)
        return 1

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
        wav_path = tmpdir_path / "audio.wav"
        extract_audio(input_path, wav_path)
        detector = PopDetector(Path(args.model), device=args.device)
        impact_times = detector.find_impacts(wav_path)
        person_detector = PersonDetector(args.device)

        swings: List[Swing] = []
        for i, t in enumerate(impact_times):
            start = t - PRE_CONTACT_BUFFER
            end = t + POST_CONTACT_BUFFER
            frame_path = tmpdir_path / f"impact_{i}.jpg"
            extract_frame(input_path, t, frame_path)
            box = person_detector.find_box(frame_path)
            if box is None:
                _LOG.info("No person found for impact %d", i)
                continue
            crop = expand_box(box, meta["resolution"])
            swings.append(
                Swing(
                    index=len(swings),
                    start=start,
                    end=end,
                    contact=t,
                    crop=crop,
                )
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
            cut_swing(
                input_path,
                swing.start,
                swing.end,
                out_tmp,
                swing.crop,
                slowmo=args.slowmo,
            )
            clip_paths.append(out_tmp)

        if args.clips:
            for i, src_path in enumerate(clip_paths):
                dest = output_dir / f"{base}_swing{i}.mp4"
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
                json.dump(
                    {
                        "video": str(input_path.name),
                        "sample_rate": SAMPLE_RATE,
                        "swings": records,
                    },
                    fh,
                    indent=2,
                )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    setup_logging(args)
    check_ffmpeg()

    input_path = Path(args.input)

    if input_path.is_dir():
        rc = 0
        for path in sorted(input_path.iterdir()):
            if path.is_file():
                result = process_video(path, args)
                rc = rc or result
        return rc

    return process_video(input_path, args)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""prepare_swing_frames.py
-------------------------
Extract labelled frames for training the swing classification model.

The script expects a directory of tennis videos each with a companion ``.json``
file containing a ``"shots"`` list of dictionaries with ``"time"`` (in seconds)
and ``"type"`` (``forehand``/``backhand``/``volley``/``serve``).

For every shot frame, an image is extracted using ``ffmpeg`` and stored in a
subdirectory named after the shot type. Additional ``no_shot`` frames are
sampled near each shot to balance the dataset.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys
from tqdm import tqdm
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utilities import PersonDetector, expand_box, probe_duration

# -----------------------  constants / defaults  -----------------------
NEG_PER_POS = 1            # number of no_shot samples per positive
MIN_OFF     = 1            # negatives are at least this far from impact
MAX_OFF     = 2            # and at most this far from impact


def extract_frame(mp4_path: pathlib.Path, timestamp: float, out_path: pathlib.Path, detector: PersonDetector, resolution: tuple[int, int]) -> None:
    """Extract a single frame using ffmpeg and crop to the detected person."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_frame = out_path.parent / f"temp_{out_path.name}"
    cmd = [
        "ffmpeg",
        "-hide_banner",  # Suppress verbose output
        "-v",
        "error",
        "-ss",
        f"{timestamp}",
        "-i",
        str(mp4_path),
        "-frames:v",
        "1",
        str(temp_frame),
        "-y",
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e.stderr.strip()}")
        raise

    box = detector.find_box(temp_frame)
    if box is not None:
        x1, y1, w, h = expand_box(box, resolution)
        crop_cmd = [
            "ffmpeg",
            "-hide_banner",  # Suppress verbose output
            "-i",
            str(temp_frame),
            "-vf",
            f"crop={w}:{h}:{x1}:{y1}",
            str(out_path),
            "-y",
        ]
        try:
            subprocess.run(crop_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during ffmpeg crop execution: {e.stderr.strip()}")
            raise
    else:
        temp_frame.rename(out_path)  # Save the uncropped frame if no person is detected

    temp_frame.unlink(missing_ok=True)


def sample_neg(t: float) -> float:
    """Return a timestamp slightly away from ``t`` for a no_shot sample."""
    off = random.uniform(MIN_OFF, MAX_OFF)
    return t + random.choice([-off, off])


def probe_video_metadata(video: pathlib.Path) -> dict:
    """Retrieve video metadata including resolution using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    meta = json.loads(result.stdout)
    v_stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    return {
        "resolution": (v_stream["width"], v_stream["height"]),
    }


# ----------------------------------------------------------------------
def main(videos_dir: str, out_dir: str, neg_per_pos: int) -> None:
    """Process ``videos_dir`` and write labelled frames to ``out_dir``."""

    videos = sorted(pathlib.Path(videos_dir).glob("*.MOV")) + \
             sorted(pathlib.Path(videos_dir).glob("*.mp4")) + \
             sorted(pathlib.Path(videos_dir).glob("*.mov"))

    pos_count = 0
    neg_count = 0
    detector = PersonDetector(device="mps")  # Default to MPS for macOS

    # Calculate total shots to process
    total_shots = 0
    for mp4 in videos:
        json_path = mp4.with_suffix(".json")
        if not json_path.exists():
            continue

        data = json.load(open(json_path))
        shots = data.get("shots", [])
        if not shots:
            continue

        for shot in shots:
            t = float(shot.get("time"))
            label = shot.get("type")
            if label is None:
                continue
            # Check if the main frame for this shot already exists
            main_frame_path = pathlib.Path(out_dir) / label / f"{mp4.stem}_{t:.3f}.jpg"
            if not main_frame_path.exists():
                total_shots += 1

    # Initialize tqdm progress bar
    with tqdm(total=total_shots, desc="Processing shots", unit="shot") as pbar:
        for mp4 in videos:
            json_path = mp4.with_suffix(".json")
            if not json_path.exists():
                continue

            data = json.load(open(json_path))
            shots = data.get("shots", [])
            if not shots:
                continue

            duration = probe_duration(mp4)
            meta = probe_video_metadata(mp4)
            resolution = meta["resolution"]
            # Gather all shot times
            shot_times = [float(shot.get("time")) for shot in shots if shot.get("type") is not None]

            for shot in shots:
                t = float(shot.get("time"))
                label = shot.get("type")
                if label is None:
                    continue
                # Extract three frames: t with random offsets
                random_offsets = [random.uniform(-0.06, -0.01), 0, random.uniform(0.01, 0.04)]
                for dt in random_offsets:
                    frame_t = t + dt
                    if 0 <= frame_t <= duration:
                        out_path = pathlib.Path(out_dir) / label / f"{mp4.stem}_{frame_t:.3f}.jpg"
                        if out_path.exists():
                            continue  # Skip if the frame has already been extracted
                        extract_frame(mp4, frame_t, out_path, detector, resolution)
                        pos_count += 1
                pbar.update(1)  # Update progress bar for each shot processed

            neg_count += sample_negatives(
                mp4=mp4,
                duration=duration,
                shot_times=shot_times,
                out_dir=out_dir,
                num_neg=int(neg_per_pos * len(shot_times)),
                detector=detector,
                resolution=resolution
            )

    print(f"✅  Wrote {pos_count} shot frames and {neg_count} negatives to {out_dir}")


def sample_negatives(mp4: pathlib.Path, duration: float, shot_times: list[float], out_dir: str, num_neg: int, detector: PersonDetector, resolution: tuple[int, int], min_dist: float = 1.25) -> int:
    """Sample negatives from intervals at least min_dist away from any shot."""
    shot_times_sorted = sorted(shot_times)
    intervals = []
    last_end = 0.0
    for st in shot_times_sorted:
        start = max(last_end, st - min_dist)
        end = st + min_dist
        if start > last_end and start > 0:
            intervals.append((last_end, start))
        last_end = end
    if last_end < duration:
        intervals.append((last_end, duration))

    # Remove intervals that are too short to sample from
    intervals = [(a, b) for a, b in intervals if b - a > 0]
    neg_count = 0
    lengths = [b - a for a, b in intervals]
    for _ in range(num_neg):
        if not lengths or sum(lengths) == 0:
            break
        chosen = random.choices(intervals, weights=lengths, k=1)[0]
        neg_t = random.uniform(chosen[0], chosen[1])
        n_path = pathlib.Path(out_dir) / "no_shot" / f"{mp4.stem}_{neg_t:.3f}.jpg"
        extract_frame(mp4, neg_t, n_path, detector, resolution)
        neg_count += 1
    return neg_count


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract labelled frames for swing classification"
    )
    ap.add_argument(
        "--videos_dir",
        type=str,
        default="videos",
        help="directory with labelled video files (.mp4/.MOV) and matching JSON",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="dataset",
        help="directory to write extracted frames organised by label",
    )
    ap.add_argument(
        "--neg-per-pos",
        type=int,
        default=NEG_PER_POS,
        metavar="N",
        help="number of no_shot frames to sample per positive",
    )
    args = ap.parse_args()
    try:
        main(args.videos_dir, args.out_dir, args.neg_per_pos)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg/ffprobe failed: {e}")

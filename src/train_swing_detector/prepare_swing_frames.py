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

# -----------------------  constants / defaults  -----------------------
NEG_PER_POS = 1            # number of no_shot samples per positive
MIN_OFF     = 0.3          # negatives are at least this far from impact
MAX_OFF     = 0.7          # and at most this far from impact

# ----------------------------------------------------------------------
def probe_duration(path: pathlib.Path) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def extract_frame(mp4_path: pathlib.Path, timestamp: float, out_path: pathlib.Path) -> None:
    """Extract a single frame using ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{timestamp}",
        "-i",
        str(mp4_path),
        "-frames:v",
        "1",
        str(out_path),
        "-y",
    ]
    subprocess.run(cmd, check=True)


def sample_neg(t: float) -> float:
    """Return a timestamp slightly away from ``t`` for a no_shot sample."""
    off = random.uniform(MIN_OFF, MAX_OFF)
    return t + random.choice([-off, off])


# ----------------------------------------------------------------------
def main(videos_dir: str, out_dir: str, neg_per_pos: int) -> None:
    """Process ``videos_dir`` and write labelled frames to ``out_dir``."""

    videos = sorted(pathlib.Path(videos_dir).glob("*.MOV")) + \
             sorted(pathlib.Path(videos_dir).glob("*.mp4")) + \
             sorted(pathlib.Path(videos_dir).glob("*.mov"))

    pos_count = 0
    neg_count = 0

    for mp4 in videos:
        json_path = mp4.with_suffix(".json")
        if not json_path.exists():
            print(f"No JSON for {mp4.name}; skipping.")
            continue

        data = json.load(open(json_path))
        shots = data.get("shots", [])
        if not shots:
            continue

        duration = probe_duration(mp4)

        for shot in shots:
            t = float(shot.get("time"))
            label = shot.get("type")
            if label is None:
                continue
            out_path = pathlib.Path(out_dir) / label / f"{mp4.stem}_{t:.3f}.jpg"
            extract_frame(mp4, t, out_path)
            pos_count += 1

            for _ in range(neg_per_pos):
                neg_t = sample_neg(t)
                if 0 < neg_t < duration:
                    n_path = pathlib.Path(out_dir) / "no_shot" / f"{mp4.stem}_{neg_t:.3f}.jpg"
                    extract_frame(mp4, neg_t, n_path)
                    neg_count += 1

    print(f"✅  Wrote {pos_count} shot frames and {neg_count} negatives to {out_dir}")


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

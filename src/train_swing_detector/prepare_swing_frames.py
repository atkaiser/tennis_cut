#!/usr/bin/env python3
"""prepare_swing_frames.py
------------------------
Extract labelled frames from annotated videos for training the swing
classifier.

The script expects a directory of tennis videos where each ``.mp4`` or ``.MOV``
has a companion ``.json`` file. The JSON file must contain a ``"shots"`` array
listing the impact ``time`` (in seconds) and ``type`` (forehand, backhand,
volley, serve).

For every shot a single frame is extracted at the impact time. Additional
negative frames are sampled before or after the impact and further away in the
video to balance the dataset. All frames are written to sub‑folders under the
output directory named after the label. A CSV describing the extracted frames is
also produced so that ``train_swing_classifier.py`` can easily load the data.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import subprocess
import sys
from typing import Iterable, List, Tuple

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
# negative sampling parameters (seconds)
NEAR_MIN_OFF = 0.15
FAR_MIN_GAP = 1.0
FAR_MAX_GAP = 2.0

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
LABELS = ["forehand", "backhand", "volley", "serve", "no_shot"]

# ---------------------------------------------------------------------------

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


def extract_frame(video: pathlib.Path, time: float, out: pathlib.Path) -> None:
    """Extract a single frame at ``time`` seconds using ffmpeg."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-ss",
        f"{time:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        str(out),
        "-y",
    ]
    subprocess.run(cmd, check=True)


def sample_near(t: float) -> float:
    off = random.uniform(NEAR_MIN_OFF, NEAR_MIN_OFF * 2)
    return t + random.choice([-off, off])


def sample_far(t: float) -> float:
    gap = random.uniform(FAR_MIN_GAP, FAR_MAX_GAP)
    return t + random.choice([-gap, gap])


# ---------------------------------------------------------------------------

def gather_videos(directory: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in VIDEO_EXTS:
            yield path


# ---------------------------------------------------------------------------

def main(
    videos_dir: str,
    frames_dir: str,
    out_csv: str,
    neg_per_pos: int,
    far_neg_per_pos: int,
) -> None:
    vids = list(gather_videos(pathlib.Path(videos_dir)))
    rows: List[Tuple[str, str]] = []
    idx = 0
    for vid in vids:
        js = vid.with_suffix(".json")
        if not js.exists():
            print(f"No JSON for {vid.name}; skipping.")
            continue
        try:
            data = json.load(open(js))
        except Exception as e:
            print(f"Failed to load {js}: {e}; skipping.")
            continue
        shots = data.get("shots")
        if not isinstance(shots, list):
            print(f"No shots in {js.name}; skipping.")
            continue
        duration = probe_duration(vid)
        for shot in shots:
            try:
                t = float(shot["time"])
                typ = str(shot["type"])
            except Exception:
                continue
            if typ not in LABELS:
                print(f"Unknown label '{typ}' in {js.name}; skipping.")
                continue
            # positive frame
            out_path = pathlib.Path(frames_dir) / typ / f"{vid.stem}_{idx}.jpg"
            idx += 1
            extract_frame(vid, t, out_path)
            rows.append((str(out_path), typ))
            # near negatives
            for _ in range(neg_per_pos):
                nt = sample_near(t)
                if 0 <= nt <= duration:
                    out_path = (
                        pathlib.Path(frames_dir)
                        / "no_shot"
                        / f"{vid.stem}_{idx}.jpg"
                    )
                    idx += 1
                    extract_frame(vid, nt, out_path)
                    rows.append((str(out_path), "no_shot"))
            # far negatives
            for _ in range(far_neg_per_pos):
                nt = sample_far(t)
                if 0 <= nt <= duration:
                    out_path = (
                        pathlib.Path(frames_dir)
                        / "no_shot"
                        / f"{vid.stem}_{idx}.jpg"
                    )
                    idx += 1
                    extract_frame(vid, nt, out_path)
                    rows.append((str(out_path), "no_shot"))
    random.shuffle(rows)
    pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["img_path", "label"])
        w.writerows(rows)
    print(f"✅  Wrote {len(rows):,} rows ➜ {out_csv}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract labelled frames for swing classification"
    )
    ap.add_argument(
        "--videos_dir",
        type=str,
        default="videos",
        help="directory with labelled video files",
    )
    ap.add_argument(
        "--frames_dir",
        type=str,
        default="frames",
        help="where to store extracted frames grouped by label",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="meta/labled_frames.csv",
        help="CSV listing image paths and labels",
    )
    ap.add_argument(
        "--neg-per-pos",
        type=int,
        default=3,
        metavar="N",
        help="number of near negative frames per shot",
    )
    ap.add_argument(
        "--far-neg-per-pos",
        type=int,
        default=1,
        metavar="N",
        help="number of far negative frames per shot",
    )
    args = ap.parse_args()
    try:
        main(
            args.videos_dir,
            args.frames_dir,
            args.out_csv,
            args.neg_per_pos,
            args.far_neg_per_pos,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg/ffprobe failed: {e}")

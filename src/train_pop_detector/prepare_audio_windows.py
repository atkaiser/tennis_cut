#!/usr/bin/env python3
"""prepare_audio_windows.py
---------------------------
Generate training windows for the audio pop detector.

The script expects a directory of labelled tennis videos. For each ``video``
file there should be a companion ``.json`` file containing an ``"impacts"``
array with the impact timestamps (in seconds). ``ffmpeg`` is used to extract
the audio track at 48 kHz mono and short windows around each timestamp are
written to a CSV file. Additional negative windows are sampled near and far
from each impact to balance the dataset.

Usage example::

   python prepare_audio_windows.py videos/ wav/ meta/train_all.csv \
       --neg-per-pos 3 --far-neg-per-pos 1
"""

import argparse
import csv
import json
import pathlib
import random
import subprocess
import sys
from typing import List, Tuple

# -----------------------  constants / defaults  -----------------------
WIN_SEC          = 0.25         # length of each training window in seconds
SR               = 48_000       # audio sample rate used for extraction
NEG_PER_POS      = 3            # number of near-negative samples per positive
FAR_NEG_PER_POS  = 1            # number of far-negative samples per positive
NEAR_MIN_OFF     = 0.15         # near negatives are at least this far from impact
FAR_MIN_GAP      = 1.0          # far negatives start at least this many seconds away
FAR_MAX_GAP      = 2.0          # and at most this many seconds away

# ----------------------------------------------------------------------
def extract_wav(mp4_path: pathlib.Path, wav_path: pathlib.Path) -> None:
    """Create wav_path if missing; 48 kHz mono PCM."""
    if wav_path.exists():
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(mp4_path),
        "-ac", "1", "-ar", str(SR),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)

def probe_duration(path: pathlib.Path) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)

def sample_near_neg(t: float) -> float:
    """Return a timestamp slightly before or after ``t`` for a near negative."""
    off = random.uniform(NEAR_MIN_OFF, WIN_SEC/2)
    return t + random.choice([-off, off])

def sample_far_neg(t: float) -> float:
    """Return a timestamp well away from ``t`` for a far negative."""
    gap = random.uniform(FAR_MIN_GAP, FAR_MAX_GAP)
    return t + random.choice([-gap, gap])

def row(wav: pathlib.Path, start: float, label: str) -> Tuple[str, float, str]:
    """Format a CSV row describing an audio window."""
    return (str(wav), round(start, 3), label)

# ----------------------------------------------------------------------
def main(videos_dir: str, wav_dir: str, out_csv: str,
         neg_per_pos: int, far_neg_per_pos: int) -> None:
    """Process ``videos_dir`` and write a window CSV to ``out_csv``.

    For every ``.mp4``/``.mov`` file the function loads the corresponding JSON
    labels, extracts the audio to ``wav_dir`` and samples positive, near
    negative and far negative windows.
    """

    videos = sorted(pathlib.Path(videos_dir).glob("*.MOV")) + \
             sorted(pathlib.Path(videos_dir).glob("*.mp4")) + \
             sorted(pathlib.Path(videos_dir).glob("*.mov"))

    rows: List[Tuple[str, float, str]] = []

    for mp4 in videos:
        json_path = mp4.with_suffix(".json")
        if not json_path.exists():
            print(f"⚠  No JSON for {mp4.name}; skipping.")
            continue

        wav_path  = pathlib.Path(wav_dir) / f"{mp4.stem}.wav"
        extract_wav(mp4, wav_path)
        duration  = probe_duration(wav_path)          # seconds

        meta   = json.load(open(json_path))
        for t in meta["impacts"]:
            # positive
            rows.append(row(wav_path, t - WIN_SEC/2, "pos"))

            # near negatives
            for _ in range(neg_per_pos):
                neg_t = sample_near_neg(t)
                if 0 < neg_t < duration - WIN_SEC:
                    rows.append(row(wav_path, neg_t - WIN_SEC/2, "neg"))

            # far negatives
            for _ in range(far_neg_per_pos):
                neg_t = sample_far_neg(t)
                if 0 < neg_t < duration - WIN_SEC:
                    rows.append(row(wav_path, neg_t - WIN_SEC/2, "neg"))

    random.shuffle(rows)
    pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wav_path", "start", "label"])
        w.writerows(rows)

    print(f"✅  Wrote {len(rows):,} rows ➜ {out_csv}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract audio windows from labelled videos and write a CSV"
    )
    ap.add_argument(
        "videos_dir",
        help="directory with labelled video files (.mp4/.MOV) and matching JSON",
    )
    ap.add_argument(
        "wav_dir",
        help="where to store extracted 48 kHz mono wav files",
    )
    ap.add_argument(
        "out_csv",
        help="path to CSV listing wav paths, start times and labels",
    )
    ap.add_argument(
        "--neg-per-pos",
        type=int,
        default=NEG_PER_POS,
        metavar="N",
        help="number of near negatives to sample per positive",
    )
    ap.add_argument(
        "--far-neg-per-pos",
        type=int,
        default=FAR_NEG_PER_POS,
        metavar="N",
        help="number of far negatives to sample per positive",
    )
    args = ap.parse_args()
    try:
        main(args.videos_dir, args.wav_dir, args.out_csv,
             args.neg_per_pos, args.far_neg_per_pos)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg/ffprobe failed: {e}")

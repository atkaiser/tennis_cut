#!/usr/bin/env python3
"""
make_windows.py  –  Prep fast‑ai dataset from labelled videos.

Usage
-----
python make_windows.py  videos/  wav/  meta/train_all.csv  \
       --neg-per-pos 3  --far-neg-per-pos 1
"""

import argparse, csv, json, os, random, subprocess, sys, pathlib
from typing import List, Tuple

# -----------------------  constants / defaults  -----------------------
WIN_SEC          = 0.25         # window length  (s)
SR               = 48_000       # sample‑rate     (Hz)
NEG_PER_POS      = 3            # near negatives  per positive
FAR_NEG_PER_POS  = 1            # far  negatives  per positive
NEAR_MIN_OFF     = 0.15         # 0.15‑0.25  s from impact
FAR_MIN_GAP      = 1.0          # ≥1.0 s  away  (far negatives)
FAR_MAX_GAP      = 2.0          # ≤2.0 s  for far negative sampling

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
    off = random.uniform(NEAR_MIN_OFF, WIN_SEC/2)
    return t + random.choice([-off, off])

def sample_far_neg(t: float) -> float:
    gap = random.uniform(FAR_MIN_GAP, FAR_MAX_GAP)
    return t + random.choice([-gap, gap])

def row(wav: pathlib.Path, start: float, label: str) -> Tuple[str, float, str]:
    return (str(wav), round(start, 3), label)

# ----------------------------------------------------------------------
def main(videos_dir: str, wav_dir: str, out_csv: str,
         neg_per_pos: int, far_neg_per_pos: int) -> None:

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
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_dir")
    ap.add_argument("wav_dir"),
    ap.add_argument("out_csv")
    ap.add_argument("--neg-per-pos",  type=int, default=NEG_PER_POS,
                    help="near negatives per positive (default 3)")
    ap.add_argument("--far-neg-per-pos", type=int, default=FAR_NEG_PER_POS,
                    help="far  negatives per positive (default 1)")
    args = ap.parse_args()
    try:
        main(args.videos_dir, args.wav_dir, args.out_csv,
             args.neg_per_pos, args.far_neg_per_pos)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg/ffprobe failed: {e}")

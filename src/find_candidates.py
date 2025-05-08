#!/usr/bin/env python
import sys, json, pathlib, librosa, numpy as np

AUDIO_SR = 48_000            # expected sample‑rate

def main(wav_path: str):
    y, sr = librosa.load(wav_path, sr=AUDIO_SR, mono=True)
    # onset strength envelope (better than raw RMS)
    env = librosa.onset.onset_strength(y=y, sr=sr)

    peaks = librosa.util.peak_pick(
        env,
        pre_max=3,  post_max=3,      # look ±3 frames for local maximum
        pre_avg=3,  post_avg=3,      # compare against local mean
        delta=0.3,                   # threshold above the mean
        wait=5                       # minimum 5 frames between peaks
    )

    times = librosa.frames_to_time(peaks, sr=sr).round(3).tolist()
    out = pathlib.Path(wav_path).with_suffix('.json')
    json.dump(times, open(out, "w"), indent=2)
    print(f"wrote {len(times)} candidates ➜ {out}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: find_candidates.py <wav-file>")
        sys.exit(1)
    main(sys.argv[1])

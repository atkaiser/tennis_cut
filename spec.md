## 1 – Scope & Success Criteria  
**Goal:** Given an iPhone-shot tennis video, output:

* **Default:** one stitched MP4 containing *only* the player-of-interest’s swings, each clip cropped to the player and zoomed, in original frame-rate, aspect ratio preserved.
* Optional flags:
  * Separate per-swing clips instead of a single file.
  * Optional slow‑motion versions at user-specified factors (e.g. 0.5×).
  * JSON metadata file describing each swing.
* “Swing” = from **1.20 s before contact** to **0.70 s after contact**.
  Only the highest-scoring peak is kept within any two-second window so all
  swings have uniform length.

---

## 2 – System Architecture  

```
 +-------------+        +-------------------+       +-----------------+
 | Input .mp4  | ---->  | Pre-process       | ----> | Audio “pop”     |
 | (H.264/AAC) |        |  • Probe audio SR |       | classifier      |
 +-------------+        |  • Resample→48 kHz|       |  (fastai)       |
                        +-------------------+       +-----------------+
                                                            |
                                   candidate ±0.25 s windows|
                                                            v
                        +-------------------+       +-----------------+
                        | Vision verifier   |       | Swing window    |
                        |  • YOLOv8-n       | ----> | generator       |
                        |    (“person” box) |       +-----------------+
                        |  • Largest box    |               |
                        +-------------------+               v
                                                 crop &        +---------------+
                                                 zoom logic -->| FFmpeg cutter |
                                                                +---------------+
                                                                        |
                                         +------------------+-----------+--------------+
                                         |                                      |
                              stitched output.mp4                   per-swing *.mp4
                                                                                 |
                                                                                 v
                                                  optional slow-mo re-encode (--slowmo)

                                         +--------------------------------------------+
                                         | JSON metadata (<input>_swings.json)       |
                                         +--------------------------------------------+
```

*All heavy compute runs on Apple-silicon GPU via the PyTorch-MPS backend.*

---

## 3 – Core Algorithms & Libraries  

| Stage | Technique | Key libs |
|-------|-----------|----------|
| Audio resample | Auto-detect and convert to **48 kHz/mono** | `torchaudio.resample` |
| **Audio impact detector** | Fast-ai 1-D CNN on 250 ms waveform patches <br>– positives: “pop” frames <br>– negatives: other audio <br>Training set ≈ 60 min (≈ 1 500–2 000 pops) | **fastai**, **PyTorch**, `torchaudio` |
| Candidate windows | For each predicted impact, take ±0.25 s | in-code |
| **Vision verifier** | YOLOv8-n (pre-trained “person”) → accept window if any frame shows a “person” box whose center ≤ 96 px from frame centre (tune) | `ultralytics` YOLOv8 wheel |
| Player selection | Per-frame heuristic: pick **largest box** (fallback: enable SORT tracker behind a `--tracker` flag) | `numpy` |
| Crop box | Union of all accepted boxes in the swing window, padded **10 %** each side, then expanded to match source aspect ratio. |
| Clip merge | Windows that overlap **any amount** are merged. |
| Cutting & stitching | `ffmpeg` concat demuxer. All video is copy-stream-copied; audio kept unchanged. |
| Slow-mo | `ffmpeg` re-encode with reduced frame rate and repeated `atempo=0.5` filters to approximate the factor. |
| Metadata file | `<input>_swings.json`, schema:  ```{ "video": "<file>", "sample_rate": 48000, "swings":[ { "index":0,"start":12.417,"end":14.317,"contact":13.617,"crop":[x,y,w,h] }, … ] }``` |
| CLI framework | `argparse` |

---

## 4 – Command-line Interface  

```
python tennis_cut.py <input.mp4> [options]

Options
  -o, --output-dir DIR      Directory for all outputs (default: ./out/)
  --clips                   Produce individual swing_<N>.mp4 files as well
  --slowmo FACTOR [FACTOR ...]  Also generate slow-motion version(s)
  --metadata                Emit JSON metadata file
  --no-stitch               Skip the stitched video
  --tracker                 Use SORT tracking instead of “largest box”
  -v, --verbose             Info logging
  -q, --quiet               Errors only
```

*Outputs:*

* Stitched: `<basename>_swings.mp4`
* Per-swing: `<basename>_swing<N>.mp4`
* Slow-mo: suffix `_slow<F>x.mp4` where `<F>` is the factor (e.g. `_slow0.5x.mp4`)
* Metadata: `<basename>_swings.json`

**Collision rule:** if any of those filenames already exist in `--output-dir`, abort with an error code ≈ 2.

---

## 5 – Annotation GUI (PySide 6)  

| Feature | Hotkey/UI |
|---------|-----------|
| Play / Pause | Space |
| Frame ±1 | ← / → |
| Jump to next/prev audio peak | N / P |
| Zoom waveform | Mouse wheel |
| Mark / unmark impact | I |
| Undo / Redo | Ctrl-Z / Ctrl-Y |
| Save progress | Ctrl-S (writes JSON schema “A” per video) |
| Resume | drag-drop project file |

Implementation: `QtMultimedia` for synced video & audio playback; `QGraphicsView` for waveform; accelerates to 60 fps on M-GPU.

---

## 6 – Model-training Details  

### 6.1 Audio “pop” classifier  
* **Input**: 0.25 s, mono, 48 kHz (12 000 samples)  
* **Architecture**: 4 × [Conv1d-BN-ReLU-MaxPool] → GlobalAvgPool → 2-unit fc + Softmax.  
* **Loss**: Cross-entropy.  
* **Class balance**: ratio 1:3 positive:negative (sampled).  
* **Augmentations**:  -6 dB – +6 dB gain, random 15 ms white-noise, time-mask 30 ms.  
* **Epochs**: ~15, early-stop on val F1.  
* **Device**: `'mps'`.

### 6.2 Vision verifier (optional fine-tune)  
* Fine-tune YOLOv8-n on 500–1 000 labelled contact/non-contact frames (if needed).  
* Convert to `half=True` for FP16 on M-GPU.

---

## 7 – Error Handling & Logging  

| Condition | Behaviour |
|-----------|-----------|
| Missing `ffmpeg` | Exit 1; print install hint (`brew install ffmpeg`). |
| Filename collision | Exit 2; list colliding files. |
| No swings detected | Warn, exit 0, create empty JSON if `--metadata`. |
| Audio SR not read | Fallback: skip resample, warn. |
| GPU unavailable | Auto-switch to CPU with banner log. |
| Unexpected exception | Exit 99; dump stacktrace to `tennis_cut_error.log`. |

---

## 8 – Testing & QA Plan  

| Level | Tests |
|-------|-------|
| **Unit** |  • Audio resampler idempotence <br>• Impact-window extractor <br>• Crop-box padding/aspect logic <br>• CLI flag parser |
| **Model** |  • 5-fold CV on labeled audio (target F1 ≥ 0.92) <br>• Hold-out vision set accuracy ≥ 0.9 |
| **Integration** |  • Golden sample video → expected # clips & timestamps <br>• Collision path aborts <br>• Slow-mo duration doubles/quadruples |
| **Performance** |  • End-to-end runtime ≤ 1× real-time on M1 Pro (15 min video in ≤ 15 min) |
| **Manual QA** |  • Annotation GUI usability ⟶ label 10 min footage in ≤ 12 min. |
| **Regression** | GitHub Actions: run unit + short integration test on macOS runner using `--no-stitch`. |

---

## 9 – Future-proof / Extensibility Notes  

* Swap **SORT tracking** in by adding `--tracker` (already scaffolded).  
* Switch to **pose keypoints**: drop-in `rtmpose-tiny` and alter crop logic to use joint extents.  
* Allow user-drawn ROI for player of interest (`--roi-calibrate`).  
* Support cloud-GPU batch mode via ONNX export of audio & vision models.

---

## 10 – External Reference  

* AudioSet ontology includes a *“Tennis ball”* impact tag useful for future augmentation.  ([The AudioSet dataset - Google Research](https://research.google.com/audioset/dataset/index.html?utm_source=chatgpt.com))
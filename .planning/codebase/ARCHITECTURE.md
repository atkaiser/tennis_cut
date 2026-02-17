# Architecture

**Analysis Date:** 2026-02-16

## Pattern Overview

**Overall:** Script-first batch pipeline with CLI entry points and offline data prep/training utilities.

**Key Characteristics:**
- Orchestrates external tooling (`ffmpeg`/`ffprobe`) for media I/O.
- Models are trained offline and loaded at runtime via fastai exports.
- Minimal shared library layer in `src/utilities/` with helper abstractions.

## Layers

**CLI Orchestration:**
- Purpose: Run end-to-end swing extraction from raw video.
- Location: `src/tennis_cut/tennis_cut.py`
- Contains: Argument parsing, logging, pipeline orchestration, ffmpeg calls.
- Depends on: `src/utilities/core.py`, `ffmpeg`, `ffprobe`, fastai models.
- Used by: Direct CLI invocation.

**Data Preparation:**
- Purpose: Create datasets from labeled videos.
- Location: `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/prepare_swing_frames.py`
- Contains: Window/frame extraction, CSV generation, negative sampling.
- Depends on: `src/utilities/core.py`, `ffmpeg`, `ffprobe`.
- Used by: Training scripts and model development.

**Model Training:**
- Purpose: Train audio and vision classifiers.
- Location: `src/train_pop_detector/train_audio_pop.py`, `src/train_swing_detector/train_swing_classifier.py`
- Contains: fastai training loops, export of model artifacts.
- Depends on: fastai, torch, torchaudio.
- Used by: Offline training workflows.

**Annotation UI:**
- Purpose: Manual labeling of impacts and shot types.
- Location: `src/label_videos/tennis_annotate.py`, `src/label_videos/label_shots.py`
- Contains: PySide6 GUI, JSON read/write for annotations.
- Depends on: PySide6, ffprobe.
- Used by: Dataset creation and labeling.

**Shared Utilities:**
- Purpose: Central helpers for media probing and person detection.
- Location: `src/utilities/core.py`
- Contains: `PersonDetector`, `expand_box`, `probe_duration`.
- Depends on: ultralytics YOLOv8, ffprobe.
- Used by: `src/tennis_cut/tennis_cut.py`, `src/train_swing_detector/prepare_swing_frames.py`, `src/train_pop_detector/prepare_audio_windows.py`.

## Data Flow

**Swing Extraction Pipeline:**

1. CLI loads input video path(s) in `src/tennis_cut/tennis_cut.py`.
2. Audio extracted to wav via `ffmpeg` in `src/tennis_cut/tennis_cut.py`.
3. `PopDetector.find_impacts()` scores windows and produces peak timestamps in `src/tennis_cut/tennis_cut.py`.
4. `PersonDetector.find_box()` finds a person crop in `src/utilities/core.py`.
5. Optional `SwingDetector` filters non-swing frames in `src/tennis_cut/tennis_cut.py`.
6. `cut_swing()` writes clips and optional slow-mo with ffmpeg in `src/tennis_cut/tennis_cut.py`.
7. Clips stitched and JSON metadata emitted in `src/tennis_cut/tennis_cut.py`.

**Training Data Pipeline:**

1. Manual labeling produces per-video JSON in `src/label_videos/tennis_annotate.py`.
2. Audio windows CSV generated in `src/train_pop_detector/prepare_audio_windows.py`.
3. Audio model trained and exported in `src/train_pop_detector/train_audio_pop.py`.
4. Labeled frames extracted and cropped in `src/train_swing_detector/prepare_swing_frames.py`.
5. Swing classifier trained and exported in `src/train_swing_detector/train_swing_classifier.py`.

**State Management:**
- Runtime state is in-memory (lists of peaks/Swings) and persisted as files (JSON, CSV, mp4) via paths in `src/tennis_cut/tennis_cut.py` and `src/label_videos/tennis_annotate.py`.

## Key Abstractions

**PopDetector:**
- Purpose: Encapsulate audio impact detection.
- Examples: `src/tennis_cut/tennis_cut.py`
- Pattern: fastai `load_learner` + windowed inference.

**SwingDetector:**
- Purpose: Image-level swing classification filter.
- Examples: `src/tennis_cut/tennis_cut.py`
- Pattern: fastai `load_learner` with `predict` and `no_shot` filtering.

**PersonDetector:**
- Purpose: Find largest person box for cropping.
- Examples: `src/utilities/core.py`
- Pattern: ultralytics YOLOv8 inference wrapper.

**Swing Dataclass:**
- Purpose: Record detected swing metadata for output.
- Examples: `src/tennis_cut/tennis_cut.py`
- Pattern: `@dataclass` value container.

## Entry Points

**Swing Extraction CLI:**
- Location: `src/tennis_cut/tennis_cut.py`
- Triggers: `python src/tennis_cut/tennis_cut.py <input>`
- Responsibilities: Orchestrate audio detection, person crop, clip cutting, stitching, metadata output.

**Audio Window Prep CLI:**
- Location: `src/train_pop_detector/prepare_audio_windows.py`
- Triggers: `python src/train_pop_detector/prepare_audio_windows.py`
- Responsibilities: Build training CSV and wavs from labeled videos.

**Audio Model Training CLI:**
- Location: `src/train_pop_detector/train_audio_pop.py`
- Triggers: `python src/train_pop_detector/train_audio_pop.py <csv>`
- Responsibilities: Train/export audio pop detector.

**Swing Frame Prep CLI:**
- Location: `src/train_swing_detector/prepare_swing_frames.py`
- Triggers: `python src/train_swing_detector/prepare_swing_frames.py`
- Responsibilities: Extract labeled frames and negatives for swing classifier.

**Swing Model Training CLI:**
- Location: `src/train_swing_detector/train_swing_classifier.py`
- Triggers: `python src/train_swing_detector/train_swing_classifier.py <dataset>`
- Responsibilities: Train/export swing classifier.

**Annotation GUI:**
- Location: `src/label_videos/tennis_annotate.py`
- Triggers: `python src/label_videos/tennis_annotate.py <dir>`
- Responsibilities: Manual labeling of impacts and shot types.

**Legacy Labeling GUI:**
- Location: `src/label_videos/label_shots.py`
- Triggers: `python src/label_videos/label_shots.py <dir>`
- Responsibilities: Add shot types to older impact-only JSONs.

## Error Handling

**Strategy:** Fail-fast with subprocess errors and minimal recovery.

**Patterns:**
- `subprocess.CalledProcessError` surfaced or logged in `src/tennis_cut/tennis_cut.py` and `src/train_*/*.py`.
- Skip or continue on missing JSON labels in `src/train_pop_detector/prepare_audio_windows.py` and `src/train_swing_detector/prepare_swing_frames.py`.

## Cross-Cutting Concerns

**Logging:** Python `logging` configured in `src/tennis_cut/tennis_cut.py`.
**Validation:** Argument parsing with `argparse` in all CLI scripts under `src/`.
**Authentication:** Not applicable.

---

*Architecture analysis: 2026-02-16*

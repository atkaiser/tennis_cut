# Codebase Structure

**Analysis Date:** 2026-02-16

## Directory Layout

```
[project-root]/
├── src/                 # Application scripts and shared utilities
├── examples/            # Small sample media for running scripts
├── dataset/             # Extracted training frames (generated)
├── videos/              # Training videos for labeling/training
├── wavs/                # Extracted wavs from training videos (generated)
├── meta/                # Training metadata CSVs
├── models/              # Exported model artifacts (local)
├── out/                 # CLI outputs (stitched clips, metadata)
├── test_videos/         # Small videos for manual validation
├── tmp_videos/          # Ad-hoc scratch video area
├── .planning/           # GSD planning artifacts
├── pyproject.toml       # Python project config and dependencies
├── README.md            # Project overview and usage
└── spec.md              # System design and CLI spec
```

## Directory Purposes

**src:**
- Purpose: All runnable scripts and shared helpers.
- Contains: CLI tools, dataset prep, training scripts, GUI annotation.
- Key files: `src/tennis_cut/tennis_cut.py`, `src/utilities/core.py`.

**src/tennis_cut:**
- Purpose: End-to-end swing extraction CLI.
- Contains: `tennis_cut.py`, `README.md`.
- Key files: `src/tennis_cut/tennis_cut.py`.

**src/train_pop_detector:**
- Purpose: Audio training pipeline.
- Contains: `prepare_audio_windows.py`, `train_audio_pop.py`.
- Key files: `src/train_pop_detector/prepare_audio_windows.py`, `src/train_pop_detector/train_audio_pop.py`.

**src/train_swing_detector:**
- Purpose: Vision training pipeline.
- Contains: `prepare_swing_frames.py`, `train_swing_classifier.py`.
- Key files: `src/train_swing_detector/prepare_swing_frames.py`, `src/train_swing_detector/train_swing_classifier.py`.

**src/label_videos:**
- Purpose: Manual labeling tools and video prep scripts.
- Contains: `tennis_annotate.py`, `label_shots.py`, shell helpers.
- Key files: `src/label_videos/tennis_annotate.py`.

**src/utilities:**
- Purpose: Shared helpers for media probing and detection.
- Contains: `core.py`, `__init__.py`.
- Key files: `src/utilities/core.py`.

**models:**
- Purpose: Local model artifacts for audio/swing detectors.
- Contains: fastai export files (e.g., `.pth`, `.pkl`).
- Key files: Not applicable.

**dataset:**
- Purpose: Extracted frames organized by label.
- Contains: Subdirectories per label.
- Key files: Generated content only.

**meta:**
- Purpose: CSV metadata for training.
- Contains: Window listings for audio training.
- Key files: Generated CSVs.

**out:**
- Purpose: Outputs from swing extraction.
- Contains: Stitched mp4s and JSON metadata.
- Key files: Generated content only.

## Key File Locations

**Entry Points:**
- `src/tennis_cut/tennis_cut.py`: swing extraction CLI.
- `src/train_pop_detector/prepare_audio_windows.py`: audio window generator.
- `src/train_pop_detector/train_audio_pop.py`: audio model trainer.
- `src/train_swing_detector/prepare_swing_frames.py`: frame extractor.
- `src/train_swing_detector/train_swing_classifier.py`: swing model trainer.
- `src/label_videos/tennis_annotate.py`: annotation GUI.

**Configuration:**
- `pyproject.toml`: dependencies and project metadata.

**Core Logic:**
- `src/tennis_cut/tennis_cut.py`: main pipeline orchestration.
- `src/utilities/core.py`: media probing and YOLO person detection.

**Testing:**
- Not detected.

## Naming Conventions

**Files:**
- snake_case scripts (e.g., `prepare_audio_windows.py`, `train_swing_classifier.py`).

**Directories:**
- snake_case (e.g., `train_pop_detector/`, `label_videos/`).

## Where to Add New Code

**New Feature:**
- Primary code: `src/tennis_cut/tennis_cut.py` or a new module under `src/tennis_cut/`.
- Tests: Not applicable (no test harness present).

**New Component/Module:**
- Implementation: `src/utilities/` for shared helpers or a new subfolder under `src/` for a new workflow.

**Utilities:**
- Shared helpers: `src/utilities/core.py` or new module under `src/utilities/`.

## Special Directories

**.venv:**
- Purpose: Local Python virtual environment.
- Generated: Yes.
- Committed: No.

**.ruff_cache:**
- Purpose: Ruff lint cache.
- Generated: Yes.
- Committed: No.

**examples:**
- Purpose: Small media for smoke tests.
- Generated: No.
- Committed: Yes.

**models:**
- Purpose: Local model artifacts.
- Generated: Yes.
- Committed: No.

---

*Structure analysis: 2026-02-16*

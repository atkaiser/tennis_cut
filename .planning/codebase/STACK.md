# Technology Stack

**Analysis Date:** 2026-02-16

## Languages

**Primary:**
- Python 3.11.11+ - CLI tools, model training, and utilities in `src/`

**Secondary:**
- Shell (bash) - video conversion helper in `src/label_videos/convert_to_fast_vid.sh`
- Markdown - documentation in `README.md` and `src/*/README.md`

## Runtime

**Environment:**
- CPython >= 3.11.11 (declared in `pyproject.toml` and `uv.lock`)

**Package Manager:**
- uv (used in `README.md` and `AGENTS.md`)
- Lockfile: present (`uv.lock`)

## Frameworks

**Core:**
- PyTorch >= 2.6.0 - model runtime/training in `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/train_audio_pop.py`, `src/train_swing_detector/train_swing_classifier.py`
- fastai >= 2.8.1 - model training/export in `src/train_pop_detector/train_audio_pop.py`, `src/train_swing_detector/train_swing_classifier.py`
- Ultralytics (YOLOv8) >= 8.1.0 - person detection in `src/utilities/core.py`
- PySide6 >= 6.9.0 - GUI labeling tools in `src/label_videos/tennis_annotate.py`, `src/label_videos/label_shots.py`

**Testing:**
- Not detected (no test runner configured; see `AGENTS.md`)

**Build/Dev:**
- Ruff >= 0.12.0 - linting via `pyproject.toml` and `.github/workflows/ruff.yml`

## Key Dependencies

**Critical:**
- torchaudio >= 2.6.0 - waveform loading/resampling in `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/train_audio_pop.py`
- torchvision >= 0.21.0 - vision training stack in `src/train_swing_detector/train_swing_classifier.py`
- pandas >= 2.2.3 - dataset CSV handling in `src/train_pop_detector/train_audio_pop.py`, `src/tennis_cut/tennis_cut.py`
- ultralytics >= 8.1.0 - YOLO model wrapper in `src/utilities/core.py`
- PySide6 >= 6.9.0 - Qt multimedia UI in `src/label_videos/tennis_annotate.py`

**Infrastructure:**
- ffmpeg-python >= 0.2.0 - declared dependency in `pyproject.toml` (CLI usage is via `ffmpeg` binary)
- soundfile >= 0.13.1 - audio IO backend for `torchaudio`
- pyqtgraph >= 0.13.7 - GUI plotting toolkit (dependency only)
- tqdm >= 4.65.0 - progress bars in training scripts
- fasttransform >= 0.0.2 - fastai transforms in `src/train_pop_detector/train_audio_pop.py`

## Configuration

**Environment:**
- No `.env` files detected; configuration is via CLI arguments in `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/prepare_audio_windows.py`, `src/train_pop_detector/train_audio_pop.py`, `src/train_swing_detector/train_swing_classifier.py`
- Model paths default to local `models/` and data directories (`videos/`, `wavs/`, `meta/`) described in `README.md`

**Build:**
- `pyproject.toml` - project metadata and dependencies
- `uv.lock` - resolved dependency lockfile
- `.github/workflows/ruff.yml` - CI lint pipeline

## Platform Requirements

**Development:**
- `ffmpeg` and `ffprobe` installed and on PATH (required by `src/tennis_cut/tennis_cut.py` and `src/label_videos/convert_to_fast_vid.sh`)
- Optional GPU/MPS runtime for torch (`src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/train_audio_pop.py`, `src/train_swing_detector/train_swing_classifier.py`)

**Production:**
- Local CLI execution; no deployment target configured

---

*Stack analysis: 2026-02-16*

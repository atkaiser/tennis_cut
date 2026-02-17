# Stack Research

**Domain:** Local sports video highlight extraction (audio + vision + ffmpeg pipelines)
**Researched:** 2026-02-16
**Confidence:** MEDIUM

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11 | Runtime for the CLI pipeline | Project constraint; broad ecosystem support for ML/audio/video tooling. (Confidence: MEDIUM – constraint, not externally verified) |
| FFmpeg | 8.0.1 | Decode/encode, precise cutting, filter graphs | Canonical local media pipeline; latest stable release from ffmpeg.org. (Confidence: HIGH) |
| PyTorch | 2.7.0 | Model training/inference backbone | Current stable release; standard for local ML inference/training in Python. (Confidence: HIGH) |
| Ultralytics (YOLOv8) | 8.4.14 | Lightweight person detection for vision verification | Actively maintained, recent stable release with turnkey detection workflows. (Confidence: HIGH) |
| OpenCV | 4.13.0 | Frame extraction, image transforms, geometry utilities | Widely used CV toolkit; current 4.x docs indicate 4.13.0 is available. (Confidence: HIGH) |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchvision | 0.25.0 | Vision transforms/models aligned with PyTorch | Use for preprocessing, boxes, and model utilities when staying in the PyTorch stack. (Confidence: HIGH) |
| torchaudio | 2.10.0 | Audio transforms/features in PyTorch | Use for in-graph audio preprocessing; note I/O decode/encode is being consolidated into TorchCodec. (Confidence: HIGH) |
| librosa | 0.11.0 | Traditional audio features/onset detection | Use for classical DSP and feature extraction outside the PyTorch graph. (Confidence: HIGH) |
| PySide6 | 6.10.2 | Desktop annotation/labeling UI | Use for local labeling tools and review interfaces. (Confidence: HIGH) |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| ruff | Linting and formatting | Matches repo guidance; use `ruff .` and `ruff --fix .`. |
| pre-commit | Hook management | Keep linting consistent before PRs. |
| pytest | Smoke tests | Useful for CLI regression tests even if not present yet. |

## Installation

```bash
# Core
pip install "torch==2.7.0" "ultralytics==8.4.14"

# Supporting
pip install "torchvision==0.25.0" "torchaudio==2.10.0" "librosa==0.11.0" "PySide6==6.10.2"

# FFmpeg (system)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Ultralytics YOLOv8 | torchvision detection models | If you need a pure-PyTorch dependency chain and are willing to build custom training/inference loops. |
| FFmpeg CLI + ffprobe | PyAV (FFmpeg bindings) | If you need in-process frame-accurate decoding without shelling out; still use FFmpeg for final encoding. |
| PySide6 | PyQt6 | If you already have a PyQt license/commercial constraints that mandate PyQt. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| torchvision `video_reader` backend | Disabled by default unless torchvision is compiled from source with FFmpeg; adds build complexity. | FFmpeg CLI/ffprobe or PyAV for decoding. |
| torchaudio legacy I/O decode/encode APIs | Torchaudio docs note decoding/encoding consolidated into TorchCodec and APIs deprecated/removed. | FFmpeg CLI/ffprobe for I/O, or TorchCodec if adopted. |

## Stack Patterns by Variant

**If CPU-only inference:**
- Use PyTorch CPU wheels and smaller YOLO models (e.g., `yolov8n`) for fast person checks.
- Because CPU decoding + lightweight detection keeps local runs reasonable without GPU.

**If Apple Silicon/MPS available:**
- Use PyTorch with MPS backend for faster inference.
- Because it accelerates local inference without CUDA.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| torch==2.7.0 | torchvision==0.25.0 | Use PyTorch’s official selector to keep torch/vision/audio ABI-aligned. |
| torchaudio==2.10.0 | torch from official selector | Install via the same PyTorch selector to avoid binary mismatches. |

## Sources

- https://pytorch.org/get-started/locally/ — PyTorch 2.7.0 stable (HIGH)
- https://pytorch.org/vision/stable/index.html — torchvision 0.25 docs, video backend notes (HIGH)
- https://pytorch.org/audio/stable/index.html — torchaudio 2.10.0 docs, I/O deprecations (HIGH)
- https://github.com/ultralytics/ultralytics/releases — Ultralytics 8.4.14 release (HIGH)
- https://ffmpeg.org/download.html — FFmpeg 8.0.1 stable release (HIGH)
- https://docs.opencv.org/ — OpenCV 4.13.0 docs index (HIGH)
- https://librosa.org/doc/latest/index.html — librosa 0.11.0 docs (HIGH)
- https://doc.qt.io/qtforpython-6/release_notes/pyside6_release_notes.html — PySide6 6.10.2 release notes (HIGH)

---
*Stack research for: local sports video highlight extraction tools*
*Researched: 2026-02-16*

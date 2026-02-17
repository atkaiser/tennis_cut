# Project Research Summary

**Project:** Tennis Cut
**Domain:** Local, offline tennis swing/highlight extraction (CLI-first)
**Researched:** 2026-02-16
**Confidence:** MEDIUM

## Executive Summary

Tennis Cut is a local, CLI-first sports highlight extractor that finds tennis impact moments using audio and verifies them with lightweight vision, then cuts usable clips with metadata. Experts build this kind of tool as a deterministic media pipeline: probe and normalize timestamps with ffprobe/ffmpeg, run two-stage detection (audio then vision) to reduce false positives, and generate manifests that drive clip cutting, review, and training exports.

The recommended approach is to harden media I/O and a canonical timeline first, then layer detection quality and verification, and only then expand UX features like review tooling and slow-mo exports. The stack leans on Python 3.11 with FFmpeg 8.0.1 for all media I/O, PyTorch 2.7.0 plus Ultralytics YOLOv8 for inference, and OpenCV for frame sampling; results should be captured in reproducible manifests to support debugging and training.

Key risks are timestamp drift, false positives from audio-only detection, vision domain shift, and stream selection mistakes during clip export. Mitigate these by using container timestamps as the single timebase, explicit ffmpeg stream mapping, per-session noise calibration plus audio+vision gating, and confidence logging with fallback rules; validate against a golden sample after any media pipeline change.

## Key Findings

### Recommended Stack

The stack is a Python 3.11 offline pipeline with FFmpeg/ffprobe as the media backbone, PyTorch for detection models, and YOLOv8 for lightweight vision verification. OpenCV handles frame sampling, while torchaudio/librosa provide audio feature tooling. PySide6 is the preferred local UI toolkit for labeling/review workflows. Avoid torchvision's video_reader backend and torchaudio legacy I/O, since both add build complexity or are being deprecated.

**Core technologies:**
- Python 3.11: CLI pipeline runtime — aligns with project constraints and ML/audio/video tooling.
- FFmpeg 8.0.1: decode/encode and precise cutting — canonical local media pipeline.
- PyTorch 2.7.0: training/inference backbone — current stable release with broad support.
- Ultralytics YOLOv8 8.4.14: person verification — active, turnkey detection workflows.
- OpenCV 4.13.0: frame extraction and transforms — standard CV toolkit.

### Expected Features

Feature expectations center on reliable auto-detection with usable clips and deterministic outputs, plus a light correction loop. Differentiators come from multi-signal detection quality, slow-mo exports, and labeling workflows that improve models without large manual effort.

**Must have (table stakes):**
- Batch import/processing of local videos — users expect hands-free runs.
- Automatic impact detection + timestamps — core value.
- Clip extraction with pre/post buffers — makes output usable.
- Manual review/trim and overrides — essential for correction.
- Export formats + JSON/CSV metadata — interoperability and sharing.
- Basic organization (naming/folders/logs) — predictable outputs.

**Should have (competitive):**
- Audio+vision fusion for detection — improves precision/recall.
- Slow-mo export presets — improves coaching review.
- Active-learning labeling workflow — speeds model improvement.
- Reproducible runs (config snapshots) — debuggable results.

**Defer (v2+):**
- Shot-type classification — needs scaled labeled data.
- Quality scoring/auto-reject — requires calibration across footage.

### Architecture Approach

The standard architecture is a CLI orchestration layer over a media+ML core (probe/index, decode/prep, event detect, verify/track) feeding output artifacts (clip cutter, metadata manifests, labeling UI, and training exports). The key patterns are a deterministic job manifest, two-stage detection (audio then vision), and windowed temporal assembly using a single timestamp timebase.

**Major components:**
1. Probe/Index — ffprobe-derived media metadata and timebase.
2. Decode/Prep — audio extraction and frame sampling via ffmpeg/OpenCV.
3. Event Detect — audio model inference and candidate timestamps.
4. Verify/Track — YOLO-based person validation of candidates.
5. Clip Cutter + Manifests — ffmpeg trim/concat and JSON/CSV outputs.

### Critical Pitfalls

1. **Timestamp drift between audio and video** — normalize to container timestamps and keep all cuts in timestamp units.
2. **Audio-only false positives** — calibrate noise per session and gate with vision/motion signals.
3. **Vision domain shift** — log confidence, tune thresholds per capture style, and allow fallback rules.
4. **Stream selection mistakes** — always use explicit ffmpeg `-map` options and verify outputs.
5. **Train/validation leakage** — split by session/video, not by clip, and report leave-one-session-out results.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Media I/O + Canonical Timeline
**Rationale:** All detection and clipping depend on reliable timestamps and deterministic media handling.
**Delivers:** ffprobe index, ffmpeg audio/video extraction, timebase normalization, deterministic output structure.
**Addresses:** batch import, clip extraction, metadata export, basic organization.
**Avoids:** timestamp drift, stream selection mistakes.

### Phase 2: Detection Quality + Verification
**Rationale:** Detection drives clip boundaries; false positives kill trust without review workflows.
**Delivers:** audio impact detection, audio+vision gating, confidence logging, baseline review/override loop.
**Addresses:** automatic detection, manual review/trim, audio+vision fusion.
**Avoids:** audio-only false positives, vision domain shift, clip boundary context loss.

### Phase 3: Export Experience + Reproducibility
**Rationale:** Once detection is stable, improve usability and consistency for real workflows.
**Delivers:** slow-mo export path, config snapshots, reproducible manifests with ffmpeg/version logging.
**Addresses:** slow-mo presets, reproducible runs, improved export UX.
**Avoids:** slow-mo sync issues, non-reproducible outputs.

### Phase 4: Model Improvement + Advanced Features
**Rationale:** Advanced features require data and evaluation discipline after core workflow is validated.
**Delivers:** labeling workflow improvements, training exports, shot-type classification/quality scoring pilots.
**Addresses:** active-learning labeling, shot-type classification, quality scoring.
**Avoids:** train/validation leakage, overfitting from narrow datasets.

### Phase Ordering Rationale

- Media I/O and canonical timelines are foundational for all downstream detection and cutting.
- Two-stage detection depends on reliable audio windows and frame sampling; it must precede clipper polish.
- Export UX and reproducibility build on stable detection outputs.
- Advanced ML features require validated datasets and evaluation protocols.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** slow-mo export + audio handling choices vary by ffmpeg filters and target UX.
- **Phase 4:** active-learning/shot-type classification needs dataset and labeling strategy validation.

Phases with standard patterns (skip research-phase):
- **Phase 1:** ffprobe/ffmpeg media I/O and manifest-based pipelines are well-documented.
- **Phase 2:** audio-first + vision verification is a standard two-stage detection pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Most items are backed by official release docs; versions are explicit. |
| Features | MEDIUM | Competitor analysis is directional; needs user validation. |
| Architecture | MEDIUM | Pattern-based guidance aligns with standard pipelines but not validated in this repo. |
| Pitfalls | MEDIUM | Several items are experiential and need validation on project data. |

**Overall confidence:** MEDIUM

### Gaps to Address

- Detection thresholds and calibration targets: validate on representative footage across courts and lighting.
- Slow-mo export audio policy: decide on audio disable vs time-stretch and document per preset.
- Labeling workflow scope: confirm which UI improvements drive the largest accuracy gains.

## Sources

### Primary (HIGH confidence)
- https://ffmpeg.org/ffmpeg.html — filtergraphs, option ordering, stream selection.
- https://ffmpeg.org/ffprobe.html — media metadata probing.
- https://pytorch.org/get-started/locally/ — PyTorch 2.7.0 release info.
- https://pytorch.org/vision/stable/index.html — torchvision compatibility and video backend notes.
- https://pytorch.org/audio/stable/index.html — torchaudio I/O deprecations.
- https://github.com/ultralytics/ultralytics/releases — YOLOv8 8.4.14 release.
- https://docs.opencv.org/ — OpenCV 4.13.0 docs.
- https://librosa.org/doc/latest/index.html — librosa 0.11.0 docs.
- https://doc.qt.io/qtforpython-6/release_notes/pyside6_release_notes.html — PySide6 6.10.2.

### Secondary (MEDIUM confidence)
- https://www.veo.co/en-us/product/veo-editor — competitor feature expectations.
- https://playsight.com/ — competitor feature expectations.
- https://www.hudl.com/products/balltime — competitor feature expectations.

### Tertiary (LOW confidence)
- Practitioner experience with sports highlight extraction pipelines — pitfalls and heuristics.

---
*Research completed: 2026-02-16*
*Ready for roadmap: yes*

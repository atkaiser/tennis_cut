# Architecture Research

**Domain:** Local sports video analysis pipeline (CLI-first, offline)
**Researched:** 2026-02-16
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Orchestration Layer                              │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │ CLI Runner  │  │ Job Config   │  │ Progress/UI  │                     │
│  └─────┬────────┘  └─────┬────────┘  └─────┬────────┘                     │
│        │                 │                 │                              │
├────────┴─────────────────┴─────────────────┴──────────────────────────────┤
│                         Media + ML Core                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Probe/Index │  │ Decode/Prep  │  │ Event Detect │  │ Verify/Track │     │
│  └─────┬───────┘  └─────┬────────┘  └─────┬────────┘  └─────┬────────┘     │
│        │                │                 │                 │             │
├────────┴────────────────┴─────────────────┴─────────────────┴─────────────┤
│                         Outputs + Artifacts                              │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Clip Cutter │  │ Metadata/JSON│  │ Labeling UI  │  │ Train Data   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| CLI Runner | Parse args, run jobs, report progress | Python CLI entrypoints and config parsing |
| Probe/Index | Extract duration, fps, audio/video streams | `ffprobe` JSON output | 
| Decode/Prep | Extract audio, decode frames, resample | `ffmpeg` for audio/video, OpenCV for frames |
| Event Detect | Identify candidate impact timestamps | Audio model inference + signal post-process |
| Verify/Track | Filter candidates with vision person box, optional tracking | YOLO inference over sampled frames |
| Clip Cutter | Cut per-swing or stitched clips with slow-mo | `ffmpeg` trim/concat/filtergraph |
| Metadata/JSON | Persist clip boundaries, scores, labels | JSON/CSV manifests on disk |
| Labeling UI | Human-in-the-loop corrections | PySide6 UI on top of manifest |
| Train Data | Export labeled windows and datasets | Offline scripts with filesystem datasets |

## Recommended Project Structure

```
src/
├── tennis_cut/              # Core pipeline package
│   ├── cli.py               # CLI entrypoints
│   ├── config.py            # Argument and config normalization
│   ├── pipeline/            # Orchestration and job graph
│   ├── media/               # ffmpeg/ffprobe/OpenCV wrappers
│   ├── audio/               # Audio windows, impact detection
│   ├── vision/              # YOLO verification, frame sampling
│   ├── events/              # Timeline assembly, scoring, filtering
│   ├── clips/               # Clip cutting and stitch logic
│   ├── manifests/           # JSON/CSV schemas and writers
│   └── util/                # Shared helpers, logging
├── tools/                   # Labeling UI and annotation utilities
│   ├── label_ui/            # PySide6 UI
│   └── exports/             # Export helpers for training
├── train_pop_detector/      # Audio model training scripts
├── train_vision/            # Optional vision model training
└── examples/                # Example inputs and workflows
```

### Structure Rationale

- **tennis_cut/**: Keeps runtime pipeline cohesive and importable from CLI.
- **media/** and **events/**: Separates media I/O from model logic, making clip cutting and detection independently testable.
- **manifests/**: Centralizes schema and avoids fragile cross-module JSON formats.

## Architectural Patterns

### Pattern 1: Deterministic Job Manifest

**What:** Every run produces a manifest with inputs, parameters, intermediate artifacts, and outputs.
**When to use:** Any CLI workflow that may be resumed, debugged, or re-scored without re-decoding.
**Trade-offs:** Slight I/O overhead, but enables reproducibility and UI integration.

**Example:**
```python
from dataclasses import asdict

def run_job(job, media, detector, verifier, cutter, writer):
    candidates = detector.detect(media.audio_path, job.audio_config)
    verified = verifier.filter(media.video_path, candidates, job.vision_config)
    clips = cutter.cut(media.video_path, verified, job.clip_config)
    manifest = {
        "job": asdict(job),
        "candidates": candidates,
        "verified": verified,
        "clips": clips,
    }
    writer.write_manifest(job.output_dir, manifest)
    return clips
```

### Pattern 2: Two-Stage Detection (Audio -> Vision)

**What:** Use audio to propose timestamps, then vision to verify players and reduce false positives.
**When to use:** Impact sounds are strong signals but need visual validation.
**Trade-offs:** Requires frame sampling and adds inference cost, but reduces missed/false hits.

**Example:**
```python
def detect_impacts(audio_scores, frame_sampler, yolo, threshold=0.5):
    candidates = [t for t, s in audio_scores if s >= threshold]
    verified = []
    for t in candidates:
        frame = frame_sampler.sample(t)
        if yolo.has_person(frame):
            verified.append(t)
    return verified
```

### Pattern 3: Windowed Temporal Assembly

**What:** Convert event timestamps into clip windows with pre/post roll and optional slow-mo.
**When to use:** Clips should be consistent across different sources and fps.
**Trade-offs:** Needs accurate timebase mapping (fps + audio sample rate).

## Data Flow

### Request Flow

```
CLI args
    ↓
Job config → Probe/Index → Decode/Prep → Event Detect → Verify/Track
    ↓                                 ↓               ↓
Manifest writer ← Clip Cutter ← Window assembler ← Timeline scores
```

### State Management

```
Manifest (JSON/CSV)
    ↓ (read/update)
Labeling UI ↔ Review actions → Manifest updates → Training exports
```

### Key Data Flows

1. **Media ingestion:** `ffprobe` gathers streams/duration → `ffmpeg` extracts audio and frames.
2. **Impact detection:** Audio windows → model scores → candidate timestamps.
3. **Verification:** Sampled frames around timestamps → YOLO person check → verified events.
4. **Clip export:** Verified events → pre/post windows → `ffmpeg` cut/stitch outputs + metadata.
5. **Label feedback:** Manifest → UI edits → updated labels → training dataset export.

## Build Order Implications

1. **Media I/O foundation:** Implement probe/index and decode/prepare wrappers first (ffprobe/ffmpeg).
2. **Event timeline core:** Add audio detection + timebase alignment before any UI work.
3. **Verification layer:** Integrate YOLO frame sampling after timeline exists.
4. **Clipper + metadata:** Build cutter and manifest schema once events are stable.
5. **Labeling UI + training:** UI consumes manifest; training uses exports from labeled manifests.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-1k videos | Single-process pipeline, stream frames/audio to avoid large RAM spikes. |
| 1k-10k videos | Add job queue and cached intermediate artifacts to skip re-decode. |
| 10k+ videos | Batch pipeline with persistent manifest index and optional GPU queueing. |

### Scaling Priorities

1. **First bottleneck:** Video decode and re-encode in `ffmpeg` → cache intermediates and avoid rework.
2. **Second bottleneck:** Vision inference on frames → reduce sampling density and batch frames.

## Anti-Patterns

### Anti-Pattern 1: Decode-Everything-Into-Memory

**What people do:** Load full video/audio into RAM arrays for processing.
**Why it's wrong:** iPhone videos are large; memory spikes and slowdowns.
**Do this instead:** Stream frames and audio windows, cache on disk when needed.

### Anti-Pattern 2: No Canonical Timeline

**What people do:** Mix frame indices and timestamps without a single source of truth.
**Why it's wrong:** Clips drift when fps metadata differs or variable frame rate appears.
**Do this instead:** Normalize all events to timestamps derived from `ffprobe`/decode settings.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| FFmpeg | CLI calls for decode, resample, clip export | Supports filtergraphs for slow-mo and concat. |
| FFprobe | CLI metadata probe | JSON output as canonical media index. |
| OpenCV | Frame sampling from video | `VideoCapture` for frame access. |
| Ultralytics YOLO | Vision verification | Inference on sampled frames or streams. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| CLI ↔ Pipeline | Function calls + config object | Keep pipeline pure and CLI thin. |
| Pipeline ↔ Label UI | Manifest files | UI should never mutate media; only metadata. |
| Detection ↔ Clipper | Verified event list | Timeline schema is the contract. |

## Sources

- https://ffmpeg.org/ffmpeg.html (pipeline components, filtergraphs)
- https://ffmpeg.org/ffprobe.html (media metadata probing)
- https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html (frame capture API)
- https://docs.ultralytics.com/modes/predict/ (YOLO inference sources and streaming)

---
*Architecture research for: local sports video analysis pipeline*
*Researched: 2026-02-16*

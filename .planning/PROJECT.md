# Tennis Cut

## What This Is

Tennis Cut takes in tennis videos and returns clips of just the hits. It is a CLI-first pipeline that finds impact moments from audio, verifies them with a lightweight vision model, and cuts stitched or per-swing clips with optional metadata. It is built for local, offline processing of iPhone-shot match footage.

## Core Value

Given a tennis video, reliably output clean swing clips around each hit with minimal manual effort.

## Requirements

### Validated

- ✓ Detect candidate impacts using an audio "pop" model — existing
- ✓ Verify candidates with a vision model and person crop — existing
- ✓ Cut and stitch swing clips with optional metadata outputs — existing
- ✓ Provide CLI workflows for labeling and model training — existing

### Active

- [ ] Improve annotation UI ergonomics (fixed key panel height, show recent hits, quick undo)
- [ ] Enforce a minimum 0.5s separation between labeled hits to reduce collisions
- [ ] Make example media pipeline reproducible end-to-end and update README steps
- [ ] Update the spec to reflect the current pipeline and CLI flags

### Out of Scope

- Hosted web service or mobile app — keep local CLI processing for now
- Real-time/live swing detection — adds streaming complexity not needed yet

## Context

The pipeline extracts audio, detects impact windows with a fastai model, verifies them with a YOLOv8 person detector, and cuts clips via ffmpeg. There are supporting tools for labeling impacts and training audio/vision models. The project is mostly done but needs usability and documentation improvements to make the workflow smoother and more reproducible.

## Constraints

- **Tech stack**: Python 3.11 with PyTorch/fastai, ultralytics YOLOv8, PySide6 — existing tooling
- **Dependencies**: `ffmpeg` and `ffprobe` must be installed on PATH
- **Environment**: Local CLI workflows; no server-side deployment target
- **Data**: Large videos and model artifacts should stay out of git

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep a CLI-first, local workflow | Current users run the pipeline on local footage | — Pending |
| Maintain audio + vision verification approach | Balances recall and false positives with minimal compute | — Pending |

---
*Last updated: 2026-02-16 after initialization*

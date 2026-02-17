# External Integrations

**Analysis Date:** 2026-02-16

## APIs & External Services

**External Tools:**
- FFmpeg/FFprobe CLI - video/audio extraction and probing in `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/prepare_swing_frames.py`, `src/label_videos/convert_to_fast_vid.sh`, `src/label_videos/tennis_annotate.py`
  - SDK/Client: `ffmpeg` binary (invoked via `subprocess`)
  - Auth: Not applicable

## Data Storage

**Databases:**
- None (data stored on local filesystem)

**File Storage:**
- Local filesystem only
  - Raw videos: `videos/`, `test_videos/`
  - Extracted frames/audio: `dataset/`, `wavs/`
  - Training metadata: `meta/`
  - Model artifacts: `models/`
  - Output clips: `out/`

**Caching:**
- None

## Authentication & Identity

**Auth Provider:**
- None
  - Implementation: Not applicable

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- Standard output logging via `logging`/`print` in `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/train_audio_pop.py`

## CI/CD & Deployment

**Hosting:**
- Not applicable (local CLI tooling)

**CI Pipeline:**
- GitHub Actions (Ruff lint) in `.github/workflows/ruff.yml`

## Environment Configuration

**Required env vars:**
- None detected (no `os.environ` usage in `src/`)

**Secrets location:**
- Not applicable

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-02-16*

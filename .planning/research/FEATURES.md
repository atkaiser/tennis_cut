# Feature Research

**Domain:** Sports swing/highlight extraction tools (tennis-focused, offline/CLI)
**Researched:** 2026-02-16
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Batch import + processing of local videos | Users often record long sessions and expect hands-free processing | MEDIUM | CLI should accept folders, globbing, and handle mixed formats with ffmpeg/ffprobe.
| Automatic event detection (impacts/points) | Competing tools auto-detect key moments and trim dead time | HIGH | Core: reliable timestamp detection; allow sensitivity tuning to reduce false hits.
| Clip extraction with pre/post buffers | Users expect short, viewable swing clips rather than raw timestamps | MEDIUM | Needs per-clip padding and max/min length constraints.
| Manual review/trim and simple overrides | AI misses happen; users need quick correction | MEDIUM | Minimal UI or editable timeline/CSV to adjust boundaries.
| Export formats + metadata | Users want shareable videos and structured data | MEDIUM | MP4 clips + JSON/CSV with timestamps, confidence, source file.
| Basic organization (naming, folders) | Expect predictable outputs by session/date | LOW | Deterministic naming, folder structure, and logs.

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Audio+vision fusion for impact detection | Higher precision than vision-only or audio-only | HIGH | Match Tennis Cut’s current dual-signal strategy and expose diagnostics.
| Auto slow‑mo per clip | Improves technique review and shareability | MEDIUM | ffmpeg filter pipeline with configurable speed and smoothness.
| Active‑learning labeling workflow | Faster model improvement with fewer labels | HIGH | Tight loop between UI label tool and retraining scripts.
| Quality scoring + auto‑reject (occlusion, off‑court) | Reduces time spent sorting bad clips | HIGH | Use vision quality heuristics and detection confidence thresholds.
| Per‑swing metadata (forehand/backhand/serve) | Enables drill feedback and stats | HIGH | Requires classifier model and labeled data.
| Reproducible runs (config snapshots) | Makes results debuggable and shareable | LOW | Save config + version info with each run.

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Cloud upload + hosted processing | “No install, faster compute” | Breaks offline/local constraint; privacy concerns | Keep local processing; optional export bundle for sharing.
| Always-on automatic social posting | “Easy sharing” | Increases risk of unwanted exposure and content mistakes | Export clips locally; leave sharing to user tools.
| Proprietary camera lock‑in | “Best results with our hardware” | Conflicts with iPhone-shot input and limits adoption | Support standard video formats + calibration tips.
| Full match analytics dashboards | “Stats like pro tools” | Large scope, distracts from clip extraction | Export structured metadata for external analysis.

## Feature Dependencies

```
Accurate event detection
    └──requires──> Robust audio extraction + noise handling
                       └──requires──> ffmpeg/ffprobe pipeline

Clip extraction
    └──requires──> Event timestamps

Slow‑mo rendering
    └──requires──> Clip extraction

Shot‑type classification
    └──requires──> Labeled dataset + training pipeline

Quality scoring
    └──requires──> Vision model outputs + calibration thresholds

Labeling UI improvements
    └──enhances──> Dataset quality → Detection accuracy
```

### Dependency Notes

- **Clip extraction requires event timestamps:** Detection is the source of truth for segment boundaries.
- **Slow‑mo rendering requires clip extraction:** Apply filters on already‑trimmed segments to keep performance predictable.
- **Shot‑type classification requires labeled dataset + training pipeline:** Needs curated labels to avoid noisy categories.
- **Quality scoring requires vision model outputs:** Use person/ball visibility and confidence to reject unusable clips.
- **Labeling UI enhancements improve dataset quality:** Better labels directly increase detection accuracy.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] Automatic impact detection + timestamp list — core value.
- [ ] Clip extraction with pre/post buffers — makes output usable.
- [ ] Deterministic outputs (naming, folders, JSON/CSV metadata) — enables workflow integration.

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Manual review/trim UI improvements — reduce correction time.
- [ ] Slow‑mo export presets — improve coaching review.

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Shot‑type classification (forehand/backhand/serve) — needs labeled data scale.
- [ ] Quality scoring + auto‑reject — requires calibration across varied footage.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Impact detection + timestamps | HIGH | HIGH | P1 |
| Clip extraction + buffers | HIGH | MEDIUM | P1 |
| Metadata export (JSON/CSV) | HIGH | LOW | P1 |
| Manual review/trim | MEDIUM | MEDIUM | P2 |
| Slow‑mo export | MEDIUM | MEDIUM | P2 |
| Shot‑type classification | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Competitor A (Veo Editor) | Competitor B (PlaySight) | Our Approach |
|---------|---------------------------|--------------------------|--------------|
| Auto event detection | AI detects and lists key events | AI highlights and TagMe auto‑highlights | Audio+vision impact detection, offline CLI.
| Clip creation + sharing | Create clips, tag, comment, download/share | AI‑created highlights, shareable clips | Local clip export + metadata; share outside tool.
| Manual tagging/editing | Create your own events, assign players | Tag plays and bookmark events | Minimal correction UI and editable CSV/JSON.
| Playback controls | Timeline, playback speed | Instant replay, multi‑angle viewing | Simple per‑clip playback + optional slow‑mo render.

## Sources

- https://www.veo.co/en-us/product/veo-editor
- https://playsight.com/
- https://www.hudl.com/products/balltime

---
*Feature research for: tennis swing/highlight extraction tools*
*Researched: 2026-02-16*

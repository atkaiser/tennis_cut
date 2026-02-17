# Pitfalls Research

**Domain:** Sports video highlight extraction (tennis swing clip extraction)
**Researched:** 2026-02-16
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Timestamp drift between audio windows and video frames

**What goes wrong:**
Audio impact times don"t line up with the actual contact frame, so clips start late/early or miss the swing entirely.

**Why it happens:**
Pipelines assume constant frame rate or use frame index math instead of container timestamps, and then mix ffmpeg options that change timebases when extracting or re-encoding.

**How to avoid:**
Use timestamps from the container (ffprobe/ffmpeg), normalize to a single timebase, and keep all trimming/cutting in timestamp units. Add a small configurable pre-roll/post-roll and verify alignment on a known sample after any change to media IO.

**Warning signs:**
Impacts line up at the start of a video but drift later; slow-mo or re-encoded outputs shift impact by different amounts than originals.

**Phase to address:**
Phase 1 (Media IO + clip extraction baseline)

---

### Pitfall 2: Audio-only hit detection overwhelms precision

**What goes wrong:**
Footsteps, ball bounces, fence hits, and handling noise trigger false hits; users spend more time deleting clips than using them.

**Why it happens:**
Thresholds are tuned on a single environment and don"t generalize to different courts, phones, or mic positions.

**How to avoid:**
Calibrate per-session noise floor, add negative samples from non-hit moments, and use a simple multi-signal gate (audio + motion/person presence) before declaring hits.

**Warning signs:**
Hit counts spike in wind/noisy scenes; clusters of hits during dead time between points.

**Phase to address:**
Phase 2 (Detection quality + labeling workflow)

---

### Pitfall 3: Vision verification fails under domain shift

**What goes wrong:**
The person-box verification drops valid hits in backlit courts, wide shots, or doubles play, leading to missing highlights.

**Why it happens:**
A single pretrained model and fixed confidence threshold are used despite camera angle, lighting, and player distance shifts.

**How to avoid:**
Log verification confidence, tune thresholds per capture style, and provide a fallback path (e.g., keep hits with strong audio + motion even if person box is weak). Add a small curated validation set that spans lighting and framing.

**Warning signs:**
High rejection rate in specific courts or time of day; results improve dramatically by lowering confidence.

**Phase to address:**
Phase 2 (Detection quality + labeling workflow)

---

### Pitfall 4: Train/validation leakage from the same rally

**What goes wrong:**
Metrics look strong but field performance is poor because clips from the same session/rally appear in both train and validation.

**Why it happens:**
Splits are done by clip instead of by session or match; adjacent frames and audio windows are nearly identical.

**How to avoid:**
Split by session/video and track provenance for every clip. Add a "leave-one-session-out" sanity evaluation and report it alongside normal metrics.

**Warning signs:**
Validation accuracy spikes with small changes; performance drops sharply on a new court or player.

**Phase to address:**
Phase 3 (Model training + evaluation)

---

### Pitfall 5: Output stream selection mistakes silently drop audio

**What goes wrong:**
Generated clips are missing audio or contain a different stream than expected (e.g., narration or ambient track), breaking hit verification or user trust.

**Why it happens:**
ffmpeg defaults to automatic stream selection and option ordering is incorrect, so the intended audio stream isn"t explicitly mapped.

**How to avoid:**
Always specify `-map` for required audio/video streams and keep input/output option ordering explicit and consistent.

**Warning signs:**
Some clips have audio and others do not; audio channels or sample rates differ unexpectedly across outputs.

**Phase to address:**
Phase 1 (Media IO + clip extraction baseline)

---

### Pitfall 6: Non-reproducible outputs due to hidden defaults

**What goes wrong:**
Running the same command on a different machine or ffmpeg build yields different cut points, FPS, or audio behavior.

**Why it happens:**
Defaults differ across ffmpeg builds, and the pipeline doesn"t pin or record the exact parameters used.

**How to avoid:**
Record ffmpeg/ffprobe versions and full command lines in metadata, and provide a `--repro` mode that pins key options (timebase, fps, audio resample).

**Warning signs:**
Inconsistent clip durations between machines; bugs reported that you can"t reproduce locally.

**Phase to address:**
Phase 4 (Workflow reproducibility + specs)

---

### Pitfall 7: Clip boundaries ignore tennis context

**What goes wrong:**
Clips start too late or end before follow-through, making them unusable for coaching or sharing.

**Why it happens:**
Only impact timestamps are used, without sport-specific pre/post windows or overlap handling between close hits.

**How to avoid:**
Set sport-specific pre/post roll defaults, merge or stitch overlapping hits, and expose per-user presets for swing types.

**Warning signs:**
Users repeatedly re-cut clips manually; feedback mentions "missing the swing" even when hits are detected.

**Phase to address:**
Phase 2 (Detection quality + labeling workflow) and Phase 5 (User experience)

---

### Pitfall 8: Slow-motion generation breaks sync or audio quality

**What goes wrong:**
Slow-mo clips lose audio sync, create chipmunk audio, or stutter due to frame interpolation mismatches.

**Why it happens:**
Video is retimed without corresponding audio handling or the chosen filter assumes fixed FPS.

**How to avoid:**
Treat slow-mo as a separate export path: retime video and either drop audio, time-stretch audio, or annotate that audio is disabled.

**Warning signs:**
User reports of delayed hit sound; audio artifacts in slow-mo exports.

**Phase to address:**
Phase 4 (Export formats + slow-mo)

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hard-coded thresholds for impact detection | Fast tuning on a single sample | Fragile across courts and devices | Only for prototype demo clips |
| Storing only clip filenames (no provenance) | Simple outputs | No way to reproduce results or debug | Never |
| Single global confidence threshold for person box | Fewer knobs | Systematic misses in backlit or wide shots | MVP only if manual review is required |
| One-size-fits-all pre/post roll | Easier UX | Clips miss follow-through or include dead time | Acceptable for internal tests only |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| ffmpeg/ffprobe | Relying on automatic stream selection | Explicit `-map` and recorded stream indices |
| YOLOv8 model weights | Swapping model sizes without recalibrating thresholds | Re-tune confidence thresholds per model size |
| Local filesystem paths | Assuming POSIX paths and no spaces | Use `pathlib.Path` and robust quoting |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Running vision inference on every frame | Long runtimes; GPU/CPU maxed | Use audio to gate candidates, downsample frames | 30+ minute sessions |
| Re-encoding full video for each clip | Duplicate work; huge runtimes | Extract once, then clip from cached mezzanine | 50+ clips per session |
| Python-level frame loops for decoding | CPU bottlenecks, memory spikes | Use ffmpeg for decode and filter | 1080p+ and long videos |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Processing untrusted media without sandboxing | Exposure to decoder vulnerabilities | Keep ffmpeg updated; avoid running on untrusted files |
| Writing outputs to user-supplied paths without validation | Path traversal or overwrite | Validate output roots and prevent `..` paths |
| Embedding absolute file paths in metadata by default | Leaks local filesystem structure | Provide a redacted metadata option |

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Labeling UI lacks waveform + frame sync | Users can"t confidently label hits | Show aligned waveform and frame scrubber |
| No quality summary after processing | Users don"t know if results are good | Provide precision/recall proxies and clip counts |
| Overwriting outputs silently | Lost work and mistrust | Use versioned output folders and explicit overwrite flags |

## "Looks Done But Isn"t" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Clip extraction:** Often missing audio/video sync check — verify impact time within ±2 frames on a known sample
- [ ] **Detection model:** Often missing session-level validation — verify performance on a separate court/session
- [ ] **Slow-mo export:** Often missing audio handling decision — verify audio disabled or time-stretched intentionally
- [ ] **Metadata JSON:** Often missing provenance — verify model version, thresholds, and ffmpeg command recorded

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Timestamp drift | MEDIUM | Recompute timestamps from container timebase; re-cut clips from originals |
| Audio-only false positives | LOW | Raise thresholds, add noise profiles, re-run gating on existing detections |
| Domain shift in vision verification | MEDIUM | Collect a small new validation set; re-tune thresholds or add a fallback rule |
| Train/validation leakage | HIGH | Re-split by session, retrain, and discard misleading metrics |
| Stream selection mistakes | LOW | Re-run ffmpeg with explicit `-map` and verify outputs |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Timestamp drift | Phase 1 (Media IO + clip extraction baseline) | Golden sample: impact aligned within ±2 frames |
| Audio-only false positives | Phase 2 (Detection quality + labeling workflow) | False positives < target per 10 minutes |
| Vision domain shift | Phase 2 (Detection quality + labeling workflow) | Similar recall across 3 lighting/camera setups |
| Train/validation leakage | Phase 3 (Model training + evaluation) | Session-level validation report |
| Stream selection mistakes | Phase 1 (Media IO + clip extraction baseline) | Output has correct audio stream in 5 sample files |
| Non-reproducible outputs | Phase 4 (Workflow reproducibility + specs) | Re-run yields identical clip durations |
| Clip boundary context loss | Phase 5 (User experience) | User review shows "missing swing" < threshold |
| Slow-mo sync issues | Phase 4 (Export formats + slow-mo) | Audio behavior documented and verified |

## Sources

- https://ffmpeg.org/ffmpeg.html (official ffmpeg documentation: option ordering, stream selection, stream copy vs transcode)
- https://docs.ultralytics.com/models/yolov8/ (YOLOv8 model families, tasks, and usage context)
- Practitioner experience with sports highlight extraction pipelines (LOW confidence; validate with project data)

---
*Pitfalls research for: Sports video highlight extraction (tennis swing clip extraction)*
*Researched: 2026-02-16*

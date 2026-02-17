# Requirements: Tennis Cut

**Defined:** 2026-02-16
**Core Value:** Given a tennis video, reliably output clean swing clips around each hit with minimal manual effort.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Processing

- [ ] **PROC-01**: User can process a single video or a folder of videos in one command
- [ ] **PROC-02**: User can process common iPhone H.264/AAC videos without manual transcoding

### Detection

- [ ] **DET-01**: User can generate impact timestamps automatically from audio
- [ ] **DET-02**: User can run detection that verifies candidates with vision-based person detection

### Output

- [ ] **OUT-01**: User can generate swing clips with pre/post buffers around contact
- [ ] **OUT-02**: User can output a stitched highlight video and/or per-swing clips
- [ ] **OUT-03**: User can export per-swing metadata in JSON or CSV

### Review

- [ ] **REV-01**: User can review detected swings and adjust or remove false hits

### Organization

- [ ] **ORG-01**: Outputs use deterministic naming and folder structure in the output directory

### Labeling UI

- [ ] **LAB-01**: User can undo the last impact label in the annotation UI
- [ ] **LAB-02**: Annotation UI prevents new labels within 0.5 seconds of an existing label
- [ ] **LAB-03**: Annotation UI displays the last 3 labeled hits for quick review
- [ ] **LAB-04**: Annotation UI keeps the key controls panel at a fixed height while the video area resizes

### Documentation

- [ ] **DOC-01**: User can follow the README example workflow and complete it successfully
- [ ] **DOC-02**: Spec describes current pipeline behavior, flags, and output artifacts

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Export Enhancements

- **SLOW-01**: User can generate slow-motion clips at a specified factor with a documented audio policy

### Model Enhancements

- **QUAL-01**: User can score clip quality and auto-reject low-quality swings
- **SHOT-01**: User can classify swing type (forehand/backhand/serve) in metadata
- **AL-01**: User can run an active-learning labeling loop to prioritize uncertain clips

### Reproducibility

- **REP-01**: User can save and replay run configs with versioned manifests

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Cloud upload + hosted processing | Conflicts with offline/local constraint and privacy goals |
| Automatic social posting | Increases risk of unwanted exposure; sharing is manual |
| Proprietary camera lock-in | Must support standard iPhone video formats |
| Full match analytics dashboards | Large scope, not core to clip extraction |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PROC-01 | Phase 1 | Pending |
| PROC-02 | Phase 1 | Pending |
| DET-01 | Phase 2 | Pending |
| DET-02 | Phase 2 | Pending |
| OUT-01 | Phase 3 | Pending |
| OUT-02 | Phase 3 | Pending |
| OUT-03 | Phase 3 | Pending |
| REV-01 | Phase 4 | Pending |
| ORG-01 | Phase 1 | Pending |
| LAB-01 | Phase 5 | Pending |
| LAB-02 | Phase 5 | Pending |
| LAB-03 | Phase 5 | Pending |
| LAB-04 | Phase 5 | Pending |
| DOC-01 | Phase 6 | Pending |
| DOC-02 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-02-16*
*Last updated: 2026-02-16 after initial definition*

# Roadmap: Tennis Cut

## Overview

This roadmap delivers a reliable CLI pipeline for cutting tennis swing clips, starting with dependable media intake and deterministic outputs, then strengthening detection and exports, followed by review and labeling ergonomics, and finishing with documentation that makes the workflow reproducible end-to-end.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Media Intake & Deterministic Outputs** - Run local videos reliably with predictable output structure
- [ ] **Phase 2: Impact Detection** - Generate and verify impact timestamps automatically
- [ ] **Phase 3: Clip & Metadata Export** - Produce buffered swing clips, highlights, and metadata
- [ ] **Phase 4: Review & Correction** - Review detections and correct false hits
- [ ] **Phase 5: Labeling UI Ergonomics** - Faster, safer manual labeling for training data

## Phase Details

### Phase 1: Media Intake & Deterministic Outputs
**Goal**: Users can run the pipeline on local iPhone videos with predictable outputs
**Depends on**: Nothing (first phase)
**Requirements**: PROC-01, PROC-02, ORG-01
**Success Criteria** (what must be TRUE):
  1. User can process a single video or a folder of videos in one command.
  2. User can process common iPhone H.264/AAC videos without manual transcoding.
  3. Outputs use deterministic naming and folder structure in the output directory.
**Plans**: TBD

### Phase 2: Impact Detection
**Goal**: Users can automatically detect and verify impact timestamps
**Depends on**: Phase 1
**Requirements**: DET-01, DET-02
**Success Criteria** (what must be TRUE):
  1. User can generate impact timestamps automatically from audio.
  2. User can run detection that verifies candidates with vision-based person detection.
**Plans**: TBD

### Phase 3: Clip & Metadata Export
**Goal**: Users can export swing clips and metadata around detected impacts
**Depends on**: Phase 2
**Requirements**: OUT-01, OUT-02, OUT-03
**Success Criteria** (what must be TRUE):
  1. User can generate swing clips with pre/post buffers around contact.
  2. User can output a stitched highlight video and/or per-swing clips.
  3. User can export per-swing metadata in JSON or CSV.
**Plans**: TBD

### Phase 4: Review & Correction
**Goal**: Users can review detections and correct mistakes before export
**Depends on**: Phase 3
**Requirements**: REV-01
**Success Criteria** (what must be TRUE):
  1. User can review detected swings before finalizing outputs.
  2. User can adjust or remove false hits to produce a corrected set.
**Plans**: TBD

### Phase 5: Labeling UI Ergonomics
**Goal**: Users can label impacts efficiently with fewer mistakes
**Depends on**: Phase 4
**Requirements**: LAB-01, LAB-02, LAB-03, LAB-04
**Success Criteria** (what must be TRUE):
  1. User can undo the last impact label in the annotation UI.
  2. Annotation UI prevents new labels within 0.5 seconds of an existing label.
  3. Annotation UI displays the last 3 labeled hits for quick review.
  4. Annotation UI keeps the key controls panel at a fixed height while the video area resizes.
**Plans**: TBD

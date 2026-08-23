# Visual contact-frame prototype verdict

## Question

Can stock YOLO ball-and-racket detections plus neighboring-frame evidence select a trustworthy discrete forehand contact frame with an absolute confidence gate?

## Verdict

Yes, as the input to a main-player-focused temporal selection pipeline—not as a per-frame contact detector. The validated prototype:

- uses the largest person as the user and rejects rackets associated with other people;
- treats YOLO confidence as an object-plausibility signal rather than the primary contact-timing score;
- hard-rejects stationary ball detections;
- scores direct racket/ball proximity, short contact-time ball disappearance, and ball-direction change;
- lets a deterministic scorer select an existing source frame;
- uses a lightweight temporal ranker for corroboration and contact confidence; and
- omits a swing when the scorer and ranker disagree by more than one frame or its confidence falls below the operating threshold.

The temporal ranker was evaluated with complete adjacent camera-roll families held out. At the operating threshold chosen for maximum coverage with at least 95% within-one-frame precision, the 101-forehand pilot produced:

- 49 included swings (48.5% coverage);
- 47 of 49 within one frame of the weak manual label (95.9%);
- 25 of 49 on the manual frame (51.0%);
- 21 omissions below the confidence threshold; and
- 31 omissions because the deterministic scorer and temporal ranker disagreed.

The confirmed Ruud probe selected frame 635 exactly. Review examples `IMG_8640_45s` frame 4767, `IMG_8571_45s` frame 5274, and `IMG_8632_45s` frame 5191 were also selected exactly and cleared the operating threshold.

The stock detection pass for the compact pilot completed in 596 seconds on CPU. A warm round-three scoring, cross-validation, and self-contained gallery pass completed in 141 seconds. Both are comfortably inside the 60-minute prototype constraint.

## Reproduce

From a checkout of this branch with the local pilot media and `yolov8n.pt` available:

```bash
uv run visual-contact-prototype
```

The command writes disposable detection caches, per-swing results, a precision–coverage summary, and the self-contained visual review gallery under `out/contact-frame-prototype-v3/`. Generated media evidence is intentionally not committed.

## Decision carried forward

Proceed with visual contact-frame selection for forehands. Do not fine-tune the detector first: use stock detections as features, retain the deterministic discrete-frame selector, add temporal corroboration/confidence, and omit low-confidence disagreements. Keep this comparison-only; legacy `tennis-cut` timing remains unchanged.

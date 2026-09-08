# Contact detector bake-off

`prototype_contact_bakeoff.py` is disposable experiment code for the reviewed
side-groundstroke dataset. Generated frames, features, checkpoints, and reports
are stored under the ignored `out/contact-bakeoff/` directory.

Run the stages from the repository root:

```console
uv run python src/train_contact_detector/prototype_contact_bakeoff.py audit
uv run python src/train_contact_detector/prototype_contact_bakeoff.py prepare-audio --device cpu
uv run python src/train_contact_detector/prototype_contact_bakeoff.py prepare-frames --device cpu
uv run python src/train_contact_detector/prototype_contact_bakeoff.py extract-embeddings --device cpu
uv run python src/train_contact_detector/prototype_contact_bakeoff.py train-frame-probe
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-frame-probe
uv run python src/train_contact_detector/prototype_contact_bakeoff.py fine-tune-frame --device cpu --epochs 3 --learning-rate 0.0001
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-fine-tuned-fusion
uv run python src/train_contact_detector/prototype_contact_bakeoff.py train-temporal-probe --epochs 300 --learning-rate 0.001
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-temporal-probe
uv run python src/train_contact_detector/prototype_contact_bakeoff.py fine-tune-temporal --device cpu --epochs 2 --learning-rate 0.00005
uv run python src/train_contact_detector/prototype_contact_bakeoff.py build-report
uv run python src/train_contact_detector/prototype_contact_bakeoff.py prepare-objects --device cpu
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-structured
uv run python src/train_contact_detector/prototype_contact_bakeoff.py train-structured
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-structured-fusion
uv run python src/train_contact_detector/prototype_contact_bakeoff.py train-kinematic
uv run python src/train_contact_detector/prototype_contact_bakeoff.py train-family-kinematic
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-kinematic-fusion
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-kinematic-ensemble
uv run python src/train_contact_detector/prototype_contact_bakeoff.py export-kinematic-ensemble
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-kinematic-confidence
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-ambiguity-anomaly
uv run python src/train_contact_detector/prototype_contact_bakeoff.py train-racket-only
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-racket-fusion
uv run python src/train_contact_detector/prototype_contact_bakeoff.py audit-agreement
uv run python src/train_contact_detector/prototype_contact_bakeoff.py audit-contact-quality
uv run python src/train_contact_detector/prototype_contact_bakeoff.py prepare-poses --device cpu
uv run python src/train_contact_detector/prototype_contact_bakeoff.py train-pose-only
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-pose-only-fusion
uv run python src/train_contact_detector/prototype_contact_bakeoff.py train-pose-kinematic
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-pose-kinematic-fusion
uv run python src/train_contact_detector/prototype_contact_bakeoff.py evaluate-view-suitability --device mps
```

Open `out/contact-bakeoff/contact-results.html` to inspect the current best
method's exact, tolerated-neighbor, and failed selections beside the reviewed
contact states.

## Expanded side-groundstroke results

The expanded review contains 300 forehands from 20 recording families. Five
families (63 contacts) are held aside as unsupported views. The grouped
70/15/15 optimizer prioritizes normalized status balance, producing 164 train,
38 validation, and 35 test contacts; unresolved contacts are split 3/2/2.

There are 36 resolved validation contacts and 33 resolved test contacts. The
current selected prototype is an 85/15 blend of ordinary per-contact training
weights and weights that give every recording family equal total influence.
The blend is selected on validation only.

| Scorer | Validation tolerant | Validation exact | Test tolerant | Test exact |
| --- | ---: | ---: | ---: | ---: |
| Production deterministic rules | 91.7% | 66.7% | 90.9% | 42.4% |
| Kinematic structured ranker | **100.0%** | **83.3%** | 97.0% | **84.8%** |
| Family-balanced kinematic ranker | **100.0%** | 75.0% | 97.0% | 78.8% |
| Validation-selected ensemble | **100.0%** | **83.3%** | **100.0%** | **84.8%** |

The ensemble finds all 69 resolved held-out contacts within one adjacent visual
state. This is still a prototype estimate over a modest number of recording
families, not a final accuracy claim.

The exported 357 KB JSON artifact contains both tree ensembles and their blend
weight. Production inference has no scikit-learn dependency, and export checks
that its selected frame exactly matches the prototype for all 69 held-out
contacts before replacing the bundled artifact.

Wide single-swing searches are scored as one continuous region. Tiled decoding
is only an implementation detail; independently normalizing each tile produced
a confident ready-position false positive on
`ruud_forehand_side_clip_4x.mp4`. Global scoring selects frame 636, visually
within one frame of contact (roughly frames 635-636).

The separate one-class localizability gate is trained only on resolved,
in-scope training contacts. At its strict validation threshold it accepts all
69 resolved validation/test contacts, rejects both validation unresolved
contacts, and rejects one of two test unresolved contacts. Four held-out
unresolved contacts are too few to consider this gate calibrated.

A frozen image-embedding suitability classifier was rejected. Leave-one-family-
out evaluation retained all 15 supported camera views but rejected only one of
five unsupported views. Camera suitability therefore remains a separate,
unsolved problem requiring explicit view geometry or more dedicated labels.

Audio remains a candidate-window prior. It supplies a usable candidate for
299/300 reviewed contacts with 16.7 ms median absolute timing error, but adding
it to final frame scores reduced held-out accuracy.

## Proposed runtime module

The external seam should be one deep module:

```python
detect_contact(video, swing) -> ContactDecision
```

`ContactDecision` either contains one real source `frame` or an omission
`reason` (`unsupported_view`, `unlocalizable`, `insufficient_evidence`, or
`unsupported_shot_type`). The caller should not select models, fuse scores, or
manage detection caches.

The implementation owns these internal steps:

1. Reuse a decision cached by video fingerprint, swing window, shot type, and
   detector version.
2. Use audio only to narrow the candidate window.
3. Determine view/stroke support independently of contact localizability.
4. Extract stable player crops and temporal person/racket/ball evidence.
5. Select one existing source frame with the kinematic ensemble.
6. Apply the localizability gate; omit rather than return a doubtful frame.
7. Cache the frame ordinal, PTS/time base, status, reason, and detector version
   in the existing annotation JSON under a field distinct from human
   `contact_review` labels.

The implemented decision cache uses the video size and modification time, exact
rational candidate window, and a hash of the portable model. It adds only a
top-level `contact_detection_cache` field when the video's existing annotation
JSON is valid; without one it writes `<video>.<ext>.contact.json`, avoiding the
annotation filename. Both accepted exact-frame identities and omissions can be
reused.

Side-view groundstrokes are the first internal adapter. Back-view groundstrokes
and serves should become additional internal adapters only after labeled data
exists; they share `ContactDecision` and caching without forcing their mechanics
into the side-groundstroke ranker.

## Earlier prototype results

The validation and test columns each contain 19 resolved contacts from held-out
recording families. Tolerant accuracy accepts the reviewed visual state or an
immediately neighboring visual state. Exact accuracy accepts only a frame in a
reviewed contact state.

| Scorer | Validation tolerant | Test tolerant | Test exact |
| --- | ---: | ---: | ---: |
| Audio only | 57.9% | 57.9% | 26.3% |
| Frozen single-frame visual | 42.1% | 42.1% | 26.3% |
| Frozen single-frame + audio | 73.7% | **73.7%** | **47.4%** |
| Fine-tuned single-frame visual | 57.9% | 47.4% | 15.8% |
| Fine-tuned single-frame + audio | **84.2%** | 68.4% | 42.1% |
| Frozen temporal visual | 52.6% | 36.8% | 26.3% |
| Frozen temporal + audio | **84.2%** | 63.2% | 36.8% |
| Jointly tuned temporal visual | 31.6% | 26.3% | 10.5% |
| Jointly tuned temporal + audio | 78.9% | **73.7%** | 31.6% |
| Production structured features, retrained | 100.0% | 94.7% | 57.9% |
| Kinematic structured features | **100.0%** | **100.0%** | **89.5%** |
| Racket-only kinematic features | 84.2% | 68.4% | 36.8% |
| Pose-only temporal features | 68.4% | 52.6% | 15.8% |
| Pose-only + audio | 73.7% | 63.2% | 26.3% |
| Pose + kinematic structured features | **100.0%** | **100.0%** | 84.2% |

These are prototype-selection results, not final accuracy estimates. The audio
prior is candidate-conditioned: for each reviewed swing it uses the strongest
dense audio window within 250 ms of the review. It therefore measures timing
quality after associating an audio candidate with a known swing, not end-to-end
audio candidate recall.

The learned-image experiments show strong recording-family overfitting.
Increasing learned visual or temporal capacity did not improve held-out
results. Adding explicit body-relative racket and ball motion did: the
kinematic structured model found every held-out contact within tolerance and
selected 17/19 exact test states. It also found all 11 held-out contacts where
the ball was absent on the reviewed contact state, using observations elsewhere
in the local window. The racket-only ablation failed, so a completely hidden
ball needs an independent fallback. Pose-only timing also failed on this
dataset, and adding pose to the full kinematic ranker retained tolerant accuracy
but reduced exact test accuracy. Pose may still be useful for a future
stroke-specific model or ambiguity gate, but it does not belong in the
side-groundstroke localizer based on this result.

Raw peak margin is not an adequate abstention score. A validation-calibrated
margin accepted ambiguous and unsupported-view examples on test. Contact
ambiguity and camera suitability need separate gates. Detection visibility is
also not a suitability proxy: the out-of-scope 3/4-view clips generally had
strong ball and racket detections. Cross-model agreement, including an
independent pose-only estimate, still accepted ambiguous contacts and is not a
reliable ambiguity gate.

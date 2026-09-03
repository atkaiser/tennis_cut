# Tennis Cut

Tennis Cut identifies tennis swings in source video and produces focused video artifacts for reviewing them.

## Language

**User video**:
A source video containing multiple swings by the person whose technique is being compared.
_Avoid_: Player video, learner video

**Pro video**:
A source video containing one swing by the professional whose technique is used for comparison.
_Avoid_: Reference video, player video

**Swing**:
A single tennis stroke, from preparation through follow-through.
_Avoid_: Shot, when referring to the complete motion

**Contact timestamp**:
The presentation time of the contact frame selected for a swing. It is the swing's alignment anchor, not a claim that the source captured the physical instant of contact exactly.
_Avoid_: Impact time, contact point

**Contact frame**:
A source image automatically identified as the closest available representation of racket-ball contact. It need not depict the physical instant exactly.
_Avoid_: Exact contact frame, contact point, selected timestamp

**Contact confidence**:
The system's confidence, based on visual evidence, that an automatically identified contact frame is sufficiently close to contact for a useful comparison. At most two adjacent plausible contact frames are acceptable; broader or separated ambiguity is low confidence and excludes the swing from comparison.
_Avoid_: Swing confidence, detection confidence

**Comparison clip**:
An output video that presents two swings side by side at a shared playback speed, with their independently selected contact frames aligned.
_Avoid_: Comparison video, matched clip

**Comparison compilation**:
The primary silent output containing comparison clips in accepted user-swing order with hard cuts between them.
_Avoid_: Stitched compilation, comparison video

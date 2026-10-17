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

**Candidate window**:
A bounded source-time view around a plausible Swing. It retains the original
source-frame identities and is not a separately encoded video.
_Avoid_: Clip, temporary video

**Consumer-sized frame**:
An exact source frame decoded directly at the dimensions needed by a detector
or render panel. Its pixels may be scaled or cropped, while its source ordinal
and presentation timestamp remain unchanged.
_Avoid_: Source frame, when referring to the transformed image

**Contact timestamp**:
The presentation time of the contact frame selected for a swing. It is the swing's alignment anchor, not a claim that the source captured the physical instant of contact exactly.
_Avoid_: Impact time, contact point

**Contact frame**:
A source image selected as the closest available representation of racket-ball contact. User-video contact frames are selected automatically; a pro-video contact frame may be supplied manually when automatic selection fails. It need not depict the physical instant exactly.
_Avoid_: Exact contact frame, contact point, selected timestamp

**Contact-frame tolerance**:
The evaluation allowance expressed over distinct visual states rather than raw source-frame adjacency. For a Single-state contact, every source frame representing the labeled state is acceptable. For a Between-state contact, every source frame representing either bounding state is acceptable. The system still selects one contact frame.
_Avoid_: Contact interval, time tolerance

**Visual state**:
A distinct source image in a swing sequence. Frame-rate conversion may represent one visual state with multiple consecutive source frames having separate ordinals and presentation timestamps.
_Avoid_: Unique frame, exact instant

**Single-state contact**:
A swing whose local sequence establishes contact and has one visual state that is clearly its best representation. The ball need not appear in that state when surrounding states make contact clear.
_Avoid_: Single-frame contact, obvious contact, visible-ball contact

**Between-state contact**:
A swing whose local sequence establishes that physical contact falls between two adjacent visual states, with neither state uniquely best. Either state may be represented by multiple source frames.
_Avoid_: Between-frame contact, ambiguous contact, tied contact

**Unresolved contact**:
A swing whose available video cannot localize contact reliably because evidence is occluded, off-screen, truncated, or otherwise insufficient. The broader Swing may still be recognizable.
_Avoid_: Ambiguous swing, unclear contact, invalid swing

**Side-view suitability**:
Whether a swing's player scale and camera angle make it an in-scope example for the current side-view groundstroke comparison detector. An out-of-scope view can still have a resolved contact and remains useful for measuring appropriate omission.
_Avoid_: Contact confidence, contact clarity, valid contact

**Contact score**:
A relative value assigned to a candidate source frame for ranking it against other frames from the same swing. It is not a probability that the frame is acceptable and is not comparable to contact confidence.
_Avoid_: Contact probability, contact confidence

**Contact confidence**:
The system's confidence, based on visual evidence, that an automatically selected contact frame is sufficiently close to contact for a useful comparison. At most two adjacent plausible contact frames are acceptable; broader or separated ambiguity is low confidence and excludes the swing from comparison.
_Avoid_: Swing confidence, detection confidence

**Comparison clip**:
An output video that presents two swings side by side at a shared playback speed, with their independently selected contact frames aligned.
_Avoid_: Comparison video, matched clip

**Comparison compilation**:
The primary silent output containing comparison clips in accepted user-swing order with hard cuts between them.
_Avoid_: Stitched compilation, comparison video

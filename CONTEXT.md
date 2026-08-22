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
The estimated instant in a source video when the racket contacts the ball.
_Avoid_: Impact time, contact point

**Contact frame**:
The exact pro-video image selected by the user as the instant the racket contacts the ball. Its presentation time defines the pro swing's contact timestamp.
_Avoid_: Contact point, selected timestamp

**Comparison clip**:
An output video that presents two swings side by side at a shared playback speed, with their contact timestamps aligned.
_Avoid_: Comparison video, matched clip

**Comparison compilation**:
The primary silent output containing comparison clips in accepted user-swing order with hard cuts between them.
_Avoid_: Stitched compilation, comparison video

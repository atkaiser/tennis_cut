"""Candidate-level diagnostics for user swing and contact selection."""

from __future__ import annotations

import base64
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
import html
from pathlib import Path

from tennis_cut.visual_contact import (
    ContactSelection,
    RankedFrame,
    SourceFrameIdentity,
    TemporalPrediction,
    VisualFrame,
    decode_exact_frame_images,
)


@dataclass(frozen=True)
class CandidateDiagnostic:
    """The complete disposition of one audio-located swing candidate."""

    audio_candidate_index: int
    audio_timestamp: Fraction
    disposition: str
    reason: str
    audio_score: float | None = None
    shot_type: str | None = None
    contact_frame_ordinal: int | None = None
    contact_frame: SourceFrameIdentity | None = None
    contact_confidence: float = 0.0
    plausible_frames: tuple[int, ...] = ()
    deterministic_frame: int | None = None
    temporal_frame: int | None = None
    temporal_confidence: float | None = None
    confidence_threshold: float | None = None
    ranked_frames: tuple[RankedFrame, ...] = ()
    visual_frames: tuple[VisualFrame, ...] = ()


@dataclass(frozen=True)
class SwingDiagnostics:
    """All audio candidates observed for one user video."""

    source: Path
    candidates: tuple[CandidateDiagnostic, ...]


@dataclass
class _CandidateState:
    index: int
    audio_timestamp: Fraction
    audio_score: float | None = None
    disposition: str = "pending"
    reason: str = "awaiting classification"
    shot_type: str | None = None
    selection: ContactSelection | None = None
    comparison_swing_ordinal: int | None = None


@dataclass
class SwingDiagnosticsRecorder:
    """Mutable observer that preserves decisions made by the detection pipeline."""

    source: Path
    _candidates: dict[int, _CandidateState] = field(default_factory=dict)

    def record_audio_candidates(self, candidates: Iterable[object]) -> None:
        states: dict[int, _CandidateState] = {}
        for fallback_index, item in enumerate(candidates):
            timestamp = getattr(item, "timestamp", item)
            index = getattr(item, "source_index", fallback_index)
            score = getattr(item, "score", None)
            states[index] = _CandidateState(
                index,
                Fraction(timestamp).limit_denominator(48_000),
                audio_score=score,
            )
        self._candidates = states

    def omit(self, index: int, reason: str) -> None:
        candidate = self._candidates[index]
        candidate.disposition = "omitted"
        candidate.reason = reason

    def accept_swing_candidate(self, index: int, shot_type: str | None) -> None:
        candidate = self._candidates[index]
        candidate.disposition = "visual contact pending"
        candidate.reason = "accepted by person and swing classifiers"
        candidate.shot_type = shot_type

    def record_visual_selection(
        self,
        index: int,
        selection: ContactSelection,
        comparison_swing_ordinal: int | None = None,
    ) -> None:
        candidate = self._candidates[index]
        candidate.selection = selection
        if selection.frame is None:
            candidate.disposition = "omitted"
            candidate.reason = selection.omission_reason or "visual contact unavailable"
        else:
            candidate.disposition = "accepted"
            candidate.reason = "visual contact accepted"
            candidate.comparison_swing_ordinal = comparison_swing_ordinal

    def record_planning(
        self,
        rendered_swing_ordinals: set[int],
        pro_shot_type: str,
    ) -> None:
        """Record why a visually accepted swing did or did not reach rendering."""

        for candidate in self._candidates.values():
            ordinal = candidate.comparison_swing_ordinal
            if candidate.disposition != "accepted" or ordinal is None:
                continue
            if candidate.shot_type != pro_shot_type:
                candidate.disposition = "omitted"
                candidate.reason = (
                    f"shot type {candidate.shot_type or 'unclassified'} does not "
                    f"match pro shot type {pro_shot_type}"
                )
            elif ordinal not in rendered_swing_ordinals:
                candidate.disposition = "omitted"
                candidate.reason = "incomplete pre-contact or post-contact window"
            else:
                candidate.reason = "selected for comparison rendering"

    def snapshot(self) -> SwingDiagnostics:
        candidates = []
        for state in sorted(self._candidates.values(), key=lambda item: item.index):
            selection = state.selection
            decision = selection.diagnostics if selection is not None else None
            prediction: TemporalPrediction | None = (
                None if decision is None else decision.temporal_prediction
            )
            candidates.append(
                CandidateDiagnostic(
                    audio_candidate_index=state.index,
                    audio_timestamp=state.audio_timestamp,
                    disposition=state.disposition,
                    reason=state.reason,
                    audio_score=state.audio_score,
                    shot_type=state.shot_type,
                    contact_frame_ordinal=(
                        None if selection is None else selection.selected_frame
                    ),
                    contact_frame=(
                        None
                        if selection is None or selection.frame is None
                        else selection.frame.evidence.identity
                    ),
                    contact_confidence=(
                        0.0 if selection is None else selection.contact_confidence
                    ),
                    plausible_frames=(
                        () if selection is None else selection.plausible_frames
                    ),
                    deterministic_frame=(
                        None
                        if decision is None
                        else decision.deterministic.selected_frame
                    ),
                    temporal_frame=(
                        None if prediction is None else prediction.frame_ordinal
                    ),
                    temporal_confidence=(
                        None if prediction is None else prediction.confidence
                    ),
                    confidence_threshold=(
                        None if decision is None else decision.confidence_threshold
                    ),
                    ranked_frames=(
                        ()
                        if decision is None
                        else decision.deterministic.ranked_frames
                    ),
                    visual_frames=(() if decision is None else decision.frames),
                )
            )
        return SwingDiagnostics(self.source, tuple(candidates))


def _seconds(value: Fraction) -> str:
    return f"{float(value):.6f}s"


def _candidate_image(candidate: CandidateDiagnostic, source: Path) -> str:
    import cv2

    if not source.is_file():
        return ""
    wanted: set[int] = set(candidate.plausible_frames)
    centers = {
        frame
        for frame in (
            candidate.deterministic_frame,
            candidate.temporal_frame,
            candidate.contact_frame_ordinal,
        )
        if frame is not None
    }
    evidence = {frame.ordinal: frame.evidence for frame in candidate.visual_frames}
    if evidence:
        audio_nearest = min(
            evidence.values(),
            key=lambda frame: abs(frame.timestamp - candidate.audio_timestamp),
        )
        centers.add(audio_nearest.ordinal)
    for center in centers:
        wanted.update(range(center - 2, center + 3))
    wanted.update(ranked.ordinal for ranked in candidate.ranked_frames[:10])
    if not wanted:
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(candidate.audio_timestamp) * 1000)
        ok, image = capture.read()
        capture.release()
        if not ok:
            return ""
        label = f"audio candidate at {_seconds(candidate.audio_timestamp)}"
        return _image_figure(image, label, ())

    displayed_ordinals = tuple(sorted(wanted & evidence.keys()))
    decoded = decode_exact_frame_images(source, displayed_ordinals)
    figures = []
    for ordinal in displayed_ordinals:
        frame = evidence.get(ordinal)
        if frame is None:
            continue
        image = decoded[ordinal]
        tags = []
        if ordinal == candidate.deterministic_frame:
            tags.append("deterministic")
        if ordinal == candidate.temporal_frame:
            tags.append("ranker")
        if ordinal == candidate.contact_frame_ordinal:
            tags.append("chosen contact")
        if abs(frame.timestamp - candidate.audio_timestamp) == min(
            abs(item.timestamp - candidate.audio_timestamp)
            for item in evidence.values()
        ):
            tags.append("nearest audio peak")
        label = f"frame {ordinal} · {_seconds(frame.timestamp)}"
        if tags:
            label += " · " + " + ".join(tags)
        figures.append(_image_figure(image, label, frame.detections))
    return "".join(figures)


def _image_figure(image: object, label: str, detections: tuple[object, ...]) -> str:
    import cv2

    canvas = image.copy()
    colors = {"person": (90, 190, 80), "ball": (20, 210, 255), "racket": (255, 120, 40)}
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection.box)
        color = colors[detection.kind]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{detection.kind} {detection.confidence:.2f}",
            (x1, max(22, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    target_width = min(960, canvas.shape[1])
    if canvas.shape[1] != target_width:
        canvas = cv2.resize(
            canvas,
            (target_width, round(canvas.shape[0] * target_width / canvas.shape[1])),
        )
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 76])
    if not ok:
        return ""
    data = base64.b64encode(encoded).decode()
    return (
        f'<figure><img loading="lazy" src="data:image/jpeg;base64,{data}">'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
    )


def _candidate_card(candidate: CandidateDiagnostic, source: Path) -> str:
    status_class = "accepted" if candidate.disposition == "accepted" else "omitted"
    contact = "none"
    if candidate.contact_frame_ordinal is not None and candidate.contact_frame is not None:
        contact = (
            f"frame {candidate.contact_frame_ordinal} at "
            f"{_seconds(candidate.contact_frame.timestamp)}"
        )
    plausible = ", ".join(map(str, candidate.plausible_frames)) or "none"
    temporal = "not reached"
    if candidate.temporal_frame is not None:
        temporal = (
            f"frame {candidate.temporal_frame}, raw confidence "
            f"{candidate.temporal_confidence:.3f}, threshold "
            f"{candidate.confidence_threshold:.3f}"
        )
    score = (
        ""
        if candidate.audio_score is None
        else f" · audio score {candidate.audio_score:.3f}"
    )
    ranking = "".join(
        "<tr>"
        f"<td>{ranked.ordinal}</td><td>{ranked.score:.3f}</td>"
        f"<td>{ranked.direct_proximity:.3f}</td>"
        f"<td>{ranked.disappearance:.3f}</td>"
        f"<td>{ranked.trajectory:.3f}</td>"
        "</tr>"
        for ranked in candidate.ranked_frames[:10]
    )
    table = ""
    if ranking:
        table = (
            "<details><summary>Visual scoring</summary>"
            f"<p>Deterministic frame: {candidate.deterministic_frame}; temporal ranker: {html.escape(temporal)}; plausible frames {plausible}</p>"
            "<table><thead><tr><th>frame</th><th>score</th><th>direct</th><th>disappearance</th><th>trajectory</th></tr></thead>"
            f"<tbody>{ranking}</tbody></table></details>"
        )
    return f'''<article class="candidate {status_class}">
<header><div><h2>Candidate {candidate.audio_candidate_index}</h2><p>Audio peak {_seconds(candidate.audio_timestamp)}{score} · {html.escape(candidate.shot_type or "unclassified")}</p></div><strong>{html.escape(candidate.disposition)}</strong></header>
<p class="reason">{html.escape(candidate.reason)}</p>
<p>Chosen contact: {contact} · confidence {candidate.contact_confidence:.3f} · plausible frames {plausible}</p>
<div class="filmstrip">{_candidate_image(candidate, source)}</div>{table}
</article>'''


def write_swing_diagnostics_report(output: Path, diagnostics: SwingDiagnostics) -> None:
    """Write a self-contained visual audit of every audio candidate."""

    accepted = sum(candidate.disposition == "accepted" for candidate in diagnostics.candidates)
    omitted = len(diagnostics.candidates) - accepted
    cards = "".join(
        _candidate_card(candidate, diagnostics.source)
        for candidate in diagnostics.candidates
    )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Swing detection diagnostics</title>
<style>
:root{{--ink:#18201d;--muted:#68736d;--paper:#f5f2ea;--card:#fff;--line:#d9d5c9;--good:#176b51;--bad:#a43d32}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:42px 28px 100px}}h1{{font:700 42px/1.05 Georgia,serif;margin:0 0 10px}}.summary{{color:var(--muted);font-size:18px}}.candidate{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}}header{{display:flex;justify-content:space-between;gap:20px}}header h2{{margin:0}}header p{{margin:4px 0;color:var(--muted)}}header strong{{text-transform:uppercase}}.accepted header strong{{color:var(--good)}}.omitted header strong{{color:var(--bad)}}.reason{{font-weight:700;color:var(--ink)}}.filmstrip{{display:flex;overflow:auto;gap:14px;padding:12px 0;scroll-snap-type:x proximity}}figure{{margin:0;flex:0 0 min(960px,calc(100vw - 96px));scroll-snap-align:start}}figure img{{width:100%;border-radius:8px}}figcaption{{color:var(--muted)}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid var(--line);text-align:right}}td:first-child,th:first-child{{text-align:left}}@media(max-width:800px){{main{{padding:24px 14px 80px}}figure{{flex-basis:calc(100vw - 60px)}}}}
</style></head><body><main><h1>User swing diagnostics</h1><p class="summary">{len(diagnostics.candidates)} audio candidates · {accepted} accepted · {omitted} omitted · {html.escape(str(diagnostics.source))}</p>{cards}</main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


__all__ = [
    "CandidateDiagnostic",
    "SwingDiagnostics",
    "SwingDiagnosticsRecorder",
    "write_swing_diagnostics_report",
]

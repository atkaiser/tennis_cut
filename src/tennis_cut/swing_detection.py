"""Shared user-video swing detection."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterable

from PIL import Image
import torchaudio

from utilities import PersonDetector, expand_box

if TYPE_CHECKING:
    from .comparison.diagnostics import SwingDiagnosticsRecorder

if __package__:
    from .subprocess_utils import run_command
    from .visual_contact import (
        FrameTimeline,
        SourceFrameIdentity,
        StockVisualContactSelector,
        VisualContactSelector,
    )
else:
    from subprocess_utils import run_command
    from visual_contact import (
        FrameTimeline,
        SourceFrameIdentity,
        StockVisualContactSelector,
        VisualContactSelector,
    )


DEFAULT_AUDIO_MODEL = Path("models/audio_pop_logmel_large_20260512231349.pth")
DEFAULT_SHOT_MODEL = Path("models/shot_binary_classifier_20260328143535.pkl")
DEFAULT_SHOT_TYPE_MODEL = Path("models/shot_type_classifier_20260328220857.pkl")
DEFAULT_TEMPORAL_RANKER_MODEL = (
    Path(__file__).resolve().parent / "models" / "temporal_ranker.json"
)

DEFAULT_STRIDE_S = 0.05
SAMPLE_RATE = 48_000
WINDOW_DURATION = 0.25
PEAK_THRESHOLD = 0.5
INITIAL_PEAK_MIN_SEPARATION = 0.25
BOUNCE_GAP_MIN = 0.35
BOUNCE_GAP_MAX = 0.75
FINAL_PEAK_MIN_SEPARATION = 1.25
BOUNCE_COLLAPSE_REASON = "bounce normalization: earlier short-gap precursor"
FINAL_SUPPRESSION_REASON = "final audio suppression: lower-scoring nearby candidate"
BATCH_SIZE = 128
PRE_CONTACT_BUFFER = 1.20
POST_CONTACT_BUFFER = 0.70

_LOG = logging.getLogger(__name__)

__all__ = [
    "DetectedSwing",
    "DetectionConfig",
    "detect_comparison_user_swings",
    "detect_user_swings",
]


@dataclass(frozen=True)
class AudioCandidate:
    """One scored audio event carried through swing classification."""

    timestamp: float
    score: float
    source_index: int


@dataclass(frozen=True)
class DetectedSwing:
    """One accepted swing exposed to detection callers."""

    ordinal: int
    contact_timestamp: Fraction
    shot_type: str | None
    contact_frame: SourceFrameIdentity | None = None


@dataclass(frozen=True)
class DetectionConfig:
    """Model and device choices for user-video detection."""

    audio_model: Path = DEFAULT_AUDIO_MODEL
    shot_model: Path | None = DEFAULT_SHOT_MODEL
    shot_type_model: Path | None = DEFAULT_SHOT_TYPE_MODEL
    device: str | None = None
    temporal_ranker_model: Path | None = DEFAULT_TEMPORAL_RANKER_MODEL


@dataclass(frozen=True)
class LegacySwingDetails:
    """Detection details retained by the existing rendering command."""

    swing: DetectedSwing
    legacy_contact: float
    start: float
    end: float
    crop: tuple[int, int, int, int]
    audio_candidate_index: int | None = None


def _suppress_audio_candidates(
    candidates: tuple[AudioCandidate, ...] | list[AudioCandidate],
    minimum_separation: float,
) -> tuple[tuple[AudioCandidate, ...], tuple[AudioCandidate, ...]]:
    """Prefer strong audio events while retaining exact-boundary neighbors."""

    preferred = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.timestamp,
            candidate.source_index,
        ),
    )
    kept: list[AudioCandidate] = []
    omitted: list[AudioCandidate] = []
    for candidate in preferred:
        if all(
            _at_least(
                abs(candidate.timestamp - selected.timestamp),
                minimum_separation,
            )
            for selected in kept
        ):
            kept.append(candidate)
        else:
            omitted.append(candidate)
    return (
        tuple(
            sorted(
                kept,
                key=lambda candidate: (candidate.timestamp, candidate.source_index),
            )
        ),
        tuple(
            sorted(
                omitted,
                key=lambda candidate: (candidate.timestamp, candidate.source_index),
            )
        ),
    )


def _at_least(value: float, boundary: float) -> bool:
    return value > boundary or math.isclose(
        value,
        boundary,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _at_most(value: float, boundary: float) -> bool:
    return value < boundary or math.isclose(
        value,
        boundary,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def _normalize_bounce_candidates(
    candidates: tuple[AudioCandidate, ...] | list[AudioCandidate],
) -> tuple[tuple[AudioCandidate, ...], tuple[AudioCandidate, ...]]:
    """Collapse timing-only bounce-like chains to their final audio event."""

    if not candidates:
        return (), ()
    chronological = sorted(
        candidates,
        key=lambda candidate: (candidate.timestamp, candidate.source_index),
    )
    kept: list[AudioCandidate] = []
    omitted: list[AudioCandidate] = []
    chain_tail = chronological[0]
    for candidate in chronological[1:]:
        gap = candidate.timestamp - chain_tail.timestamp
        if _at_least(gap, BOUNCE_GAP_MIN) and _at_most(gap, BOUNCE_GAP_MAX):
            omitted.append(chain_tail)
            chain_tail = candidate
        else:
            kept.append(chain_tail)
            chain_tail = candidate
    kept.append(chain_tail)
    return tuple(kept), tuple(omitted)


def resolve_device(device: str | None) -> str:
    """Resolve an explicit or preferred available compute device."""

    if device is not None:
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(model_path: Path, device: str | None):
    import torch
    from fastai.learner import load_learner

    device = resolve_device(device)
    torch_device = torch.device(device)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="load_learner` uses Python's insecure pickle",
        )
        learner = load_learner(model_path, cpu=torch_device.type == "cpu")
    learner.to(torch_device)
    learner.model.eval()
    return torch_device, learner


class PopDetector:
    """Audio contact detector using the trained CNN."""

    def __init__(
        self,
        model_path: Path,
        stride_s: float = DEFAULT_STRIDE_S,
        device: str | None = None,
    ) -> None:
        self.stride_s = stride_s
        self.device, self.learner = _load_model(model_path, device)

    def score_windows(self, wav_path: Path) -> list[AudioCandidate]:
        """Score every dense 250 ms audio window before event suppression."""

        import pandas as pd
        import torch

        waveform, sample_rate = torchaudio.load(str(wav_path))
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, SAMPLE_RATE
            )
            sample_rate = SAMPLE_RATE

        window = int(sample_rate * WINDOW_DURATION)
        stride = int(sample_rate * self.stride_s)
        if waveform.shape[1] < window:
            return []

        starts = [
            sample_start / sample_rate
            for sample_start in range(0, waveform.shape[1] - window + 1, stride)
        ]
        frame = pd.DataFrame({"wav_path": str(wav_path), "start": starts})
        data_loader = self.learner.dls.test_dl(frame, bs=BATCH_SIZE)

        with torch.no_grad():
            predictions, _ = self.learner.get_preds(dl=data_loader, reorder=False)
            probabilities = predictions[:, 1]

        return [
            AudioCandidate(
                index * self.stride_s + (WINDOW_DURATION / 2),
                float(probability),
                index,
            )
            for index, probability in enumerate(probabilities)
        ]

    def find_candidates(self, wav_path: Path) -> list[AudioCandidate]:
        candidates = [
            candidate
            for candidate in self.score_windows(wav_path)
            if candidate.score > PEAK_THRESHOLD
        ]

        kept, _ = _suppress_audio_candidates(
            candidates,
            INITIAL_PEAK_MIN_SEPARATION,
        )
        _LOG.info("Detected %d initial audio event candidates", len(kept))
        _LOG.info(
            "Initial audio event candidates: %s",
            ", ".join(
                f"{candidate.timestamp:.3f}s ({candidate.score:.3f})"
                for candidate in kept
            ),
        )
        return list(kept)


class ShotDetector:
    """Binary image swing detector from a fastai model."""

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.device, self.learner = _load_model(model_path, device)

    def predict_label(self, image) -> str:
        prediction, _, _ = self.learner.predict(image)
        return str(prediction)

    def is_shot(self, image) -> bool:
        return self.predict_label(image) == "shot"


class ShotTypeClassifier:
    """Shot-type classifier that runs after binary swing detection."""

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.device, self.learner = _load_model(model_path, device)

    def predict_label(self, image) -> str:
        prediction, _, _ = self.learner.predict(image)
        return str(prediction)


def probe_video(video: Path) -> dict:
    """Return the media facts needed by legacy and shared detection callers."""

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if error.stderr:
            _LOG.error(error.stderr.strip())
        raise
    metadata = json.loads(result.stdout)
    video_stream = next(
        stream for stream in metadata["streams"] if stream["codec_type"] == "video"
    )
    audio_stream = next(
        stream for stream in metadata["streams"] if stream["codec_type"] == "audio"
    )
    numerator, denominator = video_stream["r_frame_rate"].split("/")
    return {
        "fps": float(numerator) / float(denominator),
        "resolution": (video_stream["width"], video_stream["height"]),
        "audio_codec": audio_stream.get("codec_name"),
    }


def extract_audio(video: Path, wav_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-i",
            str(video),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            str(wav_path),
            "-y",
        ]
    )


def extract_frame(video: Path, timestamp: float, output_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output_path),
            "-y",
        ]
    )


def _exact_contact_timestamp(contact: float) -> Fraction:
    """Recover the exact audio-sample-grid time represented by ``contact``."""

    return Fraction(contact).limit_denominator(SAMPLE_RATE)


def _detect_user_swings_with_details(
    user_video: Path,
    detection_config: DetectionConfig,
    media_info: dict,
    *,
    report_progress: bool,
    diagnostics: SwingDiagnosticsRecorder | None = None,
    initial_candidates: Iterable[AudioCandidate] | None = None,
    frame_extractor: Callable[[Path, float, Path], None] | None = None,
) -> tuple[LegacySwingDetails, ...]:
    selected_device = resolve_device(detection_config.device)
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        if initial_candidates is None:
            wav_path = temporary_path / "audio.wav"
            extract_audio(user_video, wav_path)
            pop_detector = PopDetector(
                detection_config.audio_model,
                device=selected_device,
            )
            candidates = pop_detector.find_candidates(wav_path)
        else:
            candidates = list(initial_candidates)
        extract_candidate_frame = frame_extractor or extract_frame
        if diagnostics is not None:
            diagnostics.record_audio_candidates(candidates)
        event_candidates, bounce_precursors = _normalize_bounce_candidates(
            candidates
        )
        for precursor in bounce_precursors:
            _LOG.info(
                "Audio candidate %d at %.3fs omitted by bounce normalization",
                precursor.source_index,
                precursor.timestamp,
            )
            if diagnostics is not None:
                diagnostics.omit(precursor.source_index, BOUNCE_COLLAPSE_REASON)
        person_detector = PersonDetector(selected_device)
        shot_detector = (
            ShotDetector(detection_config.shot_model, device=selected_device)
            if detection_config.shot_model is not None
            else None
        )
        shot_type_classifier = (
            ShotTypeClassifier(
                detection_config.shot_type_model,
                device=selected_device,
            )
            if detection_config.shot_type_model is not None
            else None
        )

        classifier_accepted: list[tuple[AudioCandidate, LegacySwingDetails]] = []
        processing_times: list[float] = []
        for candidate in event_candidates:
            candidate_index = candidate.source_index
            contact = candidate.timestamp
            started_at = time.perf_counter()
            frame_path = temporary_path / f"impact_{candidate_index}.jpg"
            extract_candidate_frame(user_video, contact, frame_path)
            box = person_detector.find_box(frame_path)
            if box is None:
                _LOG.info("No person found for impact %d", candidate_index)
                if diagnostics is not None:
                    diagnostics.omit(candidate_index, "no person found")
                continue
            crop = expand_box(box, media_info["resolution"])
            shot_type = None
            if shot_detector is not None:
                with Image.open(frame_path) as image:
                    cropped = image.crop(
                        (crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])
                    )
                    if not shot_detector.is_shot(cropped):
                        _LOG.info("Impact %d not a swing", candidate_index)
                        if diagnostics is not None:
                            diagnostics.omit(
                                candidate_index,
                                "swing classifier: not a swing",
                            )
                        continue
                    if shot_type_classifier is not None:
                        shot_type = shot_type_classifier.predict_label(cropped)
            _LOG.info(
                "Audio candidate %d at %.3fs accepted by classifiers (shot type: %s)",
                candidate_index,
                contact,
                shot_type or "unclassified",
            )
            if diagnostics is not None:
                diagnostics.accept_swing_candidate(candidate_index, shot_type)
            detected_swing = DetectedSwing(
                ordinal=len(classifier_accepted),
                contact_timestamp=_exact_contact_timestamp(contact),
                shot_type=shot_type,
            )
            classifier_accepted.append(
                (
                    candidate,
                    LegacySwingDetails(
                        swing=detected_swing,
                        legacy_contact=contact,
                        start=contact - PRE_CONTACT_BUFFER,
                        end=contact + POST_CONTACT_BUFFER,
                        crop=tuple(crop),
                        audio_candidate_index=candidate_index,
                    ),
                )
            )
            elapsed = time.perf_counter() - started_at
            processing_times.append(elapsed)
            if report_progress:
                print(
                    f"Peak {candidate_index} at {contact:.3f}s processed in "
                    f"{elapsed:.3f}s"
                )

        if report_progress and processing_times:
            average = sum(processing_times) / len(processing_times)
            print(
                f"Average processing time per peak: {average:.3f}s "
                f"({len(processing_times)} peaks)"
            )

        surviving_candidates, final_omissions = _suppress_audio_candidates(
            [candidate for candidate, _ in classifier_accepted],
            FINAL_PEAK_MIN_SEPARATION,
        )
        for candidate in final_omissions:
            _LOG.info(
                "Audio candidate %d at %.3fs omitted by final 1.25s suppression",
                candidate.source_index,
                candidate.timestamp,
            )
            if diagnostics is not None:
                diagnostics.omit(candidate.source_index, FINAL_SUPPRESSION_REASON)
        details_by_index = {
            candidate.source_index: detail
            for candidate, detail in classifier_accepted
        }
        accepted = tuple(
            replace(
                details_by_index[candidate.source_index],
                swing=replace(
                    details_by_index[candidate.source_index].swing,
                    ordinal=ordinal,
                ),
            )
            for ordinal, candidate in enumerate(surviving_candidates)
        )

    return accepted


def detect_user_swings(
    user_video: Path, detection_config: DetectionConfig
) -> tuple[DetectedSwing, ...]:
    """Detect and classify accepted swings in one user video."""

    details = _detect_user_swings_with_details(
        user_video,
        detection_config,
        probe_video(user_video),
        report_progress=False,
    )
    return tuple(detail.swing for detail in details)


def detect_comparison_user_swings(
    user_video: Path,
    detection_config: DetectionConfig,
    *,
    frame_timeline: FrameTimeline | None = None,
    contact_selector: VisualContactSelector | None = None,
    diagnostics: SwingDiagnosticsRecorder | None = None,
) -> tuple[DetectedSwing, ...]:
    """Detect accepted swings and replace audio timing with visual contact.

    This comparison-only operation reuses the existing audio candidate and
    swing classification stages. Legacy callers use ``detect_user_swings`` or
    ``detect_user_swings_for_legacy`` and therefore retain audio-derived timing.
    """

    details = _detect_user_swings_with_details(
        user_video,
        detection_config,
        probe_video(user_video),
        report_progress=False,
        diagnostics=diagnostics,
    )
    if contact_selector is None:
        ranker = None
        if detection_config.temporal_ranker_model is not None:
            from .temporal_ranker import load_temporal_ranker

            ranker = load_temporal_ranker(detection_config.temporal_ranker_model)
        selector = StockVisualContactSelector(
            device=detection_config.device,
            ranker=ranker,
            frame_timeline=frame_timeline,
        )
    else:
        selector = contact_selector
    candidate_timestamps = tuple(
        _exact_contact_timestamp(detail.legacy_contact) for detail in details
    )
    select_many = getattr(type(selector), "select_many", None)
    if callable(select_many):
        selections = select_many(selector, user_video, candidate_timestamps)
    else:
        selections = tuple(
            selector.select(user_video, candidate_timestamp)
            for candidate_timestamp in candidate_timestamps
        )
    if len(selections) != len(details):
        raise ValueError("visual contact selector returned the wrong result count")
    accepted: list[DetectedSwing] = []
    omissions: Counter[str] = Counter()
    for detail, selection in zip(details, selections, strict=True):
        candidate_index = (
            detail.audio_candidate_index
            if detail.audio_candidate_index is not None
            else detail.swing.ordinal
        )
        if diagnostics is not None:
            diagnostics.record_visual_selection(
                candidate_index,
                selection,
                None if selection.frame is None else len(accepted),
            )
        if selection.frame is None:
            reason = selection.omission_reason or "visual contact unavailable"
            omissions[reason] += 1
            decision = selection.diagnostics
            deterministic = (
                None if decision is None else decision.deterministic.selected_frame
            )
            temporal = (
                None
                if decision is None or decision.temporal_prediction is None
                else decision.temporal_prediction.frame_ordinal
            )
            _LOG.info(
                "Audio candidate %d at %.3fs omitted by visual contact: %s "
                "(deterministic frame=%s, temporal frame=%s, plausible=%s)",
                candidate_index,
                detail.legacy_contact,
                reason,
                deterministic,
                temporal,
                selection.plausible_frames or "none",
            )
            continue
        _LOG.info(
            "Audio candidate %d at %.3fs chose contact frame %d at %.6fs "
            "(confidence=%.3f, plausible=%s)",
            candidate_index,
            detail.legacy_contact,
            selection.frame.ordinal,
            float(selection.frame.timestamp),
            selection.contact_confidence,
            selection.plausible_frames or "none",
        )
        accepted.append(
            DetectedSwing(
                ordinal=len(accepted),
                contact_timestamp=selection.frame.timestamp,
                shot_type=detail.swing.shot_type,
                contact_frame=selection.frame.evidence.identity,
            )
        )
    omission_summary = ", ".join(
        f"{reason}={count}" for reason, count in sorted(omissions.items())
    ) or "none"
    _LOG.info(
        "visual contact selection: candidates=%d accepted=%d omitted=%d (%s)",
        len(details),
        len(accepted),
        sum(omissions.values()),
        omission_summary,
    )
    return tuple(accepted)


def detect_user_swings_for_legacy(
    user_video: Path,
    detection_config: DetectionConfig,
    media_info: dict,
) -> tuple[LegacySwingDetails, ...]:
    """Preserve details and progress output required by ``tennis-cut``."""

    return _detect_user_swings_with_details(
        user_video,
        detection_config,
        media_info,
        report_progress=True,
    )

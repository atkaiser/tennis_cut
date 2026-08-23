"""Shared user-video swing detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from fractions import Fraction

from PIL import Image
import torchaudio

from utilities import PersonDetector, expand_box

if __package__:
    from .subprocess_utils import run_command
    from .visual_contact import (
        SourceFrameIdentity,
        StockVisualContactSelector,
        VisualContactSelector,
    )
else:
    from subprocess_utils import run_command
    from visual_contact import (
        SourceFrameIdentity,
        StockVisualContactSelector,
        VisualContactSelector,
    )


DEFAULT_AUDIO_MODEL = Path("models/audio_pop_logmel_large_20260512231349.pth")
DEFAULT_SHOT_MODEL = Path("models/shot_binary_classifier_20260328143535.pkl")
DEFAULT_SHOT_TYPE_MODEL = Path("models/shot_type_classifier_20260328220857.pkl")

DEFAULT_STRIDE_S = 0.05
SAMPLE_RATE = 48_000
WINDOW_DURATION = 0.25
PEAK_THRESHOLD = 0.5
PEAK_MIN_SEPARATION = 2.0
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


@dataclass(frozen=True)
class LegacySwingDetails:
    """Detection details retained by the existing rendering command."""

    swing: DetectedSwing
    legacy_contact: float
    start: float
    end: float
    crop: tuple[int, int, int, int]


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

    def find_impacts(self, wav_path: Path) -> list[float]:
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

        candidates: list[tuple[float, float]] = []
        for index, probability in enumerate(probabilities):
            score = float(probability)
            if score > PEAK_THRESHOLD:
                center = index * self.stride_s + (WINDOW_DURATION / 2)
                candidates.append((center, score))

        candidates.sort(key=lambda candidate: candidate[1], reverse=True)
        kept: list[tuple[float, float]] = []
        for timestamp, score in candidates:
            if all(
                abs(timestamp - kept_timestamp) >= PEAK_MIN_SEPARATION
                for kept_timestamp, _ in kept
            ):
                kept.append((timestamp, score))
        kept.sort(key=lambda candidate: candidate[0])

        impacts = [timestamp for timestamp, _ in kept]
        _LOG.info("Detected %d audio peaks", len(impacts))
        _LOG.info("Detected peaks: " + ", ".join(f"{impact:.3f}" for impact in impacts))
        return impacts


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
) -> tuple[LegacySwingDetails, ...]:
    selected_device = resolve_device(detection_config.device)
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        wav_path = temporary_path / "audio.wav"
        extract_audio(user_video, wav_path)

        pop_detector = PopDetector(detection_config.audio_model, device=selected_device)
        impact_times = pop_detector.find_impacts(wav_path)
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

        accepted: list[LegacySwingDetails] = []
        processing_times: list[float] = []
        for candidate_index, contact in enumerate(impact_times):
            started_at = time.perf_counter()
            frame_path = temporary_path / f"impact_{candidate_index}.jpg"
            extract_frame(user_video, contact, frame_path)
            box = person_detector.find_box(frame_path)
            if box is None:
                _LOG.info("No person found for impact %d", candidate_index)
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
                        continue
                    if shot_type_classifier is not None:
                        shot_type = shot_type_classifier.predict_label(cropped)
            detected_swing = DetectedSwing(
                ordinal=len(accepted),
                contact_timestamp=_exact_contact_timestamp(contact),
                shot_type=shot_type,
            )
            accepted.append(
                LegacySwingDetails(
                    swing=detected_swing,
                    legacy_contact=contact,
                    start=contact - PRE_CONTACT_BUFFER,
                    end=contact + POST_CONTACT_BUFFER,
                    crop=tuple(crop),
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

    return tuple(accepted)


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
    contact_selector: VisualContactSelector | None = None,
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
    )
    selector = contact_selector or StockVisualContactSelector(
        device=detection_config.device
    )
    accepted: list[DetectedSwing] = []
    for detail in details:
        selection = selector.select(
            user_video, _exact_contact_timestamp(detail.legacy_contact)
        )
        if selection.frame is None:
            _LOG.info(
                "Omitting swing %d: %s",
                detail.swing.ordinal,
                selection.omission_reason or "visual contact unavailable",
            )
            continue
        accepted.append(
            DetectedSwing(
                ordinal=len(accepted),
                contact_timestamp=selection.frame.timestamp,
                shot_type=detail.swing.shot_type,
                contact_frame=selection.frame.evidence.identity,
            )
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

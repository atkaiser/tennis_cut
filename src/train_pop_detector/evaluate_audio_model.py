#!/usr/bin/env python3
"""Evaluate one audio pop detector on labeled test videos."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import warnings
from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torchaudio
from fastai.learner import load_learner

from train_audio_pop import (  # noqa: F401
    AddGaussianSNR,
    AudioLoad,
    DCOffset,
    LogMelSpectrogram,
    PreEmphasis,
    RandomGain,
    RmsNorm,
    SR,
    WIN_SEC,
)

VIDEO_EXTS = (".MOV", ".mp4", ".mov")
DEFAULT_THRESHOLD = 0.5
DEFAULT_STRIDE_S = 0.05
DEFAULT_TOLERANCE_S = 0.25
DEFAULT_MIN_SEPARATION_S = 2.0
DEFAULT_BATCH_SIZE = 128


@dataclass(frozen=True)
class VideoLabels:
    video: pathlib.Path
    json_path: pathlib.Path
    impacts: list[float]
    shot_types: dict[float, str]


@dataclass(frozen=True)
class ScoredWindow:
    video: str
    start: float
    center: float
    score: float
    y_true: int
    y_pred: int


@dataclass(frozen=True)
class Event:
    video: str
    time: float
    score: float
    matched_label_time: float | None
    timing_error: float | None


@dataclass(frozen=True)
class Miss:
    video: str
    label_time: float
    shot_type: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one audio pop model on labeled test videos."
    )
    parser.add_argument("model", type=pathlib.Path, help="Path to exported model")
    parser.add_argument("--videos-dir", type=pathlib.Path, default="videos_test")
    parser.add_argument("--wav-dir", type=pathlib.Path, default="wavs_test")
    parser.add_argument("--out-dir", type=pathlib.Path, default="meta/audio_eval")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    parser.add_argument("--stride-s", type=float, default=DEFAULT_STRIDE_S)
    parser.add_argument(
        "--min-separation-s", type=float, default=DEFAULT_MIN_SEPARATION_S
    )
    parser.add_argument("--bs", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="write summary.json, events.csv, and misses.csv to --out-dir",
    )
    parser.add_argument("--write-window-predictions", action="store_true")
    return parser.parse_args(argv)


def run_cmd(cmd: Sequence[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


def extract_wav(video_path: pathlib.Path, wav_path: pathlib.Path) -> None:
    """Extract mono 48 kHz audio if the cached wav is missing."""
    if wav_path.exists():
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            str(SR),
            str(wav_path),
        ]
    )


def list_labeled_videos(videos_dir: pathlib.Path, limit: int | None) -> list[VideoLabels]:
    videos: list[pathlib.Path] = []
    for ext in VIDEO_EXTS:
        videos.extend(sorted(videos_dir.glob(f"*{ext}")))
    if limit is not None:
        videos = videos[:limit]

    labeled: list[VideoLabels] = []
    for video in videos:
        json_path = video.with_suffix(".json")
        if not json_path.exists():
            print(f"No JSON for {video.name}; skipping.")
            continue
        data = json.loads(json_path.read_text())
        impacts = [float(t) for t in data.get("impacts", [])]
        shot_types = {
            float(shot["time"]): str(shot["type"])
            for shot in data.get("shots", [])
            if "time" in shot and "type" in shot
        }
        labeled.append(
            VideoLabels(
                video=video,
                json_path=json_path,
                impacts=impacts,
                shot_types=shot_types,
            )
        )
    return labeled


def load_model(model_path: pathlib.Path, device_name: str):
    device = torch.device(device_name)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="load_learner` uses Python's insecure pickle",
        )
        learner = load_learner(model_path, cpu=device.type == "cpu")
    learner.to(device)
    learner.model.eval()
    return learner


def positive_index(learner) -> int:
    vocab = list(getattr(learner.dls, "vocab", []))
    if "pos" in vocab:
        return vocab.index("pos")
    return 1


def window_starts(wav_path: pathlib.Path, stride_s: float) -> list[float]:
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
        sr = SR
    window_samples = int(WIN_SEC * sr)
    stride_samples = int(stride_s * sr)
    if waveform.shape[1] < window_samples:
        return []
    return [
        sample_start / sr
        for sample_start in range(
            0, waveform.shape[1] - window_samples + 1, stride_samples
        )
    ]


def is_near_impact(center: float, impacts: list[float], tolerance: float) -> bool:
    return any(abs(center - impact) <= tolerance for impact in impacts)


def score_video_windows(
    learner,
    labels: VideoLabels,
    wav_path: pathlib.Path,
    threshold: float,
    tolerance: float,
    stride_s: float,
    batch_size: int,
    pos_idx: int,
) -> list[ScoredWindow]:
    starts = window_starts(wav_path, stride_s=stride_s)
    if not starts:
        return []

    df = pd.DataFrame({"wav_path": str(wav_path), "start": starts})
    dl = learner.dls.test_dl(df, bs=batch_size)
    with torch.no_grad():
        preds, _ = learner.get_preds(dl=dl, reorder=False)
    scores = preds[:, pos_idx].cpu().numpy()

    rows: list[ScoredWindow] = []
    for start, score in zip(starts, scores):
        center = start + (WIN_SEC / 2)
        y_true = int(is_near_impact(center, labels.impacts, tolerance))
        y_pred = int(float(score) >= threshold)
        rows.append(
            ScoredWindow(
                video=labels.video.name,
                start=float(start),
                center=float(center),
                score=float(score),
                y_true=y_true,
                y_pred=y_pred,
            )
        )
    return rows


def nms_events(
    windows: list[ScoredWindow], threshold: float, min_separation_s: float
) -> list[tuple[float, float]]:
    candidates = [
        (window.center, window.score) for window in windows if window.score >= threshold
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    kept: list[tuple[float, float]] = []
    for timestamp, score in candidates:
        if all(abs(timestamp - kept_time) >= min_separation_s for kept_time, _ in kept):
            kept.append((timestamp, score))
    return sorted(kept, key=lambda item: item[0])


def match_events(
    video_name: str,
    predictions: list[tuple[float, float]],
    labels: VideoLabels,
    tolerance: float,
) -> tuple[list[Event], list[Miss]]:
    pairs: list[tuple[float, int, int]] = []
    for pred_idx, (pred_time, _) in enumerate(predictions):
        for label_idx, label_time in enumerate(labels.impacts):
            error = abs(pred_time - label_time)
            if error <= tolerance:
                pairs.append((error, pred_idx, label_idx))
    pairs.sort(key=lambda item: item[0])

    matched_preds: set[int] = set()
    matched_labels: set[int] = set()
    pred_to_label: dict[int, int] = {}
    for _, pred_idx, label_idx in pairs:
        if pred_idx in matched_preds or label_idx in matched_labels:
            continue
        matched_preds.add(pred_idx)
        matched_labels.add(label_idx)
        pred_to_label[pred_idx] = label_idx

    events: list[Event] = []
    for pred_idx, (pred_time, score) in enumerate(predictions):
        label_idx = pred_to_label.get(pred_idx)
        matched_label_time = None
        timing_error = None
        if label_idx is not None:
            matched_label_time = labels.impacts[label_idx]
            timing_error = abs(pred_time - matched_label_time)
        events.append(
            Event(
                video=video_name,
                time=pred_time,
                score=score,
                matched_label_time=matched_label_time,
                timing_error=timing_error,
            )
        )

    misses = [
        Miss(
            video=video_name,
            label_time=label_time,
            shot_type=labels.shot_types.get(label_time),
        )
        for label_idx, label_time in enumerate(labels.impacts)
        if label_idx not in matched_labels
    ]
    return events, misses


def confusion_matrix(windows: list[ScoredWindow]) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for window in windows:
        cm[window.y_true, window.y_pred] += 1
    return cm


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int | None = None) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    if tn is not None:
        total = tp + fp + fn + tn
        metrics["accuracy"] = (tp + tn) / total if total else 0.0
    return metrics


def write_outputs(
    out_dir: pathlib.Path,
    summary: dict,
    events: list[Event],
    misses: list[Miss],
    windows: list[ScoredWindow],
    write_window_predictions: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (out_dir / "events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "time",
                "score",
                "matched_label_time",
                "timing_error",
                "matched",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "video": event.video,
                    "time": f"{event.time:.3f}",
                    "score": f"{event.score:.6f}",
                    "matched_label_time": ""
                    if event.matched_label_time is None
                    else f"{event.matched_label_time:.3f}",
                    "timing_error": ""
                    if event.timing_error is None
                    else f"{event.timing_error:.3f}",
                    "matched": event.matched_label_time is not None,
                }
            )

    with (out_dir / "misses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label_time", "shot_type"])
        writer.writeheader()
        for miss in misses:
            writer.writerow(
                {
                    "video": miss.video,
                    "label_time": f"{miss.label_time:.3f}",
                    "shot_type": miss.shot_type or "",
                }
            )

    if write_window_predictions:
        with (out_dir / "window_predictions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["video", "start", "center", "score", "y_true", "y_pred"],
            )
            writer.writeheader()
            for window in windows:
                writer.writerow(
                    {
                        "video": window.video,
                        "start": f"{window.start:.3f}",
                        "center": f"{window.center:.3f}",
                        "score": f"{window.score:.6f}",
                        "y_true": window.y_true,
                        "y_pred": window.y_pred,
                    }
                )


def print_summary(summary: dict, per_video: list[dict]) -> None:
    print(f"Model: {summary['model']}")
    print(
        "Settings:",
        f"threshold={summary['threshold']}",
        f"tolerance={summary['tolerance']}",
        f"stride_s={summary['stride_s']}",
        f"min_separation_s={summary['min_separation_s']}",
    )
    print(
        "Test set:",
        f"videos={summary['test_videos']}",
        f"labeled_impacts={summary['labeled_impacts']}",
    )
    print("\nWindow confusion matrix [[tn, fp], [fn, tp]]:")
    print(np.array(summary["window_confusion_matrix"]))
    print(
        "Window metrics:",
        f"accuracy={summary['window_metrics']['accuracy']:.4f}",
        f"precision={summary['window_metrics']['precision']:.4f}",
        f"recall={summary['window_metrics']['recall']:.4f}",
        f"f1={summary['window_metrics']['f1']:.4f}",
    )
    print(
        "\nEvent metrics:",
        f"tp={summary['event_counts']['tp']}",
        f"fp={summary['event_counts']['fp']}",
        f"fn={summary['event_counts']['fn']}",
        f"precision={summary['event_metrics']['precision']:.4f}",
        f"recall={summary['event_metrics']['recall']:.4f}",
        f"f1={summary['event_metrics']['f1']:.4f}",
    )
    print(
        "Timing error:",
        f"mean={summary['timing_error']['mean']}",
        f"median={summary['timing_error']['median']}",
    )
    print("\nPer-video event summary:")
    for row in per_video:
        print(
            f"{row['video']}: labels={row['labels']} "
            f"detected={row['detected']} missed={row['missed']} "
            f"false_positives={row['false_positives']}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    labels_by_video = list_labeled_videos(args.videos_dir, limit=args.limit_videos)
    if not labels_by_video:
        raise SystemExit(f"No labeled videos found in {args.videos_dir}")

    learner = load_model(args.model, args.device)
    pos_idx = positive_index(learner)

    all_windows: list[ScoredWindow] = []
    all_events: list[Event] = []
    all_misses: list[Miss] = []
    per_video: list[dict] = []

    for labels in labels_by_video:
        wav_path = args.wav_dir / f"{labels.video.stem}.wav"
        extract_wav(labels.video, wav_path)
        windows = score_video_windows(
            learner=learner,
            labels=labels,
            wav_path=wav_path,
            threshold=args.threshold,
            tolerance=args.tolerance,
            stride_s=args.stride_s,
            batch_size=args.bs,
            pos_idx=pos_idx,
        )
        predictions = nms_events(
            windows, threshold=args.threshold, min_separation_s=args.min_separation_s
        )
        events, misses = match_events(
            labels.video.name,
            predictions=predictions,
            labels=labels,
            tolerance=args.tolerance,
        )

        all_windows.extend(windows)
        all_events.extend(events)
        all_misses.extend(misses)

        detected = sum(1 for event in events if event.matched_label_time is not None)
        false_positives = sum(
            1 for event in events if event.matched_label_time is None
        )
        per_video.append(
            {
                "video": labels.video.name,
                "labels": len(labels.impacts),
                "detected": detected,
                "missed": len(misses),
                "false_positives": false_positives,
            }
        )

    cm = confusion_matrix(all_windows)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    event_tp = sum(1 for event in all_events if event.matched_label_time is not None)
    event_fp = sum(1 for event in all_events if event.matched_label_time is None)
    event_fn = len(all_misses)
    timing_errors = [
        float(event.timing_error)
        for event in all_events
        if event.timing_error is not None
    ]

    summary = {
        "model": str(args.model),
        "videos_dir": str(args.videos_dir),
        "threshold": args.threshold,
        "tolerance": args.tolerance,
        "stride_s": args.stride_s,
        "min_separation_s": args.min_separation_s,
        "test_videos": len(labels_by_video),
        "labeled_impacts": sum(len(labels.impacts) for labels in labels_by_video),
        "window_confusion_matrix": cm.tolist(),
        "window_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "window_metrics": metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn),
        "event_counts": {"tp": event_tp, "fp": event_fp, "fn": event_fn},
        "event_metrics": metrics_from_counts(
            tp=event_tp, fp=event_fp, fn=event_fn
        ),
        "timing_error": {
            "mean": None if not timing_errors else round(mean(timing_errors), 6),
            "median": None if not timing_errors else round(median(timing_errors), 6),
        },
        "per_video": per_video,
    }

    print_summary(summary, per_video)
    if args.save_outputs or args.write_window_predictions:
        write_outputs(
            out_dir=args.out_dir,
            summary=summary,
            events=all_events,
            misses=all_misses,
            windows=all_windows,
            write_window_predictions=args.write_window_predictions,
        )
        print(f"\nSaved evaluation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

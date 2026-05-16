#!/usr/bin/env python3
"""Evaluate one binary shot/no_shot image classifier."""

from __future__ import annotations

import argparse
import pathlib
from collections import Counter
from typing import Sequence

import numpy as np

from shot_eval_common import (
    DEFAULT_AUDIO_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MIN_SEPARATION_S,
    DEFAULT_SHOT_THRESHOLD,
    DEFAULT_STRIDE_S,
    DEFAULT_TOLERANCE_S,
    DirectPrediction,
    PipelinePrediction,
    binary_metrics,
    check_ffmpeg,
    class_index,
    confusion_matrix,
    evaluate_pipeline_candidates,
    image_files_by_label,
    list_labeled_videos,
    load_fastai_model,
    metrics_from_counts,
    model_vocab,
    predict_image_paths,
    save_json,
    score_binary_image,
    timing_summary,
    write_direct_predictions,
    write_misses,
    write_pipeline_predictions,
)


LABELS = ["no_shot", "shot"]
MISS_REASON_LABELS = {
    "audio_miss": "fn_audio",
    "binary_rejected": "fn_binary",
    "crop_miss": "fn_crop",
    "unmatched_audio_peak": "fn_other",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one binary shot/no_shot model on test frames and videos."
    )
    parser.add_argument("model", type=pathlib.Path, help="Path to binary shot model")
    parser.add_argument(
        "--audio-model",
        type=pathlib.Path,
        default="models/audio_pop_logmel_large_20260512231349.pth",
        help="Path to audio pop detector used for pipeline evaluation",
    )
    parser.add_argument("--dataset-dir", type=pathlib.Path, default="dataset_test")
    parser.add_argument("--videos-dir", type=pathlib.Path, default="videos_test")
    parser.add_argument("--wav-dir", type=pathlib.Path, default="wavs_test")
    parser.add_argument("--out-dir", type=pathlib.Path, default="meta/shot_binary_eval")
    parser.add_argument("--shot-threshold", type=float, default=DEFAULT_SHOT_THRESHOLD)
    parser.add_argument("--audio-threshold", type=float, default=DEFAULT_AUDIO_THRESHOLD)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    parser.add_argument("--stride-s", type=float, default=DEFAULT_STRIDE_S)
    parser.add_argument(
        "--min-separation-s", type=float, default=DEFAULT_MIN_SEPARATION_S
    )
    parser.add_argument("--bs", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--limit-direct-images", type=int, default=None)
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="write summary.json and prediction CSVs to --out-dir",
    )
    return parser.parse_args(argv)


def binary_label(label: str) -> str:
    return "no_shot" if label == "no_shot" else "shot"


def evaluate_direct(
    learner,
    dataset_dir: pathlib.Path,
    shot_threshold: float,
    batch_size: int,
    limit: int | None,
) -> tuple[np.ndarray, dict[str, float], list[DirectPrediction]]:
    vocab = model_vocab(learner)
    shot_idx = class_index(vocab, "shot", fallback=1)
    rows = image_files_by_label(dataset_dir)
    if limit is not None:
        rows = rows[:limit]
    paths = [path for path, _ in rows]
    if not paths:
        cm = confusion_matrix([], [], LABELS)
        return cm, binary_metrics(cm), []
    _, probs = predict_image_paths(learner, paths, batch_size=batch_size)
    predictions: list[DirectPrediction] = []
    actual: list[str] = []
    predicted: list[str] = []

    for (path, raw_label), prob in zip(rows, probs):
        actual_label = binary_label(raw_label)
        shot_score = float(prob[shot_idx])
        pred_label = "shot" if shot_score >= shot_threshold else "no_shot"
        actual.append(actual_label)
        predicted.append(pred_label)
        predictions.append(
            DirectPrediction(
                image_path=str(path),
                actual_label=actual_label,
                predicted_label=pred_label,
                score=shot_score,
            )
        )

    cm = confusion_matrix(actual, predicted, LABELS)
    return cm, binary_metrics(cm), predictions


def count_by_label(labels: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items()))


def false_negative_breakdown(pipeline_misses) -> dict[str, dict[str, int]]:
    breakdown = {label: {} for label in MISS_REASON_LABELS.values()}
    for miss in pipeline_misses:
        bucket = MISS_REASON_LABELS.get(miss.reason, "fn_other")
        counts = Counter(breakdown[bucket])
        counts[miss.label] += 1
        breakdown[bucket] = dict(sorted(counts.items()))
    return breakdown


def add_per_video_error_types(
    per_video: list[dict],
    pipeline_misses,
) -> list[dict]:
    rows: list[dict] = []
    for row in per_video:
        video = row["video"]
        video_misses = [miss for miss in pipeline_misses if miss.video == video]
        with_types = dict(row)
        with_types["error_types"] = false_negative_breakdown(video_misses)
        rows.append(with_types)
    return rows


def print_type_breakdown(title: str, breakdown: dict[str, int]) -> None:
    if not breakdown:
        print(f"{title}: none")
        return
    rendered = ", ".join(f"{label}={count}" for label, count in breakdown.items())
    print(f"{title}: {rendered}")


def print_summary(summary: dict, per_video: list[dict]) -> None:
    print(f"Model: {summary['model']}")
    print(f"Audio model: {summary['audio_model']}")
    print(
        "Settings:",
        f"shot_threshold={summary['shot_threshold']}",
        f"audio_threshold={summary['audio_threshold']}",
        f"tolerance={summary['tolerance']}",
        f"stride_s={summary['stride_s']}",
        f"min_separation_s={summary['min_separation_s']}",
    )
    print(
        "Test set:",
        f"direct_images={summary['direct_images']}",
        f"videos={summary['test_videos']}",
        f"labeled_shots={summary['labeled_shots']}",
    )

    print("\nDirect frame confusion matrix [[no_shot, shot] actual x predicted]:")
    print(np.array(summary["direct_confusion_matrix"]))
    print(
        "Direct frame metrics:",
        f"accuracy={summary['direct_metrics']['accuracy']:.4f}",
        f"precision={summary['direct_metrics']['precision']:.4f}",
        f"recall={summary['direct_metrics']['recall']:.4f}",
        f"f1={summary['direct_metrics']['f1']:.4f}",
    )

    print(
        "\nPipeline metrics:",
        f"tp={summary['pipeline_counts']['tp']}",
        f"fp={summary['pipeline_counts']['fp']}",
        f"fn={summary['pipeline_counts']['fn']}",
        f"fn_audio={summary['pipeline_counts']['fn_audio']}",
        f"fn_binary={summary['pipeline_counts']['fn_binary']}",
        f"fn_crop={summary['pipeline_counts']['fn_crop']}",
        f"fn_other={summary['pipeline_counts']['fn_other']}",
        f"precision={summary['pipeline_metrics']['precision']:.4f}",
        f"recall={summary['pipeline_metrics']['recall']:.4f}",
        f"f1={summary['pipeline_metrics']['f1']:.4f}",
    )
    print(
        "Timing error:",
        f"mean={summary['timing_error']['mean']}",
        f"median={summary['timing_error']['median']}",
    )
    print("\nPipeline error type breakdown:")
    for key in ("fn_audio", "fn_binary", "fn_crop", "fn_other"):
        print_type_breakdown(key, summary["pipeline_error_types"][key])
    print("\nPer-video pipeline summary:")
    for row in per_video:
        print(
            f"{row['video']}: labels={row['labels']} audio_peaks={row['audio_peaks']} "
            f"predicted={row['pipeline_predictions']} matched={row['matched']} "
            f"missed={row['missed']} missed_audio={row['missed_audio']} "
            f"missed_binary={row['missed_binary']} missed_crop={row['missed_crop']} "
            f"missed_other={row['missed_other']} "
            f"false_positives={row['false_positives']}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    check_ffmpeg()

    shot_learner = load_fastai_model(args.model, args.device)
    audio_learner = load_fastai_model(args.audio_model, args.device)

    direct_cm, direct_metrics, direct_predictions = evaluate_direct(
        learner=shot_learner,
        dataset_dir=args.dataset_dir,
        shot_threshold=args.shot_threshold,
        batch_size=args.bs,
        limit=args.limit_direct_images,
    )

    labels_by_video = list_labeled_videos(args.videos_dir, limit=args.limit_videos)
    if not labels_by_video:
        raise SystemExit(f"No labeled videos found in {args.videos_dir}")

    shot_vocab = model_vocab(shot_learner)
    shot_idx = class_index(shot_vocab, "shot", fallback=1)

    def classify_peak(video_name, peak, img):
        _, shot_score = score_binary_image(shot_learner, img, shot_idx)
        if shot_score < args.shot_threshold:
            return PipelinePrediction(
                video=video_name,
                time=peak.time,
                audio_score=peak.score,
                shot_score=shot_score,
                predicted_label="no_shot",
                status="rejected_by_binary",
            )
        return PipelinePrediction(
            video=video_name,
            time=peak.time,
            audio_score=peak.score,
            shot_score=shot_score,
            predicted_label="shot",
        )

    pipeline_predictions, pipeline_misses, per_video = evaluate_pipeline_candidates(
        labels_by_video=labels_by_video,
        audio_learner=audio_learner,
        wav_dir=args.wav_dir,
        audio_threshold=args.audio_threshold,
        stride_s=args.stride_s,
        min_separation_s=args.min_separation_s,
        batch_size=args.bs,
        tolerance=args.tolerance,
        device=args.device,
        classify_peak=classify_peak,
    )

    tp = sum(1 for pred in pipeline_predictions if pred.status == "matched")
    fp = sum(1 for pred in pipeline_predictions if pred.status == "unmatched")
    fn = len(pipeline_misses)
    fn_audio = sum(1 for miss in pipeline_misses if miss.reason == "audio_miss")
    fn_binary = sum(1 for miss in pipeline_misses if miss.reason == "binary_rejected")
    fn_crop = sum(1 for miss in pipeline_misses if miss.reason == "crop_miss")
    fn_other = sum(
        1 for miss in pipeline_misses if miss.reason == "unmatched_audio_peak"
    )
    fn_types = false_negative_breakdown(pipeline_misses)
    per_video = add_per_video_error_types(
        per_video,
        pipeline_misses=pipeline_misses,
    )
    summary = {
        "model": str(args.model),
        "audio_model": str(args.audio_model),
        "dataset_dir": str(args.dataset_dir),
        "videos_dir": str(args.videos_dir),
        "shot_threshold": args.shot_threshold,
        "audio_threshold": args.audio_threshold,
        "tolerance": args.tolerance,
        "stride_s": args.stride_s,
        "min_separation_s": args.min_separation_s,
        "direct_images": len(direct_predictions),
        "test_videos": len(labels_by_video),
        "labeled_shots": sum(len(labels.labels) for labels in labels_by_video),
        "direct_confusion_matrix": direct_cm.tolist(),
        "direct_metrics": direct_metrics,
        "pipeline_counts": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "fn_audio": fn_audio,
            "fn_binary": fn_binary,
            "fn_crop": fn_crop,
            "fn_other": fn_other,
        },
        "pipeline_metrics": metrics_from_counts(tp=tp, fp=fp, fn=fn),
        "pipeline_error_types": fn_types,
        "timing_error": timing_summary(pipeline_predictions),
        "per_video": per_video,
    }

    print_summary(summary, per_video)
    if args.save_outputs:
        save_json(args.out_dir / "summary.json", summary)
        write_direct_predictions(
            args.out_dir / "direct_frame_predictions.csv", direct_predictions
        )
        write_pipeline_predictions(
            args.out_dir / "pipeline_predictions.csv", pipeline_predictions
        )
        write_misses(args.out_dir / "pipeline_misses.csv", pipeline_misses)
        print(f"\nSaved evaluation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

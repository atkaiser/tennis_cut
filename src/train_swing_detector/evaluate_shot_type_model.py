#!/usr/bin/env python3
"""Evaluate one shot-type image classifier."""

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
    PipelineMiss,
    PipelinePrediction,
    check_ffmpeg,
    class_index,
    confusion_matrix,
    evaluate_pipeline_candidates,
    image_files_by_label,
    list_labeled_videos,
    load_fastai_model,
    metrics_from_counts,
    model_vocab,
    multiclass_report,
    predict_image_paths,
    predict_label,
    save_json,
    score_binary_image,
    timing_summary,
    write_direct_predictions,
    write_misses,
    write_pipeline_predictions,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one shot-type model on test frames and videos."
    )
    parser.add_argument("model", type=pathlib.Path, help="Path to shot-type model")
    parser.add_argument(
        "--binary-model",
        dest="binary_model",
        type=pathlib.Path,
        default="models/shot_binary_classifier_20260328143535.pkl",
        help="Path to binary shot model used before shot-type classification",
    )
    parser.add_argument(
        "--audio-model",
        type=pathlib.Path,
        default="models/audio_pop_logmel_large_20260512231349.pth",
        help="Path to audio pop detector used for pipeline evaluation",
    )
    parser.add_argument("--dataset-dir", type=pathlib.Path, default="dataset_test")
    parser.add_argument("--videos-dir", type=pathlib.Path, default="videos_test")
    parser.add_argument("--wav-dir", type=pathlib.Path, default="wavs_test")
    parser.add_argument("--out-dir", type=pathlib.Path, default="meta/shot_type_eval")
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


def supported_type_labels(learner) -> list[str]:
    return [label for label in model_vocab(learner) if label != "no_shot"]


def evaluate_direct(
    learner,
    dataset_dir: pathlib.Path,
    labels: list[str],
    batch_size: int,
    limit: int | None,
) -> tuple[np.ndarray, list[dict], list[DirectPrediction], Counter[str]]:
    rows = image_files_by_label(dataset_dir)
    supported = set(labels)
    excluded = Counter()
    actual: list[str] = []
    predicted: list[str] = []
    selected_rows: list[tuple[pathlib.Path, str]] = []

    for path, actual_label in rows:
        if actual_label == "no_shot" or actual_label not in supported:
            excluded[actual_label] += 1
            continue
        if limit is not None and len(selected_rows) >= limit:
            break
        selected_rows.append((path, actual_label))

    paths = [path for path, _ in selected_rows]
    if not paths:
        cm = confusion_matrix(actual, predicted, labels)
        return cm, multiclass_report(cm, labels), [], excluded
    pred_labels, probs = predict_image_paths(learner, paths, batch_size=batch_size)
    predictions: list[DirectPrediction] = []
    for (path, actual_label), predicted_label, prob in zip(
        selected_rows, pred_labels, probs
    ):
        score = float(prob[labels.index(predicted_label)])
        actual.append(actual_label)
        predicted.append(predicted_label)
        predictions.append(
            DirectPrediction(
                image_path=str(path),
                actual_label=actual_label,
                predicted_label=predicted_label,
                score=score,
            )
        )

    cm = confusion_matrix(actual, predicted, labels)
    return cm, multiclass_report(cm, labels), predictions, excluded


def normalize_pipeline_results(
    predictions: list[PipelinePrediction],
    misses: list[PipelineMiss],
    supported: set[str],
) -> tuple[list[PipelinePrediction], list[PipelineMiss]]:
    normalized: list[PipelinePrediction] = []
    for pred in predictions:
        if pred.status in {"rejected_by_binary", "crop_miss"}:
            continue
        if pred.status == "matched" and pred.matched_label not in supported:
            normalized.append(
                PipelinePrediction(
                    video=pred.video,
                    time=pred.time,
                    audio_score=pred.audio_score,
                    shot_score=pred.shot_score,
                    predicted_label=pred.predicted_label,
                    matched_label_time=pred.matched_label_time,
                    matched_label=pred.matched_label,
                    timing_error=pred.timing_error,
                    status="ignored_unsupported",
                )
            )
        else:
            normalized.append(pred)
    supported_misses = [miss for miss in misses if miss.label in supported]
    return normalized, supported_misses


def pipeline_confusion(
    predictions: list[PipelinePrediction], labels: list[str]
) -> np.ndarray:
    actual: list[str] = []
    predicted: list[str] = []
    for pred in predictions:
        if pred.status != "matched" or pred.matched_label not in labels:
            continue
        if pred.predicted_label not in labels:
            continue
        actual.append(pred.matched_label)
        predicted.append(pred.predicted_label)
    return confusion_matrix(actual, predicted, labels)


def build_per_video_summary(
    labels_by_video,
    predictions: list[PipelinePrediction],
    misses: list[PipelineMiss],
    supported: set[str],
) -> list[dict]:
    predictions_by_video = {}
    misses_by_video = {}
    for pred in predictions:
        predictions_by_video.setdefault(pred.video, []).append(pred)
    for miss in misses:
        misses_by_video.setdefault(miss.video, []).append(miss)

    rows: list[dict] = []
    for video_labels in labels_by_video:
        video = video_labels.video.name
        video_predictions = predictions_by_video.get(video, [])
        supported_labels = [
            label for label in video_labels.labels if label.label in supported
        ]
        matched_supported = [
            pred
            for pred in video_predictions
            if pred.status == "matched" and pred.matched_label in supported
        ]
        correct_type = [
            pred for pred in matched_supported if pred.predicted_label == pred.matched_label
        ]
        rows.append(
            {
                "video": video,
                "supported_labels": len(supported_labels),
                "predicted": sum(
                    1
                    for pred in video_predictions
                    if pred.status != "ignored_unsupported"
                ),
                "matched_supported": len(matched_supported),
                "correct_type": len(correct_type),
                "wrong_type": len(matched_supported) - len(correct_type),
                "missed": len(misses_by_video.get(video, [])),
                "false_positives": sum(
                    1 for pred in video_predictions if pred.status == "unmatched"
                ),
                "ignored_unsupported": sum(
                    1
                    for pred in video_predictions
                    if pred.status == "ignored_unsupported"
                ),
            }
        )
    return rows


def print_class_report(rows: list[dict]) -> None:
    print("Per-class metrics:")
    print(f"{'class':<16} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}")
    for row in rows:
        print(
            f"{row['class']:<16} {row['precision']:>9.3f} "
            f"{row['recall']:>9.3f} {row['f1']:>9.3f} {row['support']:>9d}"
        )


def print_summary(summary: dict, direct_report: list[dict], per_video: list[dict]) -> None:
    print(f"Model: {summary['model']}")
    print(f"Binary model: {summary['binary_model']}")
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
        f"excluded_direct_images={summary['excluded_direct_images']}",
        f"videos={summary['test_videos']}",
        f"supported_labeled_shots={summary['supported_labeled_shots']}",
    )

    print("\nDirect shot-type confusion matrix:")
    print("Labels:", ", ".join(summary["labels"]))
    print(np.array(summary["direct_confusion_matrix"]))
    print_class_report(direct_report)

    print(
        "\nPipeline detection metrics:",
        f"matched_supported={summary['pipeline_counts']['matched_supported']}",
        f"false_positives={summary['pipeline_counts']['false_positives']}",
        f"missed={summary['pipeline_counts']['missed']}",
        f"precision={summary['pipeline_detection_metrics']['precision']:.4f}",
        f"recall={summary['pipeline_detection_metrics']['recall']:.4f}",
        f"f1={summary['pipeline_detection_metrics']['f1']:.4f}",
    )
    print(
        "Pipeline type metrics:",
        f"correct={summary['pipeline_counts']['correct_type']}",
        f"wrong={summary['pipeline_counts']['wrong_type']}",
        f"accuracy={summary['pipeline_type_accuracy']:.4f}",
    )
    print(
        "Ignored unsupported matches:",
        summary["pipeline_counts"]["ignored_unsupported"],
    )
    print(
        "Timing error:",
        f"mean={summary['timing_error']['mean']}",
        f"median={summary['timing_error']['median']}",
    )
    print("\nPipeline shot-type confusion matrix for matched supported shots:")
    print(np.array(summary["pipeline_confusion_matrix"]))
    print("\nPer-video pipeline summary:")
    for row in per_video:
        print(
            f"{row['video']}: labels={row['supported_labels']} "
            f"predicted={row['predicted']} matched={row['matched_supported']} "
            f"correct={row['correct_type']} wrong={row['wrong_type']} "
            f"missed={row['missed']} false_positives={row['false_positives']} "
            f"ignored_unsupported={row['ignored_unsupported']}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    check_ffmpeg()

    type_learner = load_fastai_model(args.model, args.device)
    shot_learner = load_fastai_model(args.binary_model, args.device)
    audio_learner = load_fastai_model(args.audio_model, args.device)
    labels = supported_type_labels(type_learner)
    supported = set(labels)

    direct_cm, direct_report, direct_predictions, excluded = evaluate_direct(
        learner=type_learner,
        dataset_dir=args.dataset_dir,
        labels=labels,
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
            return None
        predicted_label, _ = predict_label(type_learner, img)
        return PipelinePrediction(
            video=video_name,
            time=peak.time,
            audio_score=peak.score,
            shot_score=shot_score,
            predicted_label=predicted_label,
        )

    raw_predictions, raw_misses, _ = evaluate_pipeline_candidates(
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
    pipeline_predictions, pipeline_misses = normalize_pipeline_results(
        raw_predictions, raw_misses, supported=supported
    )
    per_video = build_per_video_summary(
        labels_by_video, pipeline_predictions, pipeline_misses, supported=supported
    )

    matched_supported = [
        pred
        for pred in pipeline_predictions
        if pred.status == "matched" and pred.matched_label in supported
    ]
    correct_type = [
        pred for pred in matched_supported if pred.predicted_label == pred.matched_label
    ]
    false_positives = [
        pred for pred in pipeline_predictions if pred.status == "unmatched"
    ]
    ignored_unsupported = [
        pred
        for pred in pipeline_predictions
        if pred.status == "ignored_unsupported"
    ]
    pipeline_cm = pipeline_confusion(pipeline_predictions, labels)
    supported_labeled_shots = sum(
        1
        for video_labels in labels_by_video
        for label in video_labels.labels
        if label.label in supported
    )
    pipeline_detection_metrics = metrics_from_counts(
        tp=len(matched_supported),
        fp=len(false_positives),
        fn=len(pipeline_misses),
    )
    type_accuracy = (
        len(correct_type) / len(matched_supported) if matched_supported else 0.0
    )
    summary = {
        "model": str(args.model),
        "binary_model": str(args.binary_model),
        "audio_model": str(args.audio_model),
        "dataset_dir": str(args.dataset_dir),
        "videos_dir": str(args.videos_dir),
        "labels": labels,
        "shot_threshold": args.shot_threshold,
        "audio_threshold": args.audio_threshold,
        "tolerance": args.tolerance,
        "stride_s": args.stride_s,
        "min_separation_s": args.min_separation_s,
        "direct_images": len(direct_predictions),
        "excluded_direct_images": sum(excluded.values()),
        "excluded_direct_by_label": dict(excluded),
        "test_videos": len(labels_by_video),
        "supported_labeled_shots": supported_labeled_shots,
        "direct_confusion_matrix": direct_cm.tolist(),
        "direct_report": direct_report,
        "pipeline_confusion_matrix": pipeline_cm.tolist(),
        "pipeline_counts": {
            "matched_supported": len(matched_supported),
            "correct_type": len(correct_type),
            "wrong_type": len(matched_supported) - len(correct_type),
            "false_positives": len(false_positives),
            "missed": len(pipeline_misses),
            "ignored_unsupported": len(ignored_unsupported),
        },
        "pipeline_detection_metrics": pipeline_detection_metrics,
        "pipeline_type_accuracy": type_accuracy,
        "timing_error": timing_summary(matched_supported),
        "per_video": per_video,
    }

    print_summary(summary, direct_report, per_video)
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

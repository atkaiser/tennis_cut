#!/usr/bin/env python3
"""Train a binary image classifier to detect shot vs no_shot frames."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastai.metrics import F1Score, Precision, Recall
from fastai.torch_core import defaults
from fastai.vision.all import (
    CategoryBlock,
    ClassificationInterpretation,
    DataBlock,
    ImageBlock,
    ImageDataLoaders,
    Resize,
    accuracy,
    aug_transforms,
    get_image_files,
    resnet18,
    resnet34,
    vision_learner,
)


def source_group(path: Path) -> str:
    """Return the source clip identifier from a dataset frame path."""
    stem, _, _ = path.stem.rpartition("_")
    return stem or path.stem


def grouped_splitter(
    items: list[Path], valid_pct: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split items by source clip so related frames stay together."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        groups[source_group(Path(item))].append(idx)

    group_ids = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    target_valid = max(1, round(len(items) * valid_pct))
    valid_group_ids: set[str] = set()
    valid_count = 0
    for group_id in group_ids:
        valid_group_ids.add(group_id)
        valid_count += len(groups[group_id])
        if valid_count >= target_valid:
            break

    train_idxs: list[int] = []
    valid_idxs: list[int] = []
    for group_id, idxs in groups.items():
        if group_id in valid_group_ids:
            valid_idxs.extend(idxs)
        else:
            train_idxs.extend(idxs)
    return train_idxs, valid_idxs


def binary_label(path: Path) -> str:
    """Map multiclass dataset folders onto binary shot/no_shot labels."""
    return "no_shot" if path.parent.name == "no_shot" else "shot"


def make_dataloaders(
    data_path: Path,
    bs: int,
    device: torch.device,
    img_size: int,
    valid_pct: float,
    seed: int,
) -> ImageDataLoaders:
    """Create image dataloaders for training and validation."""
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=lambda items: grouped_splitter(
            [Path(item) for item in items], valid_pct=valid_pct, seed=seed
        ),
        get_y=lambda item: binary_label(Path(item)),
        item_tfms=Resize(img_size),
        batch_tfms=aug_transforms(max_warp=0),
    )
    return dblock.dataloaders(data_path, bs=bs, device=device)


def resolve_arch(arch: str):
    """Map architecture names to fastai/torchvision constructors."""
    arch_map = {
        "resnet18": resnet18,
        "resnet34": resnet34,
    }
    if arch not in arch_map:
        raise ValueError(f"Unsupported architecture: {arch}")
    return arch_map[arch]


def print_classification_report(cm: np.ndarray, vocab: list[str]) -> None:
    """Print per-class precision/recall/F1/support from a confusion matrix."""
    print("Per-class metrics:")
    print(f"{'class':<12} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}")
    for idx, label in enumerate(vocab):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        print(f"{label:<12} {precision:>9.3f} {recall:>9.3f} {f1:>9.3f} {support:>9d}")


def move_batch_tfms_to_device(dls: ImageDataLoaders, device: torch.device) -> None:
    """Move tensor-valued batch transform state onto the requested device."""
    for tfm in dls.after_batch.fs:
        for attr in ("mean", "std"):
            value = getattr(tfm, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(tfm, attr, value.to(device))


def summarize_split(dls: ImageDataLoaders) -> tuple[int, int, int, int]:
    """Return train/valid group and image counts for logging."""
    train_items = [Path(item) for item in dls.train_ds.items]
    valid_items = [Path(item) for item in dls.valid_ds.items]
    train_groups = {source_group(item) for item in train_items}
    valid_groups = {source_group(item) for item in valid_items}
    return len(train_groups), len(valid_groups), len(train_items), len(valid_items)


def positive_class_index(vocab: list[str]) -> int:
    """Return the index used for shot-focused precision and recall."""
    return vocab.index("shot")


def export_validation_review(
    learn,
    dls: ImageDataLoaders,
    out_dir: Path,
    stamp: str,
) -> None:
    """Write validation predictions and grouped mistake folders for manual review."""
    probs, targs = learn.get_preds(dl=dls.valid)
    pred_idxs = probs.argmax(dim=1)
    vocab = list(dls.vocab)
    items = [Path(item) for item in dls.valid_ds.items]

    review_dir = out_dir / f"shot_binary_review_{stamp}"
    review_dir.mkdir(parents=True, exist_ok=True)
    csv_path = review_dir / "validation_predictions.csv"

    fn_dir = review_dir / "false_negative_shot"
    fp_dir = review_dir / "false_positive_shot"
    correct_shot_dir = review_dir / "correct_shot"
    correct_no_shot_dir = review_dir / "correct_no_shot"
    for path in (fn_dir, fp_dir, correct_shot_dir, correct_no_shot_dir):
        path.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "source_group",
                "actual_label",
                "predicted_label",
                "predicted_score",
                "shot_score",
                "no_shot_score",
                "bucket",
            ]
        )
        for item, targ_idx, pred_idx, prob in zip(
            items, targs.tolist(), pred_idxs.tolist(), probs
        ):
            actual_label = vocab[targ_idx]
            predicted_label = vocab[pred_idx]
            shot_score = float(prob[vocab.index("shot")])
            no_shot_score = float(prob[vocab.index("no_shot")])
            predicted_score = float(prob[pred_idx])
            if actual_label == "shot" and predicted_label == "no_shot":
                bucket = "false_negative_shot"
                target_dir = fn_dir
            elif actual_label == "no_shot" and predicted_label == "shot":
                bucket = "false_positive_shot"
                target_dir = fp_dir
            elif actual_label == "shot":
                bucket = "correct_shot"
                target_dir = correct_shot_dir
            else:
                bucket = "correct_no_shot"
                target_dir = correct_no_shot_dir

            link_name = target_dir / item.name
            if not link_name.exists():
                try:
                    link_name.symlink_to(item.resolve())
                except OSError:
                    shutil.copy2(item, link_name)

            writer.writerow(
                [
                    str(item),
                    source_group(item),
                    actual_label,
                    predicted_label,
                    f"{predicted_score:.6f}",
                    f"{shot_score:.6f}",
                    f"{no_shot_score:.6f}",
                    bucket,
                ]
            )

    print("Saved validation review artifacts to:", review_dir)


def main(
    data_dir: str,
    epochs: int,
    bs: int,
    lr: float | None,
    out_dir: str,
    device: str,
    arch: str,
    img_size: int,
    valid_pct: float,
    seed: int,
    inspect_errors: bool,
) -> None:
    """Train the binary shot detector on ``data_dir``."""
    torch_device = torch.device(device)
    defaults.device = torch_device

    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dls = make_dataloaders(
        data_path=data_path,
        bs=bs,
        device=torch_device,
        img_size=img_size,
        valid_pct=valid_pct,
        seed=seed,
    )
    dls.to(torch_device)
    move_batch_tfms_to_device(dls, torch_device)
    train_groups, valid_groups, train_images, valid_images = summarize_split(dls)
    print(
        "Validation split: group "
        f"({train_groups} train groups, {valid_groups} valid groups, "
        f"{train_images} train images, {valid_images} valid images)"
    )

    shot_idx = positive_class_index(list(dls.vocab))
    arch_func = resolve_arch(arch)
    learn = vision_learner(
        dls,
        arch_func,
        metrics=[
            accuracy,
            F1Score(average="macro"),
            Precision(average="binary", pos_label=shot_idx),
            Recall(average="binary", pos_label=shot_idx),
        ],
    )
    move_batch_tfms_to_device(learn.dls, torch_device)

    if lr is None:
        res = learn.lr_find()
        lr = res[0] if isinstance(res, tuple) else res

    print(f"Training {epochs} epochs @ {lr:.2e} ...")
    t0 = time.time()
    learn.fine_tune(epochs, base_lr=lr)
    print(f"Finished in {(time.time() - t0) / 60:.1f} min")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_directory = (Path.cwd() / out_path).resolve()
    export_path = out_directory / f"shot_binary_classifier_{stamp}.pkl"
    learn.export(export_path)
    print("Exported model to:", export_path)

    hist = pd.DataFrame(
        learn.recorder.values,
        columns=["train_loss", "valid_loss", "accuracy", "f1", "precision", "recall"],
    )
    hist["epoch"] = np.arange(1, len(hist) + 1)
    hist.to_csv(out_path / f"shot_binary_history_{stamp}.csv", index=False)
    print("Model & history saved.")

    interp = ClassificationInterpretation.from_learner(learn)
    cm = interp.confusion_matrix()
    print("Confusion Matrix:")
    print(cm)
    print_classification_report(np.asarray(cm), list(dls.vocab))

    if inspect_errors:
        export_validation_review(learn, dls, out_path, stamp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="mps",
    )
    ap.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "resnet34"],
        help="Model architecture",
    )
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--valid-pct", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--inspect-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write validation predictions and grouped mistake folders for manual review.",
    )
    args = ap.parse_args()

    main(
        args.data_dir,
        args.epochs,
        args.bs,
        args.lr,
        args.out_dir,
        args.device,
        args.arch,
        args.img_size,
        args.valid_pct,
        args.seed,
        args.inspect_errors,
    )

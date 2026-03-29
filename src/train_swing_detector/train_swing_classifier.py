#!/usr/bin/env python3
"""Train an image classifier to identify tennis shot types on shot-only frames."""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastai.metrics import F1Score
from fastai.losses import CrossEntropyLossFlat
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
    """Split items by source clip while preserving class coverage."""
    groups: dict[str, list[int]] = defaultdict(list)
    group_labels: dict[str, set[str]] = defaultdict(set)
    for idx, item in enumerate(items):
        path = Path(item)
        group_id = source_group(path)
        groups[group_id].append(idx)
        group_labels[group_id].add(path.parent.name)

    group_ids = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    target_valid = max(1, round(len(items) * valid_pct))
    valid_group_ids: set[str] = set()
    valid_count = 0

    total_groups_per_label = Counter()
    for labels in group_labels.values():
        for label in labels:
            total_groups_per_label[label] += 1
    valid_groups_per_label = Counter()

    labels_by_rarity = sorted(
        total_groups_per_label, key=lambda label: total_groups_per_label[label]
    )
    for label in labels_by_rarity:
        for group_id in group_ids:
            if group_id in valid_group_ids or label not in group_labels[group_id]:
                continue
            if any(
                valid_groups_per_label[group_label]
                >= total_groups_per_label[group_label] - 1
                for group_label in group_labels[group_id]
            ):
                continue
            valid_group_ids.add(group_id)
            valid_count += len(groups[group_id])
            for group_label in group_labels[group_id]:
                valid_groups_per_label[group_label] += 1
            break

    for group_id in group_ids:
        if group_id in valid_group_ids:
            continue
        if any(
            valid_groups_per_label[group_label]
            >= total_groups_per_label[group_label] - 1
            for group_label in group_labels[group_id]
        ):
            continue
        valid_group_ids.add(group_id)
        valid_count += len(groups[group_id])
        for group_label in group_labels[group_id]:
            valid_groups_per_label[group_label] += 1
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


def get_shot_image_files(data_path: Path) -> list[Path]:
    """Return image files from all shot classes, excluding ``no_shot``."""
    items: list[Path] = []
    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name == "no_shot":
            continue
        items.extend(get_image_files(class_dir))
    return items


def make_dataloaders(
    data_path: Path,
    bs: int,
    device: torch.device,
    img_size: int,
    valid_pct: float,
    seed: int,
) -> ImageDataLoaders:
    """Create image dataloaders for shot-type training and validation."""
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_shot_image_files,
        splitter=lambda items: grouped_splitter(
            [Path(item) for item in items], valid_pct=valid_pct, seed=seed
        ),
        get_y=lambda item: Path(item).parent.name,
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


def print_split_class_summary(dls: ImageDataLoaders) -> None:
    """Print class counts for the grouped train/valid split."""
    train_counts = Counter(Path(item).parent.name for item in dls.train_ds.items)
    valid_counts = Counter(Path(item).parent.name for item in dls.valid_ds.items)
    print("Split class counts:")
    print(f"{'class':<12} {'train':>8} {'valid':>8}")
    for label in dls.vocab:
        print(f"{label:<12} {train_counts[label]:>8d} {valid_counts[label]:>8d}")


def make_loss_weights(dls: ImageDataLoaders, device: torch.device) -> torch.Tensor:
    """Build inverse-frequency class weights from the training split."""
    train_counts = Counter(Path(item).parent.name for item in dls.train_ds.items)
    weights = torch.tensor(
        [1.0 / train_counts[label] for label in dls.vocab],
        dtype=torch.float32,
        device=device,
    )
    return weights / weights.mean()


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
) -> None:
    """Train the shot-type classifier on ``data_dir``."""
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
    print_split_class_summary(dls)

    arch_func = resolve_arch(arch)
    loss_weights = make_loss_weights(dls, torch_device)
    learn = vision_learner(
        dls,
        arch_func,
        loss_func=CrossEntropyLossFlat(weight=loss_weights),
        metrics=[accuracy, F1Score(average="macro")],
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
    export_path = out_directory / f"shot_type_classifier_{stamp}.pkl"
    learn.export(export_path)
    print("Exported model to:", export_path)

    hist = pd.DataFrame(
        learn.recorder.values,
        columns=["train_loss", "valid_loss", "accuracy", "f1"],
    )
    hist["epoch"] = np.arange(1, len(hist) + 1)
    hist.to_csv(out_path / f"history_{stamp}.csv", index=False)
    print("Model & history saved.")

    interp = ClassificationInterpretation.from_learner(learn)
    cm = interp.confusion_matrix()
    print("Confusion Matrix:")
    print(cm)
    print_classification_report(np.asarray(cm), list(dls.vocab))


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
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--valid-pct", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
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
    )

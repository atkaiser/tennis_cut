#!/usr/bin/env python3
"""train_swing_classifier.py
--------------------------------
Train an image classifier to identify tennis swing types.

The script expects a dataset directory with sub-folders named after the
labels (``forehand``, ``backhand``, ``volley``, ``serve`` and ``no_shot``).
It trains a small CNN using fastai and exports the learner for later use.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from fastai.torch_core import defaults
from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    aug_transforms,
    vision_learner,
    resnet18,
    accuracy,
)
from fastai.metrics import F1Score
from fastai.vision.all import ClassificationInterpretation


# ----------------------------------------------------------------------

def main(data_dir: str, epochs: int, bs: int, lr: float | None,
         out_dir: str, device: str, arch: str) -> None:
    """Train the swing classifier on ``data_dir``."""

    defaults.device = torch.device(device)

    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dls = ImageDataLoaders.from_folder(
        data_path,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(max_warp=0),
        bs=bs,
        device=device,
    )

    arch_func = resnet18 if arch == "resnet18" else arch
    learn = vision_learner(
        dls,
        arch_func,
        metrics=[accuracy, F1Score(average='macro')],
    )

    if lr is None:
        res = learn.lr_find()
        lr = res[0] if isinstance(res, tuple) else res

    print(f"Training {epochs} epochs @ {lr:.2e} ...")
    t0 = time.time()
    learn.fine_tune(epochs, base_lr=lr)
    print(f"Finished in {(time.time() - t0)/60:.1f} min")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_directory = (Path.cwd() / out_path).resolve()
    learn.export(out_directory / f"swing_classifier_{stamp}.pkl")
    print("Exported model to:", out_directory / f"swing_classifier_{stamp}.pkl")


    hist = pd.DataFrame(
        learn.recorder.values,
        columns=["train_loss", "valid_loss", "accuracy", "f1"],
    )
    hist["epoch"] = np.arange(1, len(hist) + 1)
    hist.to_csv(out_path / f"history_{stamp}.csv", index=False)
    print("Model & history saved.")

    interp = ClassificationInterpretation.from_learner(learn)
    print("Confusion Matrix:")
    print(interp.confusion_matrix())


# ----------------------------------------------------------------------
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
        help="Model architecture (passed to fastai vision_learner)",
    )
    args = ap.parse_args()

    main(args.data_dir, args.epochs, args.bs, args.lr,
         args.out_dir, args.device, args.arch)

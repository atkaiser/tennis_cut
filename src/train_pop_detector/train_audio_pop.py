#!/usr/bin/env python3
"""
train_audio_pop.py - fastai trainer for the tennis pop classifier.

This script trains the single supported model configuration:
- grouped validation split by source wav
- log-mel spectrogram features
- large 2D CNN
- random gain + Gaussian noise augmentation
- early stopping and best-checkpoint export
"""

import argparse
import pathlib
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from fastai.callback.schedule import fit_one_cycle  # noqa: F401
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.training import GradientClip
from fastai.data.block import CategoryBlock, DataBlock, TransformBlock
from fastai.data.transforms import IndexSplitter
from fastai.interpret import ClassificationInterpretation
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import F1Score, accuracy
from fastai.torch_core import TensorBase
from fasttransform.transform import Transform

WIN_SEC, SR = 0.25, 48_000
WIN_SAMP = int(WIN_SEC * SR)
THRESHOLD_GRID = np.linspace(0.1, 0.9, 17)
MONITOR_METRIC = "f1_score"
EXPERIMENT_NAME = "logmel_large"


class TensorAudio(TensorBase):
    """A 1-channel audio tensor."""


class TensorSpectrogram(TensorBase):
    """A spectrogram tensor for the 2D CNN."""


class AudioLoad(Transform):
    """Load a 0.25 s mono window starting at `row.start`."""

    def encodes(self, row):
        wav_path, start = row.wav_path, float(row.start)
        s0 = int(start * SR)
        sig, _ = torchaudio.load(wav_path, frame_offset=s0, num_frames=WIN_SAMP)
        if sig.shape[-1] < WIN_SAMP:
            pad = WIN_SAMP - sig.shape[-1]
            sig = torch.nn.functional.pad(sig, (0, pad))
        return TensorAudio(sig)


class DCOffset(Transform):
    def encodes(self, x: TensorAudio):
        return x - x.mean()


class PreEmphasis(Transform):
    def __init__(self, alpha: float = 0.97):
        self.alpha = alpha

    def encodes(self, x: TensorAudio):
        x = x.clone()
        x[..., 1:] -= self.alpha * x[..., :-1]
        return x


class RmsNorm(Transform):
    def encodes(self, x: TensorAudio):
        return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)


class RandomGain(Transform):
    order = 90

    def __init__(self, min_db: float = -6.0, max_db: float = 6.0):
        self.min_db = min_db
        self.max_db = max_db
        self.split_idx = 0

    def encodes(self, x: TensorAudio):
        db = torch.empty(1, device=x.device).uniform_(self.min_db, self.max_db)
        gain = torch.pow(10.0, db / 20.0)
        return x * gain


class AddGaussianSNR(Transform):
    order = 90

    def __init__(self, min_snr: float = 10, max_snr: float = 30):
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.split_idx = 0

    def encodes(self, x: TensorAudio):
        snr = torch.empty(1, device=x.device).uniform_(self.min_snr, self.max_snr)
        pwr = x.pow(2).mean()
        noise = torch.randn_like(x) * torch.sqrt(pwr / (10 ** (snr / 10)))
        return x + noise


class LogMelSpectrogram(Transform):
    order = 99

    def __init__(
        self,
        sample_rate: int = SR,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 64,
    ):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def encodes(self, x: TensorAudio):
        self.mel = self.mel.to(x.device)
        self.db = self.db.to(x.device)
        spec = self.db(self.mel(x))
        return TensorSpectrogram(spec)


def make_audio_block() -> TransformBlock:
    return TransformBlock(
        type_tfms=[AudioLoad(), DCOffset(), PreEmphasis(), RmsNorm()]
    )


def make_batch_tfms() -> list[Transform]:
    return [RandomGain(), AddGaussianSNR(), LogMelSpectrogram()]


def make_logmel_2d_cnn_large() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.1),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.15),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 2),
    )


def make_group_valid_idx(df: pd.DataFrame, valid_pct: float, seed: int) -> list[int]:
    """Split by source wav so neighboring windows do not leak across folds."""
    rng = np.random.default_rng(seed)
    wavs = df["wav_path"].drop_duplicates().to_numpy(copy=True)
    rng.shuffle(wavs)

    target_rows = max(1, int(round(len(df) * valid_pct)))
    valid_wavs: list[str] = []
    valid_rows = 0
    wav_counts = df["wav_path"].value_counts()
    for wav in wavs:
        valid_wavs.append(wav)
        valid_rows += int(wav_counts[wav])
        if valid_rows >= target_rows:
            break

    return df.index[df["wav_path"].isin(valid_wavs)].tolist()


def compute_class_weights(
    df: pd.DataFrame, valid_idx: list[int], device: str
) -> torch.Tensor:
    train_df = df.drop(index=valid_idx)
    counts = train_df["label"].value_counts()
    neg = max(1, int(counts.get("neg", 0)))
    pos = max(1, int(counts.get("pos", 0)))
    return torch.tensor([1.0, neg / pos], device=device)


def find_best_epoch(hist: pd.DataFrame, monitor: str) -> tuple[int, float]:
    if monitor not in hist.columns:
        raise KeyError(f"Missing monitored metric column: {monitor}")
    best_idx = int(hist[monitor].astype(float).idxmax())
    best_epoch = int(hist.loc[best_idx, "epoch"])
    best_value = float(hist.loc[best_idx, monitor])
    return best_epoch, best_value


def summarize_thresholds(
    learn: Learner,
    out_dir: pathlib.Path,
    stamp: str,
    max_fp: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    probs, targs = learn.get_preds()
    pos_probs = probs[:, 1].cpu().numpy()
    y_true = targs.cpu().numpy().astype(int)

    rows: list[dict[str, float | int | bool]] = []
    for thr in THRESHOLD_GRID:
        y_pred = (pos_probs >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "within_fp_cap": fp <= max_fp,
            }
        )

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / f"thresholds_{EXPERIMENT_NAME}_{stamp}.csv", index=False)

    valid = sweep[sweep["within_fp_cap"]]
    guardrail_failed = valid.empty
    ranked = (valid if not guardrail_failed else sweep).sort_values(
        by=["recall", "f1", "threshold"],
        ascending=[False, False, True],
    )
    best_row = ranked.iloc[0].to_dict()

    print(
        f"Selected validation threshold={best_row['threshold']:.2f} "
        f"precision={best_row['precision']:.3f} "
        f"recall={best_row['recall']:.3f} "
        f"f1={best_row['f1']:.3f}"
    )
    status = "FAILED" if guardrail_failed else "passed"
    print(f"False-positive guardrail ({max_fp}) {status}.")
    print("Threshold sweep:")
    print(
        sweep[
            ["threshold", "precision", "recall", "f1", "fp", "fn", "within_fp_cap"]
        ].to_string(index=False)
    )
    print("Thresholded Confusion Matrix:")
    print(
        np.array(
            [
                [int(best_row["tn"]), int(best_row["fp"])],
                [int(best_row["fn"]), int(best_row["tp"])],
            ]
        )
    )
    return sweep, best_row


def main(args: argparse.Namespace):
    csv_path = args.csv_path
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    valid_idx = make_group_valid_idx(df, valid_pct=args.valid_pct, seed=args.seed)

    dblock = DataBlock(
        blocks=(make_audio_block(), CategoryBlock),
        get_x=lambda r: r,
        get_y=lambda r: r["label"],
        splitter=IndexSplitter(valid_idx),
        batch_tfms=make_batch_tfms(),
    )
    dls = dblock.dataloaders(df, bs=args.bs, device=args.device)
    pos_w = compute_class_weights(df, valid_idx=valid_idx, device=args.device)
    print(
        "Split summary:",
        f"train_wavs={df.drop(index=valid_idx)['wav_path'].nunique()}",
        f"valid_wavs={df.loc[valid_idx, 'wav_path'].nunique()}",
        f"class_weights={pos_w.tolist()}",
        f"experiment={EXPERIMENT_NAME}",
    )

    checkpoint_name = f"best_{EXPERIMENT_NAME}"
    callbacks = [
        GradientClip(max_norm=args.grad_clip),
        SaveModelCallback(
            monitor=MONITOR_METRIC,
            comp=np.greater,
            fname=checkpoint_name,
            with_opt=True,
        ),
        EarlyStoppingCallback(
            monitor=MONITOR_METRIC,
            comp=np.greater,
            patience=args.early_stop_patience,
        ),
    ]
    learn = Learner(
        dls,
        make_logmel_2d_cnn_large(),
        loss_func=CrossEntropyLossFlat(weight=pos_w),
        metrics=[accuracy, F1Score()],
        path=out_dir,
        model_dir=".",
        cbs=callbacks,
    )

    lr = args.lr
    print(f"Training {args.epochs} epochs @ {lr:.2e} ...")
    t0 = time.time()
    learn.fit_one_cycle(args.epochs, lr_max=lr)
    print(f"Finished in {(time.time() - t0) / 60:.1f} min")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    hist = pd.DataFrame(
        learn.recorder.values,
        columns=["train_loss", "valid_loss", "accuracy", "f1"],
    )
    hist["epoch"] = np.arange(1, len(hist) + 1)
    hist["experiment"] = EXPERIMENT_NAME
    hist["monitor_metric"] = MONITOR_METRIC
    best_epoch, best_value = find_best_epoch(hist, monitor="f1")
    hist["best_epoch"] = best_epoch
    hist["best_f1"] = best_value
    export_name = f"audio_pop_{EXPERIMENT_NAME}_{stamp}.pth"

    learn.load(checkpoint_name, with_opt=True)
    learn.export(export_name)
    checkpoint_path = out_dir / f"{checkpoint_name}.pth"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(
        "Best checkpoint reloaded:",
        f"epoch={best_epoch}",
        f"{MONITOR_METRIC}={best_value:.6f}",
    )
    print("Model saved.")

    interp = ClassificationInterpretation.from_learner(learn)
    print("Confusion Matrix:")
    print(interp.confusion_matrix())
    summarize_thresholds(
        learn,
        out_dir=out_dir,
        stamp=stamp,
        max_fp=args.max_fp,
    )
    interp.plot_confusion_matrix()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    ap.add_argument("--valid-pct", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-fp", type=int, default=650)
    ap.add_argument("--early-stop-patience", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    main(args)

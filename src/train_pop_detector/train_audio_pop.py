#!/usr/bin/env python3
"""
train_audio_pop.py  –  fast ai trainer for the tennis “pop” classifier
(no fastaudio dependency)

Run, e.g.:
    python train_audio_pop.py meta/train_all.csv --epochs 15 --out-dir models
"""

import argparse
import pathlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchaudio
from fasttransform.transform import Transform
from fastai.data.block import DataBlock, CategoryBlock, TransformBlock
from fastai.data.transforms import IndexSplitter
from fastai.interpret import ClassificationInterpretation
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import F1Score, accuracy
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.torch_core import TensorBase
# Importing these patches registers the lr_find and fit_one_cycle methods on
# fastai's Learner class, so even if they aren't directly used, they are still needed
from fastai.callback.schedule import lr_find, fit_one_cycle  # noqa: F401
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
WIN_SEC, SR = 0.25, 48_000
WIN_SAMP    = int(WIN_SEC * SR)
THRESHOLD_GRID = np.linspace(0.1, 0.9, 17)
MONITOR_METRIC = "f1_score"

class TensorAudio(TensorBase):
    """A 1-channel audio tensor"""
    pass


class TensorSpectrogram(TensorBase):
    """A spectrogram tensor for the 2D CNN path."""
    pass

# --------------------------- Transforms -------------------------------
class AudioLoad(Transform):
    "Load a 0.25 s mono window starting at `row.start`"
    def encodes(self, row):
        wav_path, start = row.wav_path, float(row.start)
        s0 = int(start * SR)
        sig, _ = torchaudio.load(wav_path, frame_offset=s0, num_frames=WIN_SAMP)
        if sig.shape[-1] < WIN_SAMP:
            pad = WIN_SAMP - sig.shape[-1]
            sig = torch.nn.functional.pad(sig, (0, pad))
        return TensorAudio(sig)                            # (1, samples)

class DCOffset(Transform):
    def encodes(self, x:TensorAudio): return x - x.mean()

class PreEmphasis(Transform):
    def __init__(self, α=0.97): self.a = α
    def encodes(self, x:TensorAudio):
        x = x.clone()
        x[...,1:] -= self.a * x[...,:-1]
        return x

class RmsNorm(Transform):
    def encodes(self, x:TensorAudio):
        return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)


class LogMelSpectrogram(Transform):
    order = 99

    def __init__(
        self,
        sample_rate: int = SR,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 64,
    ) -> None:
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def encodes(self, x: TensorAudio) -> TensorSpectrogram:
        self.mel = self.mel.to(x.device)
        self.db = self.db.to(x.device)
        spec = self.mel(x)
        spec = self.db(spec)
        return TensorSpectrogram(spec)

class AddGaussianSNR(Transform):
    order = 90
    def __init__(self, min_snr=10, max_snr=30):
        self.min,self.max=min_snr,max_snr
        self.split_idx = 0
    def encodes(self, x:TensorAudio):
        snr = torch.empty(1, device=x.device).uniform_(self.min,self.max)
        pwr = x.pow(2).mean()
        noise = torch.randn_like(x) * torch.sqrt(pwr / (10**(snr/10)))
        return x + noise

class RandomGain(Transform):
    """Apply a random gain between `min_db` and `max_db`."""
    order = 90

    def __init__(self, min_db: float = -6.0, max_db: float = 6.0) -> None:
        self.min, self.max = min_db, max_db
        self.split_idx = 0

    def encodes(self, x: TensorAudio) -> TensorAudio:
        db = torch.empty(1, device=x.device).uniform_(self.min, self.max)
        gain = torch.pow(10.0, db / 20.0)
        return x * gain


class WhiteNoiseSegment(Transform):
    """Inject white noise into a random short segment."""
    order = 90

    def __init__(self, seg_ms: int = 15, amp: float = 0.02) -> None:
        self.seg_samples = int((seg_ms / 1000) * SR)
        self.amp = amp
        self.split_idx = 0

    def encodes(self, x: TensorAudio) -> TensorAudio:
        if self.seg_samples <= 0 or self.seg_samples >= x.shape[-1]:
            return x
        start = torch.randint(0, x.shape[-1] - self.seg_samples, (1,))
        noise = torch.randn(self.seg_samples, device=x.device) * self.amp
        x = x.clone()
        x[..., start : start + self.seg_samples] += noise
        return x


class TimeMask(Transform):
    """Zero out a random time span inside the audio."""
    order = 90

    def __init__(self, mask_ms: int = 30) -> None:
        self.mask_samples = int((mask_ms / 1000) * SR)
        self.split_idx = 0

    def encodes(self, x: TensorAudio) -> TensorAudio:
        if self.mask_samples <= 0 or self.mask_samples >= x.shape[-1]:
            return x
        start = torch.randint(0, x.shape[-1] - self.mask_samples, (1,))
        x = x.clone()
        x[..., start : start + self.mask_samples] = 0
        return x

def make_audio_block(feature_type: str) -> TransformBlock:
    del feature_type
    return TransformBlock(type_tfms=[AudioLoad(), DCOffset(), PreEmphasis(), RmsNorm()])

# --------------------- tiny raw‑waveform CNN --------------------------
def make_raw1d_cnn():
    return torch.nn.Sequential(
        torch.nn.Conv1d(1,  8, 11, stride=2), torch.nn.ReLU(),
        torch.nn.Conv1d(8, 16, 11, stride=2), torch.nn.ReLU(),
        torch.nn.Conv1d(16,32, 11, stride=2), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
        torch.nn.Linear(32, 2)
    )


def make_logmel_2d_cnn():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 2),
    )


def make_model(feature_type: str) -> torch.nn.Module:
    if feature_type == "logmel":
        return make_logmel_2d_cnn()
    return make_raw1d_cnn()


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


def compute_class_weights(df: pd.DataFrame, valid_idx: list[int], device: str) -> torch.Tensor:
    train_df = df.drop(index=valid_idx)
    counts = train_df["label"].value_counts()
    neg = max(1, int(counts.get("neg", 0)))
    pos = max(1, int(counts.get("pos", 0)))
    return torch.tensor([1.0, neg / pos], device=device)


def make_batch_tfms(args: argparse.Namespace) -> list[Transform]:
    tfms: list[Transform] = []
    if not args.disable_random_gain:
        tfms.append(RandomGain())
    if not args.disable_white_noise_segment:
        tfms.append(WhiteNoiseSegment())
    if not args.disable_time_mask:
        tfms.append(TimeMask())
    if not args.disable_gaussian_snr:
        tfms.append(AddGaussianSNR())
    if args.feature_type == "logmel":
        tfms.append(LogMelSpectrogram())
    return tfms


def build_experiment_name(args: argparse.Namespace) -> str:
    parts = [args.feature_type]
    if args.disable_random_gain:
        parts.append("nogain")
    if args.disable_white_noise_segment:
        parts.append("nowhitenoise")
    if args.disable_time_mask:
        parts.append("notimemask")
    if args.disable_gaussian_snr:
        parts.append("nosnr")
    return "_".join(parts)


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
    experiment_name: str,
    max_fp: int | None,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
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
                "within_fp_cap": max_fp is None or fp <= max_fp,
            }
        )

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / f"thresholds_{experiment_name}_{stamp}.csv", index=False)

    valid = sweep[sweep["within_fp_cap"]]
    guardrail_failed = valid.empty and max_fp is not None
    if max_fp is None:
        ranked = sweep.sort_values(
            by=["f1", "recall", "threshold"],
            ascending=[False, False, True],
        )
    elif not valid.empty:
        ranked = valid.sort_values(
            by=["recall", "f1", "threshold"],
            ascending=[False, False, True],
        )
    else:
        ranked = sweep.sort_values(
            by=["f1", "recall", "threshold"],
            ascending=[False, False, True],
        )
    best_row = ranked.iloc[0].to_dict()

    print(
        f"Selected validation threshold={best_row['threshold']:.2f} "
        f"precision={best_row['precision']:.3f} "
        f"recall={best_row['recall']:.3f} "
        f"f1={best_row['f1']:.3f}"
    )
    if max_fp is not None:
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
    best_row["guardrail_failed"] = guardrail_failed
    return sweep, best_row

# --------------------------- main -------------------------------------
def main(args: argparse.Namespace):
    csv_path = args.csv_path
    out_dir = pathlib.Path(args.out_dir)
    device = args.device
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df      = pd.read_csv(csv_path)
    valid_idx = make_group_valid_idx(df, valid_pct=args.valid_pct, seed=args.seed)
    experiment_name = build_experiment_name(args)

    dblock = DataBlock(
        blocks   =(make_audio_block(args.feature_type), CategoryBlock),
        get_x    = lambda r: r,                 # row passed to AudioLoad
        get_y    = lambda r: r['label'],
        splitter = IndexSplitter(valid_idx),
        batch_tfms=make_batch_tfms(args),
    )
    dls = dblock.dataloaders(df, bs=args.bs, device=device)
    pos_w = compute_class_weights(df, valid_idx=valid_idx, device=device)
    print(
        "Split summary:",
        f"train_wavs={df.drop(index=valid_idx)['wav_path'].nunique()}",
        f"valid_wavs={df.loc[valid_idx, 'wav_path'].nunique()}",
        f"class_weights={pos_w.tolist()}",
        f"experiment={experiment_name}",
    )

    checkpoint_name = f"best_{experiment_name}"
    callbacks = [
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
        make_model(args.feature_type),
        loss_func=CrossEntropyLossFlat(weight=pos_w),
        metrics=[accuracy, F1Score()],
        path=out_dir,
        model_dir=".",
        cbs=callbacks,
    )
    
    lr = args.lr
    if lr is None:                         # auto‑select LR
        res = learn.lr_find()              # may be a float *or* a tuple
        lr  = res[0] if isinstance(res, tuple) else res

    print(f"Training {args.epochs} epochs @ {lr:.2e} ...")
    t0 = time.time()
    learn.fit_one_cycle(args.epochs, lr_max=lr)
    print(f"Finished in {(time.time()-t0)/60:.1f} min")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    hist = pd.DataFrame(learn.recorder.values,
                        columns=['train_loss','valid_loss','accuracy','f1'])
    hist['epoch'] = np.arange(1, len(hist) + 1)
    hist["feature_type"] = args.feature_type
    hist["experiment"] = experiment_name
    hist["monitor_metric"] = MONITOR_METRIC
    best_epoch, best_value = find_best_epoch(hist, monitor="f1")
    hist["best_epoch"] = best_epoch
    hist["best_f1"] = best_value
    hist.to_csv(out_dir / f"history_{experiment_name}_{stamp}.csv", index=False)
    learn.load(checkpoint_name, with_opt=True)
    learn.export(out_dir / f"audio_pop_{experiment_name}_{stamp}.pth")
    print(
        "Best checkpoint reloaded:",
        f"epoch={best_epoch}",
        f"{MONITOR_METRIC}={best_value:.6f}",
    )
    print("Model & history saved.")
    interp = ClassificationInterpretation.from_learner(learn)
    print("Confusion Matrix:")
    print(interp.confusion_matrix())
    summarize_thresholds(
        learn,
        out_dir=out_dir,
        stamp=stamp,
        experiment_name=experiment_name,
        max_fp=args.max_fp,
    )
    # If you want to graph the confusion matrix:
    interp.plot_confusion_matrix()
    plt.show()

# --------------------------- CLI --------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--device", choices=["cpu","cuda","mps"], default="mps")
    ap.add_argument("--valid-pct", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-type", choices=["raw", "logmel"], default="raw")
    ap.add_argument("--max-fp", type=int, default=None)
    ap.add_argument("--early-stop-patience", type=int, default=4)
    ap.add_argument("--disable-random-gain", action="store_true")
    ap.add_argument("--disable-white-noise-segment", action="store_true")
    ap.add_argument("--disable-time-mask", action="store_true")
    ap.add_argument("--disable-gaussian-snr", action="store_true")
    args = ap.parse_args()

    main(args)

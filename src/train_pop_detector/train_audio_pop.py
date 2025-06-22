#!/usr/bin/env python3
"""
train_audio_pop.py  –  fast ai trainer for the tennis “pop” classifier
(no fastaudio dependency)

Run, e.g.:
    python train_audio_pop.py meta/train_all.csv --epochs 15 --out-dir models
"""

import argparse
import time
import pathlib
from datetime import datetime

import torch
import torchaudio
import pandas as pd
import numpy as np
from fasttransform.transform import Transform
from fastai.data.block  import DataBlock, CategoryBlock, TransformBlock
from fastai.data.transforms import IndexSplitter
from fastai.metrics     import accuracy, F1Score
from fastai.learner     import Learner
from fastai.torch_core  import TensorBase
from fastai.interpret import ClassificationInterpretation
from fastai.losses import CrossEntropyLossFlat

# ----------------------------------------------------------------------
WIN_SEC, SR = 0.25, 48_000
WIN_SAMP    = int(WIN_SEC * SR)

def kfold_indices(n:int, k:int, seed:int=42):
    """Yield validation indices for ``k`` folds."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_sizes = [n//k + (1 if i < n%k else 0) for i in range(k)]
    cur = 0
    for fs in fold_sizes:
        yield idx[cur:cur+fs]
        cur += fs

class TensorAudio(TensorBase):
    """A 1-channel audio tensor"""
    pass

# --------------------------- Transforms -------------------------------
class AudioLoad(Transform):
    "Load a 0.25 s mono window starting at `row.start`"
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
    """Apply random gain between ``min_db`` and ``max_db`` in dB."""
    order = 90
    def __init__(self, min_db=-6., max_db=6.):
        self.min_db, self.max_db = min_db, max_db
        self.split_idx = 0
    def encodes(self, x:TensorAudio):
        db = torch.empty(1, device=x.device).uniform_(self.min_db, self.max_db)
        gain = 10 ** (db / 20)
        return x * gain

class AddNoiseBurst(Transform):
    """Inject a short white-noise burst."""
    order = 90
    def __init__(self, ms=15, min_snr=10, max_snr=30):
        self.samples = int(ms/1000 * SR)
        self.min_snr, self.max_snr = min_snr, max_snr
        self.split_idx = 0
    def encodes(self, x:TensorAudio):
        start = torch.randint(0, x.shape[-1]-self.samples, (1,)).item()
        snr = torch.empty(1, device=x.device).uniform_(self.min_snr, self.max_snr)
        pwr = x.pow(2).mean()
        noise = torch.randn(self.samples, device=x.device) * torch.sqrt(pwr / (10**(snr/10)))
        x = x.clone()
        x[..., start:start+self.samples] += noise
        return x

class TimeMask(Transform):
    """Zero out a random time span."""
    order = 90
    def __init__(self, ms=30):
        self.samples = int(ms/1000 * SR)
        self.split_idx = 0
    def encodes(self, x:TensorAudio):
        start = torch.randint(0, x.shape[-1]-self.samples, (1,)).item()
        x = x.clone()
        x[..., start:start+self.samples] = 0
        return x

# Helper block so DataBlock is clean
def AudioBlock():
    return TransformBlock(type_tfms=[AudioLoad(),
                                     DCOffset(), PreEmphasis(), RmsNorm()])

# --------------------- tiny raw‑waveform CNN --------------------------
def make_raw1d_cnn():
    return torch.nn.Sequential(
        torch.nn.Conv1d(1,  8, 11, stride=2), torch.nn.ReLU(),
        torch.nn.Conv1d(8, 16, 11, stride=2), torch.nn.ReLU(),
        torch.nn.Conv1d(16,32, 11, stride=2), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
        torch.nn.Linear(32, 2)
    )

# --------------------------- main -------------------------------------
def main(csv_path:str, epochs:int, bs:int, lr:float,
         out_dir:str, device:str, folds:int):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df      = pd.read_csv(csv_path)

    if folds < 2:
        splits = [np.arange(len(df))]
    else:
        splits = list(kfold_indices(len(df), folds))

    f1_scores = []
    for i, val_idx in enumerate(splits):
        splitter_fn = IndexSplitter(val_idx)
        dblock = DataBlock(
            blocks   =(AudioBlock(), CategoryBlock),
            get_x    = lambda r: r,
            get_y    = lambda r: r['label'],
            splitter = splitter_fn,
            batch_tfms=[RandomGain(), AddNoiseBurst(), TimeMask(), AddGaussianSNR()]
        )
        dls = dblock.dataloaders(df, bs=bs, device=device)
        pos_w = torch.tensor([1., 4.], device=device)
        learn = Learner(dls, make_raw1d_cnn(),
                        loss_func=CrossEntropyLossFlat(weight=pos_w),
                        metrics=[accuracy, F1Score()])

        lr_fold = lr
        if lr_fold is None:
            res = learn.lr_find()
            lr_fold = res[0] if isinstance(res, tuple) else res

        fold_msg = f"Fold {i+1}/{len(splits)}:" if folds > 1 else ""
        print(f"{fold_msg} Training {epochs} epochs @ {lr_fold:.2e} ...")
        t0 = time.time()
        learn.fit_one_cycle(epochs, lr_max=lr_fold)
        print(f"Finished in {(time.time()-t0)/60:.1f} min")

        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = f"audio_pop_f{i+1}_{stamp}.pth" if folds > 1 else f"audio_pop_{stamp}.pth"
        learn.export(out_dir/model_name)

        hist = pd.DataFrame(learn.recorder.values,
                            columns=['train_loss','valid_loss','accuracy','f1'])
        hist['epoch']=np.arange(1,len(hist)+1)
        hist_name = f"history_f{i+1}_{stamp}.csv" if folds > 1 else f"history_{stamp}.csv"
        hist.to_csv(out_dir/hist_name, index=False)
        interp = ClassificationInterpretation.from_learner(learn)
        print("Confusion Matrix:")
        print(interp.confusion_matrix())

        val_res = learn.validate()
        if len(val_res) > 2:
            f1_scores.append(float(val_res[2]))

    if f1_scores:
        mean = np.mean(f1_scores)
        std  = np.std(f1_scores)
        print(f"Mean F1 over {len(f1_scores)} folds: {mean:.3f} ± {std:.3f}")

# --------------------------- CLI --------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--device", choices=["cpu","cuda","mps"], default="mps")
    ap.add_argument("--folds", type=int, default=1, metavar="N",
                    help="number of cross-validation folds")
    args = ap.parse_args()

    main(args.csv_path, args.epochs, args.bs,
         args.lr, args.out_dir, args.device, args.folds)

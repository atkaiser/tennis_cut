#!/usr/bin/env python3
"""
train_audio_pop.py  –  fast ai trainer for the tennis “pop” classifier
(no fastaudio dependency)

Run, e.g.:
    python train_audio_pop.py meta/train_all.csv --epochs 15 --out-dir models
"""

import argparse, time, pathlib
from datetime import datetime

import torch, torchaudio
import pandas as pd, numpy as np
from fasttransform.transform import Transform
import fastai.callback.schedule
from fastai.data.block  import DataBlock, CategoryBlock, RandomSplitter, TransformBlock
from fastai.data.core   import Datasets
from fastai.metrics     import accuracy, F1Score
from fastai.learner     import Learner
from fastai.torch_core  import TensorBase

# ----------------------------------------------------------------------
WIN_SEC, SR = 0.25, 48_000
WIN_SAMP    = int(WIN_SEC * SR)

class TensorAudio(TensorBase): "A 1‑channel audio tensor"; pass

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
        x = x.clone(); x[...,1:] -= self.a * x[...,:-1]; return x

class RmsNorm(Transform):
    def encodes(self, x:TensorAudio):
        return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)

class AddGaussianSNR(Transform):
    order = 90
    def __init__(self, min_snr=10, max_snr=30):
        self.min,self.max=min_snr,max_snr
    def encodes(self, x:TensorAudio):
        snr = torch.empty(1, device=x.device).uniform_(self.min,self.max)
        pwr = x.pow(2).mean()
        noise = torch.randn_like(x) * torch.sqrt(pwr / (10**(snr/10)))
        return x + noise

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
         out_dir:str, device:str):
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df      = pd.read_csv(csv_path)

    dblock = DataBlock(
        blocks   =(AudioBlock(), CategoryBlock),
        get_x    = lambda r: r,                 # row passed to AudioLoad
        get_y    = lambda r: r['label'],
        splitter = RandomSplitter(0.2, seed=42),
        batch_tfms=[AddGaussianSNR()]
    )
    dls = dblock.dataloaders(df, bs=bs, device=device)

    learn = Learner(dls, make_raw1d_cnn(),
                    loss_func=torch.nn.CrossEntropyLoss(),
                    metrics=[accuracy, F1Score()])
    
    if lr is None:                         # auto‑select LR
        res = learn.lr_find()              # may be a float *or* a tuple
        lr  = res[0] if isinstance(res, tuple) else res

    print(f"Training {epochs} epochs @ {lr:.2e} ...")
    t0 = time.time()
    learn.fit_one_cycle(epochs, lr_max=lr)
    print(f"Finished in {(time.time()-t0)/60:.1f} min")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    learn.export(out_dir/f"audio_pop_{stamp}.pth")

    hist = pd.DataFrame(learn.recorder.values,
                        columns=['train_loss','valid_loss','accuracy','f1'])
    hist['epoch']=np.arange(1,len(hist)+1)
    hist.to_csv(out_dir/f"history_{stamp}.csv",index=False)
    print("Model & history saved.")

# --------------------------- CLI --------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--device", choices=["cpu","cuda","mps"], default="mps")
    args = ap.parse_args()

    main(args.csv_path, args.epochs, args.bs,
         args.lr, args.out_dir, args.device)

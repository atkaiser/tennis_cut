# train_pop_detector Agent Guide

This file gives local instructions for agents working in `src/train_pop_detector`.

## Scope
- Applies to this directory and its children.
- Follow the repository-level `AGENTS.md` as well.

## Goal
- Maintain and improve the single supported audio pop training path.
- Do not reintroduce alternate model families or abandoned augmentation branches unless the user asks for that explicitly.

## Supported Training Path
- Group validation by `wav_path`.
- Use log-mel spectrogram features.
- Train the large 2D CNN in `train_audio_pop.py`.
- Keep random gain and Gaussian noise augmentation enabled.
- Use early stopping, best-checkpoint reload, gradient clipping, and threshold selection under the FP cap.

## Default Command
```bash
MPLBACKEND=Agg uv run python src/train_pop_detector/train_audio_pop.py \
  meta/labled_windows.csv \
  --epochs 15 \
  --out-dir models \
  --device mps
```

## Runtime Knobs That Still Matter
- `--lr`
- `--epochs`
- `--bs`
- `--grad-clip`
- `--early-stop-patience`
- `--max-fp`
- `--seed`
- `--valid-pct`

## Evaluation Rules
- Use the selected validation threshold from the threshold sweep, not raw argmax alone.
- Default operating constraint is `--max-fp 650`.
- When comparing runs, report:
  - learning rate
  - selected threshold
  - precision
  - recall
  - F1
  - confusion matrix or FP/FN counts

## Artifacts
- Each run writes:
  - exported model
  - threshold-sweep CSV
- The best checkpoint is still used internally during training, then deleted after export.
- Prefer `/tmp/...` for exploratory runs and `models/` for intentional outputs the user wants to keep.

## Editing Guidance
- Keep the script focused on the single supported path.
- If you change defaults, update this file and `README.md` in the same turn.

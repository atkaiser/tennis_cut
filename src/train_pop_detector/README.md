# Train Pop Detector

Scripts for preparing the dataset and training the audio impact model.

## prepare_audio_windows.py
Extracts labelled audio windows from annotated videos and writes a CSV used for training.
Example:

```bash
uv run python src/train_pop_detector/prepare_audio_windows.py videos/ wav/ meta/train_all.csv --neg-per-pos 3 --far-neg-per-pos 1
```

This creates 0.25 s windows around each labelled impact along with near and far negative samples. Audio is extracted to the given `wav/` directory.

## train_audio_pop.py
The trainer now supports one production path only:
- grouped validation split by `wav_path`
- log-mel spectrogram features
- large 2D CNN
- random gain and Gaussian noise augmentation
- early stopping on validation F1
- best-checkpoint export
- threshold selection with a false-positive cap

Recommended command:

```bash
MPLBACKEND=Agg uv run python src/train_pop_detector/train_audio_pop.py \
  meta/labled_windows.csv \
  --epochs 15 \
  --out-dir models \
  --device mps
```

Key defaults:
- `--lr 5e-4`
- `--max-fp 650`
- `--early-stop-patience 4`
- `--grad-clip 1.0`

The script saves:
- exported model
- threshold-sweep CSV

If you need to tune training behavior, keep changes limited to runtime knobs such as `--lr`, `--epochs`, `--grad-clip`, or `--early-stop-patience`.

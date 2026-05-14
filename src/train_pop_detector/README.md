# Train Pop Detector

Scripts for preparing the dataset and training the audio impact model.

## prepare_audio_windows.py
Extracts labelled audio windows from annotated videos and writes a CSV used for training.
Example:

```bash
MPLBACKEND=Agg uv run python src/train_pop_detector/prepare_audio_windows.py \
  --videos_dir videos_train \
  --wav_dir wavs_train \
  --out_csv meta/audio_train_windows.csv \
  --neg-per-pos 3 \
  --far-neg-per-pos 1
```

This creates 0.25 s windows around each labelled impact along with near and far negative samples. Audio is extracted to `wavs_train/`, and the resulting CSV should reference only train audio files.

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
  meta/audio_train_windows.csv \
  --epochs 15 \
  --out-dir models \
  --device mps \
  --lr 5e-4 \
  --max-fp 650
```

The script saves:
- exported model
- threshold-sweep CSV

If you need to tune training behavior, keep changes limited to runtime knobs such as `--lr`, `--epochs`, `--grad-clip`, or `--early-stop-patience`.

## evaluate_audio_model.py
Evaluates one exported audio model on labelled test videos. It reports a dense-window confusion matrix and event-level detected shots, missed shots, and false positives after peak suppression.

Recommended command:

```bash
uv run python src/train_pop_detector/evaluate_audio_model.py \
  models/audio_pop_logmel_large_20260512231349.pth \
  --videos-dir videos_test \
  --wav-dir wavs_test \
  --threshold 0.5
```

The important line in the output is the below:
```
Event metrics: tp=545 fp=147 fn=78 precision=0.7876 recall=0.8748 f1=0.8289
```

This was for the `models/audio_pop_logmel_large_20260512231349.pth` model as of 2026/05/14. This means that from the test videos it found 545 shots correctly, with 147 false positives. It also missed finding 78 shots.
# Train Pop Detector

Scripts for preparing the dataset and training the audio impact model.

## prepare_audio_windows.py
Extracts labelled audio windows from annotated videos and writes a CSV used for training.
Example:

```bash
uv run python src/train_pop_detector/prepare_audio_windows.py videos/ wav/ meta/train_all.csv --neg-per-pos 3 --far-neg-per-pos 1
```

This creates 0.25 s windows around each labelled impact along with near and far negative samples. Audio is extracted to the given `wav/` directory.

## train_audio_pop.py
FastAI-based trainer for the pop classifier. The default path trains the raw waveform CNN:

```bash
uv run python src/train_pop_detector/train_audio_pop.py meta/labled_windows.csv --epochs 15 --out-dir models
```

For recall-focused experiments, you can disable individual augmentations, switch to log-mel features, and enforce a false-positive guardrail during threshold selection. Example:

```bash
MPLBACKEND=Agg uv run python src/train_pop_detector/train_audio_pop.py \
  meta/labled_windows.csv \
  --epochs 15 \
  --out-dir models \
  --feature-type logmel \
  --disable-time-mask \
  --disable-white-noise-segment \
  --max-fp 650
```

The script saves the exported model, a training history CSV, and a threshold-sweep CSV under the given output directory.
It also saves the best checkpoint during training and uses early stopping on validation F1 by default.

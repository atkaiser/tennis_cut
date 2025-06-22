# Train Pop Detector

Scripts for preparing the dataset and training the audio impact model.

## prepare_audio_windows.py
Extracts labelled audio windows from annotated videos and writes a CSV used for training.
Example:

```bash
python prepare_audio_windows.py videos/ wav/ meta/train_all.csv --neg-per-pos 3 --far-neg-per-pos 1
```

This creates 0.25 s windows around each labelled impact along with near and far negative samples. Audio is extracted to the given `wav/` directory.

## train_audio_pop.py
FastAI-based trainer for the raw waveform CNN. Run:

```bash
python train_audio_pop.py meta/train_all.csv --epochs 15 --out-dir models
```

The script saves the exported model and a training history CSV under the given output directory.

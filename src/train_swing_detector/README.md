# Train Swing Detector

Scripts for preparing the dataset and training the two-stage vision pipeline. The first model predicts **shot** vs **no_shot**. The second model runs only on shot frames and predicts **forehand**, **backhand**, **volley**, or **serve**.

## prepare_swing_frames.py
Extracts labelled frames from videos and writes them to a dataset directory that `train_swing_classifier.py` can consume.

`uv run python src/train_swing_detector/prepare_swing_frames.py`
`uv run python src/train_swing_detector/prepare_swing_frames.py --videos_dir videos_test --out_dir dataset_test`

## train_swing_classifier.py
Trains the stage-two shot-type classifier on shot-only frames. It ignores the `no_shot` folder and exports a `shot_type_classifier_<timestamp>.pkl` model.

## train_shot_binary_classifier.py
Trains a binary vision model that maps all shot labels to `shot` and keeps `no_shot` as-is. This is useful when the visual task is primarily shot detection rather than shot type classification.

By default it also writes validation review artifacts under `models/shot_binary_review_<timestamp>/`, including a `validation_predictions.csv` file and folders for `false_negative_shot`, `false_positive_shot`, `correct_shot`, and `correct_no_shot` so you can inspect mistakes directly.

## prepare_shot_sequences.py
Builds a sequence dataset for temporal binary shot detection. It writes samples under `shot/` and `no_shot/`, where each sample directory contains `frame_00.jpg ... frame_0N.jpg`, plus a `manifest.csv` file used by the temporal trainer.

Supports `--seq-len 3` or `--seq-len 5` with centered offsets based on `--dt`.

## train_shot_binary_temporal_classifier.py
Trains a temporal binary classifier from sequence samples in `manifest.csv`. It stacks sequence frames channel-wise (`3 * seq_len` channels), adapts a ResNet first layer to accept the stacked input, and exports model/history plus grouped validation-review artifacts.

Typical training flow:

```bash
uv run python src/train_swing_detector/train_shot_binary_classifier.py dataset --arch resnet34 --img-size 224 --seed 148
uv run python src/train_swing_detector/train_swing_classifier.py dataset --arch resnet34 --img-size 224 --seed 148

# Temporal binary flow (3-frame)
uv run python src/train_swing_detector/prepare_shot_sequences.py --videos_dir videos --out_dir dataset_sequences --seq-len 3 --dt 0.03
uv run python src/train_swing_detector/train_shot_binary_temporal_classifier.py dataset_sequences --seq-len 3 --arch resnet34 --img-size 320 --seed 148
```

## Evaluation

The evaluators report two views of model performance:

- Direct frame metrics on `dataset_test`, using already-extracted labeled frames.
- Pipeline metrics on `videos_test`, running audio detection, person crop, then the relevant shot detector stage.

By default, evaluators only print results. Add `--save-outputs` to write `summary.json` and detailed CSVs under `--out-dir`.

### Binary shot classifier

Evaluates the binary `shot` vs `no_shot` classifier. Pipeline false negatives are split by stage:

- `fn_audio`: the audio detector did not produce a nearby peak.
- `fn_binary`: audio found a nearby peak, but the binary classifier rejected it.
- `fn_crop`: audio found a nearby peak, but person crop failed.
- `fn_other`: a nearby audio peak existed, but matching/NMS did not produce a matched accepted prediction.

```bash
uv run python src/train_swing_detector/evaluate_shot_binary_model.py \
  models/shot_binary_classifier_20260328143535.pkl
```

Run on 2025/05/15
```
uv run python src/train_swing_detector/evaluate_shot_binary_model.py models/shot_binary_classifier_20260328143535.pkl
Model: models/shot_binary_classifier_20260328143535.pkl
Audio model: models/audio_pop_logmel_large_20260512231349.pth
Settings: shot_threshold=0.5 audio_threshold=0.5 tolerance=0.25 stride_s=0.05 min_separation_s=2.0
Test set: direct_images=2488 videos=12 labeled_shots=623

Direct frame confusion matrix [[no_shot, shot] actual x predicted]:
[[ 589   34]
 [  62 1803]]
Direct frame metrics: accuracy=0.9614 precision=0.9815 recall=0.9668 f1=0.9741

Pipeline metrics: tp=526 fp=23 fn=97 fn_audio=74 fn_binary=19 fn_crop=0 fn_other=4 precision=0.9581 recall=0.8443 f1=0.8976
Timing error: mean=0.025793 median=0.019

Pipeline error type breakdown:
fn_audio: backhand=11, backhand_slice=1, forehand=32, overhead=1, serve=3, volley=26
fn_binary: backhand=2, forehand=15, volley=2
fn_crop: none
fn_other: forehand=4
```


Common options:

```bash
uv run python src/train_swing_detector/evaluate_shot_binary_model.py \
  models/shot_binary_classifier_20260328143535.pkl \
  --audio-model models/audio_pop_logmel_large_20260512231349.pth \
  --dataset-dir dataset_test \
  --videos-dir videos_test \
  --wav-dir wavs_test \
  --shot-threshold 0.5 \
  --audio-threshold 0.5
```

### Shot-type classifier

Evaluates the multi-class shot-type classifier after the binary shot filter. Direct metrics exclude `no_shot` and any labels not present in the model vocabulary, such as rare unsupported classes. Pipeline metrics report detection quality plus type accuracy on matched supported shots.

```bash
uv run python src/train_swing_detector/evaluate_shot_type_model.py \
  models/shot_type_classifier_20260328220857.pkl \
  --binary-model models/shot_binary_classifier_20260328143535.pkl \
  --audio-model models/audio_pop_logmel_large_20260512231349.pth
```

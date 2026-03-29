# Train Swing Detector

Scripts for preparing the dataset and training the two-stage vision pipeline. The first model predicts **shot** vs **no_shot**. The second model runs only on shot frames and predicts **forehand**, **backhand**, **volley**, or **serve**.

## prepare_swing_frames.py
Extracts labelled frames from videos and writes them to a dataset directory that `train_swing_classifier.py` can consume.

`uv run python src/train_swing_detector/prepare_swing_frames.py`

## train_swing_classifier.py
Trains the stage-two shot-type classifier on shot-only frames. It ignores the `no_shot` folder and exports a `shot_type_classifier_<timestamp>.pkl` model.

## train_shot_binary_classifier.py
Trains a binary vision model that maps all shot labels to `shot` and keeps `no_shot` as-is. This is useful when the visual task is primarily shot detection rather than shot type classification.

By default it also writes validation review artifacts under `models/shot_binary_review_<timestamp>/`, including a `validation_predictions.csv` file and folders for `false_negative_shot`, `false_positive_shot`, `correct_shot`, and `correct_no_shot` so you can inspect mistakes directly.

Typical training flow:

```bash
uv run python src/train_swing_detector/train_shot_binary_classifier.py dataset --arch resnet34 --img-size 224 --seed 148
uv run python src/train_swing_detector/train_swing_classifier.py dataset --arch resnet34 --img-size 224 --seed 148
```

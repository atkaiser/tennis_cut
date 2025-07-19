# Train Swing Detector

Scripts for preparing the dataset and training the swing classification model. The detector takes a single image and predicts whether it shows a **forehand**, **backhand**, **volley**, **serve**, or **no_shot**.

## prepare_swing_frames.py
Extracts labelled frames from videos and writes them to a dataset directory that `train_swing_classifier.py` can consume.

## train_swing_classifier.py
Trains a vision model on the prepared dataset and exports the classifier for use by other tools.

# Tennis Cut

This repository contains tooling for extracting tennis swing clips from raw match footage. The system uses an audio "pop" detector to find candidate impact moments and then verifies them with a lightweight vision model. Full design details are available in [spec.md](spec.md).

## Directory overview

- `src/label_videos/` – utilities for manually labeling videos.
- `src/train_pop_detector/` – scripts for preparing data and training the audio classifier.
- `src/train_swing_detector/` – scripts for preparing data and training the vision swing classifier.

See the README files inside those folders for usage examples.

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting. To check the
code locally run:

```bash
pip install -e .
ruff .
```

Ruff runs automatically on pull requests to the `main` branch via GitHub
Actions.

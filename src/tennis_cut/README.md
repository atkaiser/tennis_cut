# Tennis Cut CLI

Command-line tool for extracting tennis swing clips from a video. It relies on a
trained audio "pop" model to locate impact moments and can optionally use a
two-stage vision pipeline: a binary shot detector to filter false positives,
followed by a shot-type classifier to label detected shots. Models are produced
by `train_audio_pop.py`, `train_shot_binary_classifier.py`, and
`train_swing_classifier.py` and exported via fastai's `Learner.export`.

Run:

```bash
python tennis_cut.py <input.mp4> --audio_model path/to/audio_pop.pth [options]
```

Use `--shot-model` to supply the optional binary vision model. Add
`--shot-type-model` to label retained shots by type. `--shot-type-model`
requires `--shot-model`. Use `--device` to control the PyTorch device
(default: `mps`).

The available options match those described in `spec.md` under the *Command-line Interface* section.

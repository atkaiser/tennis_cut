# Tennis Cut CLI

Command-line tool for extracting tennis swing clips from a video. It relies on a
trained audio "pop" model to locate impact moments. The model is produced by
`train_audio_pop.py` and exported via fastai's `Learner.export`.

Run:

```bash
python tennis_cut.py <input.mp4> --model path/to/audio_pop.pth [options]
```

The available options match those described in `spec.md` under the *Command-line Interface* section.

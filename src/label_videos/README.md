# Label Videos

Utilities for creating impact annotations used to train the audio pop detector.

## convert_to_fast_vid.sh
Converts raw camera footage to a high‑frame‑rate intra-frame format so that the GUI can scrub quickly. Run the script inside the folder containing your `.MOV` files:

```bash
sh convert_to_fast_vid.sh
```

Each file is replaced with a 120 fps version encoded with `ffmpeg`.

## tennis_annotate.py
PySide6 application for manually marking ball contact frames. Launch it with the directory that holds your videos:

```bash
python tennis_annotate.py /path/to/videos
```

For every video it creates a `<video>.json` file listing the impact timestamps. Re-run the command to continue where you left off.

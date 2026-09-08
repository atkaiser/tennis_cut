# Label Videos

Utilities for creating impact annotations used to train the audio pop detector.

## Create a new labeled video
1. Add it to `tmp_videos`
2. Copy vids over to `videos` (`cp tmp_videos/* videos/.`) (Run from top level dir)
3. Run `cd tmp_videos && ../src/label_videos/convert_to_fast_vid.sh` from that directory (Note, this can take a few min per a video)
4. Run `uv sync && uv run python src/label_videos/tennis_annotate.py tmp_videos` from the top level directory
5. When done annotating all videos add the JSON to `videos` (`cp tmp_videos/*.json videos/.`) (Run from top level dir)
6. Lower quality for long term keeping (`./src/label_videos/downsize_vids.sh`) (Run from top level dir)
7. Delete what is in `tmp_videos`
(Possibly regenerate the models after doing this (see the other two train dirs for how to do this))

(Note this will destroy the originals)

## convert_to_fast_vid.sh
Converts raw camera footage to a high‑frame‑rate intra-frame format so that the GUI can scrub quickly. Run the script inside the folder containing your `.MOV` files:

```bash
sh convert_to_fast_vid.sh
```

Each file is replaced with a 120 fps version encoded with `ffmpeg`.

## tennis_annotate.py
PySide6 application for manually marking ball contact frames. Launch it with the directory that holds your videos:

```bash
uv run python tennis_annotate.py /path/to/videos
```

For every video it creates a `<video>.json` file listing the impact timestamps. Re-run the command to continue where you left off.

## Contact-frame review prototype

`prototype_contact_review.py` is a disposable, keyboard-first tool for refining
existing forehand timestamps into exact source-frame labels. From the repository
root, run:

```bash
uv run python src/label_videos/prototype_contact_review.py videos
```

The first run creates an ignored 200-shot manifest and frame cache under
`out/contact-review-prototype/`. Review stays within one video while the next
video is prepared. The tool resumes from labels already stored in the existing
video JSON files.

- Left/Right: move one frame
- Shift+Left/Right: move ten frames
- Space: play or pause the cached frame sequence
- `C`: Single-state contact at the selected visual state
- `A`: Between-state contact after the selected (earlier) visual state
- `X`: Unresolved contact
- Backspace: undo the last label from this run

Use `--rebuild-manifest` to replace the current selection, or
`--manifest-only` to create and summarize it without launching the UI.

If a completed review encountered source frames containing repeated images,
run the duplicate-aware correction pass. It first backs up the original JSON,
migrates unaffected reviews, and collapses repeated images while reviewing only
affected videos:

```bash
uv run python src/label_videos/prototype_contact_review.py videos --correct-duplicates
```

Then classify the 13 fixed camera views for the initial side-view phase:

```bash
uv run python src/label_videos/prototype_contact_review.py videos --review-suitability
```

Press `I` for an in-scope side view or `O` for a view whose angle or player
scale should be excluded from primary training and evaluated as an abstention
case instead.

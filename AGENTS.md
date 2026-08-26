# Repository Guidelines

## Project Structure & Module Organization
See `Directory overview` in main README.md

## Build, Test, and Development Commands
- Setup (Python >= 3.11):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e .`
- Lint: `ruff check .` (required before PRs; use `ruff check --fix .` to autofix).
- Sanity checks (after downloading example media):
  - Download: `wget 'https://www.dropbox.com/scl/fi/dce0wabuy0kss3xtcp7th/tester.MOV?rlkey=y8cwf7wssvswq1dxj12rxrrhw&st=i4yi7aeh&dl=0' -O examples/videos/tester.MOV`
    and `wget 'https://www.dropbox.com/scl/fi/5pxrm1y9ij8qvls07ve03/tester.wav?rlkey=rxlfhsdxigrqf3zwye6jk8b1q&st=icvmew0l&dl=0' -O examples/wavs/tester.wav`
  - `python src/train_pop_detector/prepare_audio_windows.py --videos_dir examples/videos --wav_dir /tmp/wavs --out_csv examples/meta/labled_windows.csv`
  - `python src/train_pop_detector/train_audio_pop.py examples/meta/labled_windows.csv --epochs 1 --device cpu --bs 8`
  - `python src/tennis_cut/tennis_cut.py examples/videos --model models/<audio_model>.pth --no-stitch`

## Coding Style & Naming Conventions
- Python; 4‑space indentation; use type hints where practical.
- Keep modules script‑friendly (CLI under `if __name__ == "__main__":`).
- Prefer descriptive file names (`prepare_audio_windows.py`, `train_swing_classifier.py`).

## Testing Guidelines
- Acceptance suite: follow `Acceptance tests` in `README.md`; it is required before PRs.
- Ensure `ffmpeg`/`ffprobe` are installed; GPU/MPS optional.
- Do not commit large binaries; use `examples/` download links for local tests.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject, optional scope (e.g., `train_pop: balance negatives`).
- PRs must describe problem, solution, and test steps (commands + sample output), and pass `ruff .`.
- Exclude large media/models from Git; keep artifacts under `models/` locally.

## Security & Configuration Tips
- Keep generated videos, wavs, and model checkpoints local. Commit the compact
  portable visual contact ranker under `src/tennis_cut/models/` as the sole
  bundled runtime model.
- Use `pathlib.Path` for cross‑platform paths; document new tooling in `README.md`.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `atkaiser/tennis_cut`. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the five default canonical labels. See `docs/agents/triage-labels.md`.

### Domain docs

This repository uses a single-context domain-doc layout. See `docs/agents/domain.md`.

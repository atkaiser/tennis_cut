# Codebase Concerns

**Analysis Date:** 2026-02-16

## Tech Debt

**Import path hacks instead of packaging:**
- Issue: Scripts mutate `sys.path` at runtime to import shared utilities, which breaks when run from other working directories and prevents normal packaging.
- Files: `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/prepare_swing_frames.py`
- Impact: Fragile CLI execution, harder reuse in tooling, and inconsistent module resolution.
- Fix approach: Convert `src/` to a proper package and import via package paths (e.g., `from tennis_cut.utilities import ...`).

**Duplicated ffmpeg/ffprobe logic:**
- Issue: Multiple scripts implement their own subprocess wrappers for ffmpeg/ffprobe.
- Files: `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/prepare_swing_frames.py`, `src/utilities/core.py`
- Impact: Inconsistent error handling and flags, duplicated maintenance when CLI behavior changes.
- Fix approach: Centralize ffmpeg/ffprobe helpers in a shared module and reuse consistently.

**Deprecated GUI utility still in tree:**
- Issue: Deprecated labeling tool remains in active source without guardrails or replacement note in code.
- Files: `src/label_videos/label_shots.py`
- Impact: Confusion about which labeling flow to use; risk of stale data formats.
- Fix approach: Move to `deprecated/` or remove, and document current annotation flow in `README.md`.

## Known Bugs

**Crash when videos lack audio stream:**
- Symptoms: `StopIteration` when probing streams; pipeline halts before detection.
- Files: `src/tennis_cut/tennis_cut.py`
- Trigger: Input video with no audio track.
- Workaround: Re-encode with an audio track before running.

**Output collision handling differs from spec:**
- Symptoms: Existing output files silently skip processing instead of failing fast.
- Files: `src/tennis_cut/tennis_cut.py`, `spec.md`
- Trigger: Output files already present in `--output-dir`.
- Workaround: Manually delete output files before rerunning.

## Security Considerations

**Model loading via pickle is inherently unsafe:**
- Risk: `fastai.load_learner` executes pickled objects and can run arbitrary code.
- Files: `src/tennis_cut/tennis_cut.py`
- Current mitigation: Warning suppression only; no trust boundary enforcement.
- Recommendations: Treat model files as trusted artifacts only and document trust requirement in `README.md`.

## Performance Bottlenecks

**Full-audio inference on long videos:**
- Problem: Loads entire waveform and evaluates all windows in memory.
- Files: `src/tennis_cut/tennis_cut.py`
- Cause: `torchaudio.load` loads whole file and fastai inference iterates every stride window.
- Improvement path: Stream audio windows or chunk inference to limit memory and latency.

**High ffmpeg call volume during dataset creation:**
- Problem: Two ffmpeg calls per extracted frame (extract + crop) scales poorly.
- Files: `src/train_swing_detector/prepare_swing_frames.py`
- Cause: Separate extraction and crop steps for every frame.
- Improvement path: Use a single ffmpeg call with crop filters or batch frame extraction.

## Fragile Areas

**Hard-coded YOLO model filename depends on CWD:**
- Files: `src/utilities/core.py`, `yolov8n.pt`
- Why fragile: Running from any directory other than repo root can break model loading.
- Safe modification: Resolve model path relative to module file or allow CLI override.
- Test coverage: Not covered by automated tests.

**Default device assumes MPS availability:**
- Files: `src/train_swing_detector/prepare_swing_frames.py`, `src/train_swing_detector/train_swing_classifier.py`, `src/train_pop_detector/train_audio_pop.py`
- Why fragile: Non-macOS or non-MPS machines will error with default `mps` device.
- Safe modification: Detect device availability and fall back to CPU when MPS/CUDA is absent.
- Test coverage: Not covered by automated tests.

## Scaling Limits

**Single-process, serial pipelines:**
- Current capacity: One video or one dataset loop per process, no parallelism.
- Limit: Large training sets or video batches take hours due to sequential ffmpeg/YOLO calls.
- Scaling path: Add multiprocessing for extraction and inference or allow per-file parallelism.
- Files: `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/prepare_swing_frames.py`

## Dependencies at Risk

**External binary requirement not enforced consistently:**
- Risk: `ffmpeg`/`ffprobe` not installed leads to hard failures in dataset scripts.
- Impact: Dataset generation and metadata probing fail without clear guidance.
- Migration plan: Add preflight checks similar to `check_ffmpeg` and document in `README.md`.
- Files: `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/prepare_swing_frames.py`, `src/utilities/core.py`

## Missing Critical Features

**Spec’d processing rules not implemented in CLI:**
- Problem: Overlap merging, tracker option, and collision rule are specified but not present.
- Blocks: Reproducibility against `spec.md` and expected pipeline behavior.
- Files: `spec.md`, `src/tennis_cut/tennis_cut.py`

**Annotation UI limitations called out but not addressed:**
- Problem: No minimum spacing enforcement or undo/redo in the current annotator.
- Blocks: Faster, reliable labeling workflow for dense impacts.
- Files: `README.md`, `src/label_videos/tennis_annotate.py`

## Test Coverage Gaps

**No automated test suite:**
- What's not tested: Core detection, training, and annotation flows.
- Files: `README.md`, `AGENTS.md`, `.github/workflows/ruff.yml`
- Risk: Regressions in CLI behavior and data pipeline logic go unnoticed.
- Priority: High

---

*Concerns audit: 2026-02-16*

# Testing Patterns

**Analysis Date:** 2026-02-16

## Test Framework

**Runner:**
- Not detected; `pyproject.toml` defines dependencies but no test runner configuration.
- Config: Not detected; `pyproject.toml` has no test tool config.

**Assertion Library:**
- Not applicable; no automated test framework referenced in `pyproject.toml`.

**Run Commands:**
```bash
ruff check .
python src/train_pop_detector/prepare_audio_windows.py --videos_dir examples/videos --wav_dir /tmp/wavs --out_csv examples/meta/labled_windows.csv
python src/train_pop_detector/train_audio_pop.py examples/meta/labled_windows.csv --epochs 1 --device cpu --bs 8
python src/tennis_cut/tennis_cut.py examples/videos --model models/<audio_model>.pth --no-stitch
```
Manual sanity commands are documented in `AGENTS.md`.

## Test File Organization

**Location:**
- Not detected; no test directories or files referenced in `AGENTS.md`.

**Naming:**
- Not applicable; no test files referenced in `AGENTS.md`.

**Structure:**
```
Not detected (no test directory structure referenced in `AGENTS.md`).
```

## Test Structure

**Suite Organization:**
```python
# Not applicable; `AGENTS.md` documents manual validation only.
```

**Patterns:**
- Manual CLI validation via training and inference scripts (`AGENTS.md`).

## Mocking

**Framework:** Not detected; no mocking tools referenced in `pyproject.toml`.

**Patterns:**
```python
# Not applicable; `AGENTS.md` documents manual validation only.
```

**What to Mock:**
- Not applicable; no automated tests referenced in `AGENTS.md`.

**What NOT to Mock:**
- Not applicable; no automated tests referenced in `AGENTS.md`.

## Fixtures and Factories

**Test Data:**
```python
# Not applicable; `AGENTS.md` documents manual validation only.
```

**Location:**
- Manual example data is referenced in `AGENTS.md` and `src/train_pop_detector/prepare_audio_windows.py`.

## Coverage

**Requirements:** None enforced; no coverage tooling referenced in `pyproject.toml`.

**View Coverage:**
```bash
# Not applicable; `pyproject.toml` has no coverage tooling.
```

## Test Types

**Unit Tests:**
- Not detected; `AGENTS.md` notes no formal unit test suite.

**Integration Tests:**
- Not detected; `AGENTS.md` notes manual CLI validation.

**E2E Tests:**
- Not used; `AGENTS.md` only lists manual CLI runs.

## Common Patterns

**Async Testing:**
```python
# Not applicable; `AGENTS.md` documents manual validation only.
```

**Error Testing:**
```python
# Not applicable; `AGENTS.md` documents manual validation only.
```

---

*Testing analysis: 2026-02-16*

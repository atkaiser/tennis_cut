# Coding Conventions

**Analysis Date:** 2026-02-16

## Naming Patterns

**Files:**
- Use `snake_case.py` for scripts and modules (examples: `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/train_swing_classifier.py`, `src/tennis_cut/tennis_cut.py`).

**Functions:**
- Use `snake_case` for functions and methods (examples: `src/tennis_cut/tennis_cut.py`, `src/train_swing_detector/prepare_swing_frames.py`).

**Variables:**
- Use `snake_case` for locals and params; use `UPPER_SNAKE_CASE` for constants (examples: `src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/prepare_audio_windows.py`).

**Types:**
- Use `CamelCase` for classes and type aliases (examples: `src/train_pop_detector/train_audio_pop.py`, `src/utilities/core.py`).

## Code Style

**Formatting:**
- Use 4-space indentation and script-friendly structure (documented in `AGENTS.md`).
- No formatter configuration detected (no `pyproject.toml` tool config beyond dependencies; no `.ruff.toml`).

**Linting:**
- Use `ruff check .` as the required lint step (`.github/workflows/ruff.yml`, `AGENTS.md`).
- Ruff is declared as a dependency in `pyproject.toml`.

## Import Organization

**Order:**
1. Standard library imports
2. Third-party imports
3. Local imports
   - Local imports sometimes follow a `sys.path.append(...)` shim to allow script execution from repo root (`src/tennis_cut/tennis_cut.py`, `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/prepare_swing_frames.py`).

**Path Aliases:**
- No import aliasing or package-level path alias config detected in `pyproject.toml` or `src/tennis_cut/tennis_cut.py`.

## Error Handling

**Patterns:**
- Use `subprocess.run(..., check=True)` with `try/except` and `sys.exit(...)` in CLI scripts (examples: `src/train_pop_detector/prepare_audio_windows.py`, `src/train_swing_detector/prepare_swing_frames.py`).
- Wrap shell commands in helper functions that log errors and raise `CalledProcessError` (`src/tennis_cut/tennis_cut.py`).

## Logging

**Framework:** logging module (`src/tennis_cut/tennis_cut.py`).

**Patterns:**
- Module logger named with `__name__` and configured by a CLI flag (`src/tennis_cut/tennis_cut.py`).
- Other scripts favor `print(...)` for user-facing progress (`src/train_pop_detector/train_audio_pop.py`, `src/train_swing_detector/train_swing_classifier.py`).

## Comments

**When to Comment:**
- Use top-of-file module docstrings and section dividers for scripts (`src/train_pop_detector/train_audio_pop.py`, `src/train_swing_detector/prepare_swing_frames.py`, `src/tennis_cut/tennis_cut.py`).

**JSDoc/TSDoc:**
- Not applicable (Python codebase; see `src/tennis_cut/tennis_cut.py`).

## Function Design

**Size:**
- Use script-style top-level functions and a `main(...)` entrypoint (examples: `src/tennis_cut/tennis_cut.py`, `src/train_swing_detector/train_swing_classifier.py`).

**Parameters:**
- Use typed parameters with `pathlib.Path` where appropriate (examples: `src/utilities/core.py`, `src/tennis_cut/tennis_cut.py`).

**Return Values:**
- Use explicit return codes for CLI workflows (example: `src/tennis_cut/tennis_cut.py` returns `int`).

## Module Design

**Exports:**
- Use explicit `__all__` for utility modules (`src/utilities/__init__.py`).

**Barrel Files:**
- Use `src/utilities/__init__.py` as the single barrel-style export location.

---

*Convention analysis: 2026-02-16*

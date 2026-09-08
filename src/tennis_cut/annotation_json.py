"""Lossless, atomic persistence for video annotation documents."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def load_annotation(path: Path) -> dict[str, Any]:
    """Load an annotation document without discarding unrecognized fields."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"annotation document must be a JSON object: {path}")
    return payload


def save_annotation(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace an annotation document with ``payload``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = ["load_annotation", "save_annotation"]

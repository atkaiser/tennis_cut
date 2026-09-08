#!/usr/bin/env python3
"""PROTOTYPE: review exact forehand contact frames one source video at a time.

This is intentionally disposable UI code. It answers whether the proposed
three-state review workflow is fast and unambiguous enough to label the first
contact-frame training set.

Run from the repository root:

    python src/label_videos/prototype_contact_review.py videos
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tennis_cut.annotation_json import load_annotation, save_annotation
from tennis_cut.comparison.media import inspect_media

SCHEMA_VERSION = 2
FRAME_RADIUS = 24
DEFAULT_LIMIT = 200
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
VISUAL_EQUIVALENCE_THRESHOLD = 0.05
DUPLICATE_CADENCE_FRACTION = 0.10


def _video_for_annotation(annotation_path: Path, payload: dict[str, Any]) -> Path | None:
    named = payload.get("video")
    candidates = []
    if isinstance(named, str):
        candidates.append(annotation_path.with_name(named))
        candidates.append(annotation_path.with_name(Path(named).stem + ".mp4"))
    candidates.extend(annotation_path.with_suffix(suffix) for suffix in VIDEO_EXTENSIONS)
    by_lower_name = {
        path.name.lower(): path
        for path in annotation_path.parent.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        match = by_lower_name.get(candidate.name.lower())
        if match is not None:
            return match.resolve()
    return None


def _recording_families(stems: list[str]) -> dict[str, str]:
    """Group nearby camera-roll clips, including sequences crossing a decade."""
    numbered: list[tuple[int, str]] = []
    families: dict[str, str] = {}
    for stem in stems:
        match = re.fullmatch(r"IMG_(\d+)(?:_45s)?", stem, flags=re.IGNORECASE)
        if match is None:
            families[stem] = stem.split("_", 1)[0] if stem[:4].isdigit() else stem
        else:
            numbered.append((int(match.group(1)), stem))

    clusters: list[list[tuple[int, str]]] = []
    for item in sorted(numbered):
        if not clusters or item[0] - clusters[-1][-1][0] > 3:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    for cluster in clusters:
        first, last = cluster[0][0], cluster[-1][0]
        family = f"IMG_{first:04d}-{last:04d}"
        for _, stem in cluster:
            families[stem] = family
    return families


def _assign_splits(videos: list[dict[str, Any]]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for video in videos:
        family = video["family"]
        counts[family] = counts.get(family, 0) + len(video["shots"])
    ordered = sorted(counts, key=lambda value: (-counts[value], value))
    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    assigned_counts = {name: 0 for name in targets}
    result: dict[str, str] = {}
    total = sum(counts.values()) or 1
    for family in ordered:
        split = min(
            targets,
            key=lambda name: assigned_counts[name] / (targets[name] * total),
        )
        result[family] = split
        assigned_counts[split] += counts[family]
    return result


def build_manifest(source_directory: Path, limit: int) -> dict[str, Any]:
    """Select about ``limit`` forehands while keeping review ordered by video."""
    candidates: list[dict[str, Any]] = []
    annotation_paths = sorted(source_directory.glob("*.json"))
    families = _recording_families([path.stem for path in annotation_paths])
    for annotation_path in annotation_paths:
        payload = load_annotation(annotation_path)
        video_path = _video_for_annotation(annotation_path, payload)
        if video_path is None:
            continue
        shots = []
        for shot_index, shot in enumerate(payload.get("shots", [])):
            if not isinstance(shot, dict) or shot.get("type") != "forehand":
                continue
            timestamp = shot.get("time")
            if not isinstance(timestamp, (int, float)):
                continue
            shots.append(
                {
                    "shot_index": shot_index,
                    "time": float(timestamp),
                    "type": "forehand",
                }
            )
        if shots:
            candidates.append(
                {
                    "video": str(video_path),
                    "annotation": str(annotation_path.resolve()),
                    "family": families[annotation_path.stem],
                    "shots": shots,
                }
            )

    rng = random.Random(20260903)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_family.setdefault(candidate["family"], []).append(candidate)
    for family_videos in by_family.values():
        rng.shuffle(family_videos)
    family_names = sorted(by_family)
    rng.shuffle(family_names)

    selected: list[dict[str, Any]] = []
    selected_count = 0
    while family_names and selected_count < limit:
        remaining = []
        for family in family_names:
            family_videos = by_family[family]
            if not family_videos:
                continue
            video = family_videos.pop()
            available = max(1, limit - selected_count)
            video = {**video, "shots": video["shots"][:available]}
            selected.append(video)
            selected_count += len(video["shots"])
            if family_videos:
                remaining.append(family)
            if selected_count >= limit:
                remaining.extend(name for name in family_names if name not in remaining)
                break
        family_names = list(dict.fromkeys(remaining))

    split_by_family = _assign_splits(selected)
    for video in selected:
        video["split"] = split_by_family[video["family"]]
        stat = Path(video["video"]).stat()
        video["source_fingerprint"] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    selected.sort(key=lambda item: (item["family"], item["video"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "source_directory": str(source_directory.resolve()),
        "requested_shots": limit,
        "selected_shots": sum(len(video["shots"]) for video in selected),
        "videos": selected,
    }


def finalize_reviewed_splits(manifest: dict[str, Any]) -> dict[str, str]:
    """Assign whole in-scope families after review; isolate out-of-scope views."""
    counts_by_family: dict[str, Counter[str]] = {}
    out_of_scope: set[str] = set()
    for video in manifest["videos"]:
        annotation = load_annotation(Path(video["annotation"]))
        metadata = annotation["contact_review_metadata"]
        family = video["family"]
        if metadata["side_view_suitability"] == "out_of_scope_view":
            out_of_scope.add(family)
            continue
        counts = counts_by_family.setdefault(family, Counter())
        counts.update(
            annotation["shots"][shot["shot_index"]]["contact_review"]["status"]
            for shot in video["shots"]
        )

    result = _optimize_reviewed_family_splits(counts_by_family)
    result.update({family: "out_of_scope" for family in out_of_scope})
    for video in manifest["videos"]:
        video["split"] = result[video["family"]]
    return result


def _optimize_reviewed_family_splits(
    counts_by_family: dict[str, Counter[str]],
) -> dict[str, str]:
    """Assign families with the original lexicographic balance objective."""

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    families = sorted(counts_by_family)
    splits = ("train", "validation", "test")
    statuses = ("single_state", "between_state", "unresolved")
    fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    status_totals = {
        status: sum(counts[status] for counts in counts_by_family.values())
        for status in statuses
    }
    total = sum(status_totals.values())
    choice_count = len(families) * len(splits)
    size_error_start = choice_count
    resolved_error_start = size_error_start + len(splits)
    unresolved_error_start = resolved_error_start + len(splits) * 2
    variable_count = unresolved_error_start + len(splits)

    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def choice(family_index: int, split_index: int) -> int:
        return family_index * len(splits) + split_index

    def add_constraint(
        coefficients: dict[int, float], minimum: float, maximum: float
    ) -> None:
        row = [0.0] * variable_count
        for index, value in coefficients.items():
            row[index] = value
        rows.append(row)
        lower.append(minimum)
        upper.append(maximum)

    for family_index in range(len(families)):
        add_constraint(
            {choice(family_index, split_index): 1.0 for split_index in range(3)},
            1.0,
            1.0,
        )

    for split_index, split in enumerate(splits):
        expression = {
            choice(family_index, split_index): float(
                sum(counts_by_family[family].values())
            )
            for family_index, family in enumerate(families)
        }
        target = fractions[split] * total
        error_index = size_error_start + split_index
        add_constraint({**expression, error_index: -1.0}, -np.inf, target)
        add_constraint(
            {index: -value for index, value in expression.items()}
            | {error_index: -1.0},
            -np.inf,
            -target,
        )

    for status_index, status in enumerate(statuses[:2]):
        for split_index, split in enumerate(splits):
            expression = {
                choice(family_index, split_index): float(
                    counts_by_family[family][status]
                )
                for family_index, family in enumerate(families)
            }
            target = fractions[split] * status_totals[status]
            error_index = resolved_error_start + status_index * 3 + split_index
            add_constraint({**expression, error_index: -1.0}, -np.inf, target)
            add_constraint(
                {index: -value for index, value in expression.items()}
                | {error_index: -1.0},
                -np.inf,
                -target,
            )

    for split_index, split in enumerate(splits):
        expression = {
            choice(family_index, split_index): float(
                counts_by_family[family]["unresolved"]
            )
            for family_index, family in enumerate(families)
        }
        target = fractions[split] * status_totals["unresolved"]
        error_index = unresolved_error_start + split_index
        add_constraint({**expression, error_index: -1.0}, -np.inf, target)
        add_constraint(
            {index: -value for index, value in expression.items()}
            | {error_index: -1.0},
            -np.inf,
            -target,
        )

    for split_index in (1, 2):
        for status in statuses:
            add_constraint(
                {
                    choice(family_index, split_index): float(
                        counts_by_family[family][status]
                    )
                    for family_index, family in enumerate(families)
                },
                1.0,
                np.inf,
            )

    bounds = Bounds(
        np.zeros(variable_count),
        np.concatenate(
            (np.ones(choice_count), np.full(variable_count - choice_count, np.inf))
        ),
    )
    integrality = np.concatenate(
        (np.ones(choice_count), np.zeros(variable_count - choice_count))
    )

    def solve(objective_coefficients: dict[int, float]) -> Any:
        objective = np.zeros(variable_count)
        for index, coefficient in objective_coefficients.items():
            objective[index] = coefficient
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(
                np.asarray(rows), np.asarray(lower), np.asarray(upper)
            ),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise ValueError("could not make grouped splits with evaluation coverage")
        optimum = float(objective @ result.x)
        add_constraint(
            objective_coefficients,
            -np.inf,
            optimum + 1e-7,
        )
        return result

    status_objective = {}
    for status_index, status in enumerate(statuses[:2]):
        for split_index in range(len(splits)):
            status_objective[
                resolved_error_start + status_index * len(splits) + split_index
            ] = 1 / max(1, status_totals[status])
    for split_index in range(len(splits)):
        status_objective[unresolved_error_start + split_index] = 1 / max(
            1, status_totals["unresolved"]
        )
    solve(status_objective)
    result = solve(
        {index: 1.0 for index in range(size_error_start, resolved_error_start)}
    )
    return {
        family: splits[
            max(
                range(len(splits)),
                key=lambda split_index: result.x[choice(family_index, split_index)],
            )
        ]
        for family_index, family in enumerate(families)
    }


def _frame_payload(frame: Any) -> dict[str, Any]:
    return {
        "stream_index": frame.stream_index,
        "ordinal": frame.ordinal,
        "pts": frame.pts,
        "time_base": {
            "numerator": frame.time_base.numerator,
            "denominator": frame.time_base.denominator,
        },
    }


def _cache_directory(video_entry: dict[str, Any], cache_root: Path) -> Path:
    video_path = Path(video_entry["video"])
    fingerprint = video_entry["source_fingerprint"]
    cache_key = f"{video_path.stem}-{fingerprint['size']}-{fingerprint['mtime_ns']}"
    return cache_root / cache_key


def _image_difference(first: Path, second: Path) -> float:
    import cv2
    import numpy as np

    first_image = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    second_image = cv2.imread(str(second), cv2.IMREAD_GRAYSCALE)
    if first_image is None or second_image is None:
        raise ValueError("could not read a cached review frame")
    first_small = cv2.resize(first_image, (320, 180))
    second_small = cv2.resize(second_image, (320, 180))
    return float(np.mean(cv2.absdiff(first_small, second_small)))


def _visual_states(
    ordinals: list[int], frames_by_ordinal: dict[int, dict[str, Any]]
) -> list[list[int]]:
    states: list[list[int]] = []
    for ordinal in ordinals:
        if states:
            previous = states[-1][-1]
            difference = _image_difference(
                Path(frames_by_ordinal[previous]["path"]),
                Path(frames_by_ordinal[ordinal]["path"]),
            )
            if difference < VISUAL_EQUIVALENCE_THRESHOLD:
                states[-1].append(ordinal)
                continue
        states.append([ordinal])
    return states


def duplicate_cadence_videos(
    manifest: dict[str, Any], cache_root: Path
) -> set[str]:
    """Identify videos with a recurring, strong repeated-image cadence."""
    duplicates: set[str] = set()
    for video in manifest["videos"]:
        prepared = prepare_video(video, cache_root)
        frames = {frame["ordinal"]: frame for frame in prepared["frames"]}
        examined = 0
        equivalent = 0
        for window in prepared["windows"]:
            ordinals = window["ordinals"]
            for first, second in zip(ordinals, ordinals[1:]):
                examined += 1
                if _image_difference(
                    Path(frames[first]["path"]), Path(frames[second]["path"])
                ) < VISUAL_EQUIVALENCE_THRESHOLD:
                    equivalent += 1
        if examined and equivalent / examined >= DUPLICATE_CADENCE_FRACTION:
            duplicates.add(video["video"])
    return duplicates


def _state_payload(
    ordinals: list[int], frames_by_ordinal: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "frames": [
            {key: value for key, value in frames_by_ordinal[ordinal].items() if key != "path"}
            for ordinal in ordinals
        ]
    }


def migrate_completed_reviews(
    manifest: dict[str, Any],
    cache_root: Path,
    backup_root: Path,
    duplicate_videos: set[str],
) -> tuple[int, int]:
    """Migrate unambiguous v1 reviews and retain a recoverable source backup."""
    migrated = 0
    deferred = 0
    backup_root.mkdir(parents=True, exist_ok=True)
    for video in manifest["videos"]:
        annotation_path = Path(video["annotation"])
        backup_path = backup_root / annotation_path.name
        if not backup_path.exists():
            shutil.copy2(annotation_path, backup_path)
        if video["video"] in duplicate_videos:
            payload = load_annotation(annotation_path)
            deferred += sum(
                payload["shots"][selected["shot_index"]]
                .get("contact_review", {})
                .get("status")
                in {"single_frame", "between_frame"}
                for selected in video["shots"]
            )
            continue
        prepared = prepare_video(video, cache_root)
        frames = {frame["ordinal"]: frame for frame in prepared["frames"]}
        payload = load_annotation(annotation_path)
        changed = False
        for selected, window in zip(video["shots"], prepared["windows"], strict=True):
            shot = payload["shots"][selected["shot_index"]]
            review = shot.get("contact_review")
            if not isinstance(review, dict):
                continue
            status = review.get("status")
            if status in {"single_state", "between_state", "unresolved"}:
                continue
            if status == "single_frame":
                ordinal = int(review["frame"]["ordinal"])
                shot["contact_review"] = {
                    "status": "single_state",
                    "frame": review["frame"],
                    "visual_states": [_state_payload([ordinal], frames)],
                }
            elif status == "between_frame":
                ordinal = int(review["frame"]["ordinal"])
                following = ordinal + 1
                if following not in frames or following not in window["ordinals"]:
                    raise ValueError(
                        f"missing following frame for {annotation_path.name} shot "
                        f"{selected['shot_index']}"
                    )
                shot["contact_review"] = {
                    "status": "between_state",
                    "frame": review["frame"],
                    "visual_states": [
                        _state_payload([ordinal], frames),
                        _state_payload([following], frames),
                    ],
                }
            else:
                continue
            changed = True
            migrated += 1
        if changed:
            metadata = payload.get("contact_review_metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            payload["contact_review_metadata"] = {
                **metadata,
                "schema_version": SCHEMA_VERSION,
            }
            save_annotation(annotation_path, payload)
    return migrated, deferred


def _merge_ranges(ordinals: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for ordinal in sorted(set(ordinals)):
        if not ranges or ordinal > ranges[-1][1] + 1:
            ranges.append((ordinal, ordinal))
        else:
            ranges[-1] = (ranges[-1][0], ordinal)
    return ranges


def _cache_matches_video_entry(
    cached: dict[str, Any], video_entry: dict[str, Any]
) -> bool:
    windows = cached.get("windows")
    frames = cached.get("frames")
    shots = video_entry["shots"]
    return (
        cached.get("video") == video_entry["video"]
        and isinstance(windows, list)
        and len(windows) == len(shots)
        and all(
            isinstance(window, dict)
            and all(window.get(key) == value for key, value in shot.items())
            for shot, window in zip(shots, windows, strict=True)
        )
        and isinstance(frames, list)
        and all(
            isinstance(frame, dict)
            and isinstance(frame.get("path"), str)
            and Path(frame["path"]).is_file()
            for frame in frames
        )
    )


def prepare_video(video_entry: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    """Inspect and decode every requested window in one FFmpeg traversal."""
    video_path = Path(video_entry["video"])
    cache_directory = _cache_directory(video_entry, cache_root)
    index_path = cache_directory / "frames.json"
    if index_path.is_file():
        cached = json.loads(index_path.read_text())
        if _cache_matches_video_entry(cached, video_entry):
            return cached

    media = inspect_media(video_path)
    requested_ordinals: list[int] = []
    windows = []
    for shot in video_entry["shots"]:
        timestamp = Fraction(str(shot["time"]))
        center = min(media.frames, key=lambda frame: abs(frame.timestamp - timestamp))
        lower = max(0, center.ordinal - FRAME_RADIUS)
        upper = min(len(media.frames) - 1, center.ordinal + FRAME_RADIUS)
        ordinals = list(range(lower, upper + 1))
        requested_ordinals.extend(ordinals)
        windows.append({**shot, "center_ordinal": center.ordinal, "ordinals": ordinals})

    cache_directory.mkdir(parents=True, exist_ok=True)
    for stale_frame in cache_directory.glob("frame_*.jpg"):
        stale_frame.unlink()
    unique_ordinals = sorted(set(requested_ordinals))
    selection = "+".join(
        f"between(n\\,{lower}\\,{upper})"
        for lower, upper in _merge_ranges(unique_ordinals)
    )
    pattern = cache_directory / "frame_%06d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-noautorotate",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            f"select={selection}",
            "-fps_mode",
            "passthrough",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(pattern),
            "-y",
        ],
        check=True,
    )
    paths = sorted(cache_directory.glob("frame_*.jpg"))
    if len(paths) != len(unique_ordinals):
        raise RuntimeError(
            f"decoded {len(paths)} frames for {len(unique_ordinals)} ordinals"
        )
    frames = [
        {**_frame_payload(media.frames[ordinal]), "path": str(path.resolve())}
        for ordinal, path in zip(unique_ordinals, paths, strict=True)
    ]
    result = {"video": str(video_path), "windows": windows, "frames": frames}
    index_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


class ClickableFrame(QLabel):
    clicked = Signal(int)

    def __init__(self, position: int) -> None:
        super().__init__()
        self.position = position
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def mousePressEvent(self, event: Any) -> None:
        self.clicked.emit(self.position)
        super().mousePressEvent(event)


class ReviewWindow(QWidget):
    def __init__(
        self,
        manifest: dict[str, Any],
        cache_root: Path,
        *,
        collapse_duplicates: bool = False,
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.cache_root = cache_root
        self.collapse_duplicates = collapse_duplicates
        self.video_index = 0
        self.shot_position = 0
        self.frame_position = 0
        self.prepared: dict[int, dict[str, Any]] = {}
        self.preparing: dict[int, Future[dict[str, Any]]] = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.last_change: tuple[Path, int, object] | None = None
        self.reviewed_keys = {
            (video_index, shot_position)
            for video_index, video in enumerate(manifest["videos"])
            for shot_position, shot in enumerate(video["shots"])
            if self._review_for(video, shot) is not None
        }
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(70)
        self.play_timer.timeout.connect(self._play_step)

        self.setWindowTitle("PROTOTYPE contact-frame review")
        self.main_image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.main_image.setMinimumSize(QSize(900, 500))
        self.status = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.filmstrip = QHBoxLayout()
        button_row = QHBoxLayout()
        for label, callback in (
            ("C  Single-state", lambda: self._record("single_state")),
            ("A  Between-state", lambda: self._record("between_state")),
            ("X  Unresolved", lambda: self._record("unresolved")),
            ("Backspace  Undo", self._undo),
        ):
            button = QPushButton(label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(callback)
            button_row.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.main_image, 1)
        layout.addLayout(self.filmstrip)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addLayout(button_row)
        self.showMaximized()
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self._poll_preparation)
        self.poll_timer.start()
        self._advance_past_reviewed()
        self._request_video(self.video_index)
        self._render()

    def closeEvent(self, event: Any) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def _request_video(self, index: int) -> None:
        if not 0 <= index < len(self.manifest["videos"]):
            return
        if index in self.prepared or index in self.preparing:
            return
        self.preparing[index] = self.executor.submit(
            prepare_video, self.manifest["videos"][index], self.cache_root
        )

    def _poll_preparation(self) -> None:
        for index, future in list(self.preparing.items()):
            if not future.done():
                continue
            del self.preparing[index]
            try:
                self.prepared[index] = future.result()
            except Exception as error:
                QMessageBox.critical(self, "Frame preparation failed", str(error))
                QApplication.instance().quit()
                return
            if self.collapse_duplicates:
                prepared = self.prepared[index]
                frames = {frame["ordinal"]: frame for frame in prepared["frames"]}
                for window in prepared["windows"]:
                    window["visual_states"] = _visual_states(window["ordinals"], frames)
            if index == self.video_index:
                self._center_on_label()
                self._render()
                self._request_video(index + 1)

    def _current_shot(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        prepared = self.prepared.get(self.video_index)
        if prepared is None:
            return None
        entry = self.manifest["videos"][self.video_index]
        return entry["shots"][self.shot_position], prepared["windows"][self.shot_position]

    def _frames_by_ordinal(self) -> dict[int, dict[str, Any]]:
        return {
            frame["ordinal"]: frame for frame in self.prepared[self.video_index]["frames"]
        }

    def _states(self, window: dict[str, Any]) -> list[list[int]]:
        if self.collapse_duplicates:
            return window["visual_states"]
        return [[ordinal] for ordinal in window["ordinals"]]

    def _center_on_label(self) -> None:
        current = self._current_shot()
        if current is None:
            return
        _, window = current
        self.frame_position = next(
            index
            for index, state in enumerate(self._states(window))
            if window["center_ordinal"] in state
        )

    def _review_for(self, video_entry: dict[str, Any], shot: dict[str, Any]) -> Any:
        payload = load_annotation(Path(video_entry["annotation"]))
        current = payload["shots"][shot["shot_index"]]
        if current.get("time") != shot["time"] or current.get("type") != shot["type"]:
            raise RuntimeError("shot changed since the review manifest was created")
        review = current.get("contact_review")
        if not isinstance(review, dict):
            return None
        status = review.get("status")
        if status == "unresolved":
            return review
        if status not in {"single_state", "between_state"}:
            return None
        frame = review.get("frame")
        required = {"stream_index", "ordinal", "pts", "time_base"}
        states = review.get("visual_states")
        expected_states = 1 if status == "single_state" else 2
        return (
            review
            if (
                isinstance(frame, dict)
                and required <= frame.keys()
                and isinstance(states, list)
                and len(states) == expected_states
            )
            else None
        )

    def _advance_past_reviewed(self) -> None:
        while self.video_index < len(self.manifest["videos"]):
            entry = self.manifest["videos"][self.video_index]
            while self.shot_position < len(entry["shots"]):
                if (self.video_index, self.shot_position) not in self.reviewed_keys:
                    return
                self.shot_position += 1
            self.video_index += 1
            self.shot_position = 0

    def _render(self) -> None:
        total = self.manifest["selected_shots"]
        complete = len(self.reviewed_keys)
        self.progress.setRange(0, total)
        self.progress.setValue(complete)
        self.progress.setFormat(f"{complete} / {total} reviewed")
        if self.video_index >= len(self.manifest["videos"]):
            self.main_image.setText("Review complete")
            self.status.setText("All selected forehands have contact-review labels.")
            return
        entry = self.manifest["videos"][self.video_index]
        if self.video_index not in self.prepared:
            self.main_image.setText(f"Preparing {Path(entry['video']).name}…")
            self.status.setText("Scanning frame identities and building the resumable cache.")
            return
        current = self._current_shot()
        if current is None:
            return
        shot, window = current
        states = self._states(window)
        state = states[self.frame_position]
        ordinal = state[0]
        frame = self._frames_by_ordinal()[ordinal]
        pixmap = QPixmap(frame["path"])
        self.main_image.setPixmap(
            pixmap.scaled(
                self.main_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        while self.filmstrip.count():
            item = self.filmstrip.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        lower = max(0, self.frame_position - 4)
        upper = min(len(states), self.frame_position + 5)
        for position in range(lower, upper):
            item_state = states[position]
            item_ordinal = item_state[0]
            thumbnail = ClickableFrame(position)
            thumbnail.setFixedSize(150, 95)
            thumbnail.setPixmap(
                QPixmap(self._frames_by_ordinal()[item_ordinal]["path"]).scaled(
                    thumbnail.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )
            thumbnail.setStyleSheet(
                "border: 4px solid #f4c542" if position == self.frame_position else ""
            )
            thumbnail.setToolTip(
                f"frames {item_state[0]}–{item_state[-1]}"
                if len(item_state) > 1
                else f"frame {item_ordinal}"
            )
            thumbnail.clicked.connect(self._choose_frame)
            self.filmstrip.addWidget(thumbnail)
        self.status.setText(
            f"{Path(entry['video']).name}  •  shot {self.shot_position + 1}/{len(entry['shots'])}"
            f"  •  state frames {state[0]}–{state[-1]}  •  label time {shot['time']:.3f}s\n"
            "←/→ visual state  •  Shift+←/→ 10 states  •  Space play/pause  •  "
            "C single-state  •  A between-state (current is earlier)  •  X unresolved"
        )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render()

    def _choose_frame(self, position: int) -> None:
        self.frame_position = position
        self._render()

    def _move_frame(self, delta: int) -> None:
        current = self._current_shot()
        if current is None:
            return
        _, window = current
        states = self._states(window)
        self.frame_position = max(
            0, min(len(states) - 1, self.frame_position + delta)
        )
        self._render()

    def _play_step(self) -> None:
        current = self._current_shot()
        if current is None:
            return
        _, window = current
        if self.frame_position >= len(self._states(window)) - 1:
            self.play_timer.stop()
            return
        self._move_frame(1)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            amount = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            self._move_frame(-amount if event.key() == Qt.Key.Key_Left else amount)
        elif event.key() == Qt.Key.Key_Space:
            if self.play_timer.isActive():
                self.play_timer.stop()
            else:
                self.play_timer.start()
        elif event.key() == Qt.Key.Key_C:
            self._record("single_state")
        elif event.key() == Qt.Key.Key_A:
            self._record("between_state")
        elif event.key() == Qt.Key.Key_X:
            self._record("unresolved")
        elif event.key() == Qt.Key.Key_Backspace:
            self._undo()
        else:
            super().keyPressEvent(event)

    def _record(self, status: str) -> None:
        current = self._current_shot()
        if current is None:
            return
        shot, window = current
        entry = self.manifest["videos"][self.video_index]
        annotation_path = Path(entry["annotation"])
        payload = load_annotation(annotation_path)
        stored_shot = payload["shots"][shot["shot_index"]]
        if stored_shot.get("time") != shot["time"] or stored_shot.get("type") != shot["type"]:
            raise RuntimeError("shot changed since the review manifest was created")
        previous = stored_shot.get("contact_review", _MISSING)
        review: dict[str, Any] = {"status": status}
        if status != "unresolved":
            states = self._states(window)
            if status == "between_state" and self.frame_position + 1 >= len(states):
                QMessageBox.information(
                    self,
                    "Choose the earlier state",
                    "A between-state label needs a following visual state.",
                )
                return
            selected_states = [states[self.frame_position]]
            if status == "between_state":
                selected_states.append(states[self.frame_position + 1])
            ordinal = selected_states[0][0]
            review["frame"] = {
                key: value
                for key, value in self._frames_by_ordinal()[ordinal].items()
                if key != "path"
            }
            review["visual_states"] = [
                _state_payload(state, self._frames_by_ordinal())
                for state in selected_states
            ]
        stored_shot["contact_review"] = review
        metadata = payload.get("contact_review_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        payload["contact_review_metadata"] = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "source_fingerprint": entry["source_fingerprint"],
        }
        save_annotation(annotation_path, payload)
        self.last_change = (annotation_path, shot["shot_index"], previous)
        self.reviewed_keys.add((self.video_index, self.shot_position))
        self.play_timer.stop()
        self.shot_position += 1
        if self.shot_position >= len(entry["shots"]):
            self.video_index += 1
            self.shot_position = 0
        self._advance_past_reviewed()
        self._request_video(self.video_index)
        if self.video_index in self.prepared:
            self._request_video(self.video_index + 1)
        self._center_on_label()
        self._render()

    def _undo(self) -> None:
        if self.last_change is None:
            return
        annotation_path, shot_index, previous = self.last_change
        payload = load_annotation(annotation_path)
        shot = payload["shots"][shot_index]
        if previous is _MISSING:
            shot.pop("contact_review", None)
        else:
            shot["contact_review"] = previous
        save_annotation(annotation_path, payload)
        self.last_change = None
        for video_index, entry in enumerate(self.manifest["videos"]):
            if Path(entry["annotation"]) != annotation_path:
                continue
            for position, candidate in enumerate(entry["shots"]):
                if candidate["shot_index"] == shot_index:
                    self.reviewed_keys.discard((video_index, position))
                    self.video_index = video_index
                    self.shot_position = position
                    self._request_video(video_index)
                    self._center_on_label()
                    self._render()
                    return


class SuitabilityWindow(QWidget):
    """PROTOTYPE: classify fixed-camera videos for the side-view phase."""

    def __init__(self, manifest: dict[str, Any], cache_root: Path) -> None:
        super().__init__()
        self.manifest = manifest
        self.cache_root = cache_root
        self.video_index = 0
        self.sample_position = 0
        self.last_change: tuple[Path, object] | None = None
        self.reviewed = {
            index
            for index, video in enumerate(manifest["videos"])
            if self._scope(video) is not None
        }
        self.main_image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.main_image.setMinimumSize(QSize(900, 500))
        self.samples = QHBoxLayout()
        self.status = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.progress = QProgressBar()
        buttons = QHBoxLayout()
        for label, callback in (
            ("I  In-scope side view", lambda: self._record_scope("in_scope")),
            ("O  Out-of-scope view", lambda: self._record_scope("out_of_scope_view")),
            ("Backspace  Undo", self._undo_scope),
        ):
            button = QPushButton(label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.main_image, 1)
        layout.addLayout(self.samples)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)
        self.setWindowTitle("PROTOTYPE side-view suitability review")
        self.showMaximized()
        self._advance_reviewed()
        self._render_scope()

    def _scope(self, video: dict[str, Any]) -> str | None:
        payload = load_annotation(Path(video["annotation"]))
        metadata = payload.get("contact_review_metadata")
        if not isinstance(metadata, dict):
            return None
        scope = metadata.get("side_view_suitability")
        return scope if scope in {"in_scope", "out_of_scope_view"} else None

    def _advance_reviewed(self) -> None:
        while self.video_index in self.reviewed:
            self.video_index += 1

    def _sample_frames(self, video: dict[str, Any]) -> list[dict[str, Any]]:
        prepared = prepare_video(video, self.cache_root)
        frames = {frame["ordinal"]: frame for frame in prepared["frames"]}
        payload = load_annotation(Path(video["annotation"]))
        samples = []
        for selected, window in zip(video["shots"], prepared["windows"], strict=True):
            review = payload["shots"][selected["shot_index"]].get("contact_review", {})
            ordinal = review.get("frame", {}).get("ordinal", window["center_ordinal"])
            samples.append(frames[ordinal])
        if len(samples) <= 5:
            return samples
        indexes = (0, len(samples) // 4, len(samples) // 2, 3 * len(samples) // 4, -1)
        return [samples[index] for index in indexes]

    def _render_scope(self) -> None:
        total = len(self.manifest["videos"])
        self.progress.setRange(0, total)
        self.progress.setValue(len(self.reviewed))
        self.progress.setFormat(f"{len(self.reviewed)} / {total} videos reviewed")
        if self.video_index >= total:
            self.main_image.setText("Suitability review complete")
            self.status.setText("All selected camera views have a side-view scope label.")
            return
        video = self.manifest["videos"][self.video_index]
        samples = self._sample_frames(video)
        self.sample_position = min(self.sample_position, len(samples) - 1)
        selected = samples[self.sample_position]
        pixmap = QPixmap(selected["path"])
        self.main_image.setPixmap(
            pixmap.scaled(
                self.main_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        while self.samples.count():
            item = self.samples.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for position, sample in enumerate(samples):
            thumbnail = ClickableFrame(position)
            thumbnail.setFixedSize(180, 105)
            thumbnail.setPixmap(
                QPixmap(sample["path"]).scaled(
                    thumbnail.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )
            thumbnail.setStyleSheet(
                "border: 4px solid #f4c542" if position == self.sample_position else ""
            )
            thumbnail.clicked.connect(self._choose_sample)
            self.samples.addWidget(thumbnail)
        self.status.setText(
            f"{Path(video['video']).name}  •  video {self.video_index + 1}/{total}\n"
            "I: in-scope side view  •  O: out-of-scope angle/distance  •  "
            "click samples to inspect the fixed camera view"
        )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._render_scope()

    def _choose_sample(self, position: int) -> None:
        self.sample_position = position
        self._render_scope()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_I:
            self._record_scope("in_scope")
        elif event.key() == Qt.Key.Key_O:
            self._record_scope("out_of_scope_view")
        elif event.key() == Qt.Key.Key_Backspace:
            self._undo_scope()
        else:
            super().keyPressEvent(event)

    def _record_scope(self, scope: str) -> None:
        if self.video_index >= len(self.manifest["videos"]):
            return
        video = self.manifest["videos"][self.video_index]
        annotation_path = Path(video["annotation"])
        payload = load_annotation(annotation_path)
        metadata = payload.get("contact_review_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        previous = metadata.get("side_view_suitability", _MISSING)
        payload["contact_review_metadata"] = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "side_view_suitability": scope,
        }
        save_annotation(annotation_path, payload)
        self.last_change = (annotation_path, previous)
        self.reviewed.add(self.video_index)
        self.video_index += 1
        self.sample_position = 0
        self._advance_reviewed()
        self._render_scope()

    def _undo_scope(self) -> None:
        if self.last_change is None:
            return
        annotation_path, previous = self.last_change
        payload = load_annotation(annotation_path)
        metadata = payload["contact_review_metadata"]
        if previous is _MISSING:
            metadata.pop("side_view_suitability", None)
        else:
            metadata["side_view_suitability"] = previous
        save_annotation(annotation_path, payload)
        self.last_change = None
        for index, video in enumerate(self.manifest["videos"]):
            if Path(video["annotation"]) == annotation_path:
                self.reviewed.discard(index)
                self.video_index = index
                self.sample_position = 0
                self._render_scope()
                return


_MISSING = object()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("out/contact-review-prototype")
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="create or inspect the manifest without launching the review UI",
    )
    review_mode = parser.add_mutually_exclusive_group()
    review_mode.add_argument(
        "--correct-duplicates",
        action="store_true",
        help="migrate completed reviews and re-review duplicate-cadence videos",
    )
    review_mode.add_argument(
        "--prepare-corrections",
        action="store_true",
        help="detect duplicate cadence, back up labels, and migrate without opening UI",
    )
    review_mode.add_argument(
        "--review-suitability",
        action="store_true",
        help="classify each fixed camera view for the side-view phase",
    )
    review_mode.add_argument(
        "--finalize-splits",
        action="store_true",
        help="regenerate grouped experiment splits from completed review labels",
    )
    args = parser.parse_args()
    manifest_path = args.work_dir / "manifest.json"
    if args.rebuild_manifest or not manifest_path.is_file():
        manifest = build_manifest(args.directory, args.limit)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Created {manifest_path}: {manifest['selected_shots']} forehands in "
            f"{len(manifest['videos'])} videos"
        )
    else:
        manifest = json.loads(manifest_path.read_text())
    if args.finalize_splits:
        assignments = finalize_reviewed_splits(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        counts = Counter(
            video["split"] for video in manifest["videos"] for _ in video["shots"]
        )
        print(f"Finalized {len(assignments)} family splits: {dict(counts)}")
        return 0
    if args.manifest_only:
        split_counts = {
            split: sum(
                len(video["shots"])
                for video in manifest["videos"]
                if video["split"] == split
            )
            for split in ("train", "validation", "test")
        }
        print(
            f"Manifest has {manifest['selected_shots']} forehands in "
            f"{len(manifest['videos'])} videos: {split_counts}"
        )
        return 0
    if not manifest["videos"]:
        sys.exit("No matching forehands with source video files were found.")
    cache_root = args.work_dir / "cache"
    if args.correct_duplicates or args.prepare_corrections:
        duplicate_videos = set(manifest.get("duplicate_cadence_videos", []))
        if not duplicate_videos:
            duplicate_videos = duplicate_cadence_videos(manifest, cache_root)
            manifest["duplicate_cadence_videos"] = sorted(duplicate_videos)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        migrated, deferred = migrate_completed_reviews(
            manifest,
            cache_root,
            args.work_dir / "schema-v1-backup",
            duplicate_videos,
        )
        print(
            f"Migrated {migrated} reviews; {deferred} contacts require the "
            "duplicate-aware correction pass."
        )
        if args.prepare_corrections:
            return 0
        correction_videos = [
            video for video in manifest["videos"] if video["video"] in duplicate_videos
        ]
        manifest = {
            **manifest,
            "videos": correction_videos,
            "selected_shots": sum(len(video["shots"]) for video in correction_videos),
        }
    app = QApplication.instance() or QApplication(sys.argv)
    if args.review_suitability:
        window = SuitabilityWindow(manifest, cache_root)
    else:
        window = ReviewWindow(
            manifest,
            cache_root,
            collapse_duplicates=args.correct_duplicates,
        )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""THROWAWAY PROTOTYPE: rank discrete visual contact frames.

Question: can stock YOLO ball/racket detections plus neighboring-frame evidence
select a useful forehand contact frame with an absolute confidence gate?

Run the complete compact pilot with ``uv run visual-contact-prototype``. Results
and disposable detection caches are written under ``out/contact-frame-prototype``.
This intentionally does not integrate with the production comparison workflow.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

BALL_CLASS = 32
RACKET_CLASS = 38
PERSON_CLASS = 0
CLASS_NAMES = {PERSON_CLASS: "person", BALL_CLASS: "ball", RACKET_CLASS: "racket"}
DETECTOR_CACHE_VERSION = 1
SCORER_VERSION = 3
TIE_SCORE_TOLERANCE = 1e-6
TEMPORAL_EXACT_AGREEMENT_BONUS = 0.25


@dataclass(frozen=True)
class Swing:
    source: Path
    label_time: float
    label_frame: int
    fps: float
    group: str = "pilot"

    @property
    def key(self) -> str:
        return f"{self.source.stem}-{self.label_frame}"


@dataclass(frozen=True)
class RankedFrame:
    ordinal: int
    score: float
    direct_evidence: float
    disappearance_evidence: float
    trajectory_evidence: float


@dataclass(frozen=True)
class Selection:
    selected_frame: int | None
    confidence: float
    plausible_frames: tuple[int, ...]
    reason: str | None
    ranking: tuple[RankedFrame, ...]


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_box_distance(point: tuple[float, float], box: list[float]) -> float:
    x = max(box[0], min(point[0], box[2]))
    y = max(box[1], min(point[1], box[3]))
    return distance(point, (x, y))


def choose_player(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    people = [d for d in detections if d["class_id"] == PERSON_CLASS]
    return max(people, key=lambda d: box_area(d["box"]), default=None)


def credible_confidence(confidence: float) -> float:
    """Use detector confidence as a plausibility gate, not a timing score."""
    return max(0.0, min(1.0, (confidence - 0.05) / 0.35))


def prepare_frame_evidence(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Associate detections with the main player and flag stationary ball spots."""
    prepared: list[dict[str, Any]] = []
    for frame in frames:
        player = choose_player(frame["detections"])
        if player is None:
            prepared.append({**frame, "player": None, "rackets": [], "balls": []})
            continue
        player_box = player["box"]
        player_center = box_center(player_box)
        diagonal = max(1.0, math.hypot(player_box[2] - player_box[0], player_box[3] - player_box[1]))
        people = [d for d in frame["detections"] if d["class_id"] == PERSON_CLASS]
        rackets = []
        for racket in (d for d in frame["detections"] if d["class_id"] == RACKET_CLASS):
            center = box_center(racket["box"])
            relation = point_box_distance(center, player_box) / diagonal
            main_ownership = relation + 0.15 * distance(center, player_center) / diagonal
            other_ownership = []
            for person in people:
                if person is player:
                    continue
                person_box = person["box"]
                person_center = box_center(person_box)
                person_diagonal = max(1.0, math.hypot(person_box[2] - person_box[0], person_box[3] - person_box[1]))
                other_ownership.append(
                    point_box_distance(center, person_box) / person_diagonal
                    + 0.15 * distance(center, person_center) / person_diagonal
                )
            belongs_to_player = not other_ownership or main_ownership <= min(other_ownership) + 0.03
            if relation <= 0.45 and belongs_to_player:
                quality = credible_confidence(racket["confidence"]) * (1 - relation / 0.6)
                rackets.append({**racket, "relation": relation, "quality": quality})
        rackets = sorted(rackets, key=lambda racket: racket["quality"], reverse=True)[:1]
        balls = []
        for ball in (d for d in frame["detections"] if d["class_id"] == BALL_CLASS):
            if distance(box_center(ball["box"]), player_center) / diagonal <= 1.35:
                balls.append({**ball, "diagonal": diagonal})
        prepared.append({**frame, "player": player, "rackets": rackets, "balls": balls, "diagonal": diagonal})

    for index, frame in enumerate(prepared):
        for ball in frame["balls"]:
            center = box_center(ball["box"])
            diagonal = ball["diagonal"]
            stationary_hits = 0
            motion_sides: set[int] = set()
            for offset in range(-6, 7):
                if offset == 0 or not 0 <= index + offset < len(prepared):
                    continue
                candidates = prepared[index + offset]["balls"]
                if not candidates:
                    continue
                nearest = min(distance(center, box_center(other["box"])) for other in candidates)
                if nearest <= 0.008 * diagonal:
                    stationary_hits += 1
                if 0.003 * diagonal <= nearest <= 0.16 * diagonal * abs(offset):
                    motion_sides.add(-1 if offset < 0 else 1)
            support = len(motion_sides) / 2
            ball["stationary"] = stationary_hits >= 5
            ball["quality"] = 0.0 if ball["stationary"] else credible_confidence(ball["confidence"]) * (0.45 + 0.55 * support)
    return prepared


def best_direct_pair(frame: dict[str, Any]) -> tuple[float, float, tuple[float, float] | None]:
    best = (0.0, 1.0, None)
    diagonal = frame.get("diagonal", 1.0)
    for racket in frame["rackets"]:
        for ball in frame["balls"]:
            if ball["quality"] < 0.05:
                continue
            gap = point_box_distance(box_center(ball["box"]), racket["box"]) / diagonal
            proximity = math.exp(-gap / 0.10)
            evidence = math.sqrt(max(0.0, ball["quality"] * racket["quality"])) * proximity
            candidate = (evidence, gap, box_center(ball["box"]))
            if candidate[0] > best[0]:
                best = candidate
    return best


def disappearance_evidence(index: int, frames: list[dict[str, Any]]) -> float:
    frame = frames[index]
    if not frame["rackets"]:
        return 0.0
    racket = max(frame["rackets"], key=lambda item: item["quality"])
    before: tuple[int, dict[str, Any]] | None = None
    after: tuple[int, dict[str, Any]] | None = None
    for gap in range(1, 5):
        if before is None and index - gap >= 0 and frames[index - gap]["balls"]:
            moving = [b for b in frames[index - gap]["balls"] if b["quality"] >= 0.05]
            if moving:
                before = (index - gap, max(moving, key=lambda item: item["quality"]))
        if after is None and index + gap < len(frames) and frames[index + gap]["balls"]:
            moving = [b for b in frames[index + gap]["balls"] if b["quality"] >= 0.05]
            if moving:
                after = (index + gap, max(moving, key=lambda item: item["quality"]))
    if before is None or after is None:
        return 0.0
    before_i, before_ball = before
    after_i, after_ball = after
    before_center = box_center(before_ball["box"])
    after_center = box_center(after_ball["box"])
    diagonal = frame["diagonal"]
    travel = distance(before_center, after_center) / diagonal
    if not 0.008 <= travel <= 0.55:
        return 0.0
    fraction = (index - before_i) / (after_i - before_i)
    interpolated = (
        before_center[0] + (after_center[0] - before_center[0]) * fraction,
        before_center[1] + (after_center[1] - before_center[1]) * fraction,
    )
    gap = point_box_distance(interpolated, racket["box"]) / diagonal
    quality = (before_ball["quality"] * after_ball["quality"] * racket["quality"]) ** (1 / 3)
    gap_penalty = 1 - 0.08 * (after_i - before_i - 2)
    return max(0.0, quality * math.exp(-gap / 0.12) * gap_penalty)


def primary_ball_center(frame: dict[str, Any]) -> tuple[float, float] | None:
    usable = [ball for ball in frame["balls"] if ball["quality"] >= 0.05]
    if not usable:
        return None
    return box_center(max(usable, key=lambda ball: ball["quality"])["box"])


def trajectory_evidence(index: int, frames: list[dict[str, Any]]) -> float:
    """Return evidence that the main ball changes direction around this frame."""
    centers = [primary_ball_center(frame) for frame in frames]
    current = centers[index]
    if current is not None:
        before = next((centers[i] for i in range(index - 1, max(-1, index - 3), -1) if centers[i] is not None), None)
        after = next((centers[i] for i in range(index + 1, min(len(centers), index + 3)) if centers[i] is not None), None)
        if before is None or after is None:
            return 0.0
        incoming = (current[0] - before[0], current[1] - before[1])
        outgoing = (after[0] - current[0], after[1] - current[1])
    else:
        before_points = [centers[i] for i in range(max(0, index - 3), index) if centers[i] is not None]
        after_points = [centers[i] for i in range(index + 1, min(len(centers), index + 4)) if centers[i] is not None]
        if len(before_points) < 2 or len(after_points) < 2:
            return 0.0
        incoming = (before_points[-1][0] - before_points[-2][0], before_points[-1][1] - before_points[-2][1])
        outgoing = (after_points[1][0] - after_points[0][0], after_points[1][1] - after_points[0][1])
    incoming_length = math.hypot(*incoming)
    outgoing_length = math.hypot(*outgoing)
    if incoming_length < 2 or outgoing_length < 2:
        return 0.0
    cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (incoming_length * outgoing_length)
    return max(0.0, min(1.0, (0.5 - cosine) / 1.5))


def rank_contact_frames(frames: list[dict[str, Any]]) -> Selection:
    """Pure deterministic scorer; replace this without changing extraction/reporting."""
    prepared = prepare_frame_evidence(frames)
    if not any(frame["rackets"] for frame in prepared):
        return Selection(None, 0.0, (), "no player-related racket", ())
    if not any(any(ball["quality"] >= 0.05 for ball in frame["balls"]) for frame in prepared):
        return Selection(None, 0.0, (), "no moving ball evidence", ())

    direct = [best_direct_pair(frame) for frame in prepared]
    ranking = []
    for index, frame in enumerate(prepared):
        direct_evidence = direct[index][0]
        disappearance = disappearance_evidence(index, prepared)
        trajectory = trajectory_evidence(index, prepared)
        racket_quality = max((r["quality"] for r in frame["rackets"]), default=0.0)
        contact = max(direct_evidence, min(1.0, 1.15 * disappearance))
        dropout_bonus = 0.16 * min(1.0, disappearance / 0.35) if primary_ball_center(frame) is None else 0.0
        score = min(1.0, 0.55 * contact + 0.30 * trajectory * contact + 0.15 * racket_quality + dropout_bonus)
        ranking.append(RankedFrame(frame["ordinal"], score, direct_evidence, disappearance, trajectory))

    ranking.sort(key=lambda item: (-item.score, item.ordinal))
    strongest = ranking[0]
    if strongest.score < 0.12:
        return Selection(None, strongest.score, (), "weak visual evidence", tuple(ranking))
    tied = [item for item in ranking if item.score >= strongest.score - TIE_SCORE_TOLERANCE]
    selected = min(tied, key=lambda item: item.ordinal)
    plausible = tuple(sorted(item.ordinal for item in ranking if item.score >= max(strongest.score * 0.90, strongest.score - 0.055)))
    ambiguous = len(plausible) > 2 or (len(plausible) == 2 and plausible[1] != plausible[0] + 1)
    runner_up = ranking[1].score if len(ranking) > 1 else 0.0
    margin = max(0.0, (strongest.score - runner_up) / max(strongest.score, 1e-6))
    confidence = min(1.0, strongest.score * (0.72 + 0.28 * min(1.0, margin / 0.20)))
    if ambiguous:
        confidence *= 0.35
        return Selection(selected.ordinal, confidence, plausible, "broad or separated ambiguity", tuple(ranking))
    return Selection(selected.ordinal, confidence, plausible, None, tuple(ranking))


def session_family(source: Path) -> str:
    """Keep adjacent camera-roll recordings together during prototype validation."""
    match = re.search(r"IMG_(\d+)", source.stem)
    if match is None:
        return source.stem
    ordinal = int(match.group(1))
    for name, lower, upper in (
        ("857", 8570, 8579),
        ("861", 8610, 8619),
        ("863", 8630, 8649),
        ("867", 8670, 8679),
        ("911", 9110, 9119),
        ("912", 9120, 9139),
    ):
        if lower <= ordinal <= upper:
            return name
    return source.stem


def temporal_features(frames: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    prepared = prepare_frame_evidence(frames)
    direct = [best_direct_pair(frame) for frame in prepared]
    base = []
    for index, frame in enumerate(prepared):
        base.append(
            [
                direct[index][0],
                min(direct[index][1], 1.0),
                disappearance_evidence(index, prepared),
                trajectory_evidence(index, prepared),
                max((racket["quality"] for racket in frame["rackets"]), default=0.0),
                max((ball["quality"] for ball in frame["balls"]), default=0.0),
                float(primary_ball_center(frame) is None),
            ]
        )
    base_array = np.asarray(base)
    features = [
        [
            value
            for offset in range(-4, 5)
            for value in base_array[min(len(base_array) - 1, max(0, index + offset))]
        ]
        for index in range(len(base_array))
    ]
    return np.asarray([frame["ordinal"] for frame in prepared]), np.asarray(features)


def apply_temporal_corroboration(records: list[dict[str, Any]]) -> None:
    """Apply family-held-out temporal predictions to pilot selections in place."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    pilot = [record for record in records if record["swing"].group == "pilot"]
    families = {session_family(record["swing"].source) for record in pilot}
    if len(families) < 2:
        return
    for record in pilot:
        ordinals, features = temporal_features(record["frames"])
        record["temporal_ordinals"] = ordinals
        record["temporal_features"] = features

    for held_family in sorted(families):
        training_features = []
        targets = []
        for record in pilot:
            swing: Swing = record["swing"]
            if session_family(swing.source) == held_family:
                continue
            ordinals = record["temporal_ordinals"]
            features = record["temporal_features"]
            label_index = int(np.flatnonzero(ordinals == swing.label_frame)[0])
            for index in range(max(0, label_index - 12), min(len(features), label_index + 13)):
                training_features.append(features[index])
                targets.append(math.exp(-abs(index - label_index) / 0.8))
        model = HistGradientBoostingRegressor(
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=2,
            random_state=7,
        ).fit(np.asarray(training_features), np.asarray(targets))

        for record in pilot:
            swing: Swing = record["swing"]
            if session_family(swing.source) != held_family:
                continue
            selection: Selection = record["selection"]
            ordinals = record["temporal_ordinals"]
            scores = model.predict(record["temporal_features"])
            learned_index = int(np.argmax(scores))
            learned_frame = int(ordinals[learned_index])
            local_score = float(scores[max(0, learned_index - 1) : learned_index + 2].max())
            outside = np.concatenate((scores[: max(0, learned_index - 1)], scores[min(len(scores), learned_index + 2) :]))
            local_margin = local_score - float(outside.max()) if len(outside) else local_score
            record["temporal_frame"] = learned_frame

            if selection.selected_frame is None or abs(selection.selected_frame - learned_frame) > 1:
                record["selection"] = replace(selection, confidence=0.0, reason="temporal ranker disagrees")
                continue
            confidence = local_margin
            if selection.selected_frame == learned_frame:
                confidence += TEMPORAL_EXACT_AGREEMENT_BONUS
            plausible = tuple(sorted({selection.selected_frame, learned_frame}))
            record["selection"] = replace(
                selection,
                confidence=max(0.0, min(1.0, confidence)),
                plausible_frames=plausible,
                reason=None,
            )


def video_metadata(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video metadata for {path}")
    return fps, frame_count


def load_pilot(videos_dir: Path, limit: int | None) -> list[Swing]:
    swings: list[Swing] = []
    for metadata in sorted(videos_dir.glob("*_45s.json")):
        if metadata.stem == "IMG_8631_45s":
            continue
        source = metadata.with_suffix(".mp4")
        if not source.exists():
            continue
        fps, frame_count = video_metadata(source)
        payload = json.loads(metadata.read_text())
        for shot in payload.get("shots", []):
            if shot.get("type") != "forehand":
                continue
            label_time = float(shot["time"])
            label_frame = min(frame_count - 1, max(0, math.floor(label_time * fps + 0.5)))
            swings.append(Swing(source, label_time, label_frame, fps))
            if limit is not None and len(swings) >= limit:
                return swings
    return swings


def decode_window(swing: Swing, radius: int) -> list[tuple[int, Any]]:
    capture = cv2.VideoCapture(str(swing.source))
    start = max(0, swing.label_frame - radius)
    end = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, swing.label_frame + radius)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    decoded = []
    for ordinal in range(start, end + 1):
        ok, image = capture.read()
        if not ok:
            break
        decoded.append((ordinal, image))
    capture.release()
    return decoded


def decode_ordinals(source: Path, ordinals: set[int]) -> list[tuple[int, Any]]:
    capture = cv2.VideoCapture(str(source))
    decoded = []
    start = min(ordinals)
    end = max(ordinals)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for ordinal in range(start, end + 1):
        ok, image = capture.read()
        if not ok:
            break
        if ordinal in ordinals:
            decoded.append((ordinal, image))
    capture.release()
    return decoded


def cache_signature(swing: Swing, radius: int, imgsz: int, conf: float) -> str:
    stat = swing.source.stat()
    payload = f"{swing.source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{swing.label_frame}|{radius}|{imgsz}|{conf}|{DETECTOR_CACHE_VERSION}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def detection_cache_path(cache_dir: Path, swing: Swing, radius: int, imgsz: int, conf: float) -> Path:
    return cache_dir / f"{swing.key}-{cache_signature(swing, radius, imgsz, conf)}.json"


def detect_window(
    model: YOLO,
    swing: Swing,
    decoded: list[tuple[int, Any]],
    cache_dir: Path,
    radius: int,
    imgsz: int,
    conf: float,
    batch_size: int,
    device: str,
    no_cache: bool,
) -> list[dict[str, Any]]:
    cache_path = detection_cache_path(cache_dir, swing, radius, imgsz, conf)
    if cache_path.exists() and not no_cache:
        return json.loads(cache_path.read_text())["frames"]
    frames = []
    for batch_start in range(0, len(decoded), batch_size):
        batch = decoded[batch_start : batch_start + batch_size]
        results = model.predict(
            [image for _, image in batch],
            classes=[PERSON_CLASS, BALL_CLASS, RACKET_CLASS],
            conf=conf,
            iou=0.5,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for (ordinal, _), result in zip(batch, results, strict=True):
            detections = []
            if result.boxes is not None:
                for xyxy, class_id, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    strict=True,
                ):
                    detections.append({"class_id": int(class_id), "confidence": round(float(confidence), 5), "box": [round(float(v), 2) for v in xyxy]})
            frames.append({"ordinal": ordinal, "detections": detections})
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"frames": frames}, separators=(",", ":")))
    return frames


def choose_threshold(records: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]], bool]:
    eligible = [r for r in records if r["selection"].selected_frame is not None and r["selection"].reason is None]
    confidences = sorted({r["selection"].confidence for r in eligible}, reverse=True)
    curve = []
    for threshold in confidences + [0.0]:
        included = [r for r in eligible if r["selection"].confidence >= threshold]
        correct = sum(abs(r["selection"].selected_frame - r["swing"].label_frame) <= 1 for r in included)
        curve.append({"threshold": threshold, "included": len(included), "coverage": len(included) / len(records) if records else 0.0, "precision": correct / len(included) if included else 1.0})
    feasible = [point for point in curve if point["included"] > 0 and point["precision"] >= 0.95]
    if not feasible:
        maximum = max((r["selection"].confidence for r in eligible), default=0.0)
        return math.nextafter(maximum, math.inf), curve, False
    winner = max(feasible, key=lambda point: (point["included"], -point["threshold"]))
    return winner["threshold"], curve, True


def detection_key(detection: dict[str, Any]) -> tuple[int, float, tuple[float, ...]]:
    return detection["class_id"], detection["confidence"], tuple(detection["box"])


def annotated_data_url(image: Any, frame: dict[str, Any], prepared: dict[str, Any], label: str) -> str:
    canvas = image.copy()
    colors = {PERSON_CLASS: (120, 190, 70), BALL_CLASS: (20, 210, 255), RACKET_CLASS: (255, 120, 40)}
    accepted = {
        detection_key(detection)
        for detection in ([prepared["player"]] if prepared["player"] is not None else [])
        + prepared["rackets"]
        + [ball for ball in prepared["balls"] if ball["quality"] >= 0.05]
    }
    stationary = {detection_key(ball) for ball in prepared["balls"] if ball["stationary"]}
    for detection in sorted(frame["detections"], key=lambda item: detection_key(item) in accepted):
        x1, y1, x2, y2 = (int(v) for v in detection["box"])
        key = detection_key(detection)
        is_accepted = key in accepted
        color = colors[detection["class_id"]] if is_accepted else (135, 135, 135)
        thickness = 3 if is_accepted else 1
        prefix = "" if is_accepted else ("ignored stationary " if key in stationary else "ignored ")
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        text = f"{prefix}{CLASS_NAMES[detection['class_id']]} {detection['confidence']:.2f}"
        cv2.putText(canvas, text, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, thickness, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 54), (15, 20, 25), -1)
    cv2.putText(canvas, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    target_width = 960
    canvas = cv2.resize(canvas, (target_width, round(canvas.shape[0] * target_width / canvas.shape[1])))
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 76])
    if not ok:
        raise RuntimeError("could not encode gallery frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode()


def build_card(record: dict[str, Any], threshold: float) -> str:
    swing: Swing = record["swing"]
    selection: Selection = record["selection"]
    decoded = dict(record["decoded"])
    frames = {frame["ordinal"]: frame for frame in record["frames"]}
    prepared_frames = {frame["ordinal"]: frame for frame in record["prepared_frames"]}
    selected = selection.selected_frame
    wanted = {swing.label_frame - 1, swing.label_frame, swing.label_frame + 1}
    if selected is not None:
        wanted.update({selected - 1, selected, selected + 1})
    images = []
    for ordinal in sorted(wanted):
        if ordinal not in decoded or ordinal not in frames:
            continue
        tags = []
        if ordinal == swing.label_frame:
            tags.append("manual")
        if ordinal == selected:
            tags.append("selected")
        label = f"frame {ordinal}" + (" · " + " + ".join(tags) if tags else "")
        images.append(f'<figure><img loading="lazy" src="{annotated_data_url(decoded[ordinal], frames[ordinal], prepared_frames[ordinal], label)}"><figcaption>{html.escape(label)}</figcaption></figure>')
    error = None if selected is None else selected - swing.label_frame
    included = selection.reason is None and selection.confidence >= threshold
    outcome = "omitted" if not included else ("within one" if abs(error) <= 1 else "miss")
    rank_rows = "".join(
        f"<tr><td>{r.ordinal}</td><td>{r.score:.3f}</td><td>{r.direct_evidence:.3f}</td><td>{r.disappearance_evidence:.3f}</td><td>{r.trajectory_evidence:.3f}</td></tr>"
        for r in selection.ranking[:8]
    )
    reason = selection.reason or ("below operating threshold" if not included else "included")
    return f'''<article class="swing {outcome.replace(' ', '-')}">
      <header><div><h3>{html.escape(swing.source.name)} · label frame {swing.label_frame}</h3><p>selected {selected if selected is not None else 'none'} · error {error if error is not None else '—'} · confidence {selection.confidence:.3f} · {html.escape(reason)}</p></div><span>{outcome}</span></header>
      <div class="filmstrip">{''.join(images)}</div>
      <details><summary>Full scoring state</summary><p>Plausible frames: {selection.plausible_frames or 'none'}</p><table><thead><tr><th>frame</th><th>score</th><th>direct</th><th>disappearance</th><th>trajectory</th></tr></thead><tbody>{rank_rows}</tbody></table></details>
    </article>'''


def write_report(output: Path, records: list[dict[str, Any]], threshold: float, curve: list[dict[str, Any]], feasible: bool, elapsed: float) -> None:
    included = [r for r in records if r["selection"].reason is None and r["selection"].confidence >= threshold]
    exact = sum(r["selection"].selected_frame == r["swing"].label_frame for r in included)
    within_one = sum(abs(r["selection"].selected_frame - r["swing"].label_frame) <= 1 for r in included)
    cards = "".join(build_card(record, threshold) for record in sorted(records, key=lambda r: (r["selection"].reason is None and r["selection"].confidence >= threshold, -(abs((r["selection"].selected_frame or -99999) - r["swing"].label_frame)))))
    curve_rows = "".join(f"<tr><td>{p['threshold']:.3f}</td><td>{p['included']}</td><td>{p['coverage']:.1%}</td><td>{p['precision']:.1%}</td></tr>" for p in curve)
    warning = "" if feasible else '<p class="warning">No non-empty threshold met 95% within-one-frame precision; the operating point omits every swing.</p>'
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual contact-frame prototype</title>
<style>
:root{{--ink:#18201d;--muted:#68736d;--paper:#f5f2ea;--card:#fff;--line:#d9d5c9;--accent:#176b51;--bad:#a43d32}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}main{{max-width:1600px;margin:auto;padding:42px 28px 100px}}h1{{font:700 42px/1.05 Georgia,serif;margin:0 0 10px}}h2{{font:700 26px Georgia,serif}}p{{color:var(--muted)}}.question{{max-width:850px;font-size:18px}}.stats{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:12px;margin:28px 0}}.stat,.panel,.swing{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}}.stat strong{{display:block;font-size:28px}}.controls{{position:sticky;top:0;z-index:2;background:#f5f2eaeF;padding:12px 0;display:flex;gap:8px}}button{{border:1px solid var(--line);background:white;border-radius:99px;padding:8px 13px;cursor:pointer}}button.active{{background:var(--ink);color:white}}.swing{{margin:16px 0}}.swing header{{display:flex;justify-content:space-between;gap:20px}}.swing h3{{margin:0}}.swing header p{{margin:4px 0}}.swing header span{{font-weight:700;text-transform:uppercase;white-space:nowrap}}.within-one header span{{color:var(--accent)}}.miss header span,.warning{{color:var(--bad)}}.filmstrip{{display:flex;overflow:auto;gap:14px;padding:14px 0 20px;scroll-snap-type:x proximity}}figure{{margin:0;flex:0 0 min(960px,calc(100vw - 96px));scroll-snap-align:start}}figure img{{display:block;width:100%;height:auto;border-radius:8px}}figcaption{{font-size:14px;color:var(--muted);padding-top:5px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid var(--line);text-align:right}}td:first-child,th:first-child{{text-align:left}}details{{margin-top:8px}}@media(max-width:800px){{.stats{{grid-template-columns:1fr 1fr}}main{{padding:24px 14px 80px}}figure{{flex-basis:calc(100vw - 60px)}}}}
</style></head><body><main>
<h1>Visual contact-frame prototype · round 3</h1><p class="question"><strong>Question:</strong> can main-player association, hard stationary-ball rejection, stronger contact-disappearance evidence, and a family-held-out temporal ranker select trustworthy discrete forehand contact frames with an absolute confidence gate? Bright boxes contribute evidence; thin gray boxes are ignored. Manual timestamps are weak labels, and every selection is an existing source frame.</p>
{warning}<section class="stats"><div class="stat"><strong>{len(records)}</strong><span>forehands</span></div><div class="stat"><strong>{len(included) / len(records):.1%}</strong><span>coverage</span></div><div class="stat"><strong>{exact / len(included) if included else 0:.1%}</strong><span>exact-frame precision</span></div><div class="stat"><strong>{within_one / len(included) if included else 0:.1%}</strong><span>within-one precision</span></div><div class="stat"><strong>{elapsed / 60:.1f}m</strong><span>cold/warm runtime</span></div></section>
<section class="panel"><h2>Operating point</h2><p>Threshold <strong>{threshold:.3f}</strong>, chosen for maximum coverage subject to at least 95% of included selections landing on the manual frame or one adjacent frame.</p><details><summary>Precision–coverage curve</summary><table><thead><tr><th>threshold</th><th>included</th><th>coverage</th><th>within-one precision</th></tr></thead><tbody>{curve_rows}</tbody></table></details></section>
<div class="controls"><button class="active" data-filter="all">All</button><button data-filter="within-one">Within one</button><button data-filter="miss">Misses</button><button data-filter="omitted">Omitted</button></div>
<section id="gallery">{cards}</section>
<script>document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.swing').forEach(x=>x.hidden=b.dataset.filter!=='all'&&!x.classList.contains(b.dataset.filter))}})</script>
</main></body></html>'''
    output.write_text(document)


def evaluation_summary(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    included = [r for r in records if r["selection"].reason is None and r["selection"].confidence >= threshold]
    exact = sum(r["selection"].selected_frame == r["swing"].label_frame for r in included)
    within_one = sum(abs(r["selection"].selected_frame - r["swing"].label_frame) <= 1 for r in included)
    omission_reasons: dict[str, int] = {}
    for record in records:
        selection = record["selection"]
        reason = selection.reason
        if reason is None and selection.confidence < threshold:
            reason = "below operating threshold"
        if reason is not None:
            omission_reasons[reason] = omission_reasons.get(reason, 0) + 1
    return {
        "included": len(included),
        "omitted": len(records) - len(included),
        "coverage": len(included) / len(records),
        "exact_frame_precision": exact / len(included) if included else 0.0,
        "within_one_frame_precision": within_one / len(included) if included else 0.0,
        "omission_reasons": omission_reasons,
    }


def serializable_results(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    results = []
    for record in records:
        swing: Swing = record["swing"]
        selection: Selection = record["selection"]
        included = selection.reason is None and selection.confidence >= threshold
        results.append(
            {
                "source": str(swing.source),
                "manual_timestamp": swing.label_time,
                "manual_frame": swing.label_frame,
                "selected_frame": selection.selected_frame,
                "temporal_frame": record.get("temporal_frame"),
                "frame_error": None if selection.selected_frame is None else selection.selected_frame - swing.label_frame,
                "confidence": selection.confidence,
                "included": included,
                "omission_reason": selection.reason or (None if included else "below operating threshold"),
                "plausible_frames": selection.plausible_frames,
                "top_ranking": [
                    {
                        "frame": ranked.ordinal,
                        "score": ranked.score,
                        "direct": ranked.direct_evidence,
                        "disappearance": ranked.disappearance_evidence,
                        "trajectory": ranked.trajectory_evidence,
                    }
                    for ranked in selection.ranking[:8]
                ],
            }
        )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=Path("videos_train"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/contact-frame-prototype-v3"))
    parser.add_argument("--detection-cache-dir", type=Path, help="reuse detections from another prototype round")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu", help="Ultralytics device (cpu, mps, 0, ...)")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--detector-confidence", type=float, default=0.05)
    parser.add_argument("--window-ms", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="run only the first N forehands for a smoke test")
    parser.add_argument("--probe-video", type=Path, help="run a separate qualitative probe instead of the pilot")
    parser.add_argument("--probe-frame", type=int, help="known contact frame for --probe-video")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    if (args.probe_video is None) != (args.probe_frame is None):
        sys.exit("--probe-video and --probe-frame must be supplied together")
    if args.probe_video is not None:
        fps, frame_count = video_metadata(args.probe_video)
        if not 0 <= args.probe_frame < frame_count:
            sys.exit(f"--probe-frame must be between 0 and {frame_count - 1}")
        swings = [Swing(args.probe_video, args.probe_frame / fps, args.probe_frame, fps, "probe")]
    else:
        swings = load_pilot(args.videos_dir, args.limit)
    if not swings:
        sys.exit("no pilot forehands found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    records = []
    cache_dir = args.detection_cache_dir or args.output_dir / "detections"
    print(f"Visual contact prototype: {len(swings)} forehands, {args.imgsz}px, {args.device}")
    for index, swing in enumerate(swings, 1):
        radius = max(1, round(args.window_ms / 1000 * swing.fps))
        cache_path = detection_cache_path(cache_dir, swing, radius, args.imgsz, args.detector_confidence)
        if cache_path.exists() and not args.no_cache:
            frames = json.loads(cache_path.read_text())["frames"]
            decoded = []
        else:
            decoded = decode_window(swing, radius)
            frames = detect_window(model, swing, decoded, cache_dir, radius, args.imgsz, args.detector_confidence, args.batch_size, args.device, args.no_cache)
        selection = rank_contact_frames(frames)
        gallery_ordinals = {swing.label_frame - 1, swing.label_frame, swing.label_frame + 1}
        if selection.selected_frame is not None:
            gallery_ordinals.update({selection.selected_frame - 1, selection.selected_frame, selection.selected_frame + 1})
        gallery_frames = (
            [(ordinal, image) for ordinal, image in decoded if ordinal in gallery_ordinals]
            if decoded
            else decode_ordinals(swing.source, gallery_ordinals)
        )
        records.append(
            {
                "swing": swing,
                "decoded": gallery_frames,
                "frames": frames,
                "prepared_frames": prepare_frame_evidence(frames),
                "selection": selection,
            }
        )
        print(f"[{index:3}/{len(swings)}] {swing.key}: visual evidence ready", flush=True)
    apply_temporal_corroboration(records)
    for record in records:
        swing = record["swing"]
        selection = record["selection"]
        error = None if selection.selected_frame is None else selection.selected_frame - swing.label_frame
        print(f"{swing.key}: selected={selection.selected_frame} error={error} confidence={selection.confidence:.3f} reason={selection.reason or '-'}", flush=True)
    threshold, curve, feasible = choose_threshold(records)
    elapsed = time.monotonic() - started
    report = args.output_dir / "report.html"
    write_report(report, records, threshold, curve, feasible, elapsed)
    evaluation = evaluation_summary(records, threshold)
    summary = {
        "question": "Can improved main-player visual evidence plus family-held-out temporal corroboration select trustworthy discrete forehand contact frames?",
        "scorer_version": SCORER_VERSION,
        "swings": len(records),
        "operating_threshold": threshold,
        "threshold_feasible": feasible,
        "runtime_seconds": elapsed,
        "report": str(report),
        "evaluation": evaluation,
        "precision_coverage_curve": curve,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "results.json").write_text(json.dumps(serializable_results(records, threshold), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

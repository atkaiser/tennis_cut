#!/usr/bin/env python3
"""PROTOTYPE: prepare shared inputs and compare contact-frame algorithms.

The prototype is deliberately offline. Its caches and reports belong under
``out/contact-bakeoff`` and are not runtime dependencies.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
import json
import statistics
from pathlib import Path
import tempfile
from time import monotonic
from typing import Any

from tennis_cut.contact_review import ReviewDataset, ReviewedContact, load_review_dataset
from tennis_cut.swing_detection import DEFAULT_AUDIO_MODEL, PopDetector, extract_audio

DEFAULT_MANIFEST = Path("out/contact-review-prototype/manifest.json")
DEFAULT_WORK_DIRECTORY = Path("out/contact-bakeoff")
AUDIO_SEARCH_RADIUS_SECONDS = 0.25
VISUAL_EQUIVALENCE_THRESHOLD = 0.05
PLAYER_SAMPLE_COUNT = 5
OUTPUT_IMAGE_SIZE = 512
EMBEDDING_IMAGE_SIZE = 320


def contact_targets(
    contact: ReviewedContact,
    visual_states: tuple[tuple[int, ...], ...],
) -> dict[int, float]:
    """Return soft per-frame targets without overweighting duplicated frames."""

    if contact.status == "unresolved":
        return {}
    reviewed_states = tuple(
        tuple(frame.ordinal for frame in state.frames)
        for state in contact.visual_states
    )
    try:
        first_index = visual_states.index(reviewed_states[0])
    except ValueError as error:
        raise ValueError("reviewed visual state is absent from decoded window") from error

    state_masses: dict[int, float]
    if contact.status == "between_state":
        if (
            first_index + 1 >= len(visual_states)
            or visual_states[first_index + 1] != reviewed_states[1]
        ):
            raise ValueError("reviewed between-state contact is not consecutive")
        state_masses = {first_index: 0.5, first_index + 1: 0.5}
    else:
        state_masses = {first_index: 0.7}
        if first_index > 0:
            state_masses[first_index - 1] = 0.15
        if first_index + 1 < len(visual_states):
            state_masses[first_index + 1] = 0.15
        total = sum(state_masses.values())
        state_masses = {index: mass / total for index, mass in state_masses.items()}

    return {
        ordinal: mass / len(visual_states[index])
        for index, mass in state_masses.items()
        for ordinal in visual_states[index]
    }


def honor_reviewed_states(
    contact: ReviewedContact,
    automatic_states: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Keep automatic grouping except where it crosses a human-reviewed state."""

    reviewed_by_ordinal = {
        frame.ordinal: index
        for index, state in enumerate(contact.visual_states)
        for frame in state.frames
    }
    regrouped: list[list[int]] = []
    previous_key: tuple[str, int] | None = None
    for automatic_index, state in enumerate(automatic_states):
        for ordinal in state:
            reviewed_index = reviewed_by_ordinal.get(ordinal)
            key = (
                ("automatic", automatic_index)
                if reviewed_index is None
                else ("reviewed", reviewed_index)
            )
            if key != previous_key:
                regrouped.append([])
            regrouped[-1].append(ordinal)
            previous_key = key
    return tuple(tuple(state) for state in regrouped)


def select_training_indices(
    records: list[dict[str, Any]], *, negative_count: int, seed: int
) -> set[int]:
    """Keep every soft target plus a rotating mix of near and far negatives."""

    import random

    positive = {index for index, record in enumerate(records) if record["target"] > 0}
    negative = [index for index in range(len(records)) if index not in positive]
    if len(negative) <= negative_count:
        return positive | set(negative)
    chosen = {negative[0], negative[-1]} if negative_count >= 2 else {negative[0]}
    remaining = negative_count - len(chosen)
    near_count = remaining // 2
    near = sorted(
        (index for index in negative if index not in chosen),
        key=lambda index: min(abs(index - target) for target in positive),
    )
    chosen.update(near[:near_count])
    pool = [index for index in negative if index not in chosen]
    chosen.update(random.Random(seed).sample(pool, remaining - near_count))
    return positive | chosen


def listwise_contact_loss(logits: Any, targets: Any) -> Any:
    """Cross entropy between reviewed timing mass and a swing's frame ranking."""

    import torch

    return -(targets * torch.log_softmax(logits, dim=0)).sum()


def classify_selection(record: dict[str, Any]) -> str:
    """Classify a chosen frame using the review's exact and tolerant targets."""

    if record["acceptable"]:
        return "exact"
    if record["target"] > 0:
        return "tolerated"
    return "failed"


def choose_review_threshold(
    entries: list[dict[str, Any]], *, minimum_precision: float = 0.95
) -> float:
    """Maximize correct validation recall at the requested accepted precision."""

    import math

    values = sorted({float(entry["confidence"]) for entry in entries}, reverse=True)
    best: tuple[tuple[int, int, int, float], float] | None = None
    for threshold in values:
        accepted = [
            entry for entry in entries if float(entry["confidence"]) >= threshold
        ]
        correct = sum(bool(entry["correct"]) for entry in accepted)
        precision = correct / len(accepted) if accepted else 1.0
        if precision < minimum_precision:
            continue
        false = len(accepted) - correct
        key = (correct, -false, len(accepted), -threshold)
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is not None:
        return best[1]
    return math.nextafter(max(values, default=0.0), math.inf)


def stable_player_crop(
    boxes: list[tuple[int, int, int, int]],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Build one generous player-centered crop shared by a complete swing."""

    if not boxes:
        raise ValueError("at least one player box is required")
    x1 = statistics.median(box[0] for box in boxes)
    y1 = statistics.median(box[1] for box in boxes)
    x2 = statistics.median(box[2] for box in boxes)
    y2 = statistics.median(box[3] for box in boxes)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    side = round(max(2.8 * (x2 - x1), 1.5 * (y2 - y1)))
    crop_width = min(image_width, side)
    crop_height = min(image_height, side)
    left = round(center_x - crop_width / 2)
    top = round(center_y - crop_height / 2)
    left = min(max(0, left), image_width - crop_width)
    top = min(max(0, top), image_height - crop_height)
    return left, top, left + crop_width, top + crop_height


def _review_cache_directory(
    review_work_directory: Path, video_entry: dict[str, Any]
) -> Path:
    video = Path(video_entry["video"])
    fingerprint = video_entry["source_fingerprint"]
    key = f"{video.stem}-{fingerprint['size']}-{fingerprint['mtime_ns']}"
    return review_work_directory / "cache" / key


def _visual_states_for_window(
    ordinals: list[int], frames: dict[int, dict[str, Any]]
) -> tuple[tuple[int, ...], ...]:
    import cv2
    import numpy as np

    states: list[list[int]] = []
    previous_small = None
    for ordinal in ordinals:
        image = cv2.imread(frames[ordinal]["path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read cached frame {ordinal}")
        small = cv2.resize(image, (320, 180))
        if (
            states
            and float(np.mean(cv2.absdiff(previous_small, small)))
            < VISUAL_EQUIVALENCE_THRESHOLD
        ):
            states[-1].append(ordinal)
        else:
            states.append([ordinal])
        previous_small = small
    return tuple(tuple(state) for state in states)


def _sample_ordinals(ordinals: list[int]) -> tuple[int, ...]:
    if len(ordinals) <= PLAYER_SAMPLE_COUNT:
        return tuple(ordinals)
    last = len(ordinals) - 1
    return tuple(
        ordinals[round(index * last / (PLAYER_SAMPLE_COUNT - 1))]
        for index in range(PLAYER_SAMPLE_COUNT)
    )


def _largest_person_boxes(
    frame_paths: dict[int, Path], *, device: str | None
) -> dict[int, tuple[int, int, int, int]]:
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    boxes_by_ordinal: dict[int, tuple[int, int, int, int]] = {}
    items = list(frame_paths.items())
    for start in range(0, len(items), 32):
        chunk = items[start : start + 32]
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for _, path in chunk]
        if any(image is None for image in images):
            raise ValueError("could not read cached player-detection frame")
        results = model.predict(
            images,
            classes=0,
            conf=0.15,
            imgsz=640,
            device=device,
            verbose=False,
        )
        for (ordinal, _), result in zip(chunk, results, strict=True):
            detected = result.boxes
            if detected is None or detected.xyxy.numel() == 0:
                continue
            candidates = detected.xyxy.cpu().numpy()
            areas = (candidates[:, 2] - candidates[:, 0]) * (
                candidates[:, 3] - candidates[:, 1]
            )
            box = candidates[areas.argmax()]
            boxes_by_ordinal[ordinal] = tuple(round(float(value)) for value in box)
    return boxes_by_ordinal


def _resize_with_padding(image: Any, size: int = OUTPUT_IMAGE_SIZE) -> Any:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _model_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _video_cache_path(work_directory: Path, video: Path) -> Path:
    return work_directory / "audio-windows" / f"{video.stem}.json"


def _load_or_score_audio(
    detector: PopDetector,
    video: Path,
    cache_path: Path,
    model_path: Path,
) -> list[dict[str, Any]]:
    video_stat = video.stat()
    identity = {
        "video": str(video.resolve()),
        "source_fingerprint": {
            "size": video_stat.st_size,
            "mtime_ns": video_stat.st_mtime_ns,
        },
        "model": str(model_path.resolve()),
        "model_fingerprint": _model_fingerprint(model_path),
        "stride_seconds": detector.stride_s,
    }
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text())
        if payload.get("identity") == identity:
            return payload["windows"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        wav_path = Path(directory) / "audio.wav"
        extract_audio(video, wav_path)
        windows = [asdict(candidate) for candidate in detector.score_windows(wav_path)]
    cache_path.write_text(
        json.dumps({"identity": identity, "windows": windows}, separators=(",", ":"))
        + "\n"
    )
    return windows


def _contact_time(contact: ReviewedContact) -> float:
    if contact.canonical_frame is None:
        return float(contact.labeled_timestamp)
    return float(contact.canonical_frame.pts * contact.canonical_frame.time_base)


def _audio_prior(
    contact: ReviewedContact, windows: list[dict[str, Any]]
) -> dict[str, Any]:
    contact_time = _contact_time(contact)
    nearby = [
        window
        for window in windows
        if abs(float(window["timestamp"]) - contact_time)
        <= AUDIO_SEARCH_RADIUS_SECONDS
    ]
    if not nearby:
        return {
            "contact_time": contact_time,
            "candidate_time": None,
            "score": None,
            "timing_error": None,
        }
    winner = min(
        nearby,
        key=lambda window: (
            -float(window["score"]),
            abs(float(window["timestamp"]) - contact_time),
            float(window["timestamp"]),
        ),
    )
    candidate_time = float(winner["timestamp"])
    return {
        "contact_time": contact_time,
        "candidate_time": candidate_time,
        "score": float(winner["score"]),
        "timing_error": candidate_time - contact_time,
    }


def prepare_audio_priors(
    dataset: ReviewDataset,
    *,
    work_directory: Path,
    model_path: Path,
    device: str | None,
) -> Path:
    detector = PopDetector(model_path, device=device)
    contacts_by_video: dict[Path, list[ReviewedContact]] = defaultdict(list)
    for contact in dataset.contacts:
        contacts_by_video[contact.video].append(contact)

    records = []
    for position, (video, contacts) in enumerate(
        sorted(contacts_by_video.items()), start=1
    ):
        started = monotonic()
        print(f"[{position}/{len(contacts_by_video)}] scoring {video.name}", flush=True)
        windows = _load_or_score_audio(
            detector,
            video,
            _video_cache_path(work_directory, video),
            model_path,
        )
        for contact in contacts:
            records.append(
                {
                    "video": str(contact.video),
                    "shot_index": contact.shot_index,
                    "family": contact.family,
                    "split": contact.split,
                    "suitability": contact.suitability,
                    "status": contact.status,
                    **_audio_prior(contact, windows),
                }
            )
        print(f"    finished in {monotonic() - started:.1f}s", flush=True)

    output_path = work_directory / "audio-priors.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2) + "\n")
    errors = [
        abs(record["timing_error"])
        for record in records
        if record["timing_error"] is not None
    ]
    threshold_counts = Counter(
        "above_threshold"
        if record["score"] is not None and record["score"] > 0.5
        else "below_threshold"
        for record in records
    )
    errors.sort()
    median = errors[len(errors) // 2] if errors else None
    print(
        f"Wrote {output_path}: {dict(threshold_counts)}, "
        f"median absolute timing error={median}",
        flush=True,
    )
    return output_path


def prepare_frame_dataset(
    dataset: ReviewDataset,
    *,
    manifest_path: Path,
    review_work_directory: Path,
    work_directory: Path,
    device: str | None,
) -> Path:
    """Create fixed player crops and soft targets for the image prototypes."""

    import cv2

    manifest = json.loads(manifest_path.read_text())
    entries = {Path(entry["video"]): entry for entry in manifest["videos"]}
    audio_records = json.loads((work_directory / "audio-priors.json").read_text())
    audio_by_contact = {
        (Path(record["video"]), int(record["shot_index"])): record
        for record in audio_records
    }
    contacts_by_video: dict[Path, list[ReviewedContact]] = defaultdict(list)
    for contact in dataset.contacts:
        contacts_by_video[contact.video].append(contact)

    records: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    image_root = work_directory / "frame-dataset" / "images"
    for position, (video, contacts) in enumerate(
        sorted(contacts_by_video.items()), start=1
    ):
        print(f"[{position}/{len(contacts_by_video)}] cropping {video.name}", flush=True)
        entry = entries[video]
        cached = json.loads(
            (_review_cache_directory(review_work_directory, entry) / "frames.json").read_text()
        )
        frames = {int(frame["ordinal"]): frame for frame in cached["frames"]}
        windows = {int(window["shot_index"]): window for window in cached["windows"]}
        sampled_paths: dict[int, Path] = {}
        for contact in contacts:
            for ordinal in _sample_ordinals(windows[contact.shot_index]["ordinals"]):
                sampled_paths[ordinal] = Path(frames[ordinal]["path"])
        player_boxes = _largest_person_boxes(sampled_paths, device=device)

        for contact in contacts:
            window = windows[contact.shot_index]
            ordinals = [int(ordinal) for ordinal in window["ordinals"]]
            sampled = _sample_ordinals(ordinals)
            boxes = [player_boxes[ordinal] for ordinal in sampled if ordinal in player_boxes]
            if not boxes:
                omissions.append(
                    {
                        "video": str(video),
                        "shot_index": contact.shot_index,
                        "reason": "no player detected in sampled frames",
                    }
                )
                continue
            first_image = cv2.imread(frames[ordinals[0]]["path"], cv2.IMREAD_COLOR)
            if first_image is None:
                raise ValueError(f"could not read {frames[ordinals[0]]['path']}")
            height, width = first_image.shape[:2]
            crop = stable_player_crop(boxes, image_width=width, image_height=height)
            states = honor_reviewed_states(
                contact, _visual_states_for_window(ordinals, frames)
            )
            try:
                targets = contact_targets(contact, states)
            except ValueError as error:
                raise ValueError(
                    f"{video.name} shot {contact.shot_index}: {error}"
                ) from error
            state_by_ordinal = {
                ordinal: state_index
                for state_index, state in enumerate(states)
                for ordinal in state
            }
            audio = audio_by_contact[(video, contact.shot_index)]
            contact_directory = image_root / f"{video.stem}-shot-{contact.shot_index:04d}"
            contact_directory.mkdir(parents=True, exist_ok=True)
            for ordinal in ordinals:
                output_path = contact_directory / f"frame-{ordinal:06d}.jpg"
                if not output_path.is_file():
                    image = cv2.imread(frames[ordinal]["path"], cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"could not read {frames[ordinal]['path']}")
                    left, top, right, bottom = crop
                    cropped = _resize_with_padding(image[top:bottom, left:right])
                    if not cv2.imwrite(
                        str(output_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 92]
                    ):
                        raise ValueError(f"could not write {output_path}")
                frame = frames[ordinal]
                frame_time = float(
                    int(frame["pts"])
                    * Fraction(
                        int(frame["time_base"]["numerator"]),
                        int(frame["time_base"]["denominator"]),
                    )
                )
                candidate_time = audio["candidate_time"]
                records.append(
                    {
                        "image": str(output_path.resolve()),
                        "video": str(video),
                        "shot_index": contact.shot_index,
                        "family": contact.family,
                        "split": contact.split,
                        "suitability": contact.suitability,
                        "status": contact.status,
                        "ordinal": ordinal,
                        "visual_state": state_by_ordinal[ordinal],
                        "frame_time": frame_time,
                        "audio_delta": (
                            None
                            if candidate_time is None
                            else frame_time - float(candidate_time)
                        ),
                        "target": targets.get(ordinal, 0.0),
                        "acceptable": ordinal in contact.acceptable_ordinals,
                        "crop": list(crop),
                    }
                )

    output_path = work_directory / "frame-dataset" / "records.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"records": records, "omissions": omissions}, indent=2) + "\n"
    )
    print(
        f"Wrote {output_path}: {len(records)} frames, "
        f"{len(omissions)} omitted contacts",
        flush=True,
    )
    return output_path


def extract_frame_embeddings(
    *, work_directory: Path, device: str | None, batch_size: int
) -> Path:
    """Cache frozen ImageNet features for quick scorer experiments."""

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import v2

    records_path = work_directory / "frame-dataset" / "records.json"
    payload = json.loads(records_path.read_text())
    records = payload["records"]
    output_path = work_directory / "frame-dataset" / "resnet18-embeddings.pt"
    identity = {
        "records_size": records_path.stat().st_size,
        "records_mtime_ns": records_path.stat().st_mtime_ns,
        "image_size": EMBEDDING_IMAGE_SIZE,
        "images": [record["image"] for record in records],
    }
    if output_path.is_file():
        cached = torch.load(output_path, map_location="cpu", weights_only=False)
        if cached.get("identity") == identity:
            print(f"Using cached embeddings: {output_path}")
            return output_path

    selected_device = device or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    transform = v2.Compose(
        [
            v2.Resize((EMBEDDING_IMAGE_SIZE, EMBEDDING_IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    class FrameDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> Any:
            with Image.open(records[index]["image"]) as image:
                return transform(image.convert("RGB"))

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(selected_device).eval()
    loader = DataLoader(FrameDataset(), batch_size=batch_size, shuffle=False)
    batches = []
    with torch.inference_mode():
        for index, images in enumerate(loader, start=1):
            batches.append(model(images.to(selected_device)).cpu().half())
            if index % 20 == 0 or index == len(loader):
                print(f"embedded {min(index * batch_size, len(records))}/{len(records)}")
    torch.save(
        {"identity": identity, "embeddings": torch.cat(batches)}, output_path
    )
    print(f"Wrote {output_path}")
    return output_path


def _rank_metrics(
    groups: list[list[int]], records: list[dict[str, Any]], scores: Any
) -> dict[str, float]:
    exact = 0
    tolerant = 0
    for indices in groups:
        selected = indices[int(scores[indices].argmax())]
        exact += bool(records[selected]["acceptable"])
        tolerant += records[selected]["target"] > 0
    count = len(groups)
    return {
        "contacts": count,
        "exact_accuracy": exact / count if count else 0.0,
        "tolerant_accuracy": tolerant / count if count else 0.0,
    }


def train_frame_probe(
    *, work_directory: Path, epochs: int, learning_rate: float
) -> Path:
    """Fit a listwise linear probe whose score depends on one frame at a time."""

    import torch

    records_path = work_directory / "frame-dataset" / "records.json"
    records = json.loads(records_path.read_text())["records"]
    embedding_path = work_directory / "frame-dataset" / "resnet18-embeddings.pt"
    embeddings = torch.load(
        embedding_path, map_location="cpu", weights_only=False
    )["embeddings"].float()
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    groups_by_split: dict[str, list[list[int]]] = defaultdict(list)
    for indices in grouped.values():
        record = records[indices[0]]
        if record["suitability"] != "in_scope" or record["status"] == "unresolved":
            continue
        groups_by_split[record["split"]].append(indices)

    train_indices = [index for group in groups_by_split["train"] for index in group]
    mean = embeddings[train_indices].mean(dim=0)
    std = embeddings[train_indices].std(dim=0).clamp_min(1e-5)
    features = (embeddings - mean) / std
    torch.manual_seed(148)
    scorer = torch.nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(148)
    best_key = (-1.0, -1.0)
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(
            len(groups_by_split["train"]), generator=generator
        )
        total_loss = 0.0
        for position in permutation.tolist():
            indices = groups_by_split["train"][position]
            logits = scorer(features[indices]).squeeze(1)
            targets = torch.tensor(
                [records[index]["target"] for index in indices], dtype=torch.float32
            )
            loss = -(targets * torch.log_softmax(logits, dim=0)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        with torch.no_grad():
            scores = scorer(features).squeeze(1)
        validation = _rank_metrics(groups_by_split["validation"], records, scores)
        key = (validation["tolerant_accuracy"], validation["exact_accuracy"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().clone() for name, value in scorer.state_dict().items()
            }
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"epoch {epoch}: loss={total_loss / len(permutation):.4f}, "
                f"validation={validation}"
            )

    if best_state is None:
        raise RuntimeError("training produced no frame-probe checkpoint")
    scorer.load_state_dict(best_state)
    with torch.no_grad():
        scores = scorer(features).squeeze(1)
    metrics = {
        split: _rank_metrics(groups, records, scores)
        for split, groups in groups_by_split.items()
    }
    output_path = work_directory / "frame-probe.pt"
    torch.save(
        {
            "state_dict": scorer.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "metrics": metrics,
            "best_epoch": best_epoch,
            "records": str(records_path),
            "embeddings": str(embedding_path),
        },
        output_path,
    )
    report_path = work_directory / "frame-probe-metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"best validation checkpoint: epoch {best_epoch}")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path} and {report_path}")
    return output_path


def evaluate_probe_fusion(*, work_directory: Path) -> Path:
    """Tune audio/visual score fusion on validation and report untouched test."""

    import torch

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    embeddings = torch.load(
        work_directory / "frame-dataset" / "resnet18-embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )["embeddings"].float()
    artifact = torch.load(
        work_directory / "frame-probe.pt", map_location="cpu", weights_only=False
    )
    scorer = torch.nn.Linear(embeddings.shape[1], 1)
    scorer.load_state_dict(artifact["state_dict"])
    features = (embeddings - artifact["feature_mean"]) / artifact["feature_std"]
    with torch.no_grad():
        visual_scores = scorer(features).squeeze(1)

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    groups_by_split: dict[str, list[list[int]]] = defaultdict(list)
    for indices in grouped.values():
        record = records[indices[0]]
        if record["suitability"] == "in_scope" and record["status"] != "unresolved":
            groups_by_split[record["split"]].append(indices)

    best: tuple[tuple[float, float], float, float, torch.Tensor] | None = None
    for sigma in (0.025, 0.05, 0.075, 0.1, 0.15, 0.25):
        audio_scores = torch.tensor(
            [
                -0.5 * (float(record["audio_delta"]) / sigma) ** 2
                for record in records
            ]
        )
        for visual_weight in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            combined = audio_scores.clone()
            for indices in grouped.values():
                values = visual_scores[indices]
                normalized = (values - values.mean()) / values.std().clamp_min(1e-5)
                combined[indices] += visual_weight * normalized
            validation = _rank_metrics(
                groups_by_split["validation"], records, combined
            )
            key = (
                validation["tolerant_accuracy"], validation["exact_accuracy"]
            )
            if best is None or key > best[0]:
                best = (key, sigma, visual_weight, combined)
    if best is None:
        raise RuntimeError("fusion search produced no candidate")
    _, sigma, visual_weight, scores = best
    report = {
        "sigma": sigma,
        "visual_weight": visual_weight,
        "validation": _rank_metrics(groups_by_split["validation"], records, scores),
        "test": _rank_metrics(groups_by_split["test"], records, scores),
        "train": _rank_metrics(groups_by_split["train"], records, scores),
    }
    output_path = work_directory / "frame-probe-fusion.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return output_path


def fine_tune_frame_scorer(
    *,
    work_directory: Path,
    device: str | None,
    epochs: int,
    learning_rate: float,
    negative_count: int,
    image_size: int,
) -> Path:
    """Fine-tune the final ResNet block with listwise per-swing supervision."""

    import copy
    import random

    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import v2

    torch.manual_seed(148)
    random.seed(148)
    selected_device = device or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch_device = torch.device(selected_device)
    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    groups_by_split: dict[str, list[list[int]]] = defaultdict(list)
    for indices in grouped.values():
        record = records[indices[0]]
        if record["suitability"] == "in_scope" and record["status"] != "unresolved":
            groups_by_split[record["split"]].append(indices)

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                (image_size, image_size), scale=(0.82, 1.0), ratio=(0.92, 1.08)
            ),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    evaluation_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    def load_images(indices: list[int], transform: Any) -> torch.Tensor:
        images = []
        for index in indices:
            with Image.open(records[index]["image"]) as image:
                images.append(transform(image.convert("RGB")))
        return torch.stack(images).to(torch_device)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    model.to(torch_device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    def score_groups(groups: list[list[int]]) -> tuple[dict[str, float], dict[int, float]]:
        model.eval()
        scores_by_index: dict[int, float] = {}
        with torch.inference_mode():
            for indices in groups:
                logits = model(load_images(indices, evaluation_transform)).squeeze(1)
                scores_by_index.update(
                    (index, float(score))
                    for index, score in zip(indices, logits.cpu(), strict=True)
                )
        scores = torch.full((len(records),), -torch.inf)
        for index, score in scores_by_index.items():
            scores[index] = score
        return _rank_metrics(groups, records, scores), scores_by_index

    best_key = (-1.0, -1.0)
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(groups_by_split["train"])))
        random.Random(148 + epoch).shuffle(order)
        total_loss = 0.0
        for position in order:
            complete_indices = groups_by_split["train"][position]
            complete_records = [records[index] for index in complete_indices]
            local_indices = sorted(
                select_training_indices(
                    complete_records,
                    negative_count=negative_count,
                    seed=epoch * 10_000 + position,
                )
            )
            indices = [complete_indices[index] for index in local_indices]
            targets = torch.tensor(
                [records[index]["target"] for index in indices],
                dtype=torch.float32,
                device=torch_device,
            )
            optimizer.zero_grad()
            logits = model(load_images(indices, train_transform)).squeeze(1)
            loss = listwise_contact_loss(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        validation, _ = score_groups(groups_by_split["validation"])
        key = (validation["tolerant_accuracy"], validation["exact_accuracy"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch}/{epochs}: loss={total_loss / len(order):.4f}, "
            f"validation={validation}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("fine-tuning produced no checkpoint")
    model.load_state_dict(best_state)
    validation, validation_scores = score_groups(groups_by_split["validation"])
    test, test_scores = score_groups(groups_by_split["test"])
    metrics = {
        "best_epoch": best_epoch,
        "validation": validation,
        "test": test,
    }
    output_path = work_directory / "fine-tuned-frame-scorer.pt"
    torch.save(
        {
            "architecture": "resnet18",
            "image_size": image_size,
            "state_dict": model.state_dict(),
            "metrics": metrics,
        },
        output_path,
    )
    (work_directory / "fine-tuned-frame-scores.json").write_text(
        json.dumps(
            {
                "validation": validation_scores,
                "test": test_scores,
            },
            indent=2,
        )
        + "\n"
    )
    (work_directory / "fine-tuned-frame-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return output_path


def _evaluate_stored_fusion(
    *, work_directory: Path, scores_name: str, output_name: str
) -> Path:
    """Tune audio fusion for stored visual scores without consulting test."""

    import torch

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    stored_scores = json.loads(
        (work_directory / scores_name).read_text()
    )
    visual_scores = {
        split: {int(index): float(score) for index, score in values.items()}
        for split, values in stored_scores.items()
    }
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    groups_by_split: dict[str, list[list[int]]] = defaultdict(list)
    for indices in grouped.values():
        record = records[indices[0]]
        if (
            record["split"] in visual_scores
            and record["suitability"] == "in_scope"
            and record["status"] != "unresolved"
        ):
            groups_by_split[record["split"]].append(indices)

    candidates = []
    for sigma in (0.025, 0.05, 0.075, 0.1, 0.15, 0.25):
        for visual_weight in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            metrics_by_split = {}
            for split in ("validation", "test"):
                combined = torch.full((len(records),), -torch.inf)
                for indices in groups_by_split[split]:
                    visual = torch.tensor(
                        [visual_scores[split][index] for index in indices]
                    )
                    visual = (visual - visual.mean()) / visual.std().clamp_min(1e-5)
                    audio = torch.tensor(
                        [
                            -0.5
                            * (float(records[index]["audio_delta"]) / sigma) ** 2
                            for index in indices
                        ]
                    )
                    combined[indices] = audio + visual_weight * visual
                metrics_by_split[split] = _rank_metrics(
                    groups_by_split[split], records, combined
                )
            candidates.append((sigma, visual_weight, metrics_by_split))
    sigma, visual_weight, metrics = max(
        candidates,
        key=lambda candidate: (
            candidate[2]["validation"]["tolerant_accuracy"],
            candidate[2]["validation"]["exact_accuracy"],
            -candidate[1],
        ),
    )
    report = {"sigma": sigma, "visual_weight": visual_weight, **metrics}
    output_path = work_directory / output_name
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return output_path


def evaluate_fine_tuned_fusion(*, work_directory: Path) -> Path:
    return _evaluate_stored_fusion(
        work_directory=work_directory,
        scores_name="fine-tuned-frame-scores.json",
        output_name="fine-tuned-frame-fusion.json",
    )


def evaluate_temporal_fusion(*, work_directory: Path) -> Path:
    return _evaluate_stored_fusion(
        work_directory=work_directory,
        scores_name="temporal-frame-scores.json",
        output_name="temporal-frame-fusion.json",
    )


def train_temporal_probe(
    *, work_directory: Path, epochs: int, learning_rate: float
) -> Path:
    """Train a small temporal convolution over cached per-frame embeddings."""

    import copy
    import random

    import torch

    torch.manual_seed(148)
    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    embeddings = torch.load(
        work_directory / "frame-dataset" / "resnet18-embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )["embeddings"].float()
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    groups_by_split: dict[str, list[list[int]]] = defaultdict(list)
    for indices in grouped.values():
        record = records[indices[0]]
        if record["suitability"] == "in_scope" and record["status"] != "unresolved":
            groups_by_split[record["split"]].append(indices)
    train_indices = [index for group in groups_by_split["train"] for index in group]
    mean = embeddings[train_indices].mean(dim=0)
    std = embeddings[train_indices].std(dim=0).clamp_min(1e-5)
    features = (embeddings - mean) / std

    class TemporalScorer(torch.nn.Module):
        def __init__(self, feature_count: int) -> None:
            super().__init__()
            self.project = torch.nn.Linear(feature_count * 3, 64)
            self.temporal = torch.nn.Sequential(
                torch.nn.Conv1d(64, 64, kernel_size=5, padding=2),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Conv1d(64, 64, kernel_size=5, padding=2),
                torch.nn.GELU(),
            )
            self.head = torch.nn.Conv1d(64, 1, kernel_size=1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            previous = torch.zeros_like(inputs)
            following = torch.zeros_like(inputs)
            previous[:, 1:] = inputs[:, 1:] - inputs[:, :-1]
            following[:, :-1] = inputs[:, 1:] - inputs[:, :-1]
            projected = torch.nn.functional.gelu(
                self.project(torch.cat((inputs, previous, following), dim=-1))
            )
            temporal = self.temporal(projected.transpose(1, 2))
            return self.head(temporal + projected.transpose(1, 2)).squeeze(1)

    def tensors(groups: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.stack([features[indices] for indices in groups])
        targets = torch.tensor(
            [
                [records[index]["target"] for index in indices]
                for indices in groups
            ],
            dtype=torch.float32,
        )
        return inputs, targets

    train_inputs, train_targets = tensors(groups_by_split["train"])
    validation_inputs, _ = tensors(groups_by_split["validation"])
    model = TemporalScorer(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def metrics_for(split: str, scores: torch.Tensor) -> dict[str, float]:
        flat = torch.full((len(records),), -torch.inf)
        for indices, values in zip(groups_by_split[split], scores, strict=True):
            flat[indices] = values
        return _rank_metrics(groups_by_split[split], records, flat)

    best_key = (-1.0, -1.0)
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    batch_size = 16
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_inputs)))
        random.Random(148 + epoch).shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            inputs = train_inputs[selected]
            if epoch > 1:
                inputs = inputs + torch.randn_like(inputs) * 0.015
            targets = train_targets[selected]
            optimizer.zero_grad()
            logits = model(inputs)
            loss = -(
                targets * torch.log_softmax(logits, dim=1)
            ).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
        model.eval()
        with torch.inference_mode():
            validation_scores = model(validation_inputs)
        validation = metrics_for("validation", validation_scores)
        key = (validation["tolerant_accuracy"], validation["exact_accuracy"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"epoch {epoch}: loss={total_loss / len(order):.4f}, "
                f"validation={validation}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("temporal training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    score_payload: dict[str, dict[int, float]] = {}
    metrics: dict[str, Any] = {"best_epoch": best_epoch}
    with torch.inference_mode():
        for split in ("validation", "test"):
            inputs, _ = tensors(groups_by_split[split])
            split_scores = model(inputs)
            metrics[split] = metrics_for(split, split_scores)
            score_payload[split] = {
                index: float(score)
                for indices, scores in zip(
                    groups_by_split[split], split_scores, strict=True
                )
                for index, score in zip(indices, scores, strict=True)
            }
    output_path = work_directory / "temporal-frame-probe.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "metrics": metrics,
        },
        output_path,
    )
    (work_directory / "temporal-frame-scores.json").write_text(
        json.dumps(score_payload, indent=2) + "\n"
    )
    (work_directory / "temporal-frame-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    return output_path


def fine_tune_temporal_scorer(
    *,
    work_directory: Path,
    device: str | None,
    epochs: int,
    learning_rate: float,
    image_size: int,
) -> Path:
    """Jointly tune visual and temporal features across each complete swing."""

    import copy
    import random

    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import v2

    torch.manual_seed(148)
    selected_device = device or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch_device = torch.device(selected_device)
    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    groups_by_split: dict[str, list[list[int]]] = defaultdict(list)
    for indices in grouped.values():
        record = records[indices[0]]
        if record["suitability"] == "in_scope" and record["status"] != "unresolved":
            groups_by_split[record["split"]].append(indices)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )

    def load_sequence(indices: list[int], *, augment: bool) -> torch.Tensor:
        images = []
        for index in indices:
            with Image.open(records[index]["image"]) as image:
                images.append(transform(image.convert("RGB")))
        sequence = torch.stack(images)
        if augment:
            rng = random.random
            if rng() < 0.5:
                sequence = sequence.flip(-1)
            sequence = sequence * (0.9 + 0.2 * rng())
            channel_mean = sequence.mean(dim=(-2, -1), keepdim=True)
            sequence = (sequence - channel_mean) * (0.9 + 0.2 * rng()) + channel_mean
        return sequence.to(torch_device)

    class JointTemporalScorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
            encoder.fc = torch.nn.Identity()
            self.encoder = encoder
            self.project = torch.nn.Linear(512 * 3, 64)
            self.temporal = torch.nn.Sequential(
                torch.nn.Conv1d(64, 64, kernel_size=5, padding=2),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Conv1d(64, 64, kernel_size=5, padding=2),
                torch.nn.GELU(),
            )
            self.head = torch.nn.Conv1d(64, 1, kernel_size=1)

        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            features = self.encoder(sequence).unsqueeze(0)
            previous = torch.zeros_like(features)
            following = torch.zeros_like(features)
            previous[:, 1:] = features[:, 1:] - features[:, :-1]
            following[:, :-1] = features[:, 1:] - features[:, :-1]
            projected = torch.nn.functional.gelu(
                self.project(torch.cat((features, previous, following), dim=-1))
            )
            temporal = self.temporal(projected.transpose(1, 2))
            return self.head(temporal + projected.transpose(1, 2)).flatten()

    model = JointTemporalScorer().to(torch_device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.encoder.layer4.parameters():
        parameter.requires_grad = True
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    def set_training_mode() -> None:
        model.train()
        for module in model.encoder.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def score_groups(
        groups: list[list[int]], split: str
    ) -> tuple[dict[str, float], dict[int, float]]:
        model.eval()
        flat = torch.full((len(records),), -torch.inf)
        stored = {}
        with torch.inference_mode():
            for indices in groups:
                scores = model(load_sequence(indices, augment=False)).cpu()
                flat[indices] = scores
                stored.update(
                    (index, float(score))
                    for index, score in zip(indices, scores, strict=True)
                )
        return _rank_metrics(groups_by_split[split], records, flat), stored

    best_key = (-1.0, -1.0)
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        order = list(range(len(groups_by_split["train"])))
        random.Random(148 + epoch).shuffle(order)
        total_loss = 0.0
        for position in order:
            set_training_mode()
            indices = groups_by_split["train"][position]
            targets = torch.tensor(
                [records[index]["target"] for index in indices],
                dtype=torch.float32,
                device=torch_device,
            )
            optimizer.zero_grad()
            scores = model(load_sequence(indices, augment=True))
            loss = listwise_contact_loss(scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        validation, _ = score_groups(groups_by_split["validation"], "validation")
        key = (validation["tolerant_accuracy"], validation["exact_accuracy"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch {epoch}/{epochs}: loss={total_loss / len(order):.4f}, "
            f"validation={validation}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("joint temporal training produced no checkpoint")
    model.load_state_dict(best_state)
    validation, validation_scores = score_groups(
        groups_by_split["validation"], "validation"
    )
    test, test_scores = score_groups(groups_by_split["test"], "test")
    metrics = {"best_epoch": best_epoch, "validation": validation, "test": test}
    output_path = work_directory / "fine-tuned-temporal-scorer.pt"
    torch.save(
        {
            "architecture": "resnet18-temporal-conv",
            "image_size": image_size,
            "state_dict": model.state_dict(),
            "metrics": metrics,
        },
        output_path,
    )
    (work_directory / "fine-tuned-temporal-scores.json").write_text(
        json.dumps(
            {"validation": validation_scores, "test": test_scores}, indent=2
        )
        + "\n"
    )
    (work_directory / "fine-tuned-temporal-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    return output_path


def build_contact_report(*, work_directory: Path) -> Path:
    """Build a self-contained diagnostic index referencing cached crop images."""

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    stored_scores = json.loads(
        (work_directory / "ensemble-frame-scores.json").read_text()
    )
    visual_scores = {
        split: {int(index): float(score) for index, score in values.items()}
        for split, values in stored_scores.items()
    }
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    contacts = []
    report_path = work_directory / "contact-results.html"

    def image_path(record: dict[str, Any]) -> str:
        return str(Path(record["image"]).relative_to(work_directory.resolve()))

    for indices in grouped.values():
        first = records[indices[0]]
        if (
            first["split"] not in {"validation", "test"}
            or first["suitability"] != "in_scope"
            or first["status"] == "unresolved"
        ):
            continue
        selected_position = max(
            range(len(indices)),
            key=lambda position: visual_scores[first["split"]][indices[position]],
        )
        selected_index = indices[selected_position]
        selected = records[selected_index]
        exact_records = [records[index] for index in indices if records[index]["acceptable"]]
        exact_states = sorted({int(record["visual_state"]) for record in exact_records})
        selected_state = int(selected["visual_state"])
        state_error = min(
            (selected_state - state for state in exact_states), key=abs
        )
        closest_exact = min(
            exact_records,
            key=lambda record: abs(int(record["ordinal"]) - int(selected["ordinal"])),
        )
        lower = max(0, selected_position - 3)
        upper = min(len(indices), selected_position + 4)
        neighborhood = [records[index] for index in indices[lower:upper]]
        contacts.append(
            {
                "video": Path(first["video"]).name,
                "shot_index": int(first["shot_index"]),
                "split": first["split"],
                "status": first["status"],
                "outcome": classify_selection(selected),
                "selected": {
                    "image": image_path(selected),
                    "ordinal": int(selected["ordinal"]),
                    "state": selected_state,
                },
                "reviewed": [
                    {
                        "image": image_path(record),
                        "ordinal": int(record["ordinal"]),
                        "state": int(record["visual_state"]),
                    }
                    for record in exact_records
                ],
                "neighborhood": [
                    {
                        "image": image_path(record),
                        "ordinal": int(record["ordinal"]),
                        "selected": record is selected,
                        "reviewed": bool(record["acceptable"]),
                    }
                    for record in neighborhood
                ],
                "state_error": state_error,
                "timing_error_ms": round(
                    (float(selected["frame_time"]) - float(closest_exact["frame_time"]))
                    * 1000,
                    1,
                ),
                "audio_error_ms": round(
                    -float(selected["audio_delta"])
                    + float(selected["frame_time"])
                    - float(closest_exact["frame_time"]),
                    4,
                )
                * 1000,
            }
        )
    contacts.sort(
        key=lambda contact: (
            {"failed": 0, "tolerated": 1, "exact": 2}[contact["outcome"]],
            contact["split"],
            contact["video"],
            contact["shot_index"],
        )
    )
    data = json.dumps(contacts, separators=(",", ":")).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contact-frame results</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #0c111b; color: #e7edf7; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 18px 24px; background: #121a29f2; border-bottom: 1px solid #29344a; }}
h1 {{ margin: 0 0 5px; font-size: 22px; }}
.sub {{ color: #aab6ca; font-size: 13px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 14px; align-items: center; }}
select, input {{ background: #0c111b; color: #e7edf7; border: 1px solid #3a4965; border-radius: 6px; padding: 7px 9px; }}
#summary {{ margin-left: auto; font-variant-numeric: tabular-nums; }}
main {{ display: grid; gap: 16px; padding: 20px; }}
.card {{ background: #151e2e; border: 1px solid #2b3850; border-left: 5px solid; border-radius: 9px; padding: 16px; }}
.card.exact {{ border-left-color: #34c77b; }} .card.tolerated {{ border-left-color: #e7b34c; }} .card.failed {{ border-left-color: #ed6670; }}
.title {{ display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: baseline; margin-bottom: 13px; }}
.badge {{ border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 700; text-transform: uppercase; }}
.exact .badge {{ background: #123c2b; color: #78e8aa; }} .tolerated .badge {{ background: #493817; color: #ffd37a; }} .failed .badge {{ background: #4a2027; color: #ff9299; }}
.comparison {{ display: grid; grid-template-columns: minmax(220px, 1fr) minmax(220px, 2fr); gap: 16px; }}
.panel {{ background: #0e1522; border-radius: 7px; padding: 10px; }}
.panel h3 {{ font-size: 13px; color: #afbdd2; margin: 0 0 8px; }}
.frames {{ display: flex; flex-wrap: wrap; gap: 8px; }}
figure {{ margin: 0; }} figure img {{ display: block; width: min(300px, 100%); border-radius: 5px; }}
figcaption {{ color: #aab6ca; font-size: 11px; margin-top: 4px; }}
.strip {{ display: grid; grid-template-columns: repeat(7, minmax(75px, 1fr)); gap: 5px; margin-top: 12px; }}
.strip img {{ width: 100%; border: 2px solid transparent; border-radius: 3px; box-sizing: border-box; }}
.strip .picked img {{ border-color: #66a8ff; }} .strip .truth img {{ outline: 2px solid #34c77b; }}
.metrics {{ color: #aab6ca; font-size: 12px; margin-top: 10px; }}
@media (max-width: 760px) {{ .comparison {{ grid-template-columns: 1fr; }} .strip {{ grid-template-columns: repeat(4, 1fr); }} #summary {{ width: 100%; margin-left: 0; }} }}
</style>
</head>
<body>
<header><h1>Contact-frame results</h1>
<div class="sub">Ordinary + family-balanced kinematic ensemble · held-out in-scope resolved contacts</div>
<div class="controls">
<label>Outcome <select id="outcome"><option value="all">All</option><option value="failed" selected>Failed</option><option value="tolerated">Tolerated neighbor</option><option value="exact">Exact</option></select></label>
<label>Split <select id="split"><option value="all">Validation + test</option><option value="validation">Validation</option><option value="test">Test</option></select></label>
<label>Find <input id="search" placeholder="video or shot"></label><span id="summary"></span>
</div></header><main id="cards"></main>
<script>
const contacts={data};
const escapeHtml=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const figure=(frame,label)=>`<figure><a href="${{frame.image}}"><img loading="lazy" src="${{frame.image}}"></a><figcaption>${{escapeHtml(label)}} · frame ${{frame.ordinal}} · state ${{frame.state}}</figcaption></figure>`;
function render() {{
 const outcome=document.querySelector('#outcome').value, split=document.querySelector('#split').value, search=document.querySelector('#search').value.toLowerCase();
 const shown=contacts.filter(c=>(outcome==='all'||c.outcome===outcome)&&(split==='all'||c.split===split)&&(`${{c.video}} ${{c.shot_index}}`.toLowerCase().includes(search)));
 document.querySelector('#summary').textContent=`${{shown.length}} shown · ${{shown.filter(c=>c.outcome==='exact').length}} exact · ${{shown.filter(c=>c.outcome==='tolerated').length}} tolerated · ${{shown.filter(c=>c.outcome==='failed').length}} failed`;
 document.querySelector('#cards').innerHTML=shown.map(c=>`<article class="card ${{c.outcome}}">
 <div class="title"><span class="badge">${{c.outcome}}</span><strong>${{escapeHtml(c.video)}} · shot ${{c.shot_index}}</strong><span>${{c.split}} · ${{c.status.replace('_',' ')}}</span></div>
 <div class="comparison"><section class="panel"><h3>Selected frame</h3>${{figure(c.selected,'prediction')}}</section><section class="panel"><h3>Reviewed contact state${{c.reviewed.length>1?'s':''}}</h3><div class="frames">${{c.reviewed.map((f,i)=>figure(f,`reviewed ${{i+1}}`)).join('')}}</div></section></div>
 <div class="strip">${{c.neighborhood.map(f=>`<figure class="${{f.selected?'picked ':''}}${{f.reviewed?'truth':''}}"><a href="${{f.image}}"><img loading="lazy" src="${{f.image}}"></a><figcaption>${{f.ordinal}}</figcaption></figure>`).join('')}}</div>
 <div class="metrics">Prediction offset: ${{c.state_error>0?'+':''}}${{c.state_error}} visual states · ${{c.timing_error_ms>0?'+':''}}${{c.timing_error_ms}} ms · audio-to-contact offset ${{c.audio_error_ms>0?'+':''}}${{c.audio_error_ms.toFixed(1)}} ms</div>
 </article>`).join('') || '<p>No contacts match these filters.</p>';
}}
document.querySelectorAll('select,input').forEach(element=>element.addEventListener('input',render)); render();
</script></body></html>"""
    report_path.write_text(html)
    counts = Counter(contact["outcome"] for contact in contacts)
    print(f"Wrote {report_path}: {dict(counts)}")
    return report_path


def prepare_object_detections(
    *, work_directory: Path, device: str | None, batch_size: int
) -> Path:
    """Cache COCO player, ball, and racket observations on stable crops."""

    import cv2
    from ultralytics import YOLO

    records_path = work_directory / "frame-dataset" / "records.json"
    records = json.loads(records_path.read_text())["records"]
    output_path = work_directory / "object-detections.json"
    partial_path = work_directory / "object-detections.partial.json"
    identity = {
        "records_size": records_path.stat().st_size,
        "records_mtime_ns": records_path.stat().st_mtime_ns,
        "model": str(Path("yolov8n.pt").resolve()),
        "classes": [0, 32, 38],
        "confidence": 0.05,
        "image_size": 640,
    }
    if output_path.is_file():
        cached = json.loads(output_path.read_text())
        if cached.get("identity") == identity:
            print(f"Using cached detections: {output_path}")
            return output_path

    detected_records = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text())
        if partial.get("identity") == identity:
            detected_records = partial.get("detections", [])
            print(f"Resuming after {len(detected_records)} cached frames")

    model = YOLO("yolov8n.pt")
    names = {0: "person", 32: "ball", 38: "racket"}
    for start in range(len(detected_records), len(records), batch_size):
        batch = records[start : start + batch_size]
        images = [cv2.imread(record["image"], cv2.IMREAD_COLOR) for record in batch]
        if any(image is None for image in images):
            raise ValueError("could not read frame-dataset image")
        results = model.predict(
            images,
            classes=list(names),
            conf=0.05,
            iou=0.5,
            imgsz=640,
            device=device,
            verbose=False,
        )
        for result in results:
            detections = []
            if result.boxes is not None:
                for box, kind, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    strict=True,
                ):
                    detections.append(
                        {
                            "kind": names[int(kind)],
                            "box": [round(float(value), 3) for value in box],
                            "confidence": round(float(confidence), 6),
                        }
                    )
            detected_records.append(detections)
        completed = min(start + batch_size, len(records))
        partial_path.write_text(
            json.dumps(
                {"identity": identity, "detections": detected_records},
                separators=(",", ":"),
            )
            + "\n"
        )
        if completed % (batch_size * 10) == 0 or completed == len(records):
            print(f"detected {completed}/{len(records)}", flush=True)
    output_path.write_text(
        json.dumps(
            {"identity": identity, "detections": detected_records},
            separators=(",", ":"),
        )
        + "\n"
    )
    partial_path.unlink(missing_ok=True)
    print(f"Wrote {output_path}")
    return output_path


def prepare_pose_detections(
    *, work_directory: Path, device: str | None, batch_size: int
) -> Path:
    """Cache COCO body keypoints on every stable player crop."""

    import cv2
    from ultralytics import YOLO

    records_path = work_directory / "frame-dataset" / "records.json"
    records = json.loads(records_path.read_text())["records"]
    output_path = work_directory / "pose-detections.json"
    partial_path = work_directory / "pose-detections.partial.json"
    identity = {
        "records_size": records_path.stat().st_size,
        "records_mtime_ns": records_path.stat().st_mtime_ns,
        "model": str(Path("yolov8n-pose.pt").resolve()),
        "confidence": 0.1,
        "image_size": 640,
    }
    if output_path.is_file():
        cached = json.loads(output_path.read_text())
        if cached.get("identity") == identity:
            print(f"Using cached poses: {output_path}")
            return output_path

    detected_records = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text())
        if partial.get("identity") == identity:
            detected_records = partial.get("detections", [])
            print(f"Resuming after {len(detected_records)} cached frames")

    model = YOLO("yolov8n-pose.pt")
    for start in range(len(detected_records), len(records), batch_size):
        batch = records[start : start + batch_size]
        images = [cv2.imread(record["image"], cv2.IMREAD_COLOR) for record in batch]
        if any(image is None for image in images):
            raise ValueError("could not read frame-dataset image")
        results = model.predict(
            images,
            conf=0.1,
            iou=0.5,
            imgsz=640,
            device=device,
            verbose=False,
        )
        for result in results:
            people = []
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                box_confidences = result.boxes.conf.cpu().tolist()
                points = result.keypoints.xy.cpu().tolist()
                point_confidences = result.keypoints.conf
                confidences = (
                    point_confidences.cpu().tolist()
                    if point_confidences is not None
                    else [[0.0] * len(person) for person in points]
                )
                for box, box_confidence, person, person_confidences in zip(
                    boxes, box_confidences, points, confidences, strict=True
                ):
                    people.append(
                        {
                            "box": [round(float(value), 3) for value in box],
                            "confidence": round(float(box_confidence), 6),
                            "keypoints": [
                                [
                                    round(float(point[0]), 3),
                                    round(float(point[1]), 3),
                                    round(float(confidence), 6),
                                ]
                                for point, confidence in zip(
                                    person, person_confidences, strict=True
                                )
                            ],
                        }
                    )
            detected_records.append(people)
        completed = min(start + batch_size, len(records))
        partial_path.write_text(
            json.dumps(
                {"identity": identity, "detections": detected_records},
                separators=(",", ":"),
            )
            + "\n"
        )
        if completed % (batch_size * 10) == 0 or completed == len(records):
            print(f"posed {completed}/{len(records)}", flush=True)
    output_path.write_text(
        json.dumps(
            {"identity": identity, "detections": detected_records},
            separators=(",", ":"),
        )
        + "\n"
    )
    partial_path.unlink(missing_ok=True)
    print(f"Wrote {output_path}")
    return output_path


def _visual_frames_for_indices(
    records: list[dict[str, Any]],
    detections: list[list[dict[str, Any]]],
    indices: list[int],
) -> tuple[Any, ...]:
    from tennis_cut.visual_contact import Detection, FrameEvidence, VisualFrame

    return tuple(
        VisualFrame(
            FrameEvidence(
                ordinal=int(records[index]["ordinal"]),
                timestamp=Fraction(str(records[index]["frame_time"])),
                detections=tuple(
                    Detection(
                        item["kind"],
                        tuple(float(value) for value in item["box"]),
                        float(item["confidence"]),
                    )
                    for item in detections[index]
                ),
            )
        )
        for index in indices
    )


def evaluate_structured_baseline(*, work_directory: Path) -> Path:
    """Evaluate the production scoring rules on cached stable-crop detections."""

    from tennis_cut.visual_contact import rank_contact_frames, select_contact_frame

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    detections = json.loads(
        (work_directory / "object-detections.json").read_text()
    )["detections"]
    if len(detections) != len(records):
        raise ValueError("object detection cache does not match frame records")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)

    details = []
    for indices in grouped.values():
        first = records[indices[0]]
        if first["suitability"] != "in_scope":
            continue
        frames = _visual_frames_for_indices(records, detections, indices)
        by_ordinal = {int(records[index]["ordinal"]): records[index] for index in indices}
        kinds = Counter(
            item["kind"] for index in indices for item in detections[index]
        )
        exact_indices = [index for index in indices if records[index]["acceptable"]]
        exact_kinds = Counter(
            item["kind"] for index in exact_indices for item in detections[index]
        )
        deterministic = rank_contact_frames(frames)
        selection = select_contact_frame(frames)

        def outcome(ordinal: int | None) -> str:
            if ordinal is None:
                return "omitted"
            return classify_selection(by_ordinal[ordinal])

        details.append(
            {
                "video": first["video"],
                "shot_index": int(first["shot_index"]),
                "split": first["split"],
                "status": first["status"],
                "has_person": kinds["person"] > 0,
                "has_racket": kinds["racket"] > 0,
                "has_ball": kinds["ball"] > 0,
                "contact_has_racket": exact_kinds["racket"] > 0,
                "contact_has_ball": exact_kinds["ball"] > 0,
                "deterministic_frame": deterministic.selected_frame,
                "deterministic_outcome": outcome(deterministic.selected_frame),
                "deterministic_reason": deterministic.omission_reason,
                "production_frame": selection.selected_frame,
                "production_outcome": outcome(selection.selected_frame),
                "production_reason": selection.omission_reason,
            }
        )

    report: dict[str, Any] = {"contacts": details, "summary": {}}
    for split in ("train", "validation", "test"):
        selected = [
            item
            for item in details
            if item["split"] == split and item["status"] != "unresolved"
        ]
        report["summary"][split] = {
            "contacts": len(selected),
            "any_racket": sum(item["has_racket"] for item in selected),
            "any_ball": sum(item["has_ball"] for item in selected),
            "contact_racket": sum(item["contact_has_racket"] for item in selected),
            "contact_ball": sum(item["contact_has_ball"] for item in selected),
            "deterministic_outcomes": dict(
                Counter(item["deterministic_outcome"] for item in selected)
            ),
            "deterministic_reasons": dict(
                Counter(item["deterministic_reason"] or "accepted" for item in selected)
            ),
            "production_outcomes": dict(
                Counter(item["production_outcome"] for item in selected)
            ),
            "production_reasons": dict(
                Counter(item["production_reason"] or "accepted" for item in selected)
            ),
        }
    output_path = work_directory / "structured-baseline.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return output_path


def _kinematic_feature_vectors(
    frames: tuple[Any, ...], *, racket_only: bool = False
) -> tuple[tuple[float, ...], ...]:
    """Add explicit body-relative racket/ball motion to production evidence."""

    import math

    from tennis_cut.temporal_ranker import CONTEXT_RADIUS
    from tennis_cut.visual_contact import (
        _center,
        _moving_ball_center,
        _prepare_frames,
        extract_temporal_features,
    )

    prepared = _prepare_frames(frames)
    production = extract_temporal_features(frames)
    racket_positions: list[tuple[float, float] | None] = []
    ball_positions: list[tuple[float, float] | None] = []
    racket_quality = []
    ball_quality = []
    gaps = []
    for frame in prepared:
        if frame.player is None:
            racket_positions.append(None)
            ball_positions.append(None)
            racket_quality.append(0.0)
            ball_quality.append(0.0)
            gaps.append(float("nan"))
            continue
        player_center = _center(frame.player.box)
        racket = frame.rackets[0] if frame.rackets else None
        racket_center = None if racket is None else _center(racket["box"])
        ball_center = _moving_ball_center(frame)

        def relative(point: tuple[float, float] | None) -> tuple[float, float] | None:
            if point is None:
                return None
            return (
                (point[0] - player_center[0]) / frame.diagonal,
                (point[1] - player_center[1]) / frame.diagonal,
            )

        relative_racket = relative(racket_center)
        relative_ball = relative(ball_center)
        racket_positions.append(relative_racket)
        ball_positions.append(relative_ball)
        racket_quality.append(0.0 if racket is None else float(racket["quality"]))
        usable_balls = [
            ball for ball in frame.balls if float(ball.get("quality", 0.0)) >= 0.05
        ]
        ball_quality.append(
            max((float(ball["quality"]) for ball in usable_balls), default=0.0)
        )
        gaps.append(
            float("nan")
            if relative_racket is None or relative_ball is None
            else math.dist(relative_racket, relative_ball)
        )

    def motion(
        positions: list[tuple[float, float] | None], index: int
    ) -> tuple[float, float, float, float]:
        previous = next(
            (
                (other, positions[other])
                for other in range(index - 1, max(-1, index - 4), -1)
                if positions[other] is not None
            ),
            None,
        )
        following = next(
            (
                (other, positions[other])
                for other in range(index + 1, min(len(positions), index + 4))
                if positions[other] is not None
            ),
            None,
        )
        current = positions[index]
        if previous is None or following is None:
            return (float("nan"),) * 4
        previous_index, previous_point = previous
        following_index, following_point = following
        assert previous_point is not None and following_point is not None
        dx = (following_point[0] - previous_point[0]) / (
            following_index - previous_index
        )
        dy = (following_point[1] - previous_point[1]) / (
            following_index - previous_index
        )
        speed = math.hypot(dx, dy)
        if current is None:
            return abs(dx), abs(dy), speed, float("nan")
        incoming = (
            (current[0] - previous_point[0]) / (index - previous_index),
            (current[1] - previous_point[1]) / (index - previous_index),
        )
        outgoing = (
            (following_point[0] - current[0]) / (following_index - index),
            (following_point[1] - current[1]) / (following_index - index),
        )
        denominator = max(1e-8, math.hypot(*incoming) * math.hypot(*outgoing))
        turn = sum(a * b for a, b in zip(incoming, outgoing, strict=True)) / denominator
        return abs(dx), abs(dy), speed, turn

    base = []
    for index, feature in enumerate(production):
        racket = racket_positions[index]
        ball = ball_positions[index]
        racket_motion = motion(racket_positions, index)
        ball_motion = motion(ball_positions, index)
        previous_gap = gaps[index - 1] if index else float("nan")
        following_gap = gaps[index + 1] if index + 1 < len(gaps) else float("nan")
        gap_change = (
            float("nan")
            if math.isnan(previous_gap) or math.isnan(following_gap)
            else abs(following_gap - previous_gap) / 2
        )
        racket_values = (
                float(racket is None),
                racket_quality[index],
                float("nan") if racket is None else abs(racket[0]),
                float("nan") if racket is None else racket[1],
                float("nan") if racket is None else math.hypot(*racket),
                *racket_motion,
        )
        ball_values = (
                float(ball is None),
                ball_quality[index],
                float("nan") if ball is None else abs(ball[0]),
                float("nan") if ball is None else ball[1],
                *ball_motion,
                gaps[index],
                gap_change,
        )
        base.append(
            ((feature.racket_quality,) if racket_only else feature.values)
            + racket_values
            + (() if racket_only else ball_values)
        )
    vectors = []
    for index in range(len(base)):
        vector = []
        for offset in range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1):
            vector.extend(base[min(len(base) - 1, max(0, index + offset))])
        vectors.append(tuple(vector))
    return tuple(vectors)


def _pose_feature_vectors(
    frames: tuple[Any, ...], pose_detections: list[list[dict[str, Any]]]
) -> tuple[tuple[float, ...], ...]:
    """Describe arm mechanics independently of ball and racket visibility."""

    import math

    from tennis_cut.temporal_ranker import CONTEXT_RADIUS
    from tennis_cut.visual_contact import _center, _prepare_frames

    prepared = _prepare_frames(frames)

    def iou(first: tuple[float, ...], second: list[float]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(
            0.0, first[3] - first[1]
        )
        second_area = max(0.0, second[2] - second[0]) * max(
            0.0, second[3] - second[1]
        )
        return intersection / max(1e-8, first_area + second_area - intersection)

    selected_poses: list[dict[str, Any] | None] = []
    wrist_tracks: list[list[tuple[float, float] | None]] = [[], []]
    for frame, people in zip(prepared, pose_detections, strict=True):
        if not people:
            selected = None
        elif frame.player is None:
            selected = max(
                people,
                key=lambda person: (person["box"][2] - person["box"][0])
                * (person["box"][3] - person["box"][1]),
            )
        else:
            selected = max(people, key=lambda person: iou(frame.player.box, person["box"]))
        selected_poses.append(selected)
        for track, keypoint_index in zip(wrist_tracks, (9, 10), strict=True):
            if selected is None:
                track.append(None)
                continue
            point = selected["keypoints"][keypoint_index]
            track.append(
                (float(point[0]), float(point[1]))
                if float(point[2]) >= 0.15
                else None
            )

    def motion(
        positions: list[tuple[float, float] | None], index: int, scale: float
    ) -> tuple[float, float, float, float]:
        previous = next(
            (
                (other, positions[other])
                for other in range(index - 1, max(-1, index - 4), -1)
                if positions[other] is not None
            ),
            None,
        )
        following = next(
            (
                (other, positions[other])
                for other in range(index + 1, min(len(positions), index + 4))
                if positions[other] is not None
            ),
            None,
        )
        current = positions[index]
        if previous is None or following is None:
            return (float("nan"),) * 4
        previous_index, previous_point = previous
        following_index, following_point = following
        assert previous_point is not None and following_point is not None
        incoming = None
        outgoing = None
        if current is not None:
            incoming = (
                (current[0] - previous_point[0]) / (index - previous_index),
                (current[1] - previous_point[1]) / (index - previous_index),
            )
            outgoing = (
                (following_point[0] - current[0]) / (following_index - index),
                (following_point[1] - current[1]) / (following_index - index),
            )
        dx = (following_point[0] - previous_point[0]) / (
            (following_index - previous_index) * scale
        )
        dy = (following_point[1] - previous_point[1]) / (
            (following_index - previous_index) * scale
        )
        turn = float("nan")
        if incoming is not None and outgoing is not None:
            denominator = max(1e-8, math.hypot(*incoming) * math.hypot(*outgoing))
            turn = sum(
                first * second
                for first, second in zip(incoming, outgoing, strict=True)
            ) / denominator
        return abs(dx), abs(dy), math.hypot(dx, dy), turn

    def point(
        keypoints: list[list[float]], index: int
    ) -> tuple[float, float] | None:
        value = keypoints[index]
        return (
            (float(value[0]), float(value[1]))
            if float(value[2]) >= 0.15
            else None
        )

    def distance(
        first: tuple[float, float] | None,
        second: tuple[float, float] | None,
        scale: float,
    ) -> float:
        return (
            float("nan")
            if first is None or second is None
            else math.dist(first, second) / scale
        )

    base = []
    for frame_index, (frame, pose) in enumerate(
        zip(prepared, selected_poses, strict=True)
    ):
        if pose is None:
            base.append((1.0,) + (float("nan"),) * 36)
            continue
        keypoints = pose["keypoints"]
        shoulders = (point(keypoints, 5), point(keypoints, 6))
        hips = (point(keypoints, 11), point(keypoints, 12))
        visible_torso = [item for item in (*shoulders, *hips) if item is not None]
        torso_center = (
            _center(tuple(pose["box"]))
            if not visible_torso
            else (
                sum(item[0] for item in visible_torso) / len(visible_torso),
                sum(item[1] for item in visible_torso) / len(visible_torso),
            )
        )
        scale = max(1.0, frame.diagonal)
        torso_values = (
            0.0,
            float(pose["confidence"]),
            sum(float(item[2]) for item in keypoints) / len(keypoints),
            distance(shoulders[0], shoulders[1], scale),
            distance(hips[0], hips[1], scale),
            distance(shoulders[0], hips[0], scale),
            distance(shoulders[1], hips[1], scale),
            float("nan")
            if None in shoulders
            else (shoulders[1][1] - shoulders[0][1]) / scale,
            float("nan")
            if None in hips
            else (hips[1][1] - hips[0][1]) / scale,
        )
        arm_values = []
        for arm_index, (shoulder_index, elbow_index, wrist_index) in enumerate(
            ((5, 7, 9), (6, 8, 10))
        ):
            shoulder = point(keypoints, shoulder_index)
            elbow = point(keypoints, elbow_index)
            wrist = point(keypoints, wrist_index)
            bend = float("nan")
            if shoulder is not None and elbow is not None and wrist is not None:
                upper = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
                lower = (wrist[0] - elbow[0], wrist[1] - elbow[1])
                denominator = max(1e-8, math.hypot(*upper) * math.hypot(*lower))
                bend = sum(
                    first * second
                    for first, second in zip(upper, lower, strict=True)
                ) / denominator
            arm_values.extend(
                (
                    float(shoulder is None or elbow is None or wrist is None),
                    float(keypoints[shoulder_index][2]),
                    float(keypoints[elbow_index][2]),
                    float(keypoints[wrist_index][2]),
                    float("nan")
                    if wrist is None
                    else abs(wrist[0] - torso_center[0]) / scale,
                    float("nan")
                    if wrist is None
                    else (wrist[1] - torso_center[1]) / scale,
                    distance(shoulder, elbow, scale),
                    distance(elbow, wrist, scale),
                    distance(shoulder, wrist, scale),
                    bend,
                    *motion(wrist_tracks[arm_index], frame_index, scale),
                )
            )
        values = torso_values + tuple(arm_values)
        assert len(values) == 37
        base.append(values)

    vectors = []
    for index in range(len(base)):
        vector = []
        for offset in range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1):
            vector.extend(base[min(len(base) - 1, max(0, index + offset))])
        vectors.append(tuple(vector))
    return tuple(vectors)


def train_structured_ranker(
    *,
    work_directory: Path,
    kinematic: bool = False,
    racket_only: bool = False,
    pose: bool = False,
    pose_only: bool = False,
    family_balanced: bool = False,
) -> Path:
    """Fit a grouped tree ranker over the production structured feature context."""

    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    from tennis_cut.temporal_ranker import all_temporal_feature_vectors
    from tennis_cut.visual_contact import extract_temporal_features

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    detections = json.loads(
        (work_directory / "object-detections.json").read_text()
    )["detections"]
    poses = (
        json.loads((work_directory / "pose-detections.json").read_text())[
            "detections"
        ]
        if pose or pose_only
        else None
    )
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    groups_by_split: dict[str, list[list[int]]] = defaultdict(list)
    vectors_by_index: dict[int, tuple[float, ...]] = {}
    for indices in grouped.values():
        first = records[indices[0]]
        if first["suitability"] != "in_scope" or first["status"] == "unresolved":
            continue
        frames = _visual_frames_for_indices(records, detections, indices)
        if pose or pose_only:
            assert poses is not None
            pose_vectors = _pose_feature_vectors(
                frames, [poses[index] for index in indices]
            )
            if pose_only:
                vectors = pose_vectors
            else:
                kinematic_vectors = _kinematic_feature_vectors(frames)
                vectors = tuple(
                    kinematic_vector + pose_vector
                    for kinematic_vector, pose_vector in zip(
                        kinematic_vectors, pose_vectors, strict=True
                    )
                )
        elif kinematic:
            vectors = _kinematic_feature_vectors(frames, racket_only=racket_only)
        else:
            vectors = all_temporal_feature_vectors(extract_temporal_features(frames))
        vectors_by_index.update(zip(indices, vectors, strict=True))
        groups_by_split[first["split"]].append(indices)

    train_groups = groups_by_split["train"]
    train_indices = [index for group in train_groups for index in group]
    x_train = np.asarray([vectors_by_index[index] for index in train_indices])
    y_train = np.asarray([records[index]["target"] for index in train_indices])
    validation_indices = [
        index for group in groups_by_split["validation"] for index in group
    ]
    x_validation = np.asarray(
        [vectors_by_index[index] for index in validation_indices]
    )

    best: tuple[tuple[float, float], dict[str, Any], Any] | None = None
    train_family_counts = Counter(
        records[group[0]]["family"] for group in train_groups
    )
    for positive_fraction in (0.5, 0.7, 0.85):
        weights = []
        for group in train_groups:
            positives = sum(records[index]["target"] > 0 for index in group)
            negatives = len(group) - positives
            family_scale = (
                1 / train_family_counts[records[group[0]]["family"]]
                if family_balanced
                else 1.0
            )
            weights.extend(
                family_scale * positive_fraction / positives
                if records[index]["target"] > 0
                else family_scale * (1 - positive_fraction) / negatives
                for index in group
            )
        for max_leaf_nodes in (7, 15, 31):
            for l2_regularization in (1.0, 5.0, 10.0):
                model = HistGradientBoostingRegressor(
                    max_iter=160,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    random_state=148,
                ).fit(x_train, y_train, sample_weight=np.asarray(weights))
                predictions = model.predict(x_validation)
                flat = np.full(len(records), -np.inf)
                flat[validation_indices] = predictions
                metrics = _rank_metrics(
                    groups_by_split["validation"], records, flat
                )
                key = (metrics["tolerant_accuracy"], metrics["exact_accuracy"])
                parameters = {
                    "positive_fraction": positive_fraction,
                    "max_leaf_nodes": max_leaf_nodes,
                    "l2_regularization": l2_regularization,
                }
                if best is None or key > best[0]:
                    best = (key, parameters, model)
    if best is None:
        raise RuntimeError("structured ranker search produced no model")
    _, parameters, model = best
    score_payload = {}
    metrics = {"parameters": parameters}
    for split in ("validation", "test"):
        indices = [index for group in groups_by_split[split] for index in group]
        predictions = model.predict(
            np.asarray([vectors_by_index[index] for index in indices])
        )
        flat = np.full(len(records), -np.inf)
        flat[indices] = predictions
        metrics[split] = _rank_metrics(groups_by_split[split], records, flat)
        score_payload[split] = {
            index: float(score)
            for index, score in zip(indices, predictions, strict=True)
        }
    prefix = (
        "pose-only"
        if pose_only
        else "pose-kinematic"
        if pose
        else "family-kinematic"
        if family_balanced
        else "racket"
        if racket_only
        else "kinematic"
        if kinematic
        else "structured"
    )
    model_path = work_directory / f"{prefix}-ranker.joblib"
    joblib.dump(
        {"model": model, "parameters": parameters, "feature_version": 1}, model_path
    )
    (work_directory / f"{prefix}-frame-scores.json").write_text(
        json.dumps(score_payload, indent=2) + "\n"
    )
    output_path = work_directory / f"{prefix}-ranker-metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return output_path


def evaluate_kinematic_confidence(*, work_directory: Path) -> Path:
    """Calibrate abstention on validation and audit test and unsupported views."""

    import joblib
    import numpy as np

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    detections = json.loads(
        (work_directory / "object-detections.json").read_text()
    )["detections"]
    model = joblib.load(work_directory / "kinematic-ranker.joblib")["model"]
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)

    entries = []
    for indices in grouped.values():
        frames = _visual_frames_for_indices(records, detections, indices)
        vectors = _kinematic_feature_vectors(frames)
        scores = np.asarray(model.predict(np.asarray(vectors)), dtype=float)
        selected_position = int(np.argmax(scores))
        selected_index = indices[selected_position]
        selected = records[selected_index]
        selected_state = int(selected["visual_state"])
        outside = [
            score
            for index, score in zip(indices, scores, strict=True)
            if abs(int(records[index]["visual_state"]) - selected_state) > 1
        ]
        confidence = float(scores[selected_position] - max(outside, default=0.0))
        resolved = selected["status"] != "unresolved"
        entries.append(
            {
                "video": selected["video"],
                "shot_index": int(selected["shot_index"]),
                "split": selected["split"],
                "suitability": selected["suitability"],
                "status": selected["status"],
                "selected_ordinal": int(selected["ordinal"]),
                "outcome": classify_selection(selected) if resolved else "unresolved",
                "correct": bool(resolved and selected["target"] > 0),
                "exact": bool(resolved and selected["acceptable"]),
                "confidence": confidence,
            }
        )

    validation = [
        entry
        for entry in entries
        if entry["suitability"] == "in_scope" and entry["split"] == "validation"
    ]
    threshold = choose_review_threshold(validation)

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        resolved = [entry for entry in selected if entry["status"] != "unresolved"]
        accepted = [entry for entry in selected if entry["confidence"] >= threshold]
        accepted_resolved = [
            entry for entry in accepted if entry["status"] != "unresolved"
        ]
        correct = sum(entry["correct"] for entry in accepted)
        return {
            "contacts": len(selected),
            "resolved": len(resolved),
            "accepted": len(accepted),
            "accepted_resolved": len(accepted_resolved),
            "correct_accepted": correct,
            "accepted_precision": correct / len(accepted) if accepted else 1.0,
            "resolved_correct_recall": correct / len(resolved) if resolved else 0.0,
            "exact_resolved_recall": (
                sum(entry["exact"] for entry in accepted_resolved) / len(resolved)
                if resolved
                else 0.0
            ),
            "unresolved_false_accepts": sum(
                entry["status"] == "unresolved" for entry in accepted
            ),
        }

    report = {
        "threshold": threshold,
        "validation": summarize(validation),
        "test": summarize(
            [
                entry
                for entry in entries
                if entry["suitability"] == "in_scope" and entry["split"] == "test"
            ]
        ),
        "out_of_scope": summarize(
            [entry for entry in entries if entry["suitability"] != "in_scope"]
        ),
        "contacts": entries,
    }
    output_path = work_directory / "kinematic-confidence.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "contacts"}, indent=2))
    return output_path


def evaluate_ambiguity_anomaly(*, work_directory: Path) -> Path:
    """Test one-class evidence quality as a contact-localizability gate."""

    import math

    import joblib
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    from tennis_cut.visual_contact import _moving_ball_center, _prepare_frames

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    detections = json.loads(
        (work_directory / "object-detections.json").read_text()
    )["detections"]
    ordinary_ranker = joblib.load(work_directory / "kinematic-ranker.joblib")[
        "model"
    ]
    family_ranker = joblib.load(work_directory / "family-kinematic-ranker.joblib")[
        "model"
    ]
    ensemble = json.loads(
        (work_directory / "kinematic-ensemble-metrics.json").read_text()
    )
    ordinary_weight = float(ensemble["ordinary_weight"])
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)

    feature_names = (
        "ball_state_fraction",
        "racket_state_fraction",
        "both_state_fraction",
        "near_ball_states",
        "near_racket_states",
        "minimum_ball_racket_gap",
        "median_ball_racket_gap",
        "longest_ball_missing_run",
        "longest_racket_missing_run",
        "selected_state_fraction",
        "selected_audio_delta",
        "score_margin",
        "score_standard_deviation",
        "score_range",
        "near_peak_state_span",
    )

    def longest_false(values: list[bool]) -> int:
        longest = 0
        current = 0
        for value in values:
            current = 0 if value else current + 1
            longest = max(longest, current)
        return longest

    entries = []
    for indices in grouped.values():
        frames = _visual_frames_for_indices(records, detections, indices)
        prepared = _prepare_frames(frames)
        vectors = np.asarray(_kinematic_feature_vectors(frames))
        ordinary_scores = np.asarray(ordinary_ranker.predict(vectors), dtype=float)
        family_scores = np.asarray(family_ranker.predict(vectors), dtype=float)
        ordinary_scores = (ordinary_scores - ordinary_scores.mean()) / max(
            1e-8, float(ordinary_scores.std())
        )
        family_scores = (family_scores - family_scores.mean()) / max(
            1e-8, float(family_scores.std())
        )
        scores = (
            ordinary_weight * ordinary_scores
            + (1 - ordinary_weight) * family_scores
        )
        selected_position = int(np.argmax(scores))
        selected = records[indices[selected_position]]
        selected_state = int(selected["visual_state"])
        by_state: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"ball": False, "racket": False, "gaps": []}
        )
        for position, frame in enumerate(prepared):
            state = int(records[indices[position]]["visual_state"])
            racket = frame.rackets[0] if frame.rackets else None
            ball = _moving_ball_center(frame)
            by_state[state]["ball"] |= ball is not None
            by_state[state]["racket"] |= racket is not None
            if frame.player is not None and racket is not None and ball is not None:
                left, top, right, bottom = racket["box"]
                racket_center = ((left + right) / 2, (top + bottom) / 2)
                by_state[state]["gaps"].append(
                    math.dist(ball, racket_center) / frame.diagonal
                )
        states = sorted(by_state)
        ball_present = [bool(by_state[state]["ball"]) for state in states]
        racket_present = [bool(by_state[state]["racket"]) for state in states]
        near_states = [state for state in states if abs(state - selected_state) <= 2]
        gaps = sorted(
            gap for observation in by_state.values() for gap in observation["gaps"]
        )
        outside = [
            score
            for position, score in enumerate(scores)
            if abs(int(records[indices[position]]["visual_state"]) - selected_state)
            > 1
        ]
        peak = float(scores[selected_position])
        near_peak_positions = [
            position
            for position, score in enumerate(scores)
            if score >= peak - max(1e-6, float(np.std(scores)) * 0.25)
        ]
        near_peak_states = [
            int(records[indices[position]]["visual_state"])
            for position in near_peak_positions
        ]
        vector = (
            sum(ball_present) / len(states),
            sum(racket_present) / len(states),
            sum(ball and racket for ball, racket in zip(ball_present, racket_present, strict=True))
            / len(states),
            sum(by_state[state]["ball"] for state in near_states),
            sum(by_state[state]["racket"] for state in near_states),
            min(gaps, default=float("nan")),
            gaps[len(gaps) // 2] if gaps else float("nan"),
            longest_false(ball_present) / len(states),
            longest_false(racket_present) / len(states),
            selected_state / max(1, states[-1]),
            abs(float(selected["audio_delta"] or 0.0)),
            peak - max(outside, default=peak),
            float(np.std(scores)),
            float(np.ptp(scores)),
            (max(near_peak_states) - min(near_peak_states)) / max(1, states[-1]),
        )
        entries.append(
            {
                "video": selected["video"],
                "shot_index": int(selected["shot_index"]),
                "split": selected["split"],
                "suitability": selected["suitability"],
                "status": selected["status"],
                "correct": bool(
                    selected["status"] != "unresolved" and selected["target"] > 0
                ),
                "exact": bool(
                    selected["status"] != "unresolved" and selected["acceptable"]
                ),
                "features": vector,
            }
        )

    train = [
        entry
        for entry in entries
        if entry["suitability"] == "in_scope"
        and entry["split"] == "train"
        and entry["status"] != "unresolved"
    ]
    validation = [
        entry
        for entry in entries
        if entry["suitability"] == "in_scope" and entry["split"] == "validation"
    ]
    best = None
    for max_samples in (0.5, 0.75, 1.0):
        for max_features in (0.7, 1.0):
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                IsolationForest(
                    n_estimators=300,
                    max_samples=max_samples,
                    max_features=max_features,
                    contamination="auto",
                    random_state=148,
                ),
            ).fit(np.asarray([entry["features"] for entry in train]))
            scores = model.decision_function(
                np.asarray([entry["features"] for entry in validation])
            )
            unresolved_scores = [
                float(score)
                for entry, score in zip(validation, scores, strict=True)
                if entry["status"] == "unresolved"
            ]
            threshold = max(unresolved_scores, default=float("-inf")) + 1e-9
            accepted = [
                entry
                for entry, score in zip(validation, scores, strict=True)
                if score >= threshold
            ]
            key = (
                sum(entry["correct"] for entry in accepted),
                sum(entry["exact"] for entry in accepted),
                -len(accepted),
            )
            if best is None or key > best[0]:
                best = (
                    key,
                    {"max_samples": max_samples, "max_features": max_features},
                    model,
                    threshold,
                )
    assert best is not None
    _, parameters, model, threshold = best

    validation_scores = model.decision_function(
        np.asarray([entry["features"] for entry in validation])
    )
    validation_unresolved_scores = sorted(
        float(score)
        for entry, score in zip(validation, validation_scores, strict=True)
        if entry["status"] == "unresolved"
    )
    thresholds = {
        "strict": threshold,
        "allow_one_validation_ambiguity": (
            validation_unresolved_scores[0] + 1e-9
            if len(validation_unresolved_scores) > 1
            else threshold
        ),
        "accept_all": float(np.finfo(np.float64).min),
    }

    def summarize(
        selected: list[dict[str, Any]], cutoff: float
    ) -> dict[str, Any]:
        scores = model.decision_function(
            np.asarray([entry["features"] for entry in selected])
        )
        accepted = [
            entry
            for entry, score in zip(selected, scores, strict=True)
            if score >= cutoff
        ]
        resolved = [entry for entry in selected if entry["status"] != "unresolved"]
        return {
            "contacts": len(selected),
            "resolved": len(resolved),
            "accepted": len(accepted),
            "correct_accepted": sum(entry["correct"] for entry in accepted),
            "resolved_correct_recall": sum(entry["correct"] for entry in accepted)
            / len(resolved),
            "unresolved_false_accepts": sum(
                entry["status"] == "unresolved" for entry in accepted
            ),
        }

    evaluation_sets = {
        "validation": validation,
        "test": [
            entry
            for entry in entries
            if entry["suitability"] == "in_scope" and entry["split"] == "test"
        ],
        "out_of_scope": [
            entry for entry in entries if entry["suitability"] != "in_scope"
        ],
    }
    report = {
        "feature_names": feature_names,
        "parameters": parameters,
        "operating_points": {
            name: {
                "threshold": cutoff,
                **{
                    split: summarize(selected, cutoff)
                    for split, selected in evaluation_sets.items()
                },
            }
            for name, cutoff in thresholds.items()
        },
    }
    output_path = work_directory / "ambiguity-anomaly.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return output_path


def evaluate_kinematic_ensemble(*, work_directory: Path) -> Path:
    """Blend ordinary and family-balanced rankers using validation only."""

    import numpy as np

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    ordinary = json.loads(
        (work_directory / "kinematic-frame-scores.json").read_text()
    )
    balanced = json.loads(
        (work_directory / "family-kinematic-frame-scores.json").read_text()
    )
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    groups_by_split = {
        split: [
            indices
            for indices in grouped.values()
            if records[indices[0]]["split"] == split
            and records[indices[0]]["suitability"] == "in_scope"
            and records[indices[0]]["status"] != "unresolved"
        ]
        for split in ("validation", "test")
    }

    def blended(split: str, ordinary_weight: float) -> np.ndarray:
        scores = np.full(len(records), -np.inf)
        for indices in groups_by_split[split]:
            first = np.asarray(
                [ordinary[split][str(index)] for index in indices], dtype=float
            )
            second = np.asarray(
                [balanced[split][str(index)] for index in indices], dtype=float
            )
            first = (first - first.mean()) / max(1e-8, float(first.std()))
            second = (second - second.mean()) / max(1e-8, float(second.std()))
            scores[indices] = ordinary_weight * first + (1 - ordinary_weight) * second
        return scores

    selected = None
    for ordinary_weight in np.linspace(0.0, 1.0, 21):
        scores = blended("validation", float(ordinary_weight))
        metrics = _rank_metrics(groups_by_split["validation"], records, scores)
        key = (metrics["tolerant_accuracy"], metrics["exact_accuracy"])
        if selected is None or key > selected[0]:
            selected = (key, float(ordinary_weight), metrics)
    assert selected is not None
    _, ordinary_weight, validation_metrics = selected
    split_scores = {
        split: blended(split, ordinary_weight) for split in ("validation", "test")
    }
    report = {
        "ordinary_weight": ordinary_weight,
        "family_balanced_weight": 1 - ordinary_weight,
        "validation": validation_metrics,
        "test": _rank_metrics(groups_by_split["test"], records, split_scores["test"]),
    }
    (work_directory / "ensemble-frame-scores.json").write_text(
        json.dumps(
            {
                split: {
                    index: float(scores[index])
                    for group in groups_by_split[split]
                    for index in group
                }
                for split, scores in split_scores.items()
            },
            indent=2,
        )
        + "\n"
    )
    output_path = work_directory / "kinematic-ensemble-metrics.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return output_path


def export_kinematic_ensemble(*, work_directory: Path, output: Path) -> Path:
    """Export the selected sklearn ensemble as one dependency-free artifact."""

    import joblib

    from tennis_cut.temporal_ranker import (
        KINEMATIC_VECTOR_SIZE,
        HistGradientBoostingArtifact,
        HistGradientBoostingNode,
        KinematicEnsembleArtifact,
    )

    def portable(model: Any) -> HistGradientBoostingArtifact:
        trees = tuple(
            tuple(
                HistGradientBoostingNode(
                    float(node["value"]),
                    int(node["feature_idx"]),
                    float(node["num_threshold"]),
                    bool(node["missing_go_to_left"]),
                    int(node["left"]),
                    int(node["right"]),
                    bool(node["is_leaf"]),
                )
                for node in predictors[0].nodes
            )
            for predictors in model._predictors
        )
        return HistGradientBoostingArtifact(
            baseline=float(model._baseline_prediction[0, 0]),
            trees=trees,
            supported_shot_type="forehand",
            operating_threshold=0.0,
            vector_size=KINEMATIC_VECTOR_SIZE,
        )

    ordinary = joblib.load(work_directory / "kinematic-ranker.joblib")["model"]
    family_balanced = joblib.load(
        work_directory / "family-kinematic-ranker.joblib"
    )["model"]
    metrics = json.loads(
        (work_directory / "kinematic-ensemble-metrics.json").read_text()
    )
    artifact = KinematicEnsembleArtifact(
        portable(ordinary),
        portable(family_balanced),
        float(metrics["ordinary_weight"]),
        "forehand",
    )
    artifact.save(output)
    from tennis_cut.temporal_ranker import load_temporal_ranker
    from tennis_cut.visual_contact import select_contact_frame

    ranker = load_temporal_ranker(output)
    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    detections = json.loads(
        (work_directory / "object-detections.json").read_text()
    )["detections"]
    expected_scores = json.loads(
        (work_directory / "ensemble-frame-scores.json").read_text()
    )
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    verified = 0
    for indices in grouped.values():
        first = records[indices[0]]
        split = first["split"]
        if (
            split not in ("validation", "test")
            or first["suitability"] != "in_scope"
            or first["status"] == "unresolved"
        ):
            continue
        frames = _visual_frames_for_indices(records, detections, indices)
        predicted = select_contact_frame(frames, ranker)
        expected_index = max(
            indices,
            key=lambda index: (
                float(expected_scores[split][str(index)]),
                -int(records[index]["ordinal"]),
            ),
        )
        if predicted.selected_frame != int(records[expected_index]["ordinal"]):
            raise RuntimeError(
                "portable ensemble changed a held-out frame selection for "
                f"{first['video']} contact {first['shot_index']}"
            )
        verified += 1
    print(
        f"Exported {output} ({output.stat().st_size:,} bytes); "
        f"verified {verified} held-out contacts"
    )
    return output


def evaluate_view_suitability(
    *, work_directory: Path, device: str | None
) -> Path:
    """Test frozen appearance as a fixed-camera side-view gate."""

    import numpy as np
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.transforms import v2

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[record["video"]].append(record)
    videos = sorted(by_video)
    selected_records = []
    for video in videos:
        candidates = by_video[video]
        positions = np.linspace(0, len(candidates) - 1, 5).round().astype(int)
        selected_records.extend(candidates[position] for position in positions)

    selected_device = device or (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(selected_device)
    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(selected_records), 32):
            images = torch.stack(
                [
                    transform(Image.open(record["image"]).convert("RGB"))
                    for record in selected_records[start : start + 32]
                ]
            ).to(selected_device)
            embeddings.extend(model(images).cpu().numpy())
    video_embeddings = np.asarray(embeddings).reshape(len(videos), 5, -1).mean(axis=1)
    labels = np.asarray(
        [int(by_video[video][0]["suitability"] == "in_scope") for video in videos]
    )
    probabilities = np.zeros(len(videos))
    for held_out in range(len(videos)):
        train = [index for index in range(len(videos)) if index != held_out]
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", random_state=148),
        ).fit(video_embeddings[train], labels[train])
        probabilities[held_out] = classifier.predict_proba(
            video_embeddings[held_out : held_out + 1]
        )[0, 1]
    predictions = probabilities >= 0.5
    details = [
        {
            "video": Path(video).name,
            "family": by_video[video][0]["family"],
            "suitability": by_video[video][0]["suitability"],
            "in_scope_probability": float(probability),
            "prediction": "in_scope" if prediction else "out_of_scope_view",
            "correct": bool(prediction == bool(label)),
        }
        for video, probability, prediction, label in zip(
            videos, probabilities, predictions, labels, strict=True
        )
    ]
    report = {
        "protocol": "leave-one-recording-family-out",
        "videos": len(videos),
        "in_scope": int(labels.sum()),
        "out_of_scope": int((1 - labels).sum()),
        "accuracy": float(np.mean(predictions == labels)),
        "in_scope_recall": float(np.mean(predictions[labels == 1])),
        "out_of_scope_recall": float(np.mean(~predictions[labels == 0])),
        "details": details,
    }
    output_path = work_directory / "view-suitability.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return output_path


def audit_model_agreement(*, work_directory: Path) -> Path:
    """Compare independent timing cues as a possible abstention signal."""

    import joblib
    import numpy as np
    import torch

    from tennis_cut.temporal_ranker import all_temporal_feature_vectors
    from tennis_cut.visual_contact import extract_temporal_features

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    detections = json.loads(
        (work_directory / "object-detections.json").read_text()
    )["detections"]
    poses = json.loads((work_directory / "pose-detections.json").read_text())[
        "detections"
    ]
    models = {
        name: joblib.load(work_directory / f"{name}-ranker.joblib")["model"]
        for name in (
            "structured",
            "kinematic",
            "racket",
            "pose-only",
            "pose-kinematic",
        )
    }
    embeddings = torch.load(
        work_directory / "frame-dataset" / "resnet18-embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )["embeddings"].float()
    image_artifact = torch.load(
        work_directory / "frame-probe.pt", map_location="cpu", weights_only=False
    )
    image_model = torch.nn.Linear(embeddings.shape[1], 1)
    image_model.load_state_dict(image_artifact["state_dict"])
    with torch.inference_mode():
        image_scores = image_model(
            (embeddings - image_artifact["feature_mean"])
            / image_artifact["feature_std"]
        ).squeeze(1)

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)
    entries = []
    for indices in grouped.values():
        frames = _visual_frames_for_indices(records, detections, indices)
        production_vectors = all_temporal_feature_vectors(
            extract_temporal_features(frames)
        )
        kinematic_vectors = _kinematic_feature_vectors(frames)
        racket_vectors = _kinematic_feature_vectors(frames, racket_only=True)
        pose_vectors = _pose_feature_vectors(
            frames, [poses[index] for index in indices]
        )
        vector_sets = {
            "structured": production_vectors,
            "kinematic": kinematic_vectors,
            "racket": racket_vectors,
            "pose-only": pose_vectors,
            "pose-kinematic": tuple(
                kinematic_vector + pose_vector
                for kinematic_vector, pose_vector in zip(
                    kinematic_vectors, pose_vectors, strict=True
                )
            ),
        }
        positions = {
            name: int(np.argmax(model.predict(np.asarray(vector_sets[name]))))
            for name, model in models.items()
        }
        positions["image"] = int(image_scores[indices].argmax())
        positions["audio"] = min(
            range(len(indices)),
            key=lambda position: abs(float(records[indices[position]]["audio_delta"])),
        )
        states = {
            name: int(records[indices[position]]["visual_state"])
            for name, position in positions.items()
        }
        kinematic_state = states["kinematic"]
        agreements = sum(
            abs(state - kinematic_state) <= 1
            for name, state in states.items()
            if name != "kinematic"
        )
        selected = records[indices[positions["kinematic"]]]
        entries.append(
            {
                "video": selected["video"],
                "shot_index": int(selected["shot_index"]),
                "split": selected["split"],
                "suitability": selected["suitability"],
                "status": selected["status"],
                "states": states,
                "agreements_with_kinematic": agreements,
                "state_span": max(states.values()) - min(states.values()),
                "correct": bool(
                    selected["status"] != "unresolved" and selected["target"] > 0
                ),
            }
        )
    output_path = work_directory / "model-agreement.json"
    output_path.write_text(json.dumps(entries, indent=2) + "\n")
    for split in ("validation", "test"):
        print(split)
        for entry in entries:
            if entry["suitability"] == "in_scope" and entry["split"] == split:
                print(
                    Path(entry["video"]).name,
                    entry["shot_index"],
                    entry["status"],
                    "agree",
                    entry["agreements_with_kinematic"],
                    "span",
                    entry["state_span"],
                    entry["states"],
                )
    return output_path


def audit_contact_quality(*, work_directory: Path) -> Path:
    """Expose visibility and geometry signals for a separate ambiguity gate."""

    import math

    import joblib
    import numpy as np

    from tennis_cut.visual_contact import _center, _moving_ball_center, _prepare_frames

    records = json.loads(
        (work_directory / "frame-dataset" / "records.json").read_text()
    )["records"]
    detections = json.loads(
        (work_directory / "object-detections.json").read_text()
    )["detections"]
    model = joblib.load(work_directory / "kinematic-ranker.joblib")["model"]
    agreement = {
        (entry["video"], int(entry["shot_index"])): entry
        for entry in json.loads(
            (work_directory / "model-agreement.json").read_text()
        )
    }
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record["video"], int(record["shot_index"]))].append(index)

    entries = []
    for key, indices in grouped.items():
        frames = _visual_frames_for_indices(records, detections, indices)
        prepared = _prepare_frames(frames)
        scores = np.asarray(
            model.predict(np.asarray(_kinematic_feature_vectors(frames))), dtype=float
        )
        selected_position = int(np.argmax(scores))
        selected_state = int(records[indices[selected_position]]["visual_state"])
        state_observations: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"ball": False, "racket": False, "gaps": []}
        )
        for index, frame in enumerate(prepared):
            state = int(records[indices[index]]["visual_state"])
            state_observations[state]["racket"] |= bool(frame.rackets)
            ball = _moving_ball_center(frame)
            state_observations[state]["ball"] |= ball is not None
            if frame.player is not None and frame.rackets and ball is not None:
                player_center = _center(frame.player.box)
                racket_center = _center(frame.rackets[0]["box"])
                relative_ball = (
                    (ball[0] - player_center[0]) / frame.diagonal,
                    (ball[1] - player_center[1]) / frame.diagonal,
                )
                relative_racket = (
                    (racket_center[0] - player_center[0]) / frame.diagonal,
                    (racket_center[1] - player_center[1]) / frame.diagonal,
                )
                state_observations[state]["gaps"].append(
                    math.dist(relative_ball, relative_racket)
                )
        states = sorted(state_observations)
        near = [state for state in states if abs(state - selected_state) <= 2]
        gaps = [
            gap
            for observation in state_observations.values()
            for gap in observation["gaps"]
        ]
        selected = records[indices[selected_position]]
        agreement_entry = agreement[key]
        entries.append(
            {
                "video": selected["video"],
                "shot_index": int(selected["shot_index"]),
                "split": selected["split"],
                "suitability": selected["suitability"],
                "status": selected["status"],
                "selected_state": selected_state,
                "correct": bool(
                    selected["status"] != "unresolved" and selected["target"] > 0
                ),
                "ball_state_fraction": sum(
                    state_observations[state]["ball"] for state in states
                )
                / len(states),
                "racket_state_fraction": sum(
                    state_observations[state]["racket"] for state in states
                )
                / len(states),
                "near_ball_states": sum(
                    state_observations[state]["ball"] for state in near
                ),
                "near_racket_states": sum(
                    state_observations[state]["racket"] for state in near
                ),
                "minimum_ball_racket_gap": min(gaps, default=None),
                "agreements_with_kinematic": agreement_entry[
                    "agreements_with_kinematic"
                ],
                "state_span": agreement_entry["state_span"],
            }
        )
    output_path = work_directory / "contact-quality.json"
    output_path.write_text(json.dumps(entries, indent=2) + "\n")
    for entry in entries:
        if entry["status"] == "unresolved" and entry["suitability"] == "in_scope":
            print(json.dumps(entry, indent=2))
    return output_path


def print_audit(dataset: ReviewDataset) -> None:
    print(f"valid contacts: {len(dataset.contacts)}")
    print(f"invalid contacts: {len(dataset.invalid)}")
    for split in ("train", "validation", "test"):
        print(split, dict(Counter(contact.status for contact in dataset.primary(split))))
    print("out_of_scope", len(dataset.out_of_scope))
    for reason in dataset.invalid:
        print(f"INVALID: {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "prepare-audio",
            "prepare-frames",
            "extract-embeddings",
            "train-frame-probe",
            "evaluate-frame-probe",
            "fine-tune-frame",
            "evaluate-fine-tuned-fusion",
            "train-temporal-probe",
            "evaluate-temporal-probe",
            "fine-tune-temporal",
            "build-report",
            "prepare-objects",
            "prepare-poses",
            "evaluate-structured",
            "train-structured",
            "evaluate-structured-fusion",
            "train-kinematic",
            "train-family-kinematic",
            "evaluate-kinematic-fusion",
            "evaluate-kinematic-ensemble",
            "export-kinematic-ensemble",
            "evaluate-view-suitability",
            "evaluate-kinematic-confidence",
            "evaluate-ambiguity-anomaly",
            "train-racket-only",
            "evaluate-racket-fusion",
            "audit-agreement",
            "audit-contact-quality",
            "train-pose-only",
            "evaluate-pose-only-fusion",
            "train-pose-kinematic",
            "evaluate-pose-kinematic-fusion",
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIRECTORY)
    parser.add_argument(
        "--review-work-dir",
        type=Path,
        default=Path("out/contact-review-prototype"),
    )
    parser.add_argument("--audio-model", type=Path, default=DEFAULT_AUDIO_MODEL)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negative-count", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("src/tennis_cut/models/temporal_ranker.json"),
    )
    args = parser.parse_args()
    dataset = load_review_dataset(args.manifest)
    if args.command == "audit":
        print_audit(dataset)
        return 1 if dataset.invalid else 0
    if dataset.invalid:
        print_audit(dataset)
        raise SystemExit("Refusing to prepare inputs from invalid contact reviews.")
    if args.command == "prepare-audio":
        prepare_audio_priors(
            dataset,
            work_directory=args.work_dir,
            model_path=args.audio_model,
            device=args.device,
        )
    elif args.command == "prepare-frames":
        prepare_frame_dataset(
            dataset,
            manifest_path=args.manifest,
            review_work_directory=args.review_work_dir,
            work_directory=args.work_dir,
            device=args.device,
        )
    elif args.command == "extract-embeddings":
        extract_frame_embeddings(
            work_directory=args.work_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "train-frame-probe":
        train_frame_probe(
            work_directory=args.work_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    elif args.command == "evaluate-frame-probe":
        evaluate_probe_fusion(work_directory=args.work_dir)
    elif args.command == "fine-tune-frame":
        fine_tune_frame_scorer(
            work_directory=args.work_dir,
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            negative_count=args.negative_count,
            image_size=args.image_size,
        )
    elif args.command == "evaluate-fine-tuned-fusion":
        evaluate_fine_tuned_fusion(work_directory=args.work_dir)
    elif args.command == "train-temporal-probe":
        train_temporal_probe(
            work_directory=args.work_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    elif args.command == "evaluate-temporal-probe":
        evaluate_temporal_fusion(work_directory=args.work_dir)
    elif args.command == "fine-tune-temporal":
        fine_tune_temporal_scorer(
            work_directory=args.work_dir,
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
        )
    elif args.command == "build-report":
        build_contact_report(work_directory=args.work_dir)
    elif args.command == "prepare-objects":
        prepare_object_detections(
            work_directory=args.work_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "prepare-poses":
        prepare_pose_detections(
            work_directory=args.work_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "evaluate-structured":
        evaluate_structured_baseline(work_directory=args.work_dir)
    elif args.command == "train-structured":
        train_structured_ranker(work_directory=args.work_dir)
    elif args.command == "evaluate-structured-fusion":
        _evaluate_stored_fusion(
            work_directory=args.work_dir,
            scores_name="structured-frame-scores.json",
            output_name="structured-frame-fusion.json",
        )
    elif args.command == "train-kinematic":
        train_structured_ranker(work_directory=args.work_dir, kinematic=True)
    elif args.command == "train-family-kinematic":
        train_structured_ranker(
            work_directory=args.work_dir,
            kinematic=True,
            family_balanced=True,
        )
    elif args.command == "evaluate-kinematic-fusion":
        _evaluate_stored_fusion(
            work_directory=args.work_dir,
            scores_name="kinematic-frame-scores.json",
            output_name="kinematic-frame-fusion.json",
        )
    elif args.command == "evaluate-kinematic-ensemble":
        evaluate_kinematic_ensemble(work_directory=args.work_dir)
    elif args.command == "export-kinematic-ensemble":
        export_kinematic_ensemble(
            work_directory=args.work_dir,
            output=args.artifact,
        )
    elif args.command == "evaluate-view-suitability":
        evaluate_view_suitability(
            work_directory=args.work_dir,
            device=args.device,
        )
    elif args.command == "evaluate-kinematic-confidence":
        evaluate_kinematic_confidence(work_directory=args.work_dir)
    elif args.command == "evaluate-ambiguity-anomaly":
        evaluate_ambiguity_anomaly(work_directory=args.work_dir)
    elif args.command == "train-racket-only":
        train_structured_ranker(
            work_directory=args.work_dir, kinematic=True, racket_only=True
        )
    elif args.command == "evaluate-racket-fusion":
        _evaluate_stored_fusion(
            work_directory=args.work_dir,
            scores_name="racket-frame-scores.json",
            output_name="racket-frame-fusion.json",
        )
    elif args.command == "train-pose-only":
        train_structured_ranker(work_directory=args.work_dir, pose_only=True)
    elif args.command == "evaluate-pose-only-fusion":
        _evaluate_stored_fusion(
            work_directory=args.work_dir,
            scores_name="pose-only-frame-scores.json",
            output_name="pose-only-frame-fusion.json",
        )
    elif args.command == "train-pose-kinematic":
        train_structured_ranker(work_directory=args.work_dir, pose=True)
    elif args.command == "evaluate-pose-kinematic-fusion":
        _evaluate_stored_fusion(
            work_directory=args.work_dir,
            scores_name="pose-kinematic-frame-scores.json",
            output_name="pose-kinematic-frame-fusion.json",
        )
    elif args.command == "audit-contact-quality":
        audit_contact_quality(work_directory=args.work_dir)
    else:
        audit_model_agreement(work_directory=args.work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

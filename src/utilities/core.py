from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple


def probe_duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


class PersonDetector:
    """Wrapper around YOLOv8-n person detector."""

    def __init__(self, device: str) -> None:
        from ultralytics import YOLO

        self.device = device
        self.model = YOLO("yolov8n.pt")

    def find_box(self, img_path: Path) -> Tuple[int, int, int, int] | None:
        """Return the largest person box as (x1, y1, x2, y2) integers."""
        results = self.model.predict(
            str(img_path), classes=0, device=self.device, verbose=False
        )
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy.numel() == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = areas.argmax()
        x1, y1, x2, y2 = xyxy[idx]
        return int(x1), int(y1), int(x2), int(y2)


def expand_box(box: Tuple[int, int, int, int], res: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Pad box by 20% and fit to the video aspect ratio."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    w *= 1.2
    h *= 1.2
    frame_w, frame_h = res
    aspect = frame_w / frame_h
    if w / h > aspect:
        h = w / aspect
    else:
        w = h * aspect

    # Ensure the crop doesn't exceed the frame size
    if w > frame_w:
        w = frame_w
    if h > frame_h:
        h = frame_h

    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    w = int(round(w))
    h = int(round(h))

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x1 + w > frame_w:
        x1 = max(0, frame_w - w)
    if y1 + h > frame_h:
        y1 = max(0, frame_h - h)
    return x1, y1, w, h

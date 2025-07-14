# label_shots.py

"""Label impacts from existing JSON annotations.

This utility opens a simple PySide6 GUI that shows each impact frame from
videos in a directory and lets the user assign a shot type using the
keyboard. It expects JSON files created with an older version of
``tennis_annotate.py`` that only stored impact timestamps.

Usage
-----
    python label_shots.py /path/to/videos

Key bindings
------------
    d  : forehand
    w  : backhand
    e  : volley
    r  : serve
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import List

from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


@dataclass
class ImpactItem:
    video: pathlib.Path
    json_path: pathlib.Path
    data: dict
    time: float


class ShotLabeler(QWidget):
    def __init__(self, items: List[ImpactItem]):
        super().__init__()
        self.items = items
        self.idx = 0
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(QAudioOutput())
        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        self.label = QLabel(parent=self)
        self.label.setStyleSheet("font-size:18px; padding:4px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_widget)
        layout.addWidget(self.label)
        self.showMaximized()

        self.current_video: pathlib.Path | None = None
        self._load_item()

    def _load_item(self) -> None:
        if self.idx >= len(self.items):
            print("All impacts labeled.")
            QApplication.instance().quit()
            return

        it = self.items[self.idx]
        if self.current_video != it.video:
            self.current_video = it.video
            self.player.setSource(QUrl.fromLocalFile(str(it.video)))
            self.player.play()
            QTimer.singleShot(100, self.player.pause)
        self.player.setPosition(int(it.time * 1000))
        self.player.pause()
        self._update_label()

    def _update_label(self) -> None:
        item = self.items[self.idx]
        self.label.setText(
            f"{self.idx} complete / {len(self.items)} total\n"
            f"{item.video.name} @ {item.time:.3f}s\n"
            "d: forehand   w: backhand   e: volley   r: serve"
        )

    def keyPressEvent(self, ev):
        mapping = {
            Qt.Key_D: "forehand",
            Qt.Key_W: "backhand",
            Qt.Key_E: "volley",
            Qt.Key_R: "serve",
        }
        typ = mapping.get(ev.key())
        if typ is None:
            super().keyPressEvent(ev)
            return

        it = self.items[self.idx]
        shots = it.data.setdefault("shots", [])
        shots.append({"time": it.time, "type": typ})
        json.dump(it.data, open(it.json_path, "w"), indent=2)
        print(f"[{typ}] {it.video.name} {it.time:.3f}s")

        self.idx += 1
        self._load_item()


def gather_items(directory: pathlib.Path) -> List[ImpactItem]:
    items: List[ImpactItem] = []
    for vid in sorted(directory.rglob("*")):
        if vid.suffix.lower() not in VIDEO_EXTS:
            continue
        js = vid.with_suffix(".json")
        if not js.exists():
            continue
        try:
            data = json.load(open(js))
        except Exception:
            continue
        impacts = data.get("impacts")
        if not isinstance(impacts, list):
            continue
        labeled = {float(s.get("time")) for s in data.get("shots", []) if isinstance(s, dict)}
        for t in impacts:
            if not isinstance(t, (int, float)):
                continue
            if float(t) in labeled:
                continue
            items.append(ImpactItem(video=vid, json_path=js, data=data, time=float(t)))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Label shot types for existing impact annotations")
    ap.add_argument("directory", type=pathlib.Path)
    args = ap.parse_args()

    items = gather_items(args.directory)
    if not items:
        sys.exit("No unlabeled impacts found.")
    print(f"Need to label {len(items)} impacts")

    app = QApplication.instance() or QApplication(sys.argv)
    win = ShotLabeler(items)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()

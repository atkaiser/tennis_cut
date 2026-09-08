# tennis_annotate.py
"""Annotate the exact ball‑contact moments in tennis videos.

Usage
-----
    python tennis_annotate.py /path/to/videos

Key bindings
------------
    f  : forward 1 frame
    s  : backward 1 frame
    a  : backward 10 frames

    g  : forward 250 ms
    h  : forward 500 ms
    b  : forward 1 second

    d  : mark forehand
    w  : mark backhand
    x  : mark backhand slice
    e  : mark volley
    r  : mark serve
    o  : mark overhead

There is **no candidate/skip logic**. Scrub through the clip and press the
appropriate key whenever the ball makes contact. Impacts are written to
`<video>.json` beside the source file with the selected shot type.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, List

from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

from tennis_cut.annotation_json import load_annotation, save_annotation

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

###############################################################################
# Helpers
###############################################################################


# Get frames per second from video using ffprobe (defaults to 30 if unavailable).
def probe_fps(video_path: pathlib.Path) -> float:
    """Return frames‑per‑second using ffprobe (falls back to 30 fps if unknown)."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            text=True,
        ).strip()
        if "/" in out:
            num, den = map(int, out.split("/"))
            return num / den
        return float(out)
    except Exception:
        return 30.0


###############################################################################
# Data model
###############################################################################
@dataclass
class Shot:
    time: float
    type: str
    extension_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extension_fields)
        payload.update({"time": self.time, "type": self.type})
        return payload


@dataclass
class AnnotationState:
    video: pathlib.Path
    fps: float
    impacts: List[float] = field(default_factory=list)
    shots: List[Shot] = field(default_factory=list)
    done: bool = False
    extension_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    # Calculate time step in milliseconds for a given number of frames based on the video's fps.
    def step_ms(self, frames: int) -> int:
        return round(frames * 1000 / self.fps)


def load_annotation_state(video: pathlib.Path, fps: float) -> AnnotationState:
    """Load editable state while retaining fields owned by other tools."""
    state = AnnotationState(video=video, fps=fps)
    json_path = video.with_suffix(".json")
    if not json_path.exists():
        return state

    data = load_annotation(json_path)
    state.extension_fields = {
        key: value
        for key, value in data.items()
        if key not in {"video", "impacts", "shots", "done"}
    }
    impacts = data.get("impacts")
    if isinstance(impacts, list):
        state.impacts = [
            float(impact) for impact in impacts if isinstance(impact, (int, float))
        ]
    for shot in data.get("shots", []):
        if not isinstance(shot, dict):
            continue
        timestamp = shot.get("time")
        shot_type = shot.get("type")
        if isinstance(timestamp, (int, float)) and isinstance(shot_type, str):
            state.shots.append(
                Shot(
                    time=float(timestamp),
                    type=shot_type,
                    extension_fields={
                        key: value
                        for key, value in shot.items()
                        if key not in {"time", "type"}
                    },
                )
            )
    state.done = data.get("done") is True
    return state


def annotation_payload(state: AnnotationState) -> dict[str, Any]:
    """Merge annotator-owned values into a lossless annotation document."""
    payload = dict(state.extension_fields)
    payload.update(
        {
            "video": state.video.name,
            "impacts": state.impacts,
            "shots": [shot.to_dict() for shot in state.shots],
            "done": state.done,
        }
    )
    return payload


###############################################################################
# GUI
###############################################################################
class Annotator(QWidget):
    # Initialize the annotator GUI with the provided annotation state.
    def __init__(self, st: AnnotationState):
        super().__init__()
        self.st = st
        self.setWindowTitle(st.video.name)

        # ----------------------- Media player --------------------------- #
        self.player = QMediaPlayer(self)
        self.player.setSource(QUrl.fromLocalFile(str(st.video)))
        self.player.setAudioOutput(QAudioOutput())
        self.player.errorOccurred.connect(
            lambda c, t: print("QMediaPlayer error:", c, t)
        )

        # Start/stop once so the first video frame is rendered instead of a black box
        self.player.play()
        QTimer.singleShot(100, self.player.pause)  # pause after metadata loads

        # --------------------------- Layout ----------------------------- #
        video_widget = QVideoWidget(self)
        self.player.setVideoOutput(video_widget)

        self.label = QLabel(
            "0.000 s", alignment=Qt.AlignCenter | Qt.AlignTop, parent=self
        )
        self.label.setWordWrap(True)
        self.label.setMaximumHeight(240)
        self.label.setStyleSheet("font-size:14px; padding:6px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(video_widget, 1)
        layout.addWidget(self.label, 0)

        self.showMaximized()

    # ------------------------- Key handling --------------------------- #
    # Handle key press events and trigger corresponding actions based on key bindings.
    def keyPressEvent(self, ev: QKeyEvent):
        k = ev.key()
        if k == Qt.Key_F:
            self._seek_rel(self.st.step_ms(+1))
        elif k == Qt.Key_S:
            self._seek_rel(self.st.step_ms(-1))
        elif k == Qt.Key_A:
            self._seek_rel(self.st.step_ms(-10))
        elif k == Qt.Key_G:
            self._seek_rel(250)
        elif k == Qt.Key_H:
            self._seek_rel(500)
        elif k == Qt.Key_B:
            self._seek_rel(1000)
        elif k == Qt.Key_D:
            self._mark_impact("forehand")
        elif k == Qt.Key_W:
            self._mark_impact("backhand")
        elif k == Qt.Key_X:
            self._mark_impact("backhand_slice")
        elif k == Qt.Key_E:
            self._mark_impact("volley")
        elif k == Qt.Key_R:
            self._mark_impact("serve")
        elif k == Qt.Key_O:
            self._mark_impact("overhead")
        elif k == Qt.Key_Z:
            self._mark_done()
        elif k == Qt.Key_V:
            self._seek_rel(self.st.step_ms(+10))
        elif k == Qt.Key_Q:
            sys.exit(0)
        else:
            super().keyPressEvent(ev)

    # --------------------------- Helpers ------------------------------ #
    # Seek the video playback position by the specified delta in milliseconds and update the label.
    def _seek_rel(self, delta_ms: int):
        self.player.setPosition(max(0, self.player.position() + delta_ms))
        self._update_label()

    # Record the current video position as an impact with the given shot type,
    # then save annotations and update the label.
    def _mark_impact(self, shot_type: str):
        t = round(self.player.position() / 1000.0, 3)
        self.st.impacts.append(t)
        self.st.shots.append(Shot(time=t, type=shot_type))
        print(f"[{shot_type}] {t:.3f}s  (total={len(self.st.impacts)})")
        self._save_json()
        self._update_label()

    # Mark the video as done, then save annotations and update the label.
    def _mark_done(self):
        self.st.done = True
        print(f"[done] {self.st.video.name} marked as done.")
        self._save_json()
        self._update_label()

    # Update the display label with the current playback time, total video duration, number of impacts, and keybindings.
    def _update_label(self):
        cur_s = self.player.position() / 1000.0
        total_ms = self.player.duration()
        total_s = total_ms / 1000.0 if total_ms else 0
        self.label.setText(
            f"{cur_s:.3f} s / {total_s:.3f} s   |   impacts: {len(self.st.impacts)}{' (DONE)' if self.st.done else ''}\n"
            "Keys: a:-10f  s:-1f  f:+1f  v:+10f  g:+250ms  h:+500ms  b:+1s  "
            "d:forehand  w:backhand  x:bh_slice  e:volley  r:serve  o:overhead  z:done  q:quit"
        )

    # Save the list of impact annotations to a JSON file located next to the video file.
    def _save_json(self):
        out = self.st.video.with_suffix(".json")
        save_annotation(out, annotation_payload(self.st))


###############################################################################
# Main
###############################################################################


# Launch the annotation GUI for the specified video file.
def annotate_video(path: pathlib.Path):
    state = load_annotation_state(path, probe_fps(path))

    app = QApplication.instance() or QApplication(sys.argv)
    win = Annotator(state)
    if state.impacts:
        last_impact = state.impacts[-1]
        win.player.setPosition(int(last_impact * 1000))
    win.show()

    # keep reference so GC can’t delete window
    if not hasattr(app, "_wins"):
        app._wins = []
    app._wins.append(win)

    app.exec()


# Recursively find video files with supported extensions in the given directory.
def find_videos(root: pathlib.Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


# Parse command-line arguments and execute the annotation process for each video file.
def main():
    ap = argparse.ArgumentParser(
        description="Manually annotate tennis ball impacts in videos."
    )
    ap.add_argument(
        "directory", type=pathlib.Path, help="Directory containing video files"
    )
    args = ap.parse_args()

    vids = find_videos(args.directory)
    if not vids:
        sys.exit("No video files found.")

    for v in vids:
        json_path = v.with_suffix(".json")
        if json_path.exists():
            try:
                data = load_annotation(json_path)
                if data.get("done") is True:
                    print(f"\n=== {v.name} is marked as done, skipping. ===")
                    continue
            except Exception:
                pass
        print(f"\n=== {v.name} ===")
        annotate_video(v)


if __name__ == "__main__":
    main()

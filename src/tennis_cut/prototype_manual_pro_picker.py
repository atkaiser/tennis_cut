#!/usr/bin/env python3
"""PROTOTYPE: compare three PySide6 layouts for selecting pro contact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

PRE_CONTACT = Fraction(6, 5)
POST_CONTACT = Fraction(7, 10)
SHOT_TYPES = (("D", "forehand"), ("W", "backhand"), ("E", "volley"), ("R", "serve"))
VARIANTS = {
    "A": "Keyboard-first annotator",
    "B": "Guided two-step",
    "C": "Timestamp inspector",
}


@dataclass(frozen=True)
class FrameInfo:
    ordinal: int
    pts: int


@dataclass(frozen=True)
class VideoInfo:
    stream_index: int
    time_base: Fraction
    width: int
    height: int
    frames: tuple[FrameInfo, ...]


def probe_video(video: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=index,time_base,width,height:frame=best_effort_timestamp",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    tb_num, tb_den = (int(part) for part in stream["time_base"].split("/"))
    frames = tuple(
        FrameInfo(ordinal=index, pts=int(frame["best_effort_timestamp"]))
        for index, frame in enumerate(data["frames"])
        if frame.get("best_effort_timestamp") not in (None, "N/A")
    )
    if not frames:
        raise RuntimeError("The pro video has no decoded frames with presentation timestamps")
    return VideoInfo(
        stream_index=int(stream["index"]),
        time_base=Fraction(tb_num, tb_den),
        width=int(stream["width"]),
        height=int(stream["height"]),
        frames=frames,
    )


class Picker(QMainWindow):
    def __init__(
        self,
        video: Path,
        info: VideoInfo,
        pro_speed: Fraction,
        variant: str,
        start_seconds: Fraction | None,
    ) -> None:
        super().__init__()
        self.video = video
        self.info = info
        self.pro_speed = pro_speed
        self.variant = variant
        self.index = self._nearest_frame(start_seconds) if start_seconds is not None else len(info.frames) // 2
        self.shot_type: str | None = None
        self.confirmed = False
        self.last_action = "Choose the exact contact frame, then choose the pro swing type."
        self.sidecar_path = Path(tempfile.gettempdir()) / f"{video.name}.PROTOTYPE.tennis-compare.json"
        self.capture = cv2.VideoCapture(str(video))
        self.setWindowTitle(f"PROTOTYPE — Pro contact picker — {video.name}")
        self.resize(1280, 820)
        self._load_reusable_sidecar()
        self._build()

    @property
    def current(self) -> FrameInfo:
        return self.info.frames[self.index]

    def _nearest_frame(self, seconds: Fraction) -> int:
        target_pts = seconds / self.info.time_base
        return min(
            range(len(self.info.frames)),
            key=lambda index: abs(Fraction(self.info.frames[index].pts) - target_pts),
        )

    def _source_identity(self) -> dict[str, int | str]:
        stat = self.video.stat()
        return {
            "name": self.video.name,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _sidecar_data(self) -> dict:
        return {
            "schema_version": 1,
            "source": self._source_identity(),
            "video_stream": {
                "index": self.info.stream_index,
                "time_base": {
                    "numerator": self.info.time_base.numerator,
                    "denominator": self.info.time_base.denominator,
                },
            },
            "contact_frame": {
                "decoded_ordinal": self.current.ordinal,
                "pts": self.current.pts,
            },
            "shot_type": self.shot_type,
        }

    def _load_reusable_sidecar(self) -> None:
        if not self.sidecar_path.exists():
            return
        try:
            data = json.loads(self.sidecar_path.read_text())
            if data.get("schema_version") != 1 or data.get("source") != self._source_identity():
                self.last_action = "Existing prototype sidecar is stale; choose again."
                return
            contact = data["contact_frame"]
            ordinal = int(contact["decoded_ordinal"])
            pts = int(contact["pts"])
            if self.info.frames[ordinal].pts != pts or data.get("shot_type") not in {item[1] for item in SHOT_TYPES}:
                self.last_action = "Existing prototype sidecar no longer identifies a valid decoded frame; choose again."
                return
            self.index = ordinal
            self.shot_type = data["shot_type"]
            self.last_action = "Reused the source-bound prototype sidecar. Delete it to choose again."
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            self.last_action = "Existing prototype sidecar is unreadable; choose again."

    def _clear(self) -> None:
        old = self.centralWidget()
        if old is not None:
            old.deleteLater()

    def _build(self) -> None:
        self._clear()
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 12)
        outer.setSpacing(10)

        title = QLabel("Choose pro contact frame")
        title.setStyleSheet("font-size:24px;font-weight:700")
        subtitle = QLabel(
            "One exact decoded frame · one pro swing type · no playback · source files stay untouched"
        )
        subtitle.setStyleSheet("color:#59636e")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        self.image = QLabel(alignment=Qt.AlignCenter)
        self.image.setMinimumSize(640, 360)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image.setStyleSheet("background:#080b0f;border:1px solid #303842")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding:9px;background:#eef2f5;border:1px solid #cbd2d8")
        self.metadata = QLabel()
        self.metadata.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.metadata.setWordWrap(True)
        self.metadata.setStyleSheet("font-family:monospace;background:#111820;color:#d8e1ea;padding:12px")
        self.sidecar_preview = QLabel()
        self.sidecar_preview.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sidecar_preview.setWordWrap(True)
        self.sidecar_preview.setStyleSheet("font-family:monospace;background:#111820;color:#d8e1ea;padding:12px")
        navigation = self._navigation()
        shots = self._shot_selector()
        confirmation = self._confirmation()

        if self.variant == "A":
            outer.addWidget(self.image, 1)
            outer.addLayout(navigation)
            outer.addLayout(shots)
            outer.addWidget(self.status)
            outer.addLayout(confirmation)
            legacy = QLabel(
                "Keys: a:-10 frames   s:-1 frame   f:+1 frame   v:+10 frames   "
                "d:forehand   w:backhand   e:volley   r:serve   z:confirm   q:cancel"
            )
            legacy.setAlignment(Qt.AlignCenter)
            legacy.setStyleSheet("padding:7px;color:#44505b")
            outer.addWidget(legacy)
        elif self.variant == "B":
            columns = QHBoxLayout()
            left = QVBoxLayout()
            left.addWidget(self.image, 1)
            left.addLayout(navigation)
            right = QVBoxLayout()
            right.addWidget(self._step_label("1", "Find the exact contact frame"))
            right.addWidget(self.status)
            right.addWidget(self._step_label("2", "Choose the pro swing type"))
            right.addLayout(shots)
            right.addWidget(self._step_label("3", "Confirm and save for future runs"))
            right.addLayout(confirmation)
            right.addStretch(1)
            columns.addLayout(left, 3)
            columns.addLayout(right, 2)
            outer.addLayout(columns, 1)
        else:
            columns = QGridLayout()
            columns.addWidget(self.image, 0, 0, 2, 1)
            columns.addWidget(self.metadata, 0, 1)
            columns.addWidget(self.sidecar_preview, 1, 1)
            columns.setColumnStretch(0, 3)
            columns.setColumnStretch(1, 2)
            outer.addLayout(columns, 1)
            outer.addLayout(navigation)
            action_row = QHBoxLayout()
            action_row.addLayout(shots)
            action_row.addStretch(1)
            action_row.addLayout(confirmation)
            outer.addLayout(action_row)
            outer.addWidget(self.status)

        switcher = QFrame()
        switcher.setStyleSheet("QFrame{background:#e8edf0;border:1px solid #8d99a4;border-radius:12px;padding:4px}")
        switch = QHBoxLayout(switcher)
        previous = QPushButton("←")
        previous.clicked.connect(lambda: self._cycle_variant(-1))
        next_button = QPushButton("→")
        next_button.clicked.connect(lambda: self._cycle_variant(1))
        self.variant_label = QLabel(alignment=Qt.AlignCenter)
        switch.addStretch(1)
        switch.addWidget(previous)
        switch.addWidget(self.variant_label)
        switch.addWidget(next_button)
        switch.addStretch(1)
        outer.addWidget(switcher)
        self.setCentralWidget(root)
        self._refresh()

    def _step_label(self, number: str, text: str) -> QLabel:
        label = QLabel(f"{number}  {text}")
        label.setStyleSheet("font-size:17px;font-weight:700;margin-top:10px")
        return label

    def _navigation(self) -> QHBoxLayout:
        row = QHBoxLayout()
        for label, delta in (("−10 frames (A)", -10), ("Previous frame (S)", -1), ("Next frame (F)", 1), ("+10 frames (V)", 10)):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, amount=delta: self._step(amount))
            row.addWidget(button)
        return row

    def _shot_selector(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.shot_buttons = QButtonGroup(self)
        self.shot_buttons.setExclusive(True)
        for key, shot_type in SHOT_TYPES:
            button = QPushButton(f"{shot_type.title()} ({key})")
            button.setCheckable(True)
            button.setChecked(self.shot_type == shot_type)
            button.clicked.connect(lambda checked=False, value=shot_type: self._choose_shot(value))
            self.shot_buttons.addButton(button)
            row.addWidget(button)
        return row

    def _confirmation(self) -> QHBoxLayout:
        row = QHBoxLayout()
        cancel = QPushButton("Cancel (Q)")
        cancel.clicked.connect(self.close)
        self.confirm = QPushButton("Confirm contact (Z)")
        self.confirm.clicked.connect(self._confirm)
        row.addWidget(cancel)
        row.addWidget(self.confirm)
        return row

    def _cycle_variant(self, amount: int) -> None:
        keys = list(VARIANTS)
        self.variant = keys[(keys.index(self.variant) + amount) % len(keys)]
        self._build()

    def _step(self, amount: int) -> None:
        self.index = min(max(0, self.index + amount), len(self.info.frames) - 1)
        self.confirmed = False
        self.last_action = f"Moved to decoded frame {self.current.ordinal}."
        self._refresh()

    def _choose_shot(self, shot_type: str) -> None:
        self.shot_type = shot_type
        self.confirmed = False
        self.last_action = f"Selected {shot_type}."
        self._refresh()

    def _available_footage(self) -> tuple[Fraction, Fraction]:
        before = Fraction(self.current.pts - self.info.frames[0].pts) * self.info.time_base * self.pro_speed
        after = Fraction(self.info.frames[-1].pts - self.current.pts) * self.info.time_base * self.pro_speed
        return before, after

    def _validity(self) -> tuple[bool, str]:
        before, after = self._available_footage()
        missing = []
        if before < PRE_CONTACT:
            missing.append(f"{float(PRE_CONTACT - before):.3f} real-time seconds before contact")
        if after < POST_CONTACT:
            missing.append(f"{float(POST_CONTACT - after):.3f} real-time seconds after contact")
        if missing:
            return False, "Cannot confirm: missing " + " and ".join(missing) + ". Move contact or use a longer pro video."
        if self.shot_type is None:
            return False, "Choose forehand, backhand, volley, or serve before confirming."
        return True, "Ready: the full comparison window exists and both selections are complete."

    def _render_frame(self) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current.ordinal)
        ok, frame = self.capture.read()
        if not ok:
            self.image.setText(f"Could not display decoded frame {self.current.ordinal}")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(pixmap)

    def _refresh(self) -> None:
        valid, explanation = self._validity()
        self.confirm.setEnabled(valid)
        before, after = self._available_footage()
        rational_time = Fraction(self.current.pts) * self.info.time_base
        self.status.setText(
            f"Frame {self.current.ordinal} · PTS {self.current.pts} · "
            f"time base {self.info.time_base.numerator}/{self.info.time_base.denominator} · "
            f"exact time {rational_time.numerator}/{rational_time.denominator}s\n"
            f"{self.last_action}\n{explanation}\n"
            f"Available around contact after {float(self.pro_speed):g}× normalization: "
            f"{float(before):.3f}s before / {float(after):.3f}s after."
        )
        self.metadata.setText(
            "EXACT SELECTION STATE\n\n"
            f"decoded ordinal: {self.current.ordinal}\n"
            f"presentation timestamp: {self.current.pts}\n"
            f"stream time base: {self.info.time_base.numerator}/{self.info.time_base.denominator}\n"
            f"rational source time: {rational_time.numerator}/{rational_time.denominator} seconds\n"
            f"shot type: {self.shot_type or 'not selected'}\n\n"
            "No millisecond rounding. No frame-index ÷ nominal-fps estimate."
        )
        preview = json.dumps(self._sidecar_data(), indent=2)
        self.sidecar_preview.setText(
            f"SOURCE-BOUND SIDECAR PREVIEW\n{self.video.name}.tennis-compare.json\n\n{preview}"
        )
        self.variant_label.setText(f"{self.variant} · {VARIANTS[self.variant]}   ([ / ] also switch)")
        self._render_frame()

    def _confirm(self) -> None:
        valid, _ = self._validity()
        if not valid:
            return
        self.sidecar_path.write_text(json.dumps(self._sidecar_data(), indent=2) + "\n")
        self.confirmed = True
        self.last_action = (
            f"Prototype saved {self.sidecar_path}. Production would now close the picker and continue the command."
        )
        self._refresh()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key_A:
            self._step(-10)
        elif key == Qt.Key_S:
            self._step(-1)
        elif key == Qt.Key_F:
            self._step(1)
        elif key == Qt.Key_V:
            self._step(10)
        elif key in (Qt.Key_D, Qt.Key_W, Qt.Key_E, Qt.Key_R):
            mapping = {Qt.Key_D: "forehand", Qt.Key_W: "backhand", Qt.Key_E: "volley", Qt.Key_R: "serve"}
            self._choose_shot(mapping[key])
        elif key == Qt.Key_Z and self.confirm.isEnabled():
            self._confirm()
        elif key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_BracketLeft:
            self._cycle_variant(-1)
        elif key == Qt.Key_BracketRight:
            self._cycle_variant(1)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "image"):
            self._render_frame()

    def closeEvent(self, event) -> None:
        self.capture.release()
        super().closeEvent(event)


def parse_fraction(value: str) -> Fraction:
    result = Fraction(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PROTOTYPE: manual pro contact and shot-type picker")
    parser.add_argument("video", type=Path)
    parser.add_argument("--pro-speed", type=parse_fraction, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="A")
    parser.add_argument("--start-seconds", type=parse_fraction)
    args = parser.parse_args()

    info = probe_video(args.video)
    app = QApplication.instance() or QApplication(sys.argv)
    window = Picker(args.video, info, args.pro_speed, args.variant, args.start_seconds)
    window.show()
    app.exec()
    raise SystemExit(0 if window.confirmed else 1)


if __name__ == "__main__":
    main()

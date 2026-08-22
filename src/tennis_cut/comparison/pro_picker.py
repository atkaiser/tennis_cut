"""Focused exact-frame picker for a professional swing selection."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
from typing import Protocol

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from .pro_selection import DecodedFrame, PickerSelection, PickerSession


class FrameImageReader(Protocol):
    """Read one exact decoded frame as an encoded still image."""

    def read_frame(self, source: Path, frame: DecodedFrame) -> bytes: ...


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


class _SelectionDialog(QDialog):
    def __init__(
        self,
        source: Path,
        session: PickerSession,
        frame_reader: FrameImageReader,
    ) -> None:
        super().__init__()
        if not session.inspected_media.frames:
            raise ValueError("pro video has no decoded frames")
        self._source = source
        self._session = session
        self._frame_reader = frame_reader
        self._frame_index = len(session.inspected_media.frames) // 2
        self._shot_type: str | None = None
        self.selection: PickerSelection | None = None

        self.setWindowTitle(f"Select pro contact frame — {source.name}")
        layout = QVBoxLayout(self)

        self._image = QLabel(alignment=Qt.AlignCenter)
        self._image.setObjectName("frame_image")
        self._image.setMinimumSize(640, 360)
        self._image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._image, 1)

        self._status = QLabel()
        self._status.setObjectName("status")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        navigation = QHBoxLayout()
        for name, text, offset in (
            ("navigate_back_10", "-10", -10),
            ("navigate_back_1", "-1", -1),
            ("navigate_forward_1", "+1", 1),
            ("navigate_forward_10", "+10", 10),
        ):
            button = QPushButton(text)
            button.setObjectName(name)
            button.clicked.connect(lambda checked=False, step=offset: self._move(step))
            navigation.addWidget(button)
        layout.addLayout(navigation)

        shot_types = QHBoxLayout()
        shot_group = QButtonGroup(self)
        shot_group.setExclusive(True)
        for shot_type in ("forehand", "backhand", "volley", "serve"):
            button = QPushButton(shot_type.title())
            button.setObjectName(f"shot_{shot_type}")
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, selected=shot_type: self._select_shot(selected)
            )
            shot_group.addButton(button)
            shot_types.addWidget(button)
        layout.addLayout(shot_types)

        actions = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancel")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        actions.addStretch(1)
        self._confirm = QPushButton("Confirm")
        self._confirm.setObjectName("confirm")
        self._confirm.clicked.connect(self._confirm_selection)
        actions.addWidget(self._confirm)
        layout.addLayout(actions)

        self._refresh()

    @property
    def _frame(self) -> DecodedFrame:
        return self._session.inspected_media.frames[self._frame_index]

    def _move(self, offset: int) -> None:
        last_index = len(self._session.inspected_media.frames) - 1
        self._frame_index = min(max(0, self._frame_index + offset), last_index)
        self._refresh()

    def _select_shot(self, shot_type: str) -> None:
        self._shot_type = shot_type
        self._refresh_status()

    def _refresh(self) -> None:
        pixmap = QPixmap()
        if not pixmap.loadFromData(
            self._frame_reader.read_frame(self._source, self._frame)
        ):
            raise ValueError("could not display decoded pro frame")
        display_size = self._image.size().expandedTo(self._image.minimumSize())
        self._image.setPixmap(
            pixmap.scaled(
                display_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self._refresh_status()

    def _refresh_status(self) -> None:
        frame = self._frame
        status = self._session.confirmation_status(
            frame.ordinal, self._shot_type or ""
        )
        identity = (
            f"ordinal: {frame.ordinal} | PTS: {frame.pts} | "
            f"time base: {_fraction_text(frame.time_base)} | "
            f"source time: {_fraction_text(frame.timestamp)}"
        )
        guidance = []
        if self._shot_type is None:
            guidance.append("select a shot type")
        if status.missing_before:
            guidance.append(
                f"missing before: {_fraction_text(status.missing_before)} s"
            )
        if status.missing_after:
            guidance.append(
                f"missing after: {_fraction_text(status.missing_after)} s"
            )
        if not guidance:
            guidance.append("ready to confirm")
        self._status.setText(identity + "\n" + " | ".join(guidance))
        self._confirm.setEnabled(status.can_confirm)

    def _confirm_selection(self) -> None:
        if not self._confirm.isEnabled() or self._shot_type is None:
            return
        self.selection = PickerSelection(self._frame.ordinal, self._shot_type)
        self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        navigation = {
            Qt.Key_A: -10,
            Qt.Key_S: -1,
            Qt.Key_F: 1,
            Qt.Key_V: 10,
        }
        shot_types = {
            Qt.Key_D: "forehand",
            Qt.Key_W: "backhand",
            Qt.Key_E: "volley",
            Qt.Key_R: "serve",
        }
        if event.key() in navigation:
            self._move(navigation[event.key()])
        elif event.key() in shot_types:
            shot_type = shot_types[event.key()]
            button = self.findChild(QPushButton, f"shot_{shot_type}")
            button.click()
        elif event.key() == Qt.Key_Z:
            self._confirm.click()
        elif event.key() == Qt.Key_Q:
            self.reject()
        else:
            super().keyPressEvent(event)


class QtProPicker:
    """Run the modal pro contact-frame picker when the resolver requests it."""

    def __init__(self, source: Path, frame_reader: FrameImageReader) -> None:
        self._source = source
        self._frame_reader = frame_reader

    def pick(self, session: PickerSession) -> PickerSelection | None:
        application = QApplication.instance()
        if application is None:
            application = QApplication(sys.argv[:1])
        dialog = _SelectionDialog(self._source, session, self._frame_reader)
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.selection


__all__ = ["FrameImageReader", "QtProPicker"]

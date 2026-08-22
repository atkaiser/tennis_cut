from __future__ import annotations

from fractions import Fraction
from io import BytesIO
import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton

from tennis_cut.comparison.pro_picker import QtProPicker
from tennis_cut.comparison.pro_selection import (
    DecodedFrame,
    InspectedMedia,
    PickerSelection,
    PickerSession,
)


class IdentifiableFrameReader:
    def read_frame(self, source: Path, frame: DecodedFrame) -> bytes:
        image = Image.new("RGB", (32, 18), (frame.ordinal, 40, 80))
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        return encoded.getvalue()


def picker_session() -> PickerSession:
    return PickerSession(
        inspected_media=InspectedMedia(
            tuple(
                DecodedFrame(2, ordinal * 2, 100 + ordinal, Fraction(1, 10))
                for ordinal in range(31)
            )
        ),
        pro_speed=Fraction(1),
    )


class ProPickerInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_buttons_and_keys_drive_exact_confirmation_or_cancellation(self) -> None:
        expected_confirmation = PickerSelection(ordinal=30, shot_type="serve")

        def button_confirmation(dialog: QDialog) -> None:
            status = dialog.findChild(QLabel, "status")
            image = dialog.findChild(QLabel, "frame_image")
            confirm = dialog.findChild(QPushButton, "confirm")
            self.assertIsNotNone(image.pixmap())
            self.assertGreaterEqual(image.pixmap().width(), 640)
            displayed = image.pixmap().toImage()
            self.assertEqual(
                displayed.pixelColor(
                    displayed.width() // 2, displayed.height() // 2
                ).red(),
                30,
            )
            self.assertIn("ordinal: 30", status.text())
            self.assertIn("PTS: 115", status.text())
            self.assertIn("time base: 1/10", status.text())
            self.assertIn("source time: 23/2", status.text())
            self.assertFalse(confirm.isEnabled())

            for name in (
                "navigate_back_10",
                "navigate_back_1",
                "navigate_forward_1",
            ):
                QTest.mouseClick(dialog.findChild(QPushButton, name), Qt.LeftButton)
            displayed = image.pixmap().toImage()
            self.assertEqual(
                displayed.pixelColor(
                    displayed.width() // 2, displayed.height() // 2
                ).red(),
                10,
            )
            QTest.mouseClick(
                dialog.findChild(QPushButton, "shot_forehand"), Qt.LeftButton
            )
            self.assertFalse(confirm.isEnabled())
            self.assertIn("missing before: 7/10", status.text())
            QTest.mouseClick(
                dialog.findChild(QPushButton, "navigate_forward_10"), Qt.LeftButton
            )
            displayed = image.pixmap().toImage()
            self.assertEqual(
                displayed.pixelColor(
                    displayed.width() // 2, displayed.height() // 2
                ).red(),
                30,
            )
            for shot_type in ("backhand", "volley", "serve"):
                QTest.mouseClick(
                    dialog.findChild(QPushButton, f"shot_{shot_type}"),
                    Qt.LeftButton,
                )
                self.assertTrue(confirm.isEnabled())
            QTest.mouseClick(confirm, Qt.LeftButton)

        def key_confirmation(dialog: QDialog) -> None:
            for key in (Qt.Key_A, Qt.Key_S, Qt.Key_F, Qt.Key_V):
                QTest.keyClick(dialog, key)
            for key in (Qt.Key_D, Qt.Key_W, Qt.Key_E, Qt.Key_R):
                QTest.keyClick(dialog, key)
            QTest.keyClick(dialog, Qt.Key_Z)

        def cancel_button(dialog: QDialog) -> None:
            QTest.mouseClick(
                dialog.findChild(QPushButton, "cancel"), Qt.LeftButton
            )

        scenarios = (
            ("buttons confirm", button_confirmation, expected_confirmation),
            ("keys confirm", key_confirmation, expected_confirmation),
            ("cancel button", cancel_button, None),
            ("cancel key", lambda dialog: QTest.keyClick(dialog, Qt.Key_Q), None),
            ("window close", lambda dialog: dialog.close(), None),
        )
        for name, interaction, expected in scenarios:
            with self.subTest(name=name):
                picker = QtProPicker(Path("pro.mov"), IdentifiableFrameReader())
                interaction_errors: list[BaseException] = []

                def interact() -> None:
                    dialog = QApplication.activeModalWidget()
                    assert isinstance(dialog, QDialog)
                    try:
                        interaction(dialog)
                    except BaseException as error:
                        interaction_errors.append(error)
                        dialog.reject()

                QTimer.singleShot(0, interact)
                result = picker.pick(picker_session())

                if interaction_errors:
                    raise interaction_errors[0]
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

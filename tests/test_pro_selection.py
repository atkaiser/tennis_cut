from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from tennis_cut.comparison.pro_selection import (
    DecodedFrame,
    FileSidecarStore,
    InspectedMedia,
    PickerSession,
    PickerSelection,
    ProSelection,
    SelectionCancelled,
    SelectionProcessingFailure,
    resolve_pro_selection,
)


class MemorySidecarStore:
    def __init__(self) -> None:
        self.contents: dict[Path, str] = {}

    def read(self, path: Path) -> str | None:
        return self.contents.get(path)

    def write(self, path: Path, contents: str) -> None:
        self.contents[path] = contents


class FailingSidecarStore(MemorySidecarStore):
    def write(self, path: Path, contents: str) -> None:
        raise OSError("disk is read-only")


class ScriptedPicker:
    def __init__(self, selection: PickerSelection | None) -> None:
        self.selection = selection
        self.calls = 0

    def pick(self, session: PickerSession) -> PickerSelection | None:
        self.calls += 1
        return self.selection


class CapturingPicker(ScriptedPicker):
    def __init__(self, selection: PickerSelection | None) -> None:
        super().__init__(selection)
        self.session: PickerSession | None = None

    def pick(self, session: PickerSession) -> PickerSelection | None:
        self.session = session
        return super().pick(session)


def inspected_media() -> InspectedMedia:
    time_base = Fraction(1, 10)
    return InspectedMedia(
        frames=tuple(
            DecodedFrame(
                stream_index=2,
                ordinal=ordinal,
                pts=ordinal,
                time_base=time_base,
            )
            for ordinal in range(21)
        )
    )


class ResolveProSelectionTests(unittest.TestCase):
    def test_creates_immutable_exact_selection_and_source_bound_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pro_video = Path(directory) / "pro.mov"
            pro_video.write_bytes(b"pro source")
            store = MemorySidecarStore()
            picker = ScriptedPicker(PickerSelection(ordinal=12, shot_type="forehand"))

            result = resolve_pro_selection(
                pro_video=pro_video,
                pro_speed=Fraction(1, 1),
                inspected_media=inspected_media(),
                sidecar_store=store,
                picker=picker,
            )

            self.assertEqual(
                result,
                ProSelection(
                    source=pro_video,
                    frame=DecodedFrame(2, 12, 12, Fraction(1, 10)),
                    shot_type="forehand",
                ),
            )
            assert isinstance(result, ProSelection)
            with self.assertRaises(FrozenInstanceError):
                result.shot_type = "serve"  # ty: ignore[invalid-assignment]
            sidecar_path = pro_video.with_name("pro.mov.tennis-compare.json")
            self.assertEqual(
                json.loads(store.contents[sidecar_path]),
                {
                    "schema_version": 1,
                    "source": {
                        "name": "pro.mov",
                        "size_bytes": 10,
                        "mtime_ns": pro_video.stat().st_mtime_ns,
                    },
                    "video_stream": {
                        "index": 2,
                        "time_base": {"numerator": 1, "denominator": 10},
                    },
                    "contact_frame": {"ordinal": 12, "pts": 12},
                    "shot_type": "forehand",
                },
            )
            self.assertEqual(picker.calls, 1)

    def test_reuses_valid_sidecar_without_invoking_picker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pro_video = Path(directory) / "pro.mov"
            pro_video.write_bytes(b"pro source")
            store = MemorySidecarStore()
            creation_picker = ScriptedPicker(
                PickerSelection(ordinal=12, shot_type="backhand")
            )
            expected = resolve_pro_selection(
                pro_video=pro_video,
                pro_speed=Fraction(1, 1),
                inspected_media=inspected_media(),
                sidecar_store=store,
                picker=creation_picker,
            )
            reuse_picker = ScriptedPicker(None)

            reused = resolve_pro_selection(
                pro_video=pro_video,
                pro_speed=Fraction(1, 1),
                inspected_media=inspected_media(),
                sidecar_store=store,
                picker=reuse_picker,
            )

            self.assertEqual(reused, expected)
            self.assertEqual(reuse_picker.calls, 0)

    def test_invalid_or_deleted_sidecars_invoke_picker(self) -> None:
        invalidators = {
            "deleted": lambda payload: None,
            "malformed": lambda payload: "not-json",
            "malformed time base": lambda payload: json.dumps(
                {
                    **payload,
                    "video_stream": {
                        **payload["video_stream"],
                        "time_base": {"numerator": 1, "denominator": 0},
                    },
                }
            ),
            "stale": lambda payload: json.dumps(
                {**payload, "source": {**payload["source"], "size_bytes": 999}}
            ),
            "unsupported type": lambda payload: json.dumps(
                {**payload, "shot_type": "overhead"}
            ),
            "ordinal no longer resolves to PTS": lambda payload: json.dumps(
                {
                    **payload,
                    "contact_frame": {**payload["contact_frame"], "pts": 999},
                }
            ),
            "boolean schema version": lambda payload: json.dumps(
                {**payload, "schema_version": True}
            ),
            "floating source size": lambda payload: json.dumps(
                {**payload, "source": {**payload["source"], "size_bytes": 10.0}}
            ),
            "floating ordinal": lambda payload: json.dumps(
                {
                    **payload,
                    "contact_frame": {**payload["contact_frame"], "ordinal": 12.0},
                }
            ),
            "floating PTS": lambda payload: json.dumps(
                {
                    **payload,
                    "contact_frame": {**payload["contact_frame"], "pts": 12.0},
                }
            ),
        }
        for label, invalidate in invalidators.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                pro_video = Path(directory) / "pro.mov"
                pro_video.write_bytes(b"pro source")
                store = MemorySidecarStore()
                resolve_pro_selection(
                    pro_video=pro_video,
                    pro_speed=Fraction(1, 1),
                    inspected_media=inspected_media(),
                    sidecar_store=store,
                    picker=ScriptedPicker(
                        PickerSelection(ordinal=12, shot_type="forehand")
                    ),
                )
                sidecar_path = pro_video.with_name("pro.mov.tennis-compare.json")
                payload = json.loads(store.contents[sidecar_path])
                invalid_contents = invalidate(payload)
                if invalid_contents is None:
                    del store.contents[sidecar_path]
                else:
                    store.contents[sidecar_path] = invalid_contents
                picker = ScriptedPicker(
                    PickerSelection(ordinal=13, shot_type="serve")
                )

                result = resolve_pro_selection(
                    pro_video=pro_video,
                    pro_speed=Fraction(1, 1),
                    inspected_media=inspected_media(),
                    sidecar_store=store,
                    picker=picker,
                )

                assert isinstance(result, ProSelection)
                self.assertEqual(result.frame.ordinal, 13)
                self.assertEqual(result.shot_type, "serve")
                self.assertEqual(picker.calls, 1)

    def test_reports_saved_selection_without_footage_at_current_pro_speed(self) -> None:
        media = InspectedMedia(
            frames=tuple(
                DecodedFrame(2, ordinal, ordinal, Fraction(1, 1))
                for ordinal in range(21)
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            pro_video = Path(directory) / "pro.mov"
            pro_video.touch()
            store = MemorySidecarStore()
            resolve_pro_selection(
                pro_video=pro_video,
                pro_speed=Fraction(1, 1),
                inspected_media=media,
                sidecar_store=store,
                picker=ScriptedPicker(
                    PickerSelection(ordinal=2, shot_type="forehand")
                ),
            )
            picker = ScriptedPicker(PickerSelection(ordinal=5, shot_type="forehand"))

            result = resolve_pro_selection(
                pro_video=pro_video,
                pro_speed=Fraction(1, 4),
                inspected_media=media,
                sidecar_store=store,
                picker=picker,
            )

            self.assertEqual(
                result,
                SelectionProcessingFailure(
                    stage="pro selection",
                    message=(
                        "saved selection lacks required footage at current pro speed"
                    ),
                ),
            )
            self.assertEqual(picker.calls, 0)

    def test_picker_uses_exact_pro_speed_footage_gating(self) -> None:
        media = InspectedMedia(
            frames=tuple(
                DecodedFrame(0, ordinal, ordinal, Fraction(1, 1))
                for ordinal in range(9)
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            pro_video = Path(directory) / "quarter-speed.mov"
            pro_video.touch()
            store = MemorySidecarStore()
            picker = CapturingPicker(PickerSelection(ordinal=5, shot_type="volley"))

            result = resolve_pro_selection(
                pro_video=pro_video,
                pro_speed=Fraction(1, 4),
                inspected_media=media,
                sidecar_store=store,
                picker=picker,
            )

            assert picker.session is not None
            status = picker.session.confirmation_status(5, "volley")
            self.assertTrue(status.can_confirm)
            self.assertEqual(status.available_before, Fraction(5, 4))
            self.assertEqual(status.available_after, Fraction(3, 4))
            self.assertEqual(status.missing_before, Fraction(0, 1))
            self.assertEqual(status.missing_after, Fraction(0, 1))
            self.assertIsInstance(result, ProSelection)

    def test_insufficient_or_unsupported_picker_selection_is_not_confirmed(self) -> None:
        media = InspectedMedia(
            frames=tuple(
                DecodedFrame(0, ordinal, ordinal, Fraction(1, 1))
                for ordinal in range(9)
            )
        )
        cases = ((4, "serve"), (5, "overhead"))
        for ordinal, shot_type in cases:
            with (
                self.subTest(ordinal=ordinal, shot_type=shot_type),
                tempfile.TemporaryDirectory() as directory,
            ):
                pro_video = Path(directory) / "quarter-speed.mov"
                pro_video.touch()
                store = MemorySidecarStore()

                result = resolve_pro_selection(
                    pro_video=pro_video,
                    pro_speed=Fraction(1, 4),
                    inspected_media=media,
                    sidecar_store=store,
                    picker=ScriptedPicker(
                        PickerSelection(ordinal=ordinal, shot_type=shot_type)
                    ),
                )

                self.assertIsInstance(result, SelectionProcessingFailure)
                self.assertEqual(store.contents, {})

    def test_cancellation_returns_typed_result_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pro_video = Path(directory) / "pro.mov"
            pro_video.touch()
            store = MemorySidecarStore()

            result = resolve_pro_selection(
                pro_video=pro_video,
                pro_speed=Fraction(1, 1),
                inspected_media=inspected_media(),
                sidecar_store=store,
                picker=ScriptedPicker(None),
            )

            self.assertIsInstance(result, SelectionCancelled)
            self.assertEqual(store.contents, {})

    def test_write_failure_returns_typed_processing_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pro_video = Path(directory) / "pro.mov"
            pro_video.touch()

            result = resolve_pro_selection(
                pro_video=pro_video,
                pro_speed=Fraction(1, 1),
                inspected_media=inspected_media(),
                sidecar_store=FailingSidecarStore(),
                picker=ScriptedPicker(
                    PickerSelection(ordinal=12, shot_type="forehand")
                ),
            )

            self.assertEqual(
                result,
                SelectionProcessingFailure(
                    stage="persist pro selection",
                    message="disk is read-only",
                ),
            )


class FileSidecarStoreTests(unittest.TestCase):
    def test_round_trips_sidecar_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pro.mov.tennis-compare.json"
            store = FileSidecarStore()

            store.write(path, '{"schema_version": 1}\n')

            self.assertEqual(store.read(path), '{"schema_version": 1}\n')


if __name__ == "__main__":
    unittest.main()

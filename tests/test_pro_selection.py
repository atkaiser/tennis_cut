from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path
import unittest

from tennis_cut.comparison.pro_selection import (
    DecodedFrame,
    InspectedMedia,
    ProSelection,
    SelectionProcessingFailure,
    find_pro_contact,
)
from tennis_cut.visual_contact import ContactSelection, FrameEvidence, VisualFrame


def inspected_media() -> InspectedMedia:
    return InspectedMedia(
        tuple(
            DecodedFrame(2, ordinal, ordinal, Fraction(1, 10))
            for ordinal in range(21)
        )
    )


def contact(ordinal: int) -> ContactSelection:
    frame = VisualFrame(
        FrameEvidence(
            ordinal=ordinal,
            timestamp=Fraction(ordinal, 10),
            detections=(),
            stream_index=2,
            pts=ordinal,
            time_base=Fraction(1, 10),
        )
    )
    return ContactSelection(frame, 0.9, (ordinal,), None)


class StubFinder:
    def __init__(self, result: ContactSelection) -> None:
        self.result = result
        self.calls: list[tuple[Path, Fraction, Fraction]] = []

    def select(
        self,
        source: Path,
        candidate_timestamp: Fraction,
        *,
        radius: Fraction = Fraction(2, 5),
    ) -> ContactSelection:
        self.calls.append((source, candidate_timestamp, radius))
        return self.result


class FindProContactTests(unittest.TestCase):
    def test_searches_middle_75_percent_and_finds_exact_contact(self) -> None:
        source = Path("pro.mov")
        finder = StubFinder(contact(12))

        result = find_pro_contact(
            pro_video=source,
            pro_speed=Fraction(1),
            inspected_media=inspected_media(),
            finder=finder,
        )

        self.assertEqual(
            result,
            ProSelection(source, DecodedFrame(2, 12, 12, Fraction(1, 10))),
        )
        self.assertEqual(
            finder.calls,
            [(source, Fraction(1), Fraction(3, 4))],
        )
        assert isinstance(result, ProSelection)
        with self.assertRaises(FrozenInstanceError):
            result.shot_type = "serve"  # ty: ignore[invalid-assignment]

    def test_reports_visual_finder_omission(self) -> None:
        finder = StubFinder(
            ContactSelection(None, 0.0, (), "no moving ball evidence")
        )

        result = find_pro_contact(
            pro_video=Path("pro.mov"),
            pro_speed=Fraction(1),
            inspected_media=inspected_media(),
            finder=finder,
        )

        self.assertEqual(
            result,
            SelectionProcessingFailure(
                "pro contact detection", "no moving ball evidence"
            ),
        )

    def test_rejects_contact_that_does_not_match_inspected_source(self) -> None:
        finder = StubFinder(contact(99))

        result = find_pro_contact(
            pro_video=Path("pro.mov"),
            pro_speed=Fraction(1),
            inspected_media=inspected_media(),
            finder=finder,
        )

        self.assertEqual(
            result,
            SelectionProcessingFailure(
                "pro contact detection",
                "visual contact finder returned a frame outside the pro source",
            ),
        )

    def test_rejects_contact_without_complete_normalized_window(self) -> None:
        finder = StubFinder(contact(10))

        result = find_pro_contact(
            pro_video=Path("pro.mov"),
            pro_speed=Fraction(1, 4),
            inspected_media=inspected_media(),
            finder=finder,
        )

        self.assertEqual(
            result,
            SelectionProcessingFailure(
                "pro contact detection",
                "detected contact lacks the complete comparison window",
            ),
        )

    def test_reports_empty_pro_video_without_calling_finder(self) -> None:
        finder = StubFinder(contact(0))

        result = find_pro_contact(
            pro_video=Path("pro.mov"),
            pro_speed=Fraction(1),
            inspected_media=InspectedMedia(()),
            finder=finder,
        )

        self.assertEqual(
            result,
            SelectionProcessingFailure(
                "pro contact detection", "pro video has no decoded frames"
            ),
        )
        self.assertEqual(finder.calls, [])


if __name__ == "__main__":
    unittest.main()

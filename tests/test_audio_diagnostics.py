from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from tennis_cut.audio_diagnostics import (
    AudioWindowDiagnostic,
    build_parser,
    describe_audio_windows,
    write_audio_window_report,
)
from tennis_cut.comparison.diagnostics import CandidateDiagnostic
from tennis_cut.swing_detection import (
    BOUNCE_COLLAPSE_REASON,
    FINAL_SUPPRESSION_REASON,
    AudioCandidate,
)


class AudioWindowDispositionTests(unittest.TestCase):
    def test_explains_every_audio_pipeline_stage(self) -> None:
        below = AudioCandidate(0.125, 0.1, 0)
        bounce = AudioCandidate(1.0, 0.8, 1)
        rejected = AudioCandidate(1.5, 0.7, 2)
        chosen = AudioCandidate(3.0, 0.9, 3)
        final_suppressed = AudioCandidate(4.0, 0.8, 4)
        initial_suppressed = AudioCandidate(3.05, 0.6, 5)
        all_windows = [
            below,
            bounce,
            rejected,
            chosen,
            initial_suppressed,
            final_suppressed,
        ]
        initial_candidates = (bounce, rejected, chosen, final_suppressed)
        diagnostics = (
            CandidateDiagnostic(
                1,
                Fraction(1),
                "omitted",
                BOUNCE_COLLAPSE_REASON,
                audio_score=0.8,
            ),
            CandidateDiagnostic(
                2,
                Fraction(3, 2),
                "omitted",
                "swing classifier: not a swing",
                audio_score=0.7,
            ),
            CandidateDiagnostic(
                3,
                Fraction(3),
                "visual contact pending",
                "accepted by person and swing classifiers",
                audio_score=0.9,
                shot_type="forehand",
            ),
            CandidateDiagnostic(
                4,
                Fraction(4),
                "omitted",
                FINAL_SUPPRESSION_REASON,
                audio_score=0.8,
                shot_type="forehand",
            ),
        )

        described = describe_audio_windows(
            all_windows,
            initial_candidates,
            (initial_suppressed,),
            diagnostics,
        )

        self.assertEqual(
            [item.decision for item in described],
            [
                "below threshold",
                "suppressed",
                "rejected",
                "chosen",
                "suppressed",
                "suppressed",
            ],
        )
        self.assertIn("retained later window 2", described[1].reason)
        self.assertEqual(described[2].reason, "swing classifier: not a swing")
        self.assertIn("preferred window 3", described[4].reason)
        self.assertIn("preferred window 3", described[5].reason)


class AudioWindowReportTests(unittest.TestCase):
    def test_embeds_score_bars_and_zero_based_frame_numbers(self) -> None:
        windows = (
            AudioWindowDiagnostic(
                AudioCandidate(0.125, 0.25, 0),
                "below threshold",
                "audio score ≤ 0.50",
            ),
            AudioWindowDiagnostic(
                AudioCandidate(0.175, 0.75, 1),
                "chosen",
                "accepted by person and swing classifiers",
                "forehand",
            ),
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            screenshots = (directory / "one.jpg", directory / "two.jpg")
            for screenshot in screenshots:
                screenshot.write_bytes(b"jpeg")
            output = directory / "report.html"

            write_audio_window_report(
                output,
                Path("user.mov"),
                windows,
                screenshots,
                fps=120.0,
                stride_s=0.05,
            )

            document = output.read_text()

        self.assertEqual(document.count("data:image/jpeg;base64,"), 2)
        self.assertIn('style="--score:0.250000"', document)
        self.assertIn('style="--score:0.750000"', document)
        self.assertIn('<td class="frame">15</td>', document)
        self.assertIn('<td class="frame">21</td>', document)
        self.assertIn("frame # (0-based)", document)

    def test_cli_defaults_output_beside_input(self) -> None:
        args = build_parser().parse_args(["videos/user.mov"])
        output = args.output or args.input.with_name(
            f"{args.input.stem}_audio_window_diagnostics.html"
        )

        self.assertEqual(
            output,
            Path("videos/user_audio_window_diagnostics.html"),
        )


if __name__ == "__main__":
    unittest.main()

"""Command-line adapter for the comparison workflow."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import logging
from pathlib import Path
import sys
from typing import Sequence

from tennis_cut.swing_detection import (
    DEFAULT_AUDIO_MODEL,
    DEFAULT_SHOT_MODEL,
    DEFAULT_SHOT_TYPE_MODEL,
    DEFAULT_TEMPORAL_RANKER_MODEL,
)

from .workflow import (
    ComparisonDependencies,
    ComparisonProcessingFailed,
    ComparisonRequest,
    ComparisonSelectionCancelled,
    InvalidComparisonRequest,
    SystemComparisonDependencies,
    compare_videos,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, f"{self.prog}: {message}\n")


def _fraction(value: str) -> Fraction:
    try:
        return Fraction(Decimal(value))
    except (InvalidOperation, OverflowError, ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="tennis-compare",
        description="Create contact-aligned user-to-pro swing comparisons",
    )
    parser.add_argument("user_video", type=Path)
    parser.add_argument("pro_video", type=Path)
    parser.add_argument("--pro-speed", type=_fraction, required=True, metavar="FACTOR")
    parser.add_argument("--slowmo", type=_fraction, default=Fraction(1, 16))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./processed_vids"))
    parser.add_argument("--clips", action="store_true")
    parser.add_argument("--audio-model", type=Path, default=DEFAULT_AUDIO_MODEL)
    parser.add_argument("--shot-model", type=Path, default=DEFAULT_SHOT_MODEL)
    parser.add_argument(
        "--shot-type-model", type=Path, default=DEFAULT_SHOT_TYPE_MODEL
    )
    parser.add_argument(
        "--temporal-ranker-model",
        "--visual-contact-ranker-model",
        dest="temporal_ranker_model",
        type=Path,
        default=DEFAULT_TEMPORAL_RANKER_MODEL,
        help="versioned visual contact ranker artifact (default: models/temporal_ranker.json)",
    )
    parser.add_argument(
        "--prototype",
        action="store_true",
        help="use the original HGB prototype corroborator (requires --prototype-records)",
    )
    parser.add_argument(
        "--prototype-records",
        type=Path,
        help="labeled pilot records used to train the prototype temporal ranker",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true")
    verbosity.add_argument("-q", "--quiet", action="store_true")
    return parser


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    if quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: ComparisonDependencies | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)
    if args.prototype and args.prototype_records is None:
        print("tennis-compare: --prototype requires --prototype-records PATH", file=sys.stderr)
        return 2
    ranker_model = None if args.prototype else args.temporal_ranker_model
    # Dependency fakes deliberately opt out so existing orchestration tests can
    # exercise planning without requiring a local production checkpoint.
    if dependencies is not None and ranker_model == DEFAULT_TEMPORAL_RANKER_MODEL:
        ranker_model = None
    request = ComparisonRequest(
        user_video=args.user_video,
        pro_video=args.pro_video,
        pro_speed=args.pro_speed,
        slow_motion=args.slowmo,
        output_directory=args.output_dir,
        clips=args.clips,
        audio_model=args.audio_model,
        shot_model=args.shot_model,
        shot_type_model=args.shot_type_model,
        temporal_ranker_model=ranker_model,
        prototype_records=args.prototype_records if args.prototype else None,
        device=args.device,
    )
    try:
        result = compare_videos(
            request, dependencies or SystemComparisonDependencies()
        )
    except InvalidComparisonRequest as error:
        print(f"tennis-compare: {error}", file=sys.stderr)
        return 2
    except ComparisonSelectionCancelled as error:
        print(f"tennis-compare: {error}", file=sys.stderr)
        return 1
    except ComparisonProcessingFailed as error:
        if args.verbose and error.diagnostics:
            logging.error(error.diagnostics)
        print(f"tennis-compare: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        if args.verbose:
            logging.exception("unexpected comparison failure")
        print(f"tennis-compare: processing failed: {error}", file=sys.stderr)
        return 1

    if result.published_paths and not args.quiet:
        print(result.published_paths[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

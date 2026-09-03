"""Generate self-contained diagnostics for every dense audio-model window."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import dataclass
import html
from pathlib import Path
import shutil
import subprocess
import tempfile

from .comparison.diagnostics import CandidateDiagnostic, SwingDiagnosticsRecorder
from .swing_detection import (
    BOUNCE_COLLAPSE_REASON,
    BOUNCE_GAP_MAX,
    BOUNCE_GAP_MIN,
    DEFAULT_AUDIO_MODEL,
    DEFAULT_SHOT_MODEL,
    DEFAULT_SHOT_TYPE_MODEL,
    FINAL_PEAK_MIN_SEPARATION,
    FINAL_SUPPRESSION_REASON,
    INITIAL_PEAK_MIN_SEPARATION,
    PEAK_THRESHOLD,
    WINDOW_DURATION,
    AudioCandidate,
    DetectionConfig,
    PopDetector,
    _detect_user_swings_with_details,
    _suppress_audio_candidates,
    extract_audio,
    probe_video,
)


@dataclass(frozen=True)
class AudioWindowDiagnostic:
    """One dense model window and its final pipeline disposition."""

    candidate: AudioCandidate
    decision: str
    reason: str
    shot_type: str | None = None


def _suppression_blockers(
    candidates: list[AudioCandidate] | tuple[AudioCandidate, ...],
    separation: float,
) -> dict[int, AudioCandidate]:
    kept: list[AudioCandidate] = []
    blockers: dict[int, AudioCandidate] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (-item.score, item.timestamp, item.source_index),
    ):
        blocker = next(
            (
                selected
                for selected in kept
                if abs(candidate.timestamp - selected.timestamp) < separation - 1e-9
            ),
            None,
        )
        if blocker is None:
            kept.append(candidate)
        else:
            blockers[candidate.source_index] = blocker
    return blockers


def _bounce_successors(
    candidates: tuple[AudioCandidate, ...],
) -> dict[int, AudioCandidate]:
    successors: dict[int, AudioCandidate] = {}
    chronological = sorted(candidates, key=lambda item: item.timestamp)
    if not chronological:
        return successors
    tail = chronological[0]
    for candidate in chronological[1:]:
        gap = candidate.timestamp - tail.timestamp
        if BOUNCE_GAP_MIN - 1e-9 <= gap <= BOUNCE_GAP_MAX + 1e-9:
            successors[tail.source_index] = candidate
            tail = candidate
        else:
            tail = candidate
    return successors


def describe_audio_windows(
    all_windows: list[AudioCandidate],
    initial_candidates: tuple[AudioCandidate, ...],
    initial_omissions: tuple[AudioCandidate, ...],
    diagnostics: tuple[CandidateDiagnostic, ...],
) -> tuple[AudioWindowDiagnostic, ...]:
    """Explain every window using the decisions from the production pipeline."""

    diagnostic_by_index = {
        item.audio_candidate_index: item for item in diagnostics
    }
    initial_omission_indexes = {
        candidate.source_index for candidate in initial_omissions
    }
    initial_blockers = _suppression_blockers(
        [window for window in all_windows if window.score > PEAK_THRESHOLD],
        INITIAL_PEAK_MIN_SEPARATION,
    )
    bounce_successors = _bounce_successors(initial_candidates)
    final_eligible = [
        candidate
        for candidate in initial_candidates
        if diagnostic_by_index[candidate.source_index].reason
        in {
            "accepted by person and swing classifiers",
            FINAL_SUPPRESSION_REASON,
        }
    ]
    final_blockers = _suppression_blockers(
        final_eligible,
        FINAL_PEAK_MIN_SEPARATION,
    )

    described: list[AudioWindowDiagnostic] = []
    for candidate in all_windows:
        if candidate.score <= PEAK_THRESHOLD:
            described.append(
                AudioWindowDiagnostic(
                    candidate,
                    "below threshold",
                    f"audio score ≤ {PEAK_THRESHOLD:.2f}",
                )
            )
            continue
        if candidate.source_index in initial_omission_indexes:
            blocker = initial_blockers[candidate.source_index]
            described.append(
                AudioWindowDiagnostic(
                    candidate,
                    "suppressed",
                    "initial 0.25s suppression; preferred window "
                    f"{blocker.source_index} at {blocker.timestamp:.3f}s "
                    f"with score {blocker.score:.3f}",
                )
            )
            continue

        diagnostic = diagnostic_by_index[candidate.source_index]
        if diagnostic.reason == BOUNCE_COLLAPSE_REASON:
            successor = bounce_successors[candidate.source_index]
            described.append(
                AudioWindowDiagnostic(
                    candidate,
                    "suppressed",
                    f"{BOUNCE_COLLAPSE_REASON}; retained later window "
                    f"{successor.source_index} at {successor.timestamp:.3f}s",
                    diagnostic.shot_type,
                )
            )
        elif diagnostic.reason == FINAL_SUPPRESSION_REASON:
            blocker = final_blockers[candidate.source_index]
            described.append(
                AudioWindowDiagnostic(
                    candidate,
                    "suppressed",
                    f"{FINAL_SUPPRESSION_REASON}; preferred window "
                    f"{blocker.source_index} at {blocker.timestamp:.3f}s "
                    f"with score {blocker.score:.3f}",
                    diagnostic.shot_type,
                )
            )
        elif diagnostic.disposition == "visual contact pending":
            described.append(
                AudioWindowDiagnostic(
                    candidate,
                    "chosen",
                    "accepted by person and swing classifiers",
                    diagnostic.shot_type,
                )
            )
        else:
            described.append(
                AudioWindowDiagnostic(
                    candidate,
                    "rejected",
                    diagnostic.reason,
                    diagnostic.shot_type,
                )
            )
    return tuple(described)


def _extract_screenshots(
    source: Path,
    directory: Path,
    count: int,
    stride_s: float,
) -> tuple[Path, ...]:
    pattern = directory / "window_%06d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(WINDOW_DURATION / 2),
            "-i",
            str(source),
            "-an",
            "-vf",
            f"fps={1 / stride_s},scale=320:-2",
            "-frames:v",
            str(count),
            "-q:v",
            "7",
            str(pattern),
            "-y",
        ],
        check=True,
    )
    paths = tuple(sorted(directory.glob("window_*.jpg")))
    if not paths:
        raise RuntimeError("ffmpeg did not produce any window screenshots")
    if len(paths) > count:
        raise RuntimeError(f"expected at most {count} screenshots, got {len(paths)}")
    return paths + (paths[-1],) * (count - len(paths))


def _extract_candidate_frames(
    source: Path,
    directory: Path,
    candidates: tuple[AudioCandidate, ...],
    fps: float,
    screenshots: tuple[Path, ...],
) -> dict[float, Path]:
    if not candidates:
        return {}
    pattern = directory / "candidate_%06d.jpg"
    selection = "+".join(
        f"eq(n\\,{round(candidate.timestamp * fps)})" for candidate in candidates
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select={selection}",
            "-fps_mode",
            "passthrough",
            "-q:v",
            "3",
            str(pattern),
            "-y",
        ],
        check=True,
    )
    paths = tuple(sorted(directory.glob("candidate_*.jpg")))
    if len(paths) > len(candidates):
        raise RuntimeError(
            f"expected at most {len(candidates)} candidate frames, got {len(paths)}"
        )
    frames = {
        candidate.timestamp: path
        for candidate, path in zip(candidates[: len(paths)], paths, strict=True)
    }
    frames.update(
        {
            candidate.timestamp: screenshots[candidate.source_index]
            for candidate in candidates[len(paths) :]
        }
    )
    return frames


def _image_data(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def write_audio_window_report(
    output: Path,
    source: Path,
    windows: tuple[AudioWindowDiagnostic, ...],
    screenshots: tuple[Path, ...],
    fps: float,
    stride_s: float,
) -> None:
    """Write one portable HTML table with all screenshots embedded."""

    if len(windows) != len(screenshots):
        raise ValueError("window and screenshot counts differ")
    counts = Counter(window.decision for window in windows)
    rows: list[str] = []
    for window, screenshot in zip(windows, screenshots, strict=True):
        candidate = window.candidate
        start = candidate.timestamp - WINDOW_DURATION / 2
        end = candidate.timestamp + WINDOW_DURATION / 2
        frame_number = round(candidate.timestamp * fps)
        row_class = window.decision.replace(" ", "-")
        rows.append(
            f'''<tr class="{row_class}">
<td class="numeric">{candidate.source_index}</td>
<td><img loading="lazy" src="{_image_data(screenshot)}" alt="Frame {frame_number} at {candidate.timestamp:.3f}s"></td>
<td class="numeric">{start:.3f}–{end:.3f}s</td>
<td class="numeric">{candidate.timestamp:.3f}s</td>
<td class="frame">{frame_number}</td>
<td class="score"><span style="--score:{candidate.score:.6f}">{candidate.score:.6f}</span></td>
<td><strong>{html.escape(window.decision)}</strong></td>
<td>{html.escape(window.reason)}</td>
<td>{html.escape(window.shot_type or "")}</td>
</tr>'''
        )

    summary = " · ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audio-window diagnostics · {html.escape(source.name)}</title><style>
body{{font:14px/1.35 system-ui,sans-serif;margin:0;background:#f4f1ea;color:#18201d}}main{{padding:28px}}h1{{margin:0 0 8px}}p{{color:#59635e}}
table{{border-collapse:separate;border-spacing:0;width:100%;background:white}}th{{position:sticky;top:0;background:#18201d;color:white;z-index:2}}
td,th{{padding:8px 10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:middle}}.numeric,.frame{{font-variant-numeric:tabular-nums;white-space:nowrap}}
.score{{width:260px;font-variant-numeric:tabular-nums}}.score span{{display:block;padding:7px 9px;border-radius:4px;background:linear-gradient(90deg,#e99a82 calc(var(--score) * 100%),#ece9e0 0)}}
img{{display:block;width:320px;height:auto;border-radius:4px}}tr.chosen{{background:#dcf4e8}}tr.suppressed{{background:#fff0d4}}tr.rejected{{background:#f8dddd}}tr.below-threshold{{color:#65706a}}
</style></head><body><main><h1>Audio-window diagnostics</h1>
<p>{html.escape(str(source))}<br>{len(windows)} windows · {WINDOW_DURATION * 1000:.0f} ms duration · {stride_s * 1000:.0f} ms stride · 0-based frames at {fps:g} fps · threshold &gt; {PEAK_THRESHOLD:.2f}<br>{html.escape(summary)}</p>
<table><thead><tr><th>window #</th><th>center frame</th><th>window</th><th>center</th><th>frame # (0-based)</th><th>audio score</th><th>decision</th><th>reason</th><th>shot type</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


def generate_audio_window_report(
    source: Path,
    output: Path,
    detection_config: DetectionConfig,
) -> None:
    """Run dense audio scoring and shared swing classification for one video."""

    media_info = probe_video(source)
    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name)
        wav_path = temporary / "audio.wav"
        print(f"Scoring {source.name}", flush=True)
        extract_audio(source, wav_path)
        detector = PopDetector(
            detection_config.audio_model,
            device=detection_config.device,
        )
        all_windows = detector.score_windows(wav_path)
        above_threshold = [
            window for window in all_windows if window.score > PEAK_THRESHOLD
        ]
        initial_candidates, initial_omissions = _suppress_audio_candidates(
            above_threshold,
            INITIAL_PEAK_MIN_SEPARATION,
        )
        fps = media_info["fps"]
        print(f"Extracting {len(all_windows)} center frames", flush=True)
        screenshots = _extract_screenshots(
            source,
            temporary,
            len(all_windows),
            detector.stride_s,
        )
        print(
            f"Extracting {len(initial_candidates)} classifier frames",
            flush=True,
        )
        classifier_frames = _extract_candidate_frames(
            source,
            temporary,
            initial_candidates,
            fps,
            screenshots,
        )

        def reuse_candidate_frame(
            _source: Path,
            timestamp: float,
            destination: Path,
        ) -> None:
            shutil.copyfile(classifier_frames[timestamp], destination)

        recorder = SwingDiagnosticsRecorder(source)
        print(
            f"Classifying {len(initial_candidates)} event candidates",
            flush=True,
        )
        _detect_user_swings_with_details(
            source,
            detection_config,
            media_info,
            report_progress=True,
            diagnostics=recorder,
            initial_candidates=initial_candidates,
            frame_extractor=reuse_candidate_frame,
        )
        windows = describe_audio_windows(
            all_windows,
            initial_candidates,
            initial_omissions,
            recorder.snapshot().candidates,
        )
        write_audio_window_report(
            output,
            source,
            windows,
            screenshots,
            fps,
            detector.stride_s,
        )
    print(f"Wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit every dense audio-model window in a user video",
    )
    parser.add_argument("input", type=Path, help="User video to inspect")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path (defaults beside the input video)",
    )
    parser.add_argument("--audio-model", type=Path, default=DEFAULT_AUDIO_MODEL)
    parser.add_argument("--shot-model", type=Path, default=DEFAULT_SHOT_MODEL)
    parser.add_argument(
        "--shot-type-model",
        type=Path,
        default=DEFAULT_SHOT_TYPE_MODEL,
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.input.with_name(
        f"{args.input.stem}_audio_window_diagnostics.html"
    )
    generate_audio_window_report(
        args.input,
        output,
        DetectionConfig(
            audio_model=args.audio_model,
            shot_model=args.shot_model,
            shot_type_model=args.shot_type_model,
            device=args.device,
        ),
    )


if __name__ == "__main__":
    main()

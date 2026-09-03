"""Semantic preflight and orchestration for video comparisons."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import logging
import os
from pathlib import Path
import shutil
import tempfile
from time import monotonic
from typing import Protocol

from tennis_cut.swing_detection import (
    DEFAULT_AUDIO_MODEL,
    DEFAULT_SHOT_MODEL,
    DEFAULT_SHOT_TYPE_MODEL,
    DEFAULT_TEMPORAL_RANKER_MODEL,
    DetectedSwing,
    DetectionConfig,
    detect_comparison_user_swings,
    resolve_device,
)
from tennis_cut.temporal_ranker import TemporalRankerArtifactError, load_temporal_ranker

from .media import PlayerLocator
from .planning import (
    ArtifactRequest,
    ComparisonRenderPlan,
    ComparisonSource,
    PlayerObservation,
    SelectedSourceWindow,
    build_render_plan,
    prepare_source_window,
    select_comparison_windows,
)
from .pro_selection import (
    InspectedMedia,
    ProSelection,
    SelectionProcessingFailure,
)

_LOG = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv"})


@dataclass(frozen=True)
class ComparisonRequest:
    """All user choices and model configuration for one comparison run."""

    user_video: Path
    pro_video: Path
    pro_speed: Fraction
    slow_motion: Fraction = Fraction(1, 16)
    output_directory: Path = Path("./processed_vids")
    clips: bool = False
    audio_model: Path = DEFAULT_AUDIO_MODEL
    shot_model: Path | None = DEFAULT_SHOT_MODEL
    shot_type_model: Path | None = DEFAULT_SHOT_TYPE_MODEL
    temporal_ranker_model: Path | None = DEFAULT_TEMPORAL_RANKER_MODEL
    device: str | None = None
    diagnostic_report: Path | None = None
    diagnostics_only: bool = False

    @property
    def detection_config(self) -> DetectionConfig:
        return DetectionConfig(
            audio_model=self.audio_model,
            shot_model=self.shot_model,
            shot_type_model=self.shot_type_model,
            temporal_ranker_model=self.temporal_ranker_model,
            device=self.device,
        )


@dataclass(frozen=True)
class ComparisonResult:
    """Published artifacts and the number of comparisons they contain."""

    published_paths: tuple[Path, ...]
    emitted_comparison_count: int


class InvalidComparisonRequest(ValueError):
    """A request cannot safely start processing."""


class OutputCollision(InvalidComparisonRequest):
    """One or more requested artifacts already exist."""

    def __init__(self, paths: tuple[Path, ...]) -> None:
        self.paths = paths
        super().__init__("output already exists: " + ", ".join(map(str, paths)))


class ComparisonProcessingFailed(RuntimeError):
    """A runtime stage failed after semantic preflight."""

    def __init__(
        self, stage: str, message: str, *, diagnostics: str | None = None
    ) -> None:
        self.stage = stage
        self.message = message
        self.diagnostics = diagnostics
        super().__init__(f"{stage} failed: {message}")


class ComparisonDependencies(Protocol):
    """Role-specific effects consumed by the deep workflow."""

    def executable_exists(self, name: str) -> bool: ...

    def inspect_source(self, path: Path) -> ComparisonSource: ...

    def user_has_audio(self, path: Path) -> bool: ...

    def resolve_selection(
        self,
        pro_video: Path,
        pro_speed: Fraction,
        inspected_media: InspectedMedia,
        detection_config: DetectionConfig,
    ) -> ProSelection | SelectionProcessingFailure: ...

    def detect_swings(
        self, request: ComparisonRequest, user_source: ComparisonSource
    ) -> tuple[DetectedSwing, ...]: ...

    def create_player_locator(self, device: str | None) -> PlayerLocator: ...

    def observe_players(
        self, window: SelectedSourceWindow, locator: PlayerLocator
    ) -> tuple[PlayerObservation, ...]: ...

    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None: ...


class SystemComparisonDependencies:
    """Production effects for the command-line comparison adapter."""

    def __init__(self) -> None:
        self._diagnostics_recorder = None

    def executable_exists(self, name: str) -> bool:
        return shutil.which(name) is not None

    def inspect_source(self, path: Path) -> ComparisonSource:
        from .media import inspect_comparison_source

        _LOG.info("Inspecting video metadata: %s", path)
        started = monotonic()
        source = inspect_comparison_source(path)
        _LOG.info(
            "Finished video metadata: %s (%d frames, %dx%d) in %.1fs",
            path,
            len(source.inspected_media.frames),
            source.width,
            source.height,
            monotonic() - started,
        )
        return source

    def user_has_audio(self, path: Path) -> bool:
        from .media import has_audio_stream

        _LOG.info("Checking user audio stream: %s", path)
        has_audio = has_audio_stream(path)
        _LOG.info("Finished user audio check: %s", "present" if has_audio else "absent")
        return has_audio

    def resolve_selection(
        self,
        pro_video: Path,
        pro_speed: Fraction,
        inspected_media: InspectedMedia,
        detection_config: DetectionConfig,
    ) -> ProSelection | SelectionProcessingFailure:
        from tennis_cut.visual_contact import StockVisualContactSelector

        from .pro_selection import find_pro_contact

        _LOG.info("Finding professional contact frame: %s", pro_video)
        ranker = (
            load_temporal_ranker(detection_config.temporal_ranker_model)
            if detection_config.temporal_ranker_model is not None
            else None
        )
        selection = find_pro_contact(
            pro_video=pro_video,
            pro_speed=pro_speed,
            inspected_media=inspected_media,
            finder=StockVisualContactSelector(
                device=detection_config.device,
                ranker=ranker,
                frame_timeline=inspected_media,
            ),
        )
        if isinstance(selection, ProSelection):
            _LOG.info(
                "Professional contact found: frame %d at %.6fs",
                selection.frame.ordinal,
                float(selection.frame.timestamp),
            )
        return selection

    def detect_swings(
        self, request: ComparisonRequest, user_source: ComparisonSource
    ) -> tuple[DetectedSwing, ...]:
        _LOG.info("Detecting swing candidates and visual contacts: %s", request.user_video)
        started = monotonic()
        if request.diagnostic_report is None:
            swings = detect_comparison_user_swings(
                request.user_video,
                request.detection_config,
                frame_timeline=user_source.inspected_media,
            )
            _LOG.info(
                "Finished swing detection: %d accepted swings in %.1fs",
                len(swings),
                monotonic() - started,
            )
            return swings
        from .diagnostics import (
            SwingDiagnosticsRecorder,
            write_swing_diagnostics_report,
        )

        recorder = SwingDiagnosticsRecorder(request.user_video)
        self._diagnostics_recorder = recorder
        try:
            swings = detect_comparison_user_swings(
                request.user_video,
                request.detection_config,
                frame_timeline=user_source.inspected_media,
                diagnostics=recorder,
            )
            _LOG.info(
                "Finished swing detection: %d accepted swings in %.1fs",
                len(swings),
                monotonic() - started,
            )
            return swings
        except BaseException:
            write_swing_diagnostics_report(
                request.diagnostic_report,
                recorder.snapshot(),
            )
            _LOG.info("Wrote swing diagnostics to %s", request.diagnostic_report)
            raise

    def write_swing_diagnostics(
        self,
        windows: tuple[SelectedSourceWindow, ...],
        pro_shot_type: str,
        diagnostic_report: Path | None,
    ) -> None:
        """Finish and publish an optional report before comparison rendering."""

        if diagnostic_report is None or self._diagnostics_recorder is None:
            return
        from .diagnostics import write_swing_diagnostics_report

        rendered = {
            window.swing_ordinal
            for window in windows
            if window.swing_ordinal is not None
        }
        self._diagnostics_recorder.record_planning(rendered, pro_shot_type)
        snapshot = self._diagnostics_recorder.snapshot()
        for candidate in snapshot.candidates:
            if candidate.contact_frame_ordinal is not None:
                _LOG.info(
                    "Audio candidate %d final disposition: %s (%s)",
                    candidate.audio_candidate_index,
                    candidate.disposition,
                    candidate.reason,
                )
        write_swing_diagnostics_report(diagnostic_report, snapshot)
        _LOG.info("Wrote swing diagnostics to %s", diagnostic_report)
        self._diagnostics_recorder = None

    def create_player_locator(self, device: str | None) -> PlayerLocator:
        from utilities import PersonDetector

        resolved_device = resolve_device(device)
        _LOG.info("Loading player locator on %s", resolved_device)
        return PersonDetector(resolved_device)

    def observe_players(
        self, window: SelectedSourceWindow, locator: PlayerLocator
    ) -> tuple[PlayerObservation, ...]:
        from .media import observe_players

        role = "professional" if window.swing_ordinal is None else f"user swing {window.swing_ordinal}"
        _LOG.info(
            "Locating player for %s across %d source frames",
            role,
            len(window.normalized_frames),
        )
        started = monotonic()
        observations = observe_players(window, locator)
        _LOG.info(
            "Finished player location for %s: %d observations in %.1fs",
            role,
            len(observations),
            monotonic() - started,
        )
        return observations

    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        from .media import render_comparison, render_compilation

        _LOG.info("Rendering comparison compilation with %d swing(s): %s", len(plans), primary)
        started = monotonic()
        render_compilation(plans, primary)
        _LOG.info("Finished comparison compilation in %.1fs", monotonic() - started)
        for plan, clip in zip(plans, clips):
            if plan.artifact.path != clip:
                raise ValueError("clip artifact does not match its render plan")
            _LOG.info("Rendering optional comparison clip: %s", clip)
            render_comparison(plan)


def _finite_decimal(value: Fraction) -> str:
    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator != 1:
        raise InvalidComparisonRequest("slow motion must have an exact decimal form")
    scale = max(
        _factor_count(value.denominator, 2), _factor_count(value.denominator, 5)
    )
    scaled = value.numerator * 10**scale // value.denominator
    if scale == 0:
        return str(scaled)
    digits = str(scaled).zfill(scale + 1)
    return f"{digits[:-scale]}.{digits[-scale:]}".rstrip("0").rstrip(".")


def _factor_count(value: int, factor: int) -> int:
    count = 0
    while value % factor == 0:
        value //= factor
        count += 1
    return count


def primary_output_path(request: ComparisonRequest) -> Path:
    slow_motion = _finite_decimal(request.slow_motion)
    filename = (
        f"{request.user_video.stem}_vs_{request.pro_video.stem}_"
        f"slow{slow_motion}x.mp4"
    )
    return request.output_directory / filename


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _preflight(request: ComparisonRequest, dependencies: ComparisonDependencies) -> None:
    for role, path in (("user", request.user_video), ("pro", request.pro_video)):
        if not path.exists() or not path.is_file():
            raise InvalidComparisonRequest(f"missing {role} video: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise InvalidComparisonRequest(f"unsupported {role} video: {path}")
    if request.user_video.samefile(request.pro_video):
        raise InvalidComparisonRequest("user and pro videos must be distinct files")
    if request.pro_speed <= 0:
        raise InvalidComparisonRequest("pro speed must be greater than zero")
    if not 0 < request.slow_motion <= 1:
        raise InvalidComparisonRequest("slow motion must be in (0, 1]")
    if request.shot_type_model is not None and request.shot_model is None:
        raise InvalidComparisonRequest("shot type model requires a shot model")
    if request.diagnostics_only and request.diagnostic_report is None:
        raise InvalidComparisonRequest("diagnostics-only mode requires an HTML report path")
    for model in (
        request.audio_model,
        request.shot_model,
        request.shot_type_model,
        request.temporal_ranker_model,
    ):
        if model is not None and (not model.exists() or not model.is_file()):
            raise InvalidComparisonRequest(f"missing model: {model}")
    if request.temporal_ranker_model is not None:
        try:
            load_temporal_ranker(request.temporal_ranker_model)
        except TemporalRankerArtifactError as error:
            raise InvalidComparisonRequest(f"invalid temporal ranker artifact: {error}") from error
    for executable in ("ffmpeg", "ffprobe"):
        if not dependencies.executable_exists(executable):
            raise InvalidComparisonRequest(f"required executable not found: {executable}")

    if not request.diagnostics_only:
        primary = primary_output_path(request)
        clips_directory = primary.with_name(f"{primary.stem}_clips")
        collisions = tuple(
            path
            for path in (primary, clips_directory if request.clips else None)
            if path is not None and path.exists()
        )
        if collisions:
            raise OutputCollision(collisions)
    destination = (
        request.diagnostic_report.parent
        if request.diagnostics_only and request.diagnostic_report is not None
        else request.output_directory
    )
    writable_parent = _nearest_existing_parent(destination)
    if not writable_parent.is_dir() or not os.access(
        writable_parent, os.W_OK | os.X_OK
    ):
        raise InvalidComparisonRequest(
            f"output destination is not writable: {destination}"
        )


def compare_videos(
    request: ComparisonRequest, dependencies: ComparisonDependencies
) -> ComparisonResult:
    """Validate, select, detect, plan, render, and publish comparisons."""

    _preflight(request, dependencies)
    try:
        user_source = dependencies.inspect_source(request.user_video)
        pro_source = dependencies.inspect_source(request.pro_video)
    except Exception as error:
        raise InvalidComparisonRequest(f"invalid video input: {error}") from error
    try:
        user_has_audio = dependencies.user_has_audio(request.user_video)
    except Exception as error:
        raise InvalidComparisonRequest(f"cannot inspect user audio: {error}") from error
    if not user_has_audio:
        raise InvalidComparisonRequest(f"user video has no audio: {request.user_video}")

    selection = dependencies.resolve_selection(
        request.pro_video,
        request.pro_speed,
        pro_source.inspected_media,
        request.detection_config,
    )
    if isinstance(selection, SelectionProcessingFailure):
        raise ComparisonProcessingFailed(selection.stage, selection.message)
    if selection.shot_type != "forehand":
        raise InvalidComparisonRequest(
            f"unsupported pro shot type for visual contact selection: {selection.shot_type}"
        )

    try:
        swings = dependencies.detect_swings(request, user_source)
        windows = select_comparison_windows(
            user_source=user_source,
            user_swings=swings,
            pro_source=pro_source,
            pro_selection=selection,
            pro_speed=request.pro_speed,
        )
    except Exception as error:
        raise ComparisonProcessingFailed("swing detection", str(error)) from error
    diagnostics_writer = getattr(dependencies, "write_swing_diagnostics", None)
    if callable(diagnostics_writer):
        diagnostics_writer(
            windows.user,
            selection.shot_type,
            request.diagnostic_report,
        )
    if request.diagnostics_only:
        assert request.diagnostic_report is not None
        return ComparisonResult((request.diagnostic_report,), len(windows.user))
    if not windows.user:
        return ComparisonResult((), 0)

    try:
        locator = dependencies.create_player_locator(request.device)
        prepared_pro = prepare_source_window(
            windows.pro, dependencies.observe_players(windows.pro, locator)
        )
    except Exception as error:
        raise ComparisonProcessingFailed("prepare pro window", str(error)) from error

    primary = primary_output_path(request)
    clips_directory = primary.with_name(f"{primary.stem}_clips")
    staging_parent = _nearest_existing_parent(request.output_directory.parent)
    with _staging_directory(
        staging_parent,
        primary=primary,
        clips_directory=clips_directory if request.clips else None,
        output_existed=primary.parent.exists(),
    ) as staging:
        staged_primary = staging / primary.name
        staged_clips = tuple(
            staging / clips_directory.name / f"comparison_{index:03d}.mp4"
            for index in range(1, len(windows.user) + 1)
        ) if request.clips else ()
        plans = []
        for index, user_window in enumerate(windows.user):
            try:
                prepared_user = prepare_source_window(
                    user_window,
                    dependencies.observe_players(user_window, locator),
                )
                artifact = (
                    staged_clips[index]
                    if staged_clips
                    else staging / f"comparison_{index + 1:03d}.mp4"
                )
                plans.append(
                    build_render_plan(
                        user=prepared_user,
                        pro=prepared_pro,
                        slow_motion=request.slow_motion,
                        artifact=ArtifactRequest(artifact),
                    )
                )
            except Exception as error:
                raise ComparisonProcessingFailed(
                    f"prepare user swing {user_window.swing_ordinal}", str(error)
                ) from error
        try:
            dependencies.render_artifacts(
                tuple(plans), staged_primary, staged_clips
            )
        except Exception as error:
            raise ComparisonProcessingFailed(
                "comparison rendering",
                str(error),
                diagnostics=getattr(error, "diagnostics", None),
            ) from error

        published = _publish_artifacts(
            staged_primary=staged_primary,
            staged_clips=staged_clips,
            primary=primary,
            clips_directory=clips_directory,
        )
    return ComparisonResult(published, len(plans))


@contextmanager
def _staging_directory(
    parent: Path,
    *,
    primary: Path,
    clips_directory: Path | None,
    output_existed: bool,
) -> Iterator[Path]:
    try:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix=".tennis-compare-", dir=parent
        )
    except OSError as error:
        raise ComparisonProcessingFailed("artifact publication", str(error)) from error
    body_error: BaseException | None = None
    try:
        yield Path(temporary_directory.name)
    except OSError as error:
        body_error = ComparisonProcessingFailed("artifact publication", str(error))
        raise body_error from error
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            temporary_directory.cleanup()
        except OSError as cleanup_error:
            try:
                temporary_directory.cleanup()
            except OSError as retry_error:
                if body_error is not None:
                    body_error.add_note(f"staging cleanup also failed: {retry_error}")
            if body_error is None:
                _rollback_artifacts(
                    primary=primary,
                    clips_directory=clips_directory,
                    output_directory=primary.parent,
                    output_existed=output_existed,
                )
                raise ComparisonProcessingFailed(
                    "artifact publication", str(cleanup_error)
                ) from cleanup_error


def _publish_artifacts(
    *,
    staged_primary: Path,
    staged_clips: tuple[Path, ...],
    primary: Path,
    clips_directory: Path,
) -> tuple[Path, ...]:
    if not staged_primary.is_file():
        raise ComparisonProcessingFailed(
            "comparison rendering", "renderer did not create the primary artifact"
        )
    if any(not clip.is_file() for clip in staged_clips):
        raise ComparisonProcessingFailed(
            "comparison rendering", "renderer did not create every requested clip"
        )

    late_collisions = tuple(
        path
        for path in (primary, clips_directory if staged_clips else None)
        if path is not None and path.exists()
    )
    if late_collisions:
        raise OutputCollision(late_collisions)

    output_existed = primary.parent.exists()
    primary.parent.mkdir(parents=True, exist_ok=True)
    primary_published = False
    clips_published = False
    try:
        if staged_clips:
            os.replace(staged_clips[0].parent, clips_directory)
            clips_published = True
        os.replace(staged_primary, primary)
        primary_published = True
    except OSError as error:
        _rollback_artifacts(
            primary=primary if primary_published else None,
            clips_directory=clips_directory if clips_published else None,
            output_directory=primary.parent,
            output_existed=output_existed,
        )
        raise ComparisonProcessingFailed("artifact publication", str(error)) from error

    published_clips = tuple(
        clips_directory / clip.name for clip in staged_clips
    )
    return (primary, *published_clips)


def _rollback_artifacts(
    *,
    primary: Path | None,
    clips_directory: Path | None,
    output_directory: Path,
    output_existed: bool,
) -> None:
    if primary is not None:
        try:
            primary.unlink(missing_ok=True)
        except OSError:
            pass
    if clips_directory is not None:
        shutil.rmtree(clips_directory, ignore_errors=True)
    if not output_existed:
        try:
            output_directory.rmdir()
        except OSError:
            pass

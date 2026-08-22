"""Semantic preflight and orchestration for video comparisons."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import os
from pathlib import Path
import shutil
import tempfile
from typing import Protocol

from tennis_cut.swing_detection import (
    DEFAULT_AUDIO_MODEL,
    DEFAULT_SHOT_MODEL,
    DEFAULT_SHOT_TYPE_MODEL,
    DetectedSwing,
    DetectionConfig,
    resolve_device,
)

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
    FileSidecarStore,
    InspectedMedia,
    ProSelection,
    SelectionCancelled,
    SelectionProcessingFailure,
)

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
    device: str | None = None

    @property
    def detection_config(self) -> DetectionConfig:
        return DetectionConfig(
            audio_model=self.audio_model,
            shot_model=self.shot_model,
            shot_type_model=self.shot_type_model,
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


class ComparisonSelectionCancelled(RuntimeError):
    """The user cancelled or closed pro selection."""


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
    ) -> ProSelection | SelectionCancelled | SelectionProcessingFailure: ...

    def detect_swings(self, request: ComparisonRequest) -> tuple[DetectedSwing, ...]: ...

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

    def executable_exists(self, name: str) -> bool:
        return shutil.which(name) is not None

    def inspect_source(self, path: Path) -> ComparisonSource:
        from .media import inspect_comparison_source

        return inspect_comparison_source(path)

    def user_has_audio(self, path: Path) -> bool:
        from .media import has_audio_stream

        return has_audio_stream(path)

    def resolve_selection(
        self,
        pro_video: Path,
        pro_speed: Fraction,
        inspected_media: InspectedMedia,
    ) -> ProSelection | SelectionCancelled | SelectionProcessingFailure:
        from .media import FfmpegFrameImageReader
        from .pro_picker import QtProPicker
        from .pro_selection import resolve_pro_selection

        return resolve_pro_selection(
            pro_video=pro_video,
            pro_speed=pro_speed,
            inspected_media=inspected_media,
            sidecar_store=FileSidecarStore(),
            picker=QtProPicker(pro_video, FfmpegFrameImageReader()),
        )

    def detect_swings(self, request: ComparisonRequest) -> tuple[DetectedSwing, ...]:
        from tennis_cut.swing_detection import detect_user_swings

        return detect_user_swings(request.user_video, request.detection_config)

    def create_player_locator(self, device: str | None) -> PlayerLocator:
        from utilities import PersonDetector

        return PersonDetector(resolve_device(device))

    def observe_players(
        self, window: SelectedSourceWindow, locator: PlayerLocator
    ) -> tuple[PlayerObservation, ...]:
        from .media import observe_players

        return observe_players(window, locator)

    def render_artifacts(
        self,
        plans: tuple[ComparisonRenderPlan, ...],
        primary: Path,
        clips: tuple[Path, ...],
    ) -> None:
        from .media import render_comparison, render_compilation

        render_compilation(plans, primary)
        for plan, clip in zip(plans, clips):
            if plan.artifact.path != clip:
                raise ValueError("clip artifact does not match its render plan")
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
    for model in (
        request.audio_model,
        request.shot_model,
        request.shot_type_model,
    ):
        if model is not None and (not model.exists() or not model.is_file()):
            raise InvalidComparisonRequest(f"missing model: {model}")
    for executable in ("ffmpeg", "ffprobe"):
        if not dependencies.executable_exists(executable):
            raise InvalidComparisonRequest(f"required executable not found: {executable}")

    primary = primary_output_path(request)
    clips_directory = primary.with_name(f"{primary.stem}_clips")
    collisions = tuple(
        path
        for path in (primary, clips_directory if request.clips else None)
        if path is not None and path.exists()
    )
    if collisions:
        raise OutputCollision(collisions)
    writable_parent = _nearest_existing_parent(request.output_directory)
    if not writable_parent.is_dir() or not os.access(
        writable_parent, os.W_OK | os.X_OK
    ):
        raise InvalidComparisonRequest(
            f"output destination is not writable: {request.output_directory}"
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
        request.pro_video, request.pro_speed, pro_source.inspected_media
    )
    if isinstance(selection, SelectionCancelled):
        raise ComparisonSelectionCancelled(selection.message)
    if isinstance(selection, SelectionProcessingFailure):
        raise ComparisonProcessingFailed(selection.stage, selection.message)

    try:
        swings = dependencies.detect_swings(request)
        windows = select_comparison_windows(
            user_source=user_source,
            user_swings=swings,
            pro_source=pro_source,
            pro_selection=selection,
            pro_speed=request.pro_speed,
        )
    except Exception as error:
        raise ComparisonProcessingFailed("swing detection", str(error)) from error
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
    with _staging_directory(staging_parent) as staging:
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
def _staging_directory(parent: Path) -> Iterator[Path]:
    try:
        with tempfile.TemporaryDirectory(
            prefix=".tennis-compare-", dir=parent
        ) as directory_name:
            yield Path(directory_name)
    except (ComparisonProcessingFailed, OutputCollision):
        raise
    except OSError as error:
        raise ComparisonProcessingFailed("artifact publication", str(error)) from error


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
        if primary_published:
            primary.unlink(missing_ok=True)
        if clips_published:
            shutil.rmtree(clips_directory, ignore_errors=True)
        if not output_existed:
            try:
                primary.parent.rmdir()
            except OSError:
                pass
        raise ComparisonProcessingFailed("artifact publication", str(error)) from error

    published_clips = tuple(
        clips_directory / clip.name for clip in staged_clips
    )
    return (primary, *published_clips)

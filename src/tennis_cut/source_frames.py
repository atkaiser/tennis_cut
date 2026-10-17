"""Fast exact-frame indexing and candidate-window decoding."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import json
import logging
from pathlib import Path
import subprocess
from time import monotonic
from typing import Protocol


_LOG = logging.getLogger(__name__)
MAX_PARALLEL_DECODES = 4
SEEK_PREROLL = Fraction(1)
MAX_CONTIGUOUS_TIME_GAP = Fraction(1, 4)


@dataclass(frozen=True)
class IndexedFrame:
    """One presentation-ordered source frame identified without decoding pixels."""

    stream_index: int
    ordinal: int
    pts: int
    time_base: Fraction

    @property
    def timestamp(self) -> Fraction:
        return self.pts * self.time_base


@dataclass(frozen=True)
class SourceFrameIndex:
    """Source geometry and its presentation-ordered frame identities."""

    width: int
    height: int
    frames: tuple[IndexedFrame, ...]
    method: str


class ExactFrame(Protocol):
    """Structural frame identity accepted by the candidate-window decoder."""

    stream_index: int
    ordinal: int
    pts: int
    time_base: Fraction

    @property
    def timestamp(self) -> Fraction: ...


def inspect_source_frame_index(source: Path) -> SourceFrameIndex:
    """Build a packet-derived frame index, falling back to decoded inspection."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_packets",
            "-show_entries",
            (
                "stream=index,time_base,width,height,start_pts,duration_ts,nb_frames:"
                "packet=stream_index,pts,flags"
            ),
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError("expected exactly one selected video stream")
    stream = streams[0]
    stream_index = int(stream["index"])
    time_base = Fraction(stream["time_base"])
    start_pts = (
        int(stream["start_pts"])
        if stream.get("start_pts") not in (None, "N/A")
        else None
    )
    end_pts = (
        None
        if start_pts is None or stream.get("duration_ts") in (None, "N/A")
        else start_pts + int(stream["duration_ts"])
    )
    packets = [
        packet
        for packet in payload.get("packets", [])
        if int(packet.get("stream_index", -1)) == stream_index
        and "D" not in str(packet.get("flags", ""))
    ]
    packet_pts_complete = all(
        packet.get("pts") not in (None, "N/A") for packet in packets
    )
    packet_pts = (
        [
            int(packet["pts"])
            for packet in packets
            if (start_pts is None or int(packet["pts"]) >= start_pts)
            and (end_pts is None or int(packet["pts"]) < end_pts)
        ]
        if packet_pts_complete
        else []
    )
    ordered_pts = sorted(packet_pts)
    declared_frame_count = (
        int(stream["nb_frames"])
        if stream.get("nb_frames") not in (None, "N/A")
        else None
    )
    if (
        ordered_pts
        and declared_frame_count == len(ordered_pts)
        and len(ordered_pts) == len(set(ordered_pts))
    ):
        return SourceFrameIndex(
            width=int(stream["width"]),
            height=int(stream["height"]),
            frames=tuple(
                IndexedFrame(stream_index, ordinal, pts, time_base)
                for ordinal, pts in enumerate(ordered_pts)
            ),
            method="packets",
        )
    _LOG.warning(
        "Packet timeline is not one-frame-per-PTS; falling back to decoded inspection: %s",
        source,
    )
    return _inspect_decoded_frame_index(source)


def _inspect_decoded_frame_index(source: Path) -> SourceFrameIndex:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=index,time_base,width,height:frame=stream_index,pts",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError("expected exactly one selected video stream")
    stream = streams[0]
    stream_index = int(stream["index"])
    time_base = Fraction(stream["time_base"])
    frames = tuple(
        IndexedFrame(stream_index, ordinal, int(frame["pts"]), time_base)
        for ordinal, frame in enumerate(payload.get("frames", []))
        if int(frame["stream_index"]) == stream_index
    )
    if not frames:
        raise ValueError("selected video stream has no decoded frames")
    return SourceFrameIndex(
        width=int(stream["width"]),
        height=int(stream["height"]),
        frames=frames,
        method="decoded frames",
    )


def decode_exact_frame_windows(
    source: Path,
    frames: tuple[ExactFrame, ...],
    output_directory: Path,
    *,
    max_workers: int = MAX_PARALLEL_DECODES,
    source_start_timestamp: Fraction | None = None,
    max_dimension: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
    output_size: tuple[int, int] | None = None,
) -> dict[int, Path]:
    """Decode exact windows directly into the consumer's required image shape."""

    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("maximum image dimension must be positive")
    if (crop is None) != (output_size is None):
        raise ValueError("crop and output size must be provided together")
    if max_dimension is not None and crop is not None:
        raise ValueError("maximum dimension cannot be combined with crop and resize")
    if crop is not None and any(value < 0 for value in crop[:2]):
        raise ValueError("crop origin cannot be negative")
    if crop is not None and output_size is not None and any(
        value <= 0 for value in (*crop[2:], *output_size)
    ):
        raise ValueError("crop and output dimensions must be positive")

    unique = {frame.ordinal: frame for frame in frames}
    if len(unique) != len(frames):
        raise ValueError("source frame ordinals must be unique")
    ordered = tuple(unique[ordinal] for ordinal in sorted(unique))
    if not ordered:
        return {}
    if any(
        frame.stream_index != ordered[0].stream_index
        or frame.time_base != ordered[0].time_base
        for frame in ordered
    ):
        raise ValueError("decoded frames must share one stream and time base")
    if source_start_timestamp is None:
        source_start_timestamp = inspect_source_frame_index(source).frames[0].timestamp
    groups: list[list[ExactFrame]] = [[ordered[0]]]
    for frame in ordered[1:]:
        previous = groups[-1][-1]
        if (
            frame.ordinal == previous.ordinal + 1
            and frame.timestamp - previous.timestamp <= MAX_CONTIGUOUS_TIME_GAP
        ):
            groups[-1].append(frame)
        else:
            groups.append([frame])
    output_directory.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(max_workers, len(groups)))
    _LOG.info(
        "Decoding %d exact frames from %d candidate window(s) with %d worker(s): %s",
        len(ordered),
        len(groups),
        workers,
        source,
    )
    started = monotonic()

    def decode_group(item: tuple[int, list[ExactFrame]]) -> dict[int, Path]:
        group_index, group = item
        group_directory = output_directory / f"window_{group_index:04d}"
        group_directory.mkdir()
        first = group[0]
        last = group[-1]
        seek = max(
            Fraction(),
            first.timestamp - source_start_timestamp - SEEK_PREROLL,
        )
        pattern = group_directory / "frame_%09d.png"
        filters = [f"select=between(pts\\,{first.pts}\\,{last.pts})"]
        if max_dimension is not None:
            filters.append(
                "scale="
                f"w='min(iw\\,{max_dimension})':"
                f"h='min(ih\\,{max_dimension})':"
                "force_original_aspect_ratio=decrease:force_divisible_by=2:"
                "flags=lanczos"
            )
        elif crop is not None and output_size is not None:
            x, y, width, height = crop
            output_width, output_height = output_size
            filters.extend(
                (
                    f"crop={width}:{height}:{x}:{y}",
                    f"scale={output_width}:{output_height}:flags=lanczos",
                )
            )
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-copyts",
                "-ss",
                f"{float(seek):.9f}",
                "-noautorotate",
                "-i",
                str(source),
                "-map",
                f"0:{first.stream_index}",
                "-vf",
                ",".join(filters),
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(len(group)),
                "-start_number",
                "0",
                str(pattern),
                "-y",
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "could not decode exact candidate window: " + completed.stderr.strip()
            )
        paths = tuple(sorted(group_directory.glob("frame_*.png")))
        if len(paths) != len(group):
            raise ValueError(
                f"decoded {len(paths)} of {len(group)} requested candidate frames"
            )
        return {
            frame.ordinal: path
            for frame, path in zip(group, paths, strict=True)
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        decoded_groups = tuple(executor.map(decode_group, enumerate(groups)))
    decoded = {
        ordinal: path
        for decoded_group in decoded_groups
        for ordinal, path in decoded_group.items()
    }
    _LOG.info(
        "Finished candidate-window decode: %d frames in %.1fs",
        len(decoded),
        monotonic() - started,
    )
    return decoded


__all__ = [
    "IndexedFrame",
    "SourceFrameIndex",
    "decode_exact_frame_windows",
    "inspect_source_frame_index",
]

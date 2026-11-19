"""Writing bundles from timestamped frame sources.

Each output stream is a list of source timestamps and a function that yields
the rendered frames for a given set of samples. Most streams come from an
abstract dataloader [`Sensor`][abstract_dataloader.spec.Sensor] and a
function that renders one sample:

```python
from abstract_dataloader.generic import Nearest
from multireel import Stream, write_bundle

streams = {
    "front_cam": Stream.from_sensor(trace["camera"], render_camera),
    "front_boxes": Stream.from_sensor(
        trace["detections"], render_boxes,
        profile="prores-4444", overlay_of="front_cam"),
}
manifest = write_bundle(
    "out/drive_0423_a", streams,
    sync=Nearest("__multireel_frame_clock__", tol=0.05),
    recording_id="drive_0423_a", timecode_origin=origin)
```

A [`Synchronization`][abstract_dataloader.spec.Synchronization] decides which
sample each frame shows. The writer calls it once, with the timestamps of
every stream keyed by stream ID, plus the instants of the bundle's frames
keyed by `"__multireel_frame_clock__"`. The frame clock is usually the
reference. Frames that the result leaves out are no-data frames in every
stream.

Only samples that are actually shown are rendered, and each one is rendered
once, however many frames it is shown for.
"""

import math
import re
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, TypeVar
from uuid import uuid4

import numpy as np
from abstract_dataloader import spec
from jaxtyping import Float64, Int64, UInt8

from .bundle.manifest import (
    AlphaMode,
    Manifest,
    StreamEntry,
    stream_path,
    write_manifest,
)
from .bundle.timing import Timecode
from .bundle.video import Profile, clip_name, reel_name
from .encode import EncodeOptions, Encoder, StreamFormat
from .validate.checks import validate_stream

TSample = TypeVar("TSample")


@dataclass(frozen=True)
class Stream:
    """One output stream: its frames, and how to encode them.

    Attributes:
        timestamps: source timestamps, in float seconds since the UTC epoch,
            strictly increasing.
        frames: given strictly increasing sample indices, yields the
            rendered frame of each of those samples, in order. It is called
            once per bundle, so the same `Stream` can be written more than
            once. Frames must be the same size for every sample, and RGBA if
            the profile has alpha.
        profile: encoding profile.
        label: human-readable name, used in the manifest and clip name.
        overlay_of: base stream ID, for overlays.
        alpha_mode: how the frames' alpha is stored, for overlays.
        info: free-form mapping stored in the manifest.
    """

    timestamps: Float64[np.ndarray, "N"]
    frames: Callable[
        [Int64[np.ndarray, "K"]], Iterator[UInt8[np.ndarray, "H W C"]]
    ]
    profile: Profile = "hevc-main10-420"
    label: str | None = None
    overlay_of: str | None = None
    alpha_mode: AlphaMode = "straight"
    info: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sensor(
        cls,
        sensor: spec.Sensor[TSample, Any],
        render: Callable[[TSample], UInt8[np.ndarray, "H W C"]],
        *,
        sequential: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Make a stream from an abstract dataloader sensor.

        Timestamps come from `sensor.metadata.timestamps`. With random
        access, only the shown samples are loaded, using `sensor[i]`. With
        `sequential`, every sample is read in order from `sensor.stream()`,
        which suits sources that are slow to seek. Either way, only the
        shown samples are rendered.

        Args:
            sensor: source sensor.
            render: turns one sample into one frame.
            sequential: read samples with `sensor.stream()` instead of
                `sensor[i]`.
            kwargs: other `Stream` fields, such as `profile`.

        Returns:
            The stream.
        """
        timestamps = np.asarray(sensor.metadata.timestamps, dtype=np.float64)

        def random_access(
            indices: Int64[np.ndarray, "K"],
        ) -> Iterator[UInt8[np.ndarray, "H W C"]]:
            for i in indices:
                yield render(sensor[int(i)])

        def sequential_access(
            indices: Int64[np.ndarray, "K"],
        ) -> Iterator[UInt8[np.ndarray, "H W C"]]:
            wanted = iter(indices.tolist())
            target = next(wanted, None)
            for i, sample in enumerate(sensor.stream()):
                if target is None:
                    return
                if i == target:
                    yield render(sample)
                    target = next(wanted, None)

        frames = sequential_access if sequential else random_access
        return cls(timestamps, frames, **kwargs)


def write_bundle(
    path: str | Path,
    streams: Mapping[str, Stream],
    sync: spec.Synchronization,
    *,
    recording_id: str,
    timecode_origin: datetime,
    start: float | None = None,
    end: float | None = None,
    edit_rate: Literal[10, 15, 24, 25, 30, 50, 60] = 30,
    keyframe_interval: int | None = None,
    provenance: Any = None,
    options: EncodeOptions = EncodeOptions(),
    workers: int | None = None,
) -> Manifest:
    """Render, encode and validate a bundle, then write its manifest.

    Streams are rendered in parallel threads, each feeding its own `ffmpeg`
    process. Every stream file is validated before the manifest is written.
    If anything fails, the directory is left without a manifest.

    Args:
        path: bundle directory to create.
        streams: output streams, keyed by stream ID.
        sync: chooses the sample each frame shows. The frame clock is keyed
            by `"__multireel_frame_clock__"`. For each stream, the sample
            indices it returns must not decrease from one frame to the next.
        recording_id: ID of the source recording.
        timecode_origin: timezone-aware time shown as timecode
            `01:00:00:00`, a whole number of seconds.
        start: range start, in seconds since the UTC epoch, rounded down to
            a frame. Defaults to the earliest first sample across all
            streams.
        end: range end, in seconds since the UTC epoch, rounded up to a
            frame. Defaults to the latest last sample across all streams.
        edit_rate: frames per second.
        keyframe_interval: keyframe interval `N`, in frames. Defaults to
            half a second of frames.
        provenance: free-form value stored in the manifest.
        options: encoder settings.
        workers: number of threads; defaults to one per stream.

    Returns:
        The manifest that was written.

    Raises:
        ValueError: if an argument is invalid, a stream's synchronization
            returns decreasing sample indices, or the range does not fit in
            valid timecode.
        RuntimeError: if encoding fails, or a stream file fails validation.
    """
    path = Path(path)
    for stream_id, stream in streams.items():
        if re.fullmatch(r"[a-z0-9_-]+", stream_id) is None:
            raise ValueError(f"Invalid stream ID: {stream_id!r}")
        base = stream.overlay_of
        if base is not None and (
                base not in streams or streams[base].overlay_of is not None):
            raise ValueError(
                f"Stream {stream_id!r} is an overlay of {base!r}, which is "
                f"not a stream or is itself an overlay.")
    if (path / "manifest.yaml").exists():
        raise FileExistsError(f"{path} is already a bundle.")

    if start is None:
        start = min(float(s.timestamps[0]) for s in streams.values())
    if end is None:
        end = max(float(s.timestamps[-1]) for s in streams.values())
    first_frame = math.floor(start * edit_rate)
    frame_count = math.ceil(end * edit_rate) - first_frame + 1
    start_timecode = Timecode.from_frame(
        first_frame, edit_rate, timecode_origin)
    Timecode.from_frame(  # raises if the last frame's timecode would wrap
        first_frame + frame_count - 1, edit_rate, timecode_origin)

    clock = (first_frame + np.arange(frame_count)) / edit_rate
    synced = sync({
        "__multireel_frame_clock__": clock,
        **{stream_id: s.timestamps for stream_id, s in streams.items()}})
    kept = synced["__multireel_frame_clock__"].astype(np.int64)

    manifest = Manifest(
        bundle_id=uuid4(),
        recording_id=recording_id,
        timecode_origin=timecode_origin,
        edit_rate=edit_rate,
        keyframe_interval=keyframe_interval or max(1, edit_rate // 2),
        start_timecode=start_timecode,
        frame_count=frame_count,
        streams={
            stream_id: StreamEntry(
                s.profile, s.label, s.overlay_of, s.alpha_mode, s.info)
            for stream_id, s in streams.items()},
        provenance=provenance)

    def write_stream(stream_id: str) -> None:
        sample_indices = synced[stream_id].astype(np.int64)
        if np.any(np.diff(sample_indices) < 0):
            raise ValueError(
                f"Synchronization returned decreasing sample indices for "
                f"stream {stream_id!r}.")
        shown = np.full(frame_count, -1, dtype=np.int64)
        shown[kept] = sample_indices

        stream = streams[stream_id]
        file = stream_path(path, stream_id)
        _encode(file, stream, shown, StreamFormat(
            width=0, height=0, profile=stream.profile, edit_rate=edit_rate,
            keyframe_interval=manifest.keyframe_interval,
            start_timecode=start_timecode,
            reel_name=reel_name(recording_id, stream_id),
            clip_name=clip_name(recording_id, stream_id, stream.label),
        ), options)

        violations = validate_stream(
            file, stream_id, manifest,
            ffmpeg=options.ffmpeg, ffprobe=options.ffprobe)
        if violations:
            raise RuntimeError(
                f"Stream {stream_id!r} failed validation: {violations}")

    (path / "streams").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers or len(streams)) as pool:
        futures = [pool.submit(write_stream, s) for s in streams]
        for future in futures:
            future.result()

    write_manifest(path, manifest)
    return manifest


def _encode(
    file: Path,
    stream: Stream,
    shown: Int64[np.ndarray, "F"],
    format: StreamFormat,
    options: EncodeOptions,
) -> None:
    """Render and encode one stream.

    Args:
        file: output MOV file.
        stream: stream to render.
        shown: sample shown in each frame, or `-1` for a no-data frame.
        format: stream format; its width and height are replaced by the
            size of the first rendered frame.
        options: encoder settings.
    """
    frames = stream.frames(np.unique(shown[shown >= 0]))
    pending = next(frames, None)
    if pending is None:
        raise ValueError(f"No samples are shown in {file.stem!r}.")
    height, width = pending.shape[:2]
    blank = np.zeros_like(pending)

    run_starts = np.flatnonzero(np.diff(shown, prepend=-2))
    run_lengths = np.diff(run_starts, append=len(shown))

    encoder = Encoder(
        file, replace(format, width=width, height=height), options)
    try:
        current, frame = -1, pending
        for run_start, length in zip(run_starts, run_lengths):
            sample = int(shown[run_start])
            if sample < 0:
                encoder.write(blank, int(length))
                continue
            if sample != current:
                frame = pending if pending is not None else next(frames)
                pending, current = None, sample
            encoder.write(frame, int(length))
    except BaseException:
        encoder.abort()
        raise
    encoder.close()

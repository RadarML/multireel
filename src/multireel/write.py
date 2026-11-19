"""Writing bundles from timestamped frame sources.

Each output stream is a list of source timestamps and a function that yields
the rendered frames for a given set of samples. Most streams come from an
abstract dataloader [`Sensor`][abstract_dataloader.spec.Sensor] and a
function that renders one sample:

```python
from abstract_dataloader.generic import Nearest
from multireel import Profile, Stream, write_bundle

streams = {
    "front_cam": Stream.from_sensor(trace["camera"], render_camera),
    "front_boxes": Stream.from_sensor(
        trace["detections"], render_boxes,
        profile=Profile.from_name("prores-4444"), overlay_of="front_cam"),
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

Streams are rendered, encoded and validated in parallel, one per thread by
default. Rendering usually runs Python code, which threads cannot run in
parallel, so with `processes=True` each stream runs in its own forked
process instead. Frames never cross process boundaries: each process feeds
its own `ffmpeg`.

Progress is counted in output frames handed to `ffmpeg`, per stream, in
shared memory, so it works the same with threads and processes. The calling
thread polls the counts and passes them to a `progress` callback, so
progress is always reported from the calling process.
"""

import math
import multiprocessing
import os
import re
import threading
import traceback
import warnings
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field, replace
from datetime import datetime
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Literal, Self, TypeVar
from uuid import uuid4

import numpy as np
from abstract_dataloader import spec
from jaxtyping import Float64, Int64, UInt8

from .bundle.manifest import Manifest, StreamEntry
from .bundle.timing import Timecode
from .bundle.video import Profile
from .encode import EncodeOptions, Encoder, StreamFormat
from .validate.checks import validate_stream

TSample = TypeVar("TSample")

Progress = Callable[[Mapping[str, int], int], None]
"""Progress callback: frames written so far per stream ID, and the number of
frames in every stream."""

PROGRESS_INTERVAL = 0.2
"""Seconds between progress callbacks."""


@dataclass(frozen=True)
class Stream:
    """One output stream: its frames, and how to encode them.

    Attributes:
        timestamps: source timestamps, in float seconds since the UTC epoch,
            strictly increasing.
        frames: given strictly increasing sample indices, yields the
            rendered frame of each of those samples, in order. It is called
            once per bundle, so the same `Stream` can be written more than
            once. Frames must be 8-bit and the same size for every sample,
            with an alpha channel last if the profile has alpha.
        profile: encoding profile; `hevc-main10-420` by default.
        label: human-readable name, used in the manifest and clip name.
        overlay_of: base stream ID, for overlays.
        alpha_mode: how the frames' alpha is stored, for overlays.
        info: free-form mapping stored in the manifest.
        channel_order: order of the color channels in frames, `rgb` or
            `bgr`. Alpha, if any, always comes last. `bgr` suits sources
            such as OpenCV, which decode to BGR, and saves reordering the
            channels before encoding.
    """

    timestamps: Float64[np.ndarray, "N"]
    frames: Callable[
        [Int64[np.ndarray, "K"]], Iterator[UInt8[np.ndarray, "H W C"]]
    ]
    profile: Profile = Profile.from_name("hevc-main10-420")
    label: str | None = None
    overlay_of: str | None = None
    alpha_mode: Literal["straight", "premultiplied"] = "straight"
    info: Mapping[str, Any] = field(default_factory=dict)
    channel_order: Literal["rgb", "bgr"] = "rgb"

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
    processes: bool = False,
    verify_decode: bool = True,
    progress: Progress | None = None,
) -> Manifest:
    """Render, encode and validate a bundle, then write its manifest.

    Streams are rendered in parallel threads or processes, each feeding its
    own `ffmpeg` process. Every stream file is validated before the manifest
    is written. If anything fails, the directory is left without a manifest.

    !!! warning "Processes are forked"

        With `processes`, each stream runs in a child created with `fork`,
        so streams do not need to be picklable. A child inherits only the
        thread that forks it, and any lock another thread held at that
        moment stays locked in the child. Start threads, thread pools and
        GPU contexts inside `Stream.frames`, not before calling
        `write_bundle`. A warning is issued if other threads are running.

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
        workers: number of streams written at once; defaults to all of
            them.
        processes: write each stream in its own forked process instead of
            a thread. Needs `fork`, so not available on Windows.
        verify_decode: decode every frame of each stream file when
            validating it. This is the slowest check, and can take a large
            share of the total time for long, high-resolution streams.
            Without it, the `decode` rule is not checked.
        progress: called in the calling thread every `PROGRESS_INTERVAL`
            seconds while streams are written, and once more when they are
            all written, with the frames written so far per stream and the
            number of frames in every stream. Frames are counted when they
            are handed to `ffmpeg`, so validation is not included.

    Returns:
        The manifest that was written.

    Raises:
        ValueError: if an argument is invalid, a stream's synchronization
            returns decreasing sample indices, or the range does not fit in
            valid timecode.
        RuntimeError: if encoding fails, a stream file fails validation, or
            a stream's process exits without reporting a result.
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

    # One writer per counter, so no lock; shared memory, so children's
    # counts are visible to the parent.
    stream_ids = list(streams)
    counts = multiprocessing.Array("q", len(stream_ids), lock=False)

    def report() -> None:
        if progress is not None:
            progress(dict(zip(stream_ids, counts)), frame_count)

    def write_stream(stream_id: str) -> None:
        sample_indices = synced[stream_id].astype(np.int64)
        if np.any(np.diff(sample_indices) < 0):
            raise ValueError(
                f"Synchronization returned decreasing sample indices for "
                f"stream {stream_id!r}.")
        shown = np.full(frame_count, -1, dtype=np.int64)
        shown[kept] = sample_indices

        stream = streams[stream_id]
        file = Manifest.stream_path(path, stream_id)
        _encode(file, stream, shown, StreamFormat(
            width=0, height=0, profile=stream.profile, edit_rate=edit_rate,
            keyframe_interval=manifest.keyframe_interval,
            start_timecode=start_timecode,
            reel_name=manifest.reel_name(stream_id),
            clip_name=manifest.clip_name(stream_id),
            channel_order=stream.channel_order,
        ), options, _counter(counts, stream_ids.index(stream_id)))

        violations = validate_stream(
            file, stream_id, manifest, ffmpeg=options.ffmpeg,
            ffprobe=options.ffprobe, decode=verify_decode)
        if violations:
            raise RuntimeError(
                f"Stream {stream_id!r} failed validation:\n"
                + "\n".join(violations))

    (path / "streams").mkdir(parents=True, exist_ok=True)
    if processes:
        _run_forked(
            write_stream, stream_ids, workers or len(streams), report)
    else:
        with ThreadPoolExecutor(workers or len(streams)) as pool:
            futures = [pool.submit(write_stream, s) for s in stream_ids]
            pending = set(futures)
            while pending:
                done, pending = wait_futures(
                    pending, PROGRESS_INTERVAL, FIRST_EXCEPTION)
                report()
                for future in done:
                    future.result()
    report()

    manifest.write(path)
    return manifest


def _counter(counts: Any, index: int) -> Callable[[int], None]:
    """Return a function that adds to one entry of a shared array."""

    def add(n: int) -> None:
        counts[index] += n

    return add


def _child(
    fn: Callable[[str], None], stream_id: str, conn: Connection
) -> None:
    """Run `fn(stream_id)` in a forked child and report the result.

    Sends `None` on success, or the exception and its formatted traceback.
    Exits with `os._exit`, so the child never runs the parent's exit
    handlers or flushes buffers it inherited.
    """
    code = 0
    try:
        fn(stream_id)
        conn.send(None)
    except BaseException as e:
        code = 1
        trace = traceback.format_exc()
        try:
            conn.send((e, trace))
        except Exception:  # the exception cannot be pickled
            conn.send((RuntimeError(f"{type(e).__name__}: {e}"), trace))
    finally:
        conn.close()
        os._exit(code)


def _run_forked(
    fn: Callable[[str], None], stream_ids: list[str], workers: int,
    poll: Callable[[], None],
) -> None:
    """Run `fn` once per stream, each in a forked child process.

    At most `workers` children run at once. On the first failure, the other
    children are terminated, and the failure is raised in the parent with
    the child's traceback as a note. `poll` is called in the parent every
    `PROGRESS_INTERVAL` seconds.
    """
    if threading.active_count() > 1:
        warnings.warn(
            "write_bundle(processes=True) is forking while other threads "
            "are running; locks they hold stay locked in the children.",
            RuntimeWarning, stacklevel=3)

    ctx = multiprocessing.get_context("fork")
    queue = list(reversed(stream_ids))
    running: dict[Any, tuple[str, Any, Connection]] = {}
    try:
        while queue or running:
            while queue and len(running) < workers:
                stream_id = queue.pop()
                parent, child = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=_child, args=(fn, stream_id, child), daemon=True)
                process.start()
                child.close()
                running[process.sentinel] = (stream_id, process, parent)

            ready = wait(list(running), timeout=PROGRESS_INTERVAL)
            poll()
            for sentinel in ready:
                stream_id, process, conn = running.pop(sentinel)
                process.join()
                try:
                    result = conn.recv()
                except EOFError:  # the child exited without reporting
                    result = None
                conn.close()
                if result is not None:
                    error, trace = result
                    error.add_note(
                        f"In the process for stream {stream_id!r}:\n{trace}")
                    raise error
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"The process for stream {stream_id!r} exited with "
                        f"code {process.exitcode}.")
    finally:
        # Only reached with children still running if something failed.
        for _, process, conn in running.values():
            process.terminate()
        for _, process, conn in running.values():
            process.join()
            conn.close()


def _encode(
    file: Path,
    stream: Stream,
    shown: Int64[np.ndarray, "F"],
    format: StreamFormat,
    options: EncodeOptions,
    advance: Callable[[int], None] = lambda n: None,
) -> None:
    """Render and encode one stream.

    Args:
        file: output MOV file.
        stream: stream to render.
        shown: sample shown in each frame, or `-1` for a no-data frame.
        format: stream format; its width and height are replaced by the
            size of the first rendered frame.
        options: encoder settings.
        advance: called with the number of frames after each write.
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
                advance(int(length))
                continue
            if sample != current:
                frame = pending if pending is not None else next(frames)
                pending, current = None, sample
            encoder.write(frame, int(length))
            advance(int(length))
    except BaseException:
        encoder.abort()
        raise
    encoder.close()

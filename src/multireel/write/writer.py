"""Planning and writing a bundle."""

import math
import multiprocessing
import os
import threading
import traceback
import warnings
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from abstract_dataloader import spec
from jaxtyping import Int64

from multireel.bundle import Manifest, StreamEntry, Timecode
from multireel.stream import Stream

from .encode import EncodeOptions, Encoder
from .validate import validate_stream

FRAME_CLOCK = "__multireel_frame_clock__"
"""Key of the bundle's frame instants in the synchronization's input."""

PROGRESS_INTERVAL = 0.2
"""Seconds between progress callbacks."""

Progress = Callable[[Mapping[str, int], int], None]
"""Progress callback: frames written so far per stream ID, and the number of
frames in every stream."""


class BundleWriter:
    """A planned bundle, ready to be written.

    Constructing a writer checks every argument, each stream's profile and
    the synchronization, and plans the bundle: its frame range, its
    manifest, and the sample each frame of each stream shows. Nothing is
    rendered or written until [`write`][.write]:

    ```python
    writer = BundleWriter(
        "out/drive_0423_a", streams, Nearest(FRAME_CLOCK, tol=0.05),
        recording_id="drive_0423_a", timecode_origin=first_sample_time)
    print(writer.manifest.frame_count)
    writer.write()
    ```

    Args:
        path: bundle directory to create.
        streams: output streams, keyed by stream ID.
        sync: chooses the sample each frame shows. The frame clock is keyed
            by [`FRAME_CLOCK`][multireel.write.FRAME_CLOCK]. For each
            stream, the sample indices it returns must not decrease from one
            frame to the next.
        recording_id: ID of the source recording.
        timecode_origin: time shown as timecode `01:00:00:00`: a
            timezone-aware `datetime` that is a whole number of seconds, or
            a timestamp in seconds since the UTC epoch, rounded down to a
            whole second. Bundles of the same recording must share an
            origin, so use a time that depends only on the recording, such
            as its first sample.
        start: range start, in seconds since the UTC epoch, rounded down to
            a frame. Defaults to the start of `span`.
        end: range end, in seconds since the UTC epoch, rounded up to a
            frame. Defaults to the end of `span`.
        span: default range: `all`, from the first sample of any stream to
            the last sample of any stream, or `overlap`, where every stream
            has samples.
        duration: if set, the range ends at most this many seconds after
            its start.
        edit_rate: frames per second.
        keyframe_interval: keyframe interval `N`, in frames. Defaults to
            half a second of frames.
        provenance: free-form value stored in the manifest.
        options: encoder settings.
        verify_decode: decode every frame of each stream file when
            validating it. This is the slowest check, and can take a large
            share of the total time for long, high-resolution streams.
            Without it, the `decode` rule is not checked.

    Attributes:
        path: bundle directory.
        streams: output streams, keyed by stream ID.
        options: encoder settings.
        verify_decode: whether every frame is decoded when validating.
        manifest: the manifest the bundle will have.
        shown: sample shown in each frame of each stream, or `-1` for a
            no-data frame.

    Raises:
        ValueError: if an argument or stream is invalid, the range is empty
            or does not fit in valid timecode, a stream's synchronization
            returns decreasing sample indices or shows none of its samples,
            or the encoder cannot produce a stream's profile.
    """

    def __init__(
        self,
        path: str | Path,
        streams: Mapping[str, Stream],
        sync: spec.Synchronization,
        *,
        recording_id: str,
        timecode_origin: datetime | float,
        start: float | None = None,
        end: float | None = None,
        span: Literal["all", "overlap"] = "all",
        duration: float | None = None,
        edit_rate: Literal[10, 15, 24, 25, 30, 50, 60] = 30,
        keyframe_interval: int | None = None,
        provenance: Any = None,
        options: EncodeOptions = EncodeOptions(),
        verify_decode: bool = True,
    ) -> None:
        self.path = Path(path)
        self.streams = streams
        self.options = options
        self.verify_decode = verify_decode

        if isinstance(timecode_origin, (int, float)):
            timecode_origin = datetime.fromtimestamp(
                math.floor(timecode_origin), tz=UTC)
        firsts = [float(s.timestamps[0]) for s in streams.values()]
        lasts = [float(s.timestamps[-1]) for s in streams.values()]
        if start is None:
            start = min(firsts) if span == "all" else max(firsts)
        if end is None:
            end = max(lasts) if span == "all" else min(lasts)
        if duration is not None:
            end = min(end, start + duration)
        if end < start:
            raise ValueError(f"The range from {start} to {end} is empty.")

        first_frame = math.floor(start * edit_rate)
        frame_count = math.ceil(end * edit_rate) - first_frame + 1
        Timecode.from_frame(  # raises if the last frame's timecode would wrap
            first_frame + frame_count - 1, edit_rate, timecode_origin)
        self.manifest = Manifest(
            bundle_id=uuid4(),
            recording_id=recording_id,
            timecode_origin=timecode_origin,
            edit_rate=edit_rate,
            keyframe_interval=keyframe_interval or max(1, edit_rate // 2),
            start_timecode=Timecode.from_frame(
                first_frame, edit_rate, timecode_origin),
            frame_count=frame_count,
            streams={
                stream_id: StreamEntry(
                    s.profile, s.label, s.overlay_of, s.alpha_mode, s.info)
                for stream_id, s in streams.items()},
            provenance=provenance)
        for s in streams.values():
            options.codec_args(s.profile, self.manifest.keyframe_interval)

        clock = (first_frame + np.arange(frame_count)) / edit_rate
        synced = sync({
            FRAME_CLOCK: clock,
            **{stream_id: s.timestamps for stream_id, s in streams.items()}})
        kept = synced[FRAME_CLOCK].astype(np.int64)
        self.shown: dict[str, Int64[np.ndarray, "F"]] = {}
        for stream_id in streams:
            sample_indices = synced[stream_id].astype(np.int64)
            if np.any(np.diff(sample_indices) < 0):
                raise ValueError(
                    f"Synchronization returned decreasing sample indices "
                    f"for stream {stream_id!r}.")
            shown = np.full(frame_count, -1, dtype=np.int64)
            shown[kept] = sample_indices
            if not np.any(shown >= 0):
                raise ValueError(
                    f"No samples of stream {stream_id!r} are shown.")
            self.shown[stream_id] = shown

    def write(
        self, workers: int | None = None, progress: Progress | None = None
    ) -> Manifest:
        """Render, encode and validate every stream, then write the manifest.

        Each stream is written in its own forked process, feeding its own
        `ffmpeg` process, and its file is validated before the manifest is
        written. If anything fails, the other processes are stopped, and the
        directory is left without a manifest.

        !!! warning "Processes are forked"

            Each stream runs in a child process created with `fork`, so
            streams do not need to be picklable. A child inherits only the
            thread that forks it, and any lock another thread held at that
            moment stays locked in the child. Start threads, thread pools
            and GPU contexts inside `Stream.frames`, as the
            [`Stream`][multireel.stream.Stream] constructors do, not before
            calling `write`. A warning is issued if other threads are
            running.

        Args:
            workers: number of streams written at once; defaults to all of
                them.
            progress: called every
                [`PROGRESS_INTERVAL`][multireel.write.PROGRESS_INTERVAL]
                seconds while streams are written, and once more when they
                are all written, with the frames written so far per stream
                and the number of frames in every stream. Frames are counted
                when they are handed to `ffmpeg`, so validation is not
                included.

        Returns:
            The manifest that was written.

        Raises:
            FileExistsError: if `path` is already a bundle.
            ValueError: if a stream renders no frames, or frames of the
                wrong shape or type.
            RuntimeError: if encoding fails, a stream file fails
                validation, or a stream's process exits without reporting a
                result.
        """
        if (self.path / "manifest.yaml").exists():
            raise FileExistsError(f"{self.path} is already a bundle.")
        if threading.active_count() > 1:
            warnings.warn(
                "Writing a bundle forks while other threads are running; "
                "locks they hold stay locked in the children.",
                RuntimeWarning, stacklevel=2)

        # One writer per counter, so no lock; shared memory, so children's
        # counts are visible to the parent.
        self._counts = multiprocessing.Array("q", len(self.streams), lock=False)
        (self.path / "streams").mkdir(parents=True, exist_ok=True)

        ctx = multiprocessing.get_context("fork")
        queue = list(reversed(list(self.streams)))
        running: dict[Any, tuple[str, Any, Connection]] = {}
        try:
            while queue or running:
                while queue and len(running) < (workers or len(self.streams)):
                    stream_id = queue.pop()
                    parent, child = ctx.Pipe(duplex=False)
                    process = ctx.Process(
                        target=self._child, args=(stream_id, child),
                        daemon=True)
                    process.start()
                    child.close()
                    running[process.sentinel] = (stream_id, process, parent)

                ready = wait(list(running), timeout=PROGRESS_INTERVAL)
                self._report(progress)
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
                            f"In the process for stream {stream_id!r}:\n"
                            f"{trace}")
                        raise error
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"The process for stream {stream_id!r} exited "
                            f"with code {process.exitcode}.")
        finally:
            # Only reached with children still running if something failed.
            for _, process, conn in running.values():
                process.terminate()
            for _, process, conn in running.values():
                process.join()
                conn.close()

        self._report(progress)
        self.manifest.write(self.path)
        return self.manifest

    def _report(self, progress: Progress | None) -> None:
        """Pass the frames written so far to the progress callback."""
        if progress is not None:
            progress(
                dict(zip(self.streams, self._counts)),
                self.manifest.frame_count)

    def _child(self, stream_id: str, conn: Connection) -> None:
        """Write one stream in a forked child, and report the result.

        Sends `None` on success, or the exception and its formatted
        traceback. Exits with `os._exit`, so the child never runs the
        parent's exit handlers or flushes buffers it inherited.
        """
        code = 0
        try:
            self._write_stream(stream_id)
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

    def _write_stream(self, stream_id: str) -> None:
        """Render, encode and validate one stream."""
        file = Manifest.stream_path(self.path, stream_id)
        self._encode(stream_id, file)
        violations = validate_stream(
            file, stream_id, self.manifest, ffmpeg=self.options.ffmpeg,
            ffprobe=self.options.ffprobe, decode=self.verify_decode)
        if violations:
            raise RuntimeError(
                f"Stream {stream_id!r} failed validation:\n"
                + "\n".join(violations))

    def _encode(self, stream_id: str, file: Path) -> None:
        """Render and encode one stream.

        Each shown sample is rendered once and written for as many frames
        as it is shown; no-data frames are black. The frame size is that of
        the first rendered frame. If anything fails, `ffmpeg` is stopped
        without finalizing the file, and the stream's frame iterator is
        closed, which stops its render threads.
        """
        stream, shown = self.streams[stream_id], self.shown[stream_id]
        counter = list(self.streams).index(stream_id)
        frames = stream.frames(np.unique(shown[shown >= 0]))
        with ExitStack() as stack:
            if (close := getattr(frames, "close", None)) is not None:
                stack.callback(close)

            pending = next(frames, None)
            if pending is None:
                raise ValueError(f"Stream {stream_id!r} rendered no frames.")
            height, width = pending.shape[:2]
            blank = np.zeros_like(pending)
            encoder = stack.enter_context(Encoder(
                file, self.manifest, stream_id, width, height, self.options,
                stream.channel_order))

            run_starts = np.flatnonzero(np.diff(shown, prepend=-2))
            run_lengths = np.diff(run_starts, append=len(shown))
            current, frame = -1, pending
            for run_start, length in zip(run_starts, run_lengths):
                sample = int(shown[run_start])
                if sample < 0:
                    encoder.write(blank, int(length))
                else:
                    if sample != current:
                        frame = (
                            pending if pending is not None else next(frames))
                        pending, current = None, sample
                    encoder.write(frame, int(length))
                self._counts[counter] += int(length)

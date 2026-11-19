"""Output streams: timestamps, and how to render the frames of a stream.

A [`Stream`][.Stream] is a list of source timestamps and a function that
yields the rendered frame of each requested sample, plus how to encode it.
It can be made from:

- [`random_access`][.Stream.random_access]: a function that loads one
    sample by index, for sources that can seek cheaply.
- [`sequential`][.Stream.sequential]: an iterable over every sample in
    order, for sources that are slow to seek, such as video.
- [`from_sensor`][.Stream.from_sensor]: an abstract dataloader
    [`Sensor`][abstract_dataloader.spec.Sensor], using either of the above.

Each of these loads only the samples that are shown, and renders them in a
pool of `workers` threads, keeping up to `2 * workers` samples ahead of the
encoder. The pool is started when frames are first requested, so it is
started inside the process that writes the stream.
"""

from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Self, TypeVar

import numpy as np
from abstract_dataloader import spec
from jaxtyping import Float64, Int64, UInt8

from .bundle import Profile

T = TypeVar("T")
R = TypeVar("R")


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
    def random_access(
        cls,
        timestamps: Float64[np.ndarray, "N"],
        load: Callable[[int], T],
        render: Callable[[T], UInt8[np.ndarray, "H W C"]],
        *,
        workers: int = 1,
        **kwargs: Any,
    ) -> Self:
        """Make a stream that loads each shown sample by index.

        Samples are loaded and rendered together in the worker threads.

        Args:
            timestamps: source timestamps, one per sample index.
            load: loads the sample at an index; must be thread-safe if
                `workers > 1`.
            render: turns one sample into one frame.
            workers: number of threads that load and render samples.
            kwargs: other `Stream` fields, such as `profile`.

        Returns:
            The stream.
        """

        def frames(
            indices: Int64[np.ndarray, "K"],
        ) -> Iterator[UInt8[np.ndarray, "H W C"]]:
            return _ordered_map(
                lambda i: render(load(i)), indices.tolist(), workers)

        return cls(np.asarray(timestamps, dtype=np.float64), frames, **kwargs)

    @classmethod
    def sequential(
        cls,
        timestamps: Float64[np.ndarray, "N"],
        samples: Callable[[], Iterable[T]],
        render: Callable[[T], UInt8[np.ndarray, "H W C"]],
        *,
        workers: int = 1,
        **kwargs: Any,
    ) -> Self:
        """Make a stream that reads every sample in order.

        Samples are read in the stream's own thread, and only the shown ones
        are rendered, in the worker threads. Reading stops after the last
        shown sample. To overlap reading with rendering, `samples` should
        prefetch.

        Args:
            timestamps: source timestamps, one per sample of `samples()`.
            samples: returns a new iterable over every sample, in order.
            render: turns one sample into one frame.
            workers: number of threads that render samples.
            kwargs: other `Stream` fields, such as `profile`.

        Returns:
            The stream.
        """

        def frames(
            indices: Int64[np.ndarray, "K"],
        ) -> Iterator[UInt8[np.ndarray, "H W C"]]:
            return _ordered_map(render, _select(samples(), indices), workers)

        return cls(np.asarray(timestamps, dtype=np.float64), frames, **kwargs)

    @classmethod
    def from_sensor(
        cls,
        sensor: spec.Sensor[T, Any],
        render: Callable[[T], UInt8[np.ndarray, "H W C"]],
        *,
        sequential: bool = False,
        workers: int = 1,
        **kwargs: Any,
    ) -> Self:
        """Make a stream from an abstract dataloader sensor.

        Timestamps come from `sensor.metadata.timestamps`. Samples are
        loaded with `sensor[i]`, as in [`random_access`][^.], or, with
        `sequential`, read in order from `sensor.stream()`, as in
        [`sequential`][^.].

        Args:
            sensor: source sensor.
            render: turns one sample into one frame.
            sequential: read samples with `sensor.stream()` instead of
                `sensor[i]`, for sensors that are slow to seek.
            workers: number of threads that render samples, and, unless
                `sequential`, load them.
            kwargs: other `Stream` fields, such as `profile`.

        Returns:
            The stream.
        """
        timestamps = sensor.metadata.timestamps
        if sequential:
            return cls.sequential(
                timestamps, sensor.stream, render, workers=workers, **kwargs)
        return cls.random_access(
            timestamps, sensor.__getitem__, render, workers=workers,
            **kwargs)


def _select(
    samples: Iterable[T], indices: Int64[np.ndarray, "K"]
) -> Iterator[T]:
    """Yield the samples at strictly increasing `indices` of an iterable.

    Stops reading `samples` after the last index.
    """
    wanted = iter(indices.tolist())
    target = next(wanted, None)
    if target is None:
        return
    for i, sample in enumerate(samples):
        if i == target:
            yield sample
            target = next(wanted, None)
            if target is None:
                return


def _ordered_map(
    fn: Callable[[T], R], items: Iterable[T], workers: int
) -> Generator[R, None, None]:
    """Apply `fn` to `items` in a thread pool, yielding results in order.

    Up to `2 * workers` items are processed ahead of the consumer. `items`
    is consumed in the calling thread. If the consumer stops early, pending
    items are cancelled, and running ones finish before this returns.
    """
    pending: deque[Future[R]] = deque()
    with ThreadPoolExecutor(workers) as pool:
        try:
            for item in items:
                pending.append(pool.submit(fn, item))
                if len(pending) >= 2 * workers:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()

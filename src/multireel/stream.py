"""Output streams.

A [`Stream`][.Stream] holds source timestamps and a function that renders
frames. Create one with:

- [`random_access`][.Stream.random_access]: load each sample by index. Use
    this for sources that can seek quickly.
- [`sequential`][.Stream.sequential]: read every sample in order. Use this
    for sources that are slow to seek, such as video.
- [`from_sensor`][.Stream.from_sensor]: read an abstract dataloader
    [`Sensor`][abstract_dataloader.spec.Sensor], in either of the above ways.

These only load the samples that are shown, and render them in `workers`
threads.
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

    `frames` is called with the indices of the samples to show, in
    increasing order, and yields one frame per index. Frames are `uint8`
    arrays of the same size. If the profile has alpha, it is the last
    channel.

    Attributes:
        timestamps: source timestamps in increasing order, in seconds since
            the UTC epoch.
        frames: renders the frames of the given samples.
        profile: encoding profile.
        label: display name.
        overlay_of: base stream ID, for overlays.
        alpha_mode: how alpha is stored, for overlays.
        info: any extra information to store in the manifest.
        channel_order: color channel order. Use `bgr` for OpenCV images.
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

        Args:
            timestamps: source timestamps.
            load: loads the sample at an index. Must be thread-safe if
                `workers > 1`.
            render: turns a sample into a frame.
            workers: number of threads that load and render samples.
            kwargs: other `Stream` fields.
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

        Reading stops after the last shown sample, and only shown samples
        are rendered. `samples` should prefetch, so that reading overlaps
        with rendering.

        Args:
            timestamps: source timestamps.
            samples: returns a new iterable over all samples.
            render: turns a sample into a frame.
            workers: number of threads that render samples.
            kwargs: other `Stream` fields.
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

        By default, samples are loaded by index with `sensor[i]`, as in
        [`random_access`][^.]. With `sequential`, they are read in order from
        `sensor.stream()`, as in [`sequential`][^.].

        Args:
            sensor: source sensor.
            render: turns a sample into a frame.
            sequential: read samples in order instead of by index.
            workers: number of worker threads.
            kwargs: other `Stream` fields.
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
    """Yield the samples at the given increasing indices.

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
    """Apply `fn` to `items` in a thread pool, and yield results in order.

    At most `2 * workers` items run ahead of the consumer. If the consumer
    stops early, items that have not started are cancelled.
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

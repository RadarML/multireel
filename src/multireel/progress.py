"""Progress bars for writing bundles.

Needs the `rich` extra:

```sh
pip install multireel[rich]
```
"""

from collections.abc import Mapping
from types import TracebackType
from typing import Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class _RateColumn(ProgressColumn):
    """Frames per second."""

    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        return Text(
            "" if speed is None else f"{speed:.0f} fps",
            style="progress.data.speed")


class RichProgress:
    """Rich progress bars, one per stream.

    Use as a context manager, and pass it to
    [`BundleWriter.write`][multireel.write.BundleWriter.write]:

    ```python
    with RichProgress() as progress:
        writer.write(progress=progress)
    ```

    Args:
        console: console to draw on.
    """

    def __init__(self, console: Console | None = None) -> None:
        # Bars are redrawn on each update instead of by rich's refresh
        # thread, since writing a bundle forks and must not fork while other
        # threads are running.
        self.progress = Progress(
            TextColumn("{task.description}"), BarColumn(),
            MofNCompleteColumn(), _RateColumn(), TimeElapsedColumn(),
            TimeRemainingColumn(), console=console, auto_refresh=False)
        self.tasks: dict[str, TaskID] = {}

    def __enter__(self) -> Self:
        """Start drawing."""
        self.progress.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        """Stop drawing."""
        self.progress.stop()

    def __call__(self, done: Mapping[str, int], frame_count: int) -> None:
        """Update every bar with the frames written so far."""
        for stream_id, n in done.items():
            if stream_id not in self.tasks:
                self.tasks[stream_id] = self.progress.add_task(
                    stream_id, total=frame_count)
            self.progress.update(self.tasks[stream_id], completed=n)
        self.progress.refresh()

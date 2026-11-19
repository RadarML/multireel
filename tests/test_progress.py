"""Tests for the rich progress bars."""

import io

from rich.console import Console

from multireel.progress import RichProgress


def test_rich_progress() -> None:
    """One bar per stream, updated to the latest count."""
    console = Console(file=io.StringIO(), force_terminal=True)
    with RichProgress(console) as progress:
        progress({"cam": 0, "lidar": 0}, 10)
        progress({"cam": 4, "lidar": 10}, 10)
    tasks = {t.description: t for t in progress.progress.tasks}
    assert tasks["cam"].completed == 4 and tasks["lidar"].completed == 10
    assert tasks["lidar"].finished

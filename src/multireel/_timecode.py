"""Frame timing and timecode.

Frame indices are absolute: frame `n` at edit rate `r` is at `n / r` seconds
since the UTC epoch.
"""

import re
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Timecode:
    """Non-drop-frame timecode at a given edit rate.

    `str()` formats it as `"HH:MM:SS:FF"`, and [`parse`][.parse] reads it
    back.

    Attributes:
        hours: hours, `0` to `23`.
        minutes: minutes, `0` to `59`.
        seconds: seconds, `0` to `59`.
        frames: frames, `0` to `rate - 1`.
        rate: edit rate, in frames per second.
    """

    hours: int
    minutes: int
    seconds: int
    frames: int
    rate: int

    @staticmethod
    def _origin_frame(origin: datetime, rate: int) -> int:
        """Return the absolute frame index of a timecode origin."""
        if origin.tzinfo is None or origin.microsecond != 0:
            raise ValueError(
                f"Timecode origin must be timezone-aware and a whole number "
                f"of seconds: {origin}")
        return int(origin.timestamp()) * rate

    @classmethod
    def from_frame(
        cls, frame: int, rate: int, origin: datetime
    ) -> "Timecode":
        """Return the timecode of a frame.

        `origin` is shown as `01:00:00:00`.

        Raises:
            ValueError: if the timecode would wrap past 24 hours, or `origin`
                is not a timezone-aware whole second.
        """
        offset = frame - cls._origin_frame(origin, rate) + 3600 * rate
        if not 0 <= offset < 24 * 3600 * rate:
            raise ValueError(
                f"Frame {frame} is outside the timecode range of origin "
                f"{origin}.")
        seconds, frames = divmod(offset, rate)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return cls(hours, minutes, seconds, frames, rate)

    @classmethod
    def parse(cls, text: str, rate: int) -> "Timecode":
        """Parse a `"HH:MM:SS:FF"` string.

        Raises:
            ValueError: if `text` is not a valid timecode at `rate`.
        """
        match = re.fullmatch(r"(\d\d):(\d\d):(\d\d):(\d\d)", text)
        if match is None:
            raise ValueError(f"Invalid timecode: {text!r}")
        hours, minutes, seconds, frames = map(int, match.groups())
        if hours > 23 or minutes > 59 or seconds > 59 or frames >= rate:
            raise ValueError(f"Invalid timecode at {rate} fps: {text!r}")
        return cls(hours, minutes, seconds, frames, rate)

    def to_frame(self, origin: datetime) -> int:
        """Return the frame index of this timecode."""
        offset = (
            ((self.hours * 60 + self.minutes) * 60 + self.seconds) * self.rate
            + self.frames)
        return (
            offset - 3600 * self.rate
            + self._origin_frame(origin, self.rate))

    def __str__(self) -> str:
        """Format as `"HH:MM:SS:FF"`."""
        return (
            f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}:"
            f"{self.frames:02d}")

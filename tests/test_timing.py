"""Tests for timecode arithmetic."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from multireel import Timecode

ORIGIN = datetime(2026, 11, 18, 16, 0, 0, tzinfo=UTC)


def test_origin_is_one_hour() -> None:
    """The origin's own frame is 01:00:00:00."""
    frame = int(ORIGIN.timestamp()) * 30
    assert str(Timecode.from_frame(frame, 30, ORIGIN)) == "01:00:00:00"


@pytest.mark.parametrize("rate", [10, 15, 24, 25, 30, 50, 60])
@pytest.mark.parametrize("offset", [-3600, -0.5, 0, 1.25, 723.4, 82799])
def test_round_trip(rate: int, offset: float) -> None:
    """Frame to timecode to string to timecode to frame is lossless."""
    frame = int(ORIGIN.timestamp()) * rate + int(offset * rate)
    timecode = Timecode.from_frame(frame, rate, ORIGIN)
    parsed = Timecode.parse(str(timecode), rate)
    assert parsed == timecode
    assert parsed.to_frame(ORIGIN) == frame


def test_known_value() -> None:
    """12 minutes and 15 frames after the origin."""
    frame = (int(ORIGIN.timestamp()) + 720) * 30 + 15
    assert str(Timecode.from_frame(frame, 30, ORIGIN)) == "01:12:00:15"


def test_other_timezone() -> None:
    """An origin in another timezone means the same instant."""
    local = ORIGIN.astimezone(timezone(timedelta(hours=-5)))
    frame = int(ORIGIN.timestamp()) * 30
    assert str(Timecode.from_frame(frame, 30, local)) == "01:00:00:00"


@pytest.mark.parametrize("offset", [-3601, 23 * 3600])
def test_wrap(offset: int) -> None:
    """Timecode outside 00:00:00:00 to 23:59:59:FF is rejected."""
    frame = (int(ORIGIN.timestamp()) + offset) * 30
    with pytest.raises(ValueError):
        Timecode.from_frame(frame, 30, ORIGIN)


@pytest.mark.parametrize("origin", [
    datetime(2026, 11, 18, 16, 0, 0),
    datetime(2026, 11, 18, 16, 0, 0, 500000, tzinfo=UTC),
])
def test_bad_origin(origin: datetime) -> None:
    """Naive and fractional-second origins are rejected."""
    with pytest.raises(ValueError):
        Timecode.from_frame(0, 30, origin)


@pytest.mark.parametrize("text", ["1:00:00:00", "24:00:00:00", "01:00:00:30",
                                  "01;00;00;00"])
def test_parse_invalid(text: str) -> None:
    """Malformed and out-of-range timecodes are rejected."""
    with pytest.raises(ValueError):
        Timecode.parse(text, 30)

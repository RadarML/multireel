"""Video format: encoding profiles and container naming rules."""

from dataclasses import dataclass
from typing import Literal

Profile = Literal[
    "hevc-main10-420",
    "hevc-444",
    "prores-422-lt",
    "prores-422-hq",
    "prores-4444",
]
"""Encoding profile of a stream, named as in the manifest."""


@dataclass(frozen=True)
class ProfileInfo:
    """Codec properties fixed by an encoding profile.

    Attributes:
        codec: codec family.
        chroma: chroma subsampling.
        bit_depth: bits per component.
        alpha: whether the profile carries an alpha channel.
    """

    codec: Literal["hevc", "prores"]
    chroma: Literal["420", "422", "444"]
    bit_depth: int
    alpha: bool


def profile_info(profile: Profile) -> ProfileInfo:
    """Return the codec properties of an encoding profile."""
    return {
        "hevc-main10-420": ProfileInfo("hevc", "420", 10, False),
        "hevc-444": ProfileInfo("hevc", "444", 10, False),
        "prores-422-lt": ProfileInfo("prores", "422", 10, False),
        "prores-422-hq": ProfileInfo("prores", "422", 10, False),
        "prores-4444": ProfileInfo("prores", "444", 12, True),
    }[profile]


def reel_name(recording_id: str, stream_id: str) -> str:
    """Return the reel name of a stream, `<recording_id>.<stream_id>`."""
    return f"{recording_id}.{stream_id}"


def clip_name(recording_id: str, stream_id: str, label: str | None) -> str:
    """Return the clip name of a stream, `<recording_id> <label>`.

    The stream ID is used in place of the label if there is none.
    """
    return f"{recording_id} {label or stream_id}"

"""Video format: encoding profiles and container naming rules."""

from dataclasses import dataclass
from typing import Any, Literal

ProfileName = Literal[
    "hevc-main10-420",
    "hevc-444",
    "prores-422-lt",
    "prores-422-hq",
    "prores-4444",
]
"""Name of a standard encoding profile."""


@dataclass(frozen=True)
class Profile:
    """Codec properties of a stream: a standard profile, or a custom one.

    Use [`from_name`][^.] for the standard profiles in the spec.

    Attributes:
        codec: codec family.
        chroma: chroma subsampling.
        bit_depth: bits per component, as decoded.
        alpha: whether the stream carries an alpha channel.
        level: ProRes profile, as named by `prores_ks`: `proxy`, `lt`,
            `standard`, `hq`, `4444` or `4444xq`. Not used for HEVC.
    """

    codec: Literal["hevc", "prores"]
    chroma: Literal["420", "422", "444"]
    bit_depth: int
    alpha: bool
    level: str | None = None

    @classmethod
    def from_name(cls, name: ProfileName) -> "Profile":
        """Return a standard profile.

        Raises:
            ValueError: if `name` is not a standard profile.
        """
        try:
            return _STANDARD[name]
        except KeyError:
            raise ValueError(f"Unknown profile: {name!r}") from None

    @classmethod
    def from_manifest(cls, value: Any) -> "Profile":
        """Read a profile from its manifest value.

        Args:
            value: a standard profile name, or a mapping of its fields.

        Raises:
            KeyError: if a mapping is missing a required field.
            ValueError: if a name is not a standard profile.
        """
        if isinstance(value, str):
            return cls.from_name(value)  # type: ignore
        return cls(
            codec=value["codec"], chroma=str(value["chroma"]),  # type: ignore
            bit_depth=int(value["bit_depth"]), alpha=bool(value["alpha"]),
            level=value.get("level"))

    def to_manifest(self) -> str | dict[str, Any]:
        """Return the manifest value of this profile.

        Returns:
            The standard profile name, or a mapping of the fields if this
                is a custom profile.
        """
        if self.name is not None:
            return self.name
        fields: dict[str, Any] = {
            "codec": self.codec, "chroma": self.chroma,
            "bit_depth": self.bit_depth, "alpha": self.alpha}
        if self.level is not None:
            fields["level"] = self.level
        return fields

    @property
    def name(self) -> ProfileName | None:
        """Name of the matching standard profile, or `None` if custom."""
        return next((n for n, p in _STANDARD.items() if p == self), None)


_STANDARD: dict[ProfileName, Profile] = {
    "hevc-main10-420": Profile("hevc", "420", 10, False),
    "hevc-444": Profile("hevc", "444", 10, False),
    "prores-422-lt": Profile("prores", "422", 10, False, "lt"),
    "prores-422-hq": Profile("prores", "422", 10, False, "hq"),
    "prores-4444": Profile("prores", "444", 12, True, "4444"),
}


def reel_name(recording_id: str, stream_id: str) -> str:
    """Return the reel name of a stream, `<recording_id>.<stream_id>`."""
    return f"{recording_id}.{stream_id}"


def clip_name(recording_id: str, stream_id: str, label: str | None) -> str:
    """Return the clip name of a stream, `<recording_id> <label>`.

    The stream ID is used in place of the label if there is none.
    """
    return f"{recording_id} {label or stream_id}"

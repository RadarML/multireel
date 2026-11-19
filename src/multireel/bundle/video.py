"""Video format: encoding profiles."""

from dataclasses import dataclass
from typing import Any, Literal

ProfileName = Literal[
    "hevc-main10-420",
    "hevc-444",
    "prores-422-lt",
    "prores-422-hq",
    "prores-4444",
]
"""Name of a standard profile."""


@dataclass(frozen=True)
class Profile:
    """How a stream is encoded.

    Use [`from_name`][.from_name] for one of the spec's standard profiles, or
    create a custom profile. ProRes profiles need a level: `proxy`, `lt`,
    `standard`, `hq`, `4444` or `4444xq`.

    Attributes:
        codec: codec family.
        chroma: chroma subsampling.
        bit_depth: bits per color channel.
        alpha: whether the stream has an alpha channel.
        level: ProRes level. Not used for HEVC.
    """

    codec: Literal["hevc", "prores"]
    chroma: Literal["420", "422", "444"]
    bit_depth: int
    alpha: bool
    level: str | None = None

    @classmethod
    def from_name(cls, name: ProfileName) -> "Profile":
        """Return the standard profile with this name.

        Raises:
            ValueError: if there is no such profile.
        """
        try:
            return _STANDARD[name]
        except KeyError:
            raise ValueError(f"Unknown profile: {name!r}") from None

    @classmethod
    def from_manifest(cls, value: Any) -> "Profile":
        """Read a profile from the manifest.

        Args:
            value: a standard profile name, or a mapping of fields.

        Raises:
            KeyError: if a field is missing.
            ValueError: if the name is unknown.
        """
        if isinstance(value, str):
            return cls.from_name(value)  # type: ignore
        return cls(
            codec=value["codec"], chroma=str(value["chroma"]),  # type: ignore
            bit_depth=int(value["bit_depth"]), alpha=bool(value["alpha"]),
            level=value.get("level"))

    def to_manifest(self) -> str | dict[str, Any]:
        """Return the profile as written in the manifest.

        This is the standard profile name, or a mapping of fields for a
        custom profile.
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
        """Name of the matching standard profile, if any."""
        return next((n for n, p in _STANDARD.items() if p == self), None)


_STANDARD: dict[ProfileName, Profile] = {
    "hevc-main10-420": Profile("hevc", "420", 10, False),
    "hevc-444": Profile("hevc", "444", 10, False),
    "prores-422-lt": Profile("prores", "422", 10, False, "lt"),
    "prores-422-hq": Profile("prores", "422", 10, False, "hq"),
    "prores-4444": Profile("prores", "444", 12, True, "4444"),
}

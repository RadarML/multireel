"""Checking stream files against the bundle spec.

[`write_bundle`][multireel.write.write_bundle] checks every stream file it
encodes. Each failed check is reported as a `"<rule>: <message>"` string.

- `checks`: the per-stream rules, and checking a stream file.
- `probe`: reading stream file properties with `ffprobe`. The checks only
    look at the [`StreamProbe`][.] produced there, so they can be tested
    with crafted probes instead of crafted files.
"""

from .checks import check_stream, validate_stream
from .probe import StreamProbe, decode_errors

__all__ = [
    "StreamProbe",
    "check_stream",
    "decode_errors",
    "validate_stream",
]

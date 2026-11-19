"""Checking bundles against the bundle spec.

The validator checks every MUST in the spec, and reports each failure as a
[`Violation`][.] naming the rule that was broken.

- `checks`: the rules, and checking a bundle or a stream file.
- `probe`: reading stream file properties with `ffprobe`. The checks only
    look at the [`StreamProbe`][.] produced there, so they can be tested
    with crafted probes instead of crafted files.
- `cli`: the `multireel-validate` command, also run by
    `python -m multireel.validate`.
"""

from .checks import Rule, Violation, validate_bundle, validate_stream
from .cli import main
from .probe import StreamProbe, decode_errors

__all__ = [
    "Rule",
    "StreamProbe",
    "Violation",
    "decode_errors",
    "main",
    "validate_bundle",
    "validate_stream",
]

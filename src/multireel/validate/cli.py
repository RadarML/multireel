"""The `multireel-validate` command, also run by `python -m multireel.validate`.

```sh
multireel-validate PATH [--ffmpeg FFMPEG] [--ffprobe FFPROBE]
```
"""

import argparse
from collections.abc import Sequence

from .checks import validate_bundle


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a bundle and print every violation.

    Returns:
        Exit code: `0` if the bundle is valid, and `1` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="multireel-validate",
        description="Check a bundle against the bundle spec.")
    parser.add_argument("path", help="bundle directory")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)

    violations = validate_bundle(
        args.path, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
    for v in violations:
        where = f"{v.stream}: " if v.stream else ""
        print(f"{v.rule}: {where}{v.message}")
    if not violations:
        print(f"{args.path} is a valid bundle.")
    return 1 if violations else 0

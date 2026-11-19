"""Web player for multireel bundles.

Serves a page that lists the bundles under a folder and plays the streams of
one bundle in a synced grid. Stream files are served as they are, with HTTP
range requests, so playback depends on the browser decoding each stream's
codec.

```sh
uv run --extra player python player/server.py <folder> --port 8000
```
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, jsonify, request, send_file
from werkzeug.security import safe_join

from multireel import Manifest

INDEX = Path(__file__).parent / "index.html"


def find_bundles(root: Path) -> list[Path]:
    """Return every bundle under `root`, sorted.

    A directory is a bundle if it contains `manifest.yaml`. Directories
    inside a bundle are not searched.
    """
    bundles: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "manifest.yaml" in filenames:
            bundles.append(Path(dirpath))
            dirnames.clear()
        else:
            dirnames.sort()
    return sorted(bundles)


def _bundle_summary(root: Path, bundle: Path) -> dict[str, Any]:
    """Return the bundle list entry of one bundle."""
    path = bundle.relative_to(root).as_posix()
    try:
        manifest = Manifest.read(bundle)
    except (OSError, ValueError, TypeError) as e:
        return {"path": path, "error": str(e)}
    return {
        "path": path,
        "recording_id": manifest.recording_id,
        "edit_rate": manifest.edit_rate,
        "frame_count": manifest.frame_count,
        "start_timecode": str(manifest.start_timecode),
        "n_streams": len(manifest.streams),
    }


def _manifest_json(manifest: Manifest) -> dict[str, Any]:
    """Return a manifest as the player reads it."""
    return {
        "recording_id": manifest.recording_id,
        "timecode_origin": manifest.timecode_origin.isoformat(),
        "edit_rate": manifest.edit_rate,
        "keyframe_interval": manifest.keyframe_interval,
        "start_timecode": str(manifest.start_timecode),
        "start_frame": manifest.start_timecode.to_frame(
            manifest.timecode_origin),
        "frame_count": manifest.frame_count,
        "streams": [
            {
                "id": stream_id,
                "label": entry.label,
                "profile": entry.profile.to_manifest(),
                "codec": entry.profile.codec,
                "alpha": entry.profile.alpha,
                "overlay_of": entry.overlay_of,
            }
            for stream_id, entry in manifest.streams.items()],
    }


def create_app(root: Path, max_chunk: int | None = 1 << 20) -> Flask:
    """Create the player app for the bundles under `root`.

    Browsers request video as open-ended ranges (`bytes=N-`) and cancel them
    when they seek. Over a slow link, such as an SSH tunnel, data already
    sent for the old position still has to arrive before data for the new
    one. Answering open-ended ranges with at most `max_chunk` bytes bounds
    that backlog; browsers then request the next chunk as they play.

    Args:
        root: folder to look for bundles in.
        max_chunk: largest response to an open-ended range, in bytes, or
            `None` for no limit.
    """
    root = root.resolve()
    app = Flask(__name__)

    def open_bundle(bundle: str) -> tuple[Path, Manifest]:
        joined = safe_join(str(root), bundle)
        if joined is None:
            abort(404)
        path = Path(joined)
        if not (path / "manifest.yaml").is_file():
            abort(404)
        try:
            return path, Manifest.read(path)
        except (OSError, ValueError, TypeError) as e:
            abort(500, str(e))

    @app.get("/")
    def index() -> Response:
        return send_file(INDEX, max_age=0)

    @app.get("/api/bundles")
    def bundles() -> Response:
        return jsonify([
            _bundle_summary(root, bundle) for bundle in find_bundles(root)])

    @app.get("/api/manifest/<path:bundle>")
    def manifest(bundle: str) -> Response:
        _, manifest = open_bundle(bundle)
        return jsonify(_manifest_json(manifest))

    @app.get("/stream/<path:bundle>/<stream_id>")
    def stream(bundle: str, stream_id: str) -> Response:
        path, manifest = open_bundle(bundle)
        if stream_id not in manifest.streams:
            abort(404)
        # `send_file` reads the range from the environ, so an open-ended
        # range is limited by rewriting it there.
        open_ended = re.fullmatch(
            r"bytes=(\d+)-", request.environ.get("HTTP_RANGE", "").strip())
        if max_chunk is not None and open_ended is not None:
            start = int(open_ended.group(1))
            request.environ["HTTP_RANGE"] = (
                f"bytes={start}-{start + max_chunk - 1}")
        # Chrome refuses `video/quicktime`, but plays MOV files labeled as
        # MP4, since both are ISO base media files.
        return send_file(
            Manifest.stream_path(path, stream_id), mimetype="video/mp4",
            conditional=True, max_age=0)

    return app


def main() -> None:
    """Run the player server."""
    parser = argparse.ArgumentParser(
        description="Web player for multireel bundles.")
    parser.add_argument("root", type=Path, help="Folder to look for bundles.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port.")
    parser.add_argument(
        "--chunk-mb", type=float, default=1.0,
        help="Largest response to an open-ended range request, in MiB; 0 "
        "for no limit. Smaller values make seeking faster over slow links.")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"Not a directory: {args.root}")
    max_chunk = int(args.chunk_mb * (1 << 20)) or None
    create_app(args.root, max_chunk=max_chunk).run(
        host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

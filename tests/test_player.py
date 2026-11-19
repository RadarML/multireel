"""Tests for the bundle player server in `player/server.py`."""

import importlib.util
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from flask.testing import FlaskClient

from multireel import Manifest, Profile, StreamEntry, Timecode

_spec = importlib.util.spec_from_file_location(
    "player_server", Path(__file__).parents[1] / "player" / "server.py")
assert _spec is not None and _spec.loader is not None
server = importlib.util.module_from_spec(_spec)
sys.modules["player_server"] = server
_spec.loader.exec_module(server)

ORIGIN = datetime(2026, 11, 18, 16, 0, 0, tzinfo=UTC)


def write_bundle(path: Path) -> dict[str, bytes]:
    """Write a manifest and placeholder stream files; return their bytes."""
    manifest = Manifest(
        bundle_id=uuid4(), recording_id="drive_0423_a",
        timecode_origin=ORIGIN, edit_rate=30, keyframe_interval=15,
        start_timecode=Timecode(1, 12, 0, 0, 30), frame_count=90,
        streams={
            "cam": StreamEntry(
                Profile.from_name("hevc-main10-420"), label="Front camera"),
            "boxes": StreamEntry(
                Profile.from_name("prores-4444"), overlay_of="cam"),
        })
    (path / "streams").mkdir(parents=True)
    manifest.write(path)
    data = {}
    for stream_id in manifest.streams:
        data[stream_id] = os.urandom(1000)
        Manifest.stream_path(path, stream_id).write_bytes(data[stream_id])
    return data


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Served folder, next to a bundle `outside` that must not be served."""
    write_bundle(tmp_path / "outside")
    return tmp_path / "root"


@pytest.fixture
def bundle_data(root: Path) -> dict[str, bytes]:
    """A bundle at `a/bundle`, and an incomplete one at `a/partial`."""
    data = write_bundle(root / "a" / "bundle")
    (root / "a" / "partial" / "streams").mkdir(parents=True)
    (root / "a" / "partial" / "streams" / "cam.mov").write_bytes(b"x")
    return data


@pytest.fixture
def client(root: Path, bundle_data: dict[str, bytes]) -> FlaskClient:
    """Test client for a player serving `root`."""
    return server.create_app(root).test_client()


def test_index(client: FlaskClient) -> None:
    """The root serves the player page."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.data


def test_bundles(client: FlaskClient) -> None:
    """Only directories with a manifest are listed."""
    response = client.get("/api/bundles")
    assert response.json == [{
        "path": "a/bundle", "recording_id": "drive_0423_a", "edit_rate": 30,
        "frame_count": 90, "start_timecode": "01:12:00:00", "n_streams": 2}]


def test_bundles_bad_manifest(root: Path, client: FlaskClient) -> None:
    """A manifest that does not parse is listed with an error."""
    (root / "bad").mkdir()
    (root / "bad" / "manifest.yaml").write_text("schema_version: '9.0'")
    entries = client.get("/api/bundles").json
    assert entries is not None
    bad = next(e for e in entries if e["path"] == "bad")
    assert "error" in bad


def test_manifest(client: FlaskClient) -> None:
    """The manifest round-trips the fields the player uses."""
    data = client.get("/api/manifest/a/bundle").json
    assert data is not None
    assert data["edit_rate"] == 30
    assert data["frame_count"] == 90
    assert data["start_timecode"] == "01:12:00:00"
    assert data["start_frame"] == (int(ORIGIN.timestamp()) + 720) * 30
    assert [s["id"] for s in data["streams"]] == ["cam", "boxes"]
    assert data["streams"][0]["label"] == "Front camera"
    assert data["streams"][0]["codec"] == "hevc"
    assert data["streams"][1]["overlay_of"] == "cam"
    assert data["streams"][1]["alpha"] is True
    assert data["streams"][1]["profile"] == "prores-4444"


def test_stream_full(
    client: FlaskClient, bundle_data: dict[str, bytes]
) -> None:
    """Without a range, the whole file is sent."""
    response = client.get("/stream/a/bundle/cam")
    assert response.status_code == 200
    assert response.mimetype == "video/mp4"
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.data == bundle_data["cam"]


@pytest.mark.parametrize("header,start,end", [
    ("bytes=10-99", 10, 100),
    ("bytes=900-", 900, 1000),
    ("bytes=-50", 950, 1000),
])
def test_stream_range(
    client: FlaskClient, bundle_data: dict[str, bytes],
    header: str, start: int, end: int,
) -> None:
    """Range requests return the matching slice."""
    response = client.get("/stream/a/bundle/cam", headers={"Range": header})
    assert response.status_code == 206
    assert response.headers["Content-Range"] == f"bytes {start}-{end - 1}/1000"
    assert response.data == bundle_data["cam"][start:end]


def test_stream_unsatisfiable(client: FlaskClient) -> None:
    """A range past the end of the file is rejected."""
    response = client.get(
        "/stream/a/bundle/cam", headers={"Range": "bytes=5000-"})
    assert response.status_code == 416


@pytest.mark.parametrize("url", [
    "/stream/a/bundle/missing",
    "/stream/a/partial/cam",
    "/stream/../outside/cam",
    "/stream/a/../../outside/cam",
    "/stream/%2e%2e/outside/cam",
    "/stream/%2E%2E%2Foutside/cam",
    "/api/manifest/a/partial",
    "/api/manifest/a",
    "/api/manifest/../outside",
    "/api/manifest/%2e%2e/outside",
])
def test_not_found(client: FlaskClient, url: str) -> None:
    """Unknown streams, incomplete bundles and traversal are not served."""
    assert client.get(url).status_code == 404


def test_path_within_root(client: FlaskClient) -> None:
    """A path with `..` that stays inside the root is served."""
    assert client.get("/stream/a/../a/bundle/cam").status_code == 200


@pytest.mark.parametrize("max_chunk,header,start,end", [
    (100, "bytes=10-", 10, 110),
    (100, "bytes=950-", 950, 1000),
    (100, "bytes=10-499", 10, 500),
    (100, "bytes=-300", 700, 1000),
    (None, "bytes=10-", 10, 1000),
])
def test_stream_chunk(
    root: Path, bundle_data: dict[str, bytes],
    max_chunk: int | None, header: str, start: int, end: int,
) -> None:
    """Only open-ended ranges are limited to `max_chunk` bytes."""
    client = server.create_app(root, max_chunk=max_chunk).test_client()
    response = client.get("/stream/a/bundle/cam", headers={"Range": header})
    assert response.status_code == 206
    assert response.headers["Content-Range"] == f"bytes {start}-{end - 1}/1000"
    assert response.data == bundle_data["cam"][start:end]

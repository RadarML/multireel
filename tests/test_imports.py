"""Smoke test: every public module imports."""

import importlib

import pytest

MODULES = [
    "multireel",
    "multireel.bundle",
    "multireel.bundle.manifest",
    "multireel.bundle.timing",
    "multireel.bundle.video",
    "multireel.progress",
    "multireel.render",
    "multireel.stream",
    "multireel.write",
    "multireel.write.encode",
    "multireel.write.validate",
    "multireel.write.writer",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    """Import a public module."""
    importlib.import_module(name)

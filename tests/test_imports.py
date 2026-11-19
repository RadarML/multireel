"""Smoke test: every public module imports."""

import importlib

import pytest

MODULES = [
    "multireel",
    "multireel.bundle",
    "multireel.bundle.manifest",
    "multireel.bundle.timing",
    "multireel.bundle.video",
    "multireel.encode",
    "multireel.validate",
    "multireel.validate.checks",
    "multireel.validate.probe",
    "multireel.write",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    """Import a public module."""
    importlib.import_module(name)

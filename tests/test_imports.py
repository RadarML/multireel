"""Smoke test: every public module imports."""

import importlib

import pytest

MODULES = [
    "multireel",
    "multireel.progress",
    "multireel.render",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    """Import a public module."""
    importlib.import_module(name)

# Multireel: Multimodal Rendering for Video Editing

[![GitHub](https://img.shields.io/github/v/release/RadarML/multireel)](https://github.com/RadarML/multireel)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![Supports Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
![Typed](https://img.shields.io/badge/types-typed-limegreen)
[![CI](https://github.com/RadarML/multireel/actions/workflows/ci.yml/badge.svg)](https://github.com/RadarML/multireel/actions/workflows/ci.yml)
![GitHub issues](https://img.shields.io/github/issues/RadarML/multireel)

**Multireel** is an open-source and MIT-licensed Python framework for multimodal rendering for video editing.

## Setup

=== "As Submodule"

    ```sh
    git submodule add git@github.com:RadarML/multireel.git
    uv pip install -e ./multireel
    ```

=== "Via `uv` + `pyproject.toml`"

    ```toml
    [project]
    dependencies = ["multireel"]

    [tool.uv.sources]
    multireel = { path = "./multireel", editable = true }
    ```

=== "For Development"

    ```sh
    git clone git@github.com:RadarML/multireel.git
    cd multireel
    uv sync --all-extras --frozen
    uv run pre-commit install
    ```

## Development Environment

!!! info "`multireel` uses `uv`"

    You will need [`uv`](https://docs.astral.sh/uv/getting-started/installation/):
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

**Setup**:

```sh
uv sync --all-extras
uv run pre-commit install
```

**Tests**:

```sh
uv run pytest -ra --cov=src/multireel --cov-report=html --cov-report=term -- tests
```

!!! info

    The pre-commit hooks (`ruff + pyright + pytest`) can be run manually with `uv run pre-commit run --all-files`.

**Docs**:

```sh
uv run --extra docs zensical serve
```

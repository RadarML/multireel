# Development Environment

!!! info "`multireel` uses `uv`"

    You will need [`uv`](https://docs.astral.sh/uv/getting-started/installation/):
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

## Setup

```sh
uv sync --all-extras
uv run pre-commit install
```

## Tests

```sh
uv run pytest -ra --cov=src/multireel --cov-report=html --cov-report=term -- tests
```

!!! info

    The pre-commit hooks (`ruff + pyright + pytest`) can be run manually with `uv run pre-commit run --all-files`.

## Docs

```sh
uv run --extra docs zensical serve
```

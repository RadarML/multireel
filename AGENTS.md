# AGENTS.md: AI Agent Guide for Multireel

Multireel is a multimodal rendering framework for video editing. The project is at an early stage; update this guide as the architecture takes shape.

## Layout

- `src/multireel/`: the library (src layout, built with hatchling; ships `py.typed`).
- `tests/`: pytest tests (`test_*.py`).
- `docs/`: [zensical](https://zensical.org) site (configured via `mkdocs.yml`); the API reference is generated from docstrings via mkdocstrings (`docs/api.md`). Add new pages to `nav` in `mkdocs.yml`.

## Conventions

- **Typing**: all code is fully typed and must pass `pyright`. Use `jaxtyping` for array shapes when arrays are involved.
- **Linting**: `ruff` with an 80-character line limit and pydocstyle rules (see `pyproject.toml`).
- **Docstrings**: Google style, since they are rendered by mkdocstrings. Every public module, class, and function needs one.
- **Dependencies**: managed with `uv`; commit `uv.lock`. Keep runtime dependencies minimal and put optional ones in an extra.

## Commands

```bash
uv sync --all-extras                         # Install, including dev + docs extras
uv run pre-commit install                    # ruff + pyright + pytest on every commit
uv run pre-commit run --all-files            # Run all hooks manually
uv run --extra dev ruff check src tests      # Lint
uv run --extra dev pyright src tests         # Type check
uv run --extra dev pytest -ra -- tests       # Test
uv run --extra docs zensical serve           # Serve docs locally
uv run --extra docs zensical build --strict  # Build docs as CI does
```

CI (`.github/workflows/ci.yml`) runs ruff, pyright and pytest on pushes and PRs to `main`; `.github/workflows/docs.yml` builds and deploys the docs to GitHub Pages on pushes to `main`.

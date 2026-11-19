# AGENTS.md: AI Agent Guide for Multireel

Multireel is a multimodal rendering framework for video editing. The project is at an early stage; update this guide as the architecture takes shape.

## Layout

- `src/multireel/`: the library (src layout, built with hatchling; ships `py.typed`).
- `tests/`: pytest tests (`test_*.py`).
- `docs/`: [zensical](https://zensical.org) site (configured via `mkdocs.yml`); the API reference is generated from docstrings via mkdocstrings (`docs/api/`, one page per top-level module). Add new pages to `nav` in `mkdocs.yml`.
- `docs/spec.md`: the bundle format spec (video format, time model, manifest). Code that reads or writes bundles must follow it.

## Architecture

- `bundle/`: the bundle format as code (`video`, `timing`, `manifest`, mirroring the spec's sections). Pure data types and functions; implementation defaults do not belong here.
- `write.py`: `Stream` (an [abstract dataloader](https://radarml.github.io/abstract-dataloader/) `Sensor` plus a render function) and `write_bundle`, which writes each stream in its own thread, or its own forked process with `processes=True`.
- `encode.py`: `ffmpeg` encoding.
- `validate/`: `checks` (per-stream rules; returns a list of `Violation`s) and `probe` (`ffprobe` reading). Used by `write_bundle` to check each stream file it encodes.
- `tests/synthetic.py`: synthetic sensors and pattern frames with known ground truth, for end-to-end tests.

Encoding and validation use the `ffmpeg`/`ffprobe` on `PATH`. Keep things simple: use built-in exceptions (`ValueError`, `RuntimeError`) unless a caller needs to catch a specific case, and avoid defensive layers that nothing uses yet.

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

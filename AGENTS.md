# AGENTS.md: AI Agent Guide for Multireel

Multireel is a multimodal rendering framework for video editing. The project is at an early stage; update this guide as the architecture takes shape.

## Layout

- `src/multireel/`: the library (src layout, built with hatchling; ships `py.typed`).
- `tests/`: pytest tests (`test_*.py`).
- `docs/`: [zensical](https://zensical.org) site (configured via `mkdocs.yml`); the API reference is generated from docstrings via mkdocstrings (`docs/api/`, one page per top-level module). Add new pages to `nav` in `mkdocs.yml`.
- `docs/spec.md`: the bundle format spec (video format, time model, manifest). Code that reads or writes bundles must follow it.

## Architecture

- `bundle/`: the bundle format as code (`video`, `timing`, `manifest`, mirroring the spec's sections). Pure data types and functions; implementation defaults do not belong here.
- `stream.py`: `Stream` (source timestamps plus a function that renders the shown samples) and its constructors: `random_access` (load by index), `sequential` (read in order) and `from_sensor` (an [abstract dataloader](https://radarml.github.io/abstract-dataloader/) `Sensor`, via either). Each renders in a thread pool started inside `frames()`.
- `write/`: the writing pipeline; like `bundle/`, `__init__` only re-exports every public name of its modules under one flat namespace.
  - `writer.py`: `BundleWriter`, which holds the state of one bundle. Its constructor plans the bundle (frame range, manifest, sample per frame) and raises every `ValueError` before anything is written; `write()` forks one child per stream, which renders, encodes and validates it, and polls shared-memory progress counters from the calling thread.
  - `encode.py`: `EncodeOptions` (including `codec_args`, the `ffmpeg` arguments for a profile) and `Encoder`, a context manager around one `ffmpeg` process per stream file.
  - `validate.py`: reads a stream file's properties with `ffprobe` (`StreamProbe`), checks them against the spec (`check_stream`, which returns `"<rule>: <message>"` strings), and decodes every frame.
- `render.py`: numpy helpers that turn arrays into RGB frames (`colorize`, `colorize_labels`, `resize_nearest`); colormaps by name need the `matplotlib` extra.
- `progress.py`: `RichProgress`, rich progress bars for `BundleWriter.write`; needs the `rich` extra.
- `tests/synthetic.py`: synthetic sensors and pattern frames with known ground truth, for end-to-end tests.

The library is Linux-only, since streams are written in forked processes. Encoding and validation use the `ffmpeg`/`ffprobe` on `PATH`. Keep things simple: use built-in exceptions (`ValueError`, `RuntimeError`) unless a caller needs to catch a specific case, and avoid defensive layers that nothing uses yet.

## Conventions

- **Typing**: all code is fully typed and must pass `pyright`. Use `jaxtyping` for array shapes when arrays are involved.
- **Imports**: import from other packages with absolute imports of what they re-export (`from multireel.bundle import Manifest`), not from their modules. Inside a package, import sibling modules with relative imports (`from .timing import Timecode`), to avoid circular imports through the package's `__init__`.
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

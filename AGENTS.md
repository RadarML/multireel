# AGENTS.md: AI Agent Guide for Multireel

Multireel is a multimodal rendering framework for video editing. The project is at an early stage; update this guide as the architecture takes shape.

## Layout

- `src/multireel/`: the library (src layout, built with hatchling; ships `py.typed`).
- `tests/`: pytest tests (`test_*.py`).
- `docs/`: [zensical](https://zensical.org) site (configured via `mkdocs.yml`); the API reference is generated from docstrings via mkdocstrings (`docs/api/`, one page for the top-level API and one per public submodule). Add new pages to `nav` in `mkdocs.yml`.
- `docs/spec.md`: the bundle format spec (video format, time model, manifest). Code that reads or writes bundles must follow it.

## Architecture

Every public name has exactly one import path:

- `multireel`: the core API, defined in private modules and re-exported in `__init__`'s `__all__`.
  - Bundle format (mirroring the spec's sections; pure data types and functions, no implementation defaults):
    - `_profile.py`: `Profile`, `ProfileName`.
    - `_timecode.py`: `Timecode`.
    - `_manifest.py`: `Manifest`, `StreamEntry`.
  - Writing:
    - `_stream.py`: `Stream` (source timestamps plus a function that renders the shown samples) and its constructors: `random_access` (load by index), `sequential` (read in order) and `from_sensor` (an [abstract dataloader](https://radarml.github.io/abstract-dataloader/) `Sensor`, via either). Each renders in a thread pool started inside `frames()`.
    - `_writer.py`: `BundleWriter`, which holds the state of one bundle, and its `FRAME_CLOCK` key for the synchronization. Its constructor plans the bundle (frame range, manifest, sample per frame) and raises every `ValueError` before anything is written; `write()` forks one child per stream, which renders, encodes and validates it, and polls shared-memory progress counters from the calling thread.
    - `_encode.py`: `EncodeOptions` (including the private `_codec_args`, the `ffmpeg` arguments for a profile) and the private `Encoder`, a context manager around one `ffmpeg` process per stream file.
  - Checking:
    - `_validate.py`: `validate_stream`, which reads a stream file's properties with `ffprobe` (`probe`, `StreamProbe`), checks them against the spec (`check_stream`, which returns `"<rule>: <message>"` strings), and decodes every frame (`decode_errors`). `BundleWriter` runs it on every stream it writes.
- Public submodules for separate tools, imported explicitly:
  - `render.py`: numpy helpers that turn arrays into RGB frames (`colorize`, `colorize_labels`, `resize_nearest`); colormaps by name need the `matplotlib` extra.
  - `progress.py`: `RichProgress`, rich progress bars for `BundleWriter.write`; needs the `rich` extra.
- `tests/synthetic.py`: synthetic sensors and pattern frames with known ground truth, for end-to-end tests.

New core names go in a private module and `__all__`; a new tool with its own concern or optional dependency gets a public submodule.

The library is Linux-only, since streams are written in forked processes. Encoding and validation use the `ffmpeg`/`ffprobe` on `PATH`. Keep things simple: use built-in exceptions (`ValueError`, `RuntimeError`) unless a caller needs to catch a specific case, and avoid defensive layers that nothing uses yet.

## Conventions

- **Typing**: all code is fully typed and must pass `pyright`. Use `jaxtyping` for array shapes when arrays are involved.
- **Imports**: outside the library, import from `multireel` or its public submodules (`from multireel import Manifest`), never from private modules (tests of private helpers excepted). Inside the library, import from sibling modules with relative imports (`from ._timecode import Timecode`), never from `multireel` itself, to avoid circular imports through `__init__`.
- **Linting**: `ruff` with an 80-character line limit and pydocstyle rules (see `pyproject.toml`).
- **Docstrings**: Google style, since they are rendered by mkdocstrings. Every public module, class, and function needs one. Write them in plain, direct language, and only include what callers need; implementation details go in code comments. Only document attributes that callers use; undocumented attributes are private, without needing a leading underscore. Keep `Args` and `Attributes` entries short, and explain anything longer (such as modes or how arguments interact) in the main docstring.
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

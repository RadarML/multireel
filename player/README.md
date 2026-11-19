# Multireel Player

A small web server for watching bundles in a browser: it lists every bundle
under a folder, and plays the streams of the chosen bundle side by side, in
sync.

```sh
uv run --extra player python player/server.py <folder>
```

Then open `http://localhost:8000`. See the
[documentation](https://radarml.github.io/multireel/player/) (or
`docs/player.md`) for options, controls and browser support.

# Bundle Player

The bundle player is a small web server for watching [bundles](spec.md) in a
browser. It lists every bundle under a folder, and plays the streams of the
chosen bundle side by side and in sync, with one shared play/scrub bar.

## Running the Player

The player is in the `player/` directory of the multireel repository; it is
not part of the installed `multireel` package. From a checkout of the
repository, run:

```sh
uv run --extra player python player/server.py <folder>
```

Then open `http://localhost:8000` in a browser.

!!! info "Remote machines"

    The player only listens on `localhost` by default. To watch bundles on a
    remote machine, forward the port over SSH and open
    `http://localhost:8000` on your own computer:

    ```sh
    ssh -L 8000:localhost:8000 <host>
    ```

| Option | Default | Meaning |
| --- | --- | --- |
| `<folder>` | | Folder to look for bundles in, including subfolders. |
| `--host` | `127.0.0.1` | Address to listen on. |
| `--port` | `8000` | Port to listen on. |
| `--chunk-mb` | `1` | Largest piece of video sent per request, in MiB; `0` for no limit. See [Troubleshooting](#troubleshooting). |

## Choosing a Bundle

The sidebar lists every bundle under `<folder>`. A bundle that is still being
written has no `manifest.yaml` yet, so it doesn't appear until it is finished.
Reload the page to see new bundles.

The page URL records the open bundle and frame, as `#<bundle>?f=<frame>`, so
reloading the page or sharing its URL returns to the same place.

## Playback

| Control | Action |
| --- | --- |
| Space, or click a video | Play / pause |
| Left / Right | Step one frame |
| Shift + Left / Right | Step one second |
| Home / End | Jump to the first / last frame |
| Slider | Scrub |
| Rate menu | Play at 0.25x to 2x speed |

While you drag the slider, the player shows the nearest keyframe, which is
quick to decode; when you let go, it jumps to the exact frame.

The bar below the slider shows the current [timecode](spec.md#timecode), the
frame number within the bundle, and the frame's absolute time in UTC.

## Arranging Streams

- **Show or hide** a stream with its checkbox in the sidebar.
- **Overlays** are drawn over their base stream, and have their own checkbox.
- **Reorder** streams by dragging a tile by its title bar, or by dragging its
  entry in the sidebar. The order is saved in your browser, separately for
  each bundle.

Tiles are sized to fill the window, keeping each stream's aspect ratio, and
rearrange when you resize the window or show or hide a stream.

## Browser Support

Stream files are played as they are, without conversion, so a stream only
plays if the browser can decode its [profile](spec.md#encoding-profiles):

| Profile | Safari | Chrome / Edge | Firefox |
| --- | --- | --- | --- |
| HEVC (`hevc-*`) | Yes | Only with hardware HEVC decoding, typically on macOS and Windows | No |
| ProRes (`prores-*`) | macOS only | No | No |

A stream that the browser can't decode shows a message in its tile, and the
rest of the bundle still plays.

## Troubleshooting

**A tile says "reloading after ..."**: the stream hit a network or decoding
error, and the player is reloading it. Playback waits for it, then continues.
If a stream keeps failing, its tile shows the error and a **Retry** button.

**Slow seeking**: over a slow connection, such as an SSH tunnel, try a smaller
`--chunk-mb`, such as `0.5`. If playback stutters instead, try a larger one,
such as `2` to `4`.

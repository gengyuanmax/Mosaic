# MosaicGym Toolbox playground

The Method playground calls the project's Python `mosaic_gym.toolbox.Toolbox`
directly. The browser sends a tool name, arguments and workspace image IDs; the
service passes them to `Toolbox.run([ToolCall(...)])`. The ten tool algorithms,
argument validation, image registration and RGBA semantics are the existing
MosaicGym implementation. There is no JavaScript implementation of the tools.

The frontend uses the Toolbox registry's schemas to build controls.
`gym-controls.mjs` contains only presentation labels, initial control values and
slider ranges. Numeric fields retain the Toolbox's full declared ranges.
`alpha-canvas.js` handles UI state, requests and image display.

## Run locally

From this repository, install `demo/requirements.txt` and run:

```sh
python demo/server.py --port 8767
```

Open `http://127.0.0.1:8767/#alpha-canvas`. The server imports this checkout’s package at `src`; use `--toolbox-src`
to select a different explicitly chosen source checkout. It fails if another
MosaicGym package has already been imported. Pillow, NumPy and OpenCV must be
installed in the Python environment. Startup records the actual source directory
and a digest of the Toolbox source files. The package itself is not modified.

This server serves the preview and `/api/gym/` on the same localhost origin. It
accepts only the provided example IDs and image IDs belonging to a session; it
has no arbitrary-file or upload endpoint. GitHub Pages hosts the static page and
trajectory assets. Online tool execution requires a separately deployed Python
service configured with `data-toolbox-api`. The project page uses
[the MosaicGym CPU Space](https://huggingface.co/spaces/gengyuanmax/MosaicGym);
localhost uses its own service. A
sleeping CPU Space may need time to start.
See `demo/README.md` in the repository for deployment and recovery instructions.
Recorded-trajectory replay remains independent of the playground service.

## Workspace and alpha

Each browser workspace holds a real Toolbox instance on the server. Example
files are loaded as RGBA at their stored dimensions, without another working-size
reduction. Some supplied examples are existing presentation previews; their
original dataset files are not substituted or claimed to be bit-identical.

A tool preview branches the current Toolbox state and calls the real executor.
**Add to workspace** adopts that exact result and image store. Subsequent calls
use the stored PIL images, not screenshots, thumbnails or browser-decoded copies.
Undo removes the last generated image; Reset creates a fresh Toolbox from the
selected example. Stale previews and workspace revisions cannot be committed.

Results are transported as lossless RGBA PNG. Alpha view reads the result's
actual A channel. Workspace thumbnails are display copies only. Checkerboards
are CSS backgrounds and never become image pixels or later tool inputs.
Toolbox error messages are returned unchanged. Service-level limits produce
explicit errors instead of silently resizing or changing tool parameters.

The local service allows up to 16,777,216 output pixels, 24 workspace images,
96 MiB of committed RGBA pixels, and 8 active workspaces. A workspace expires
following 30 minutes of inactivity. Parameter requests are debounced and
serialized; obsolete responses cannot replace a newer selection. The UI loads
the service only when the playground approaches the viewport.

## Examples and validation

`assets/gym-examples.json` lists four sets and their source-specific credits.
The initial MVTec-derived pair is the existing 384-pixel transparent-canvas pair.
Other examples reuse the document, rotated-text and template-rotation inputs
from the trajectory library. Input files remain unchanged.

Run `python demo/test_server.py` for the bridge
checks. They compare all ten tools on source and transparent inputs against
direct Toolbox calls in the same Python environment, including every RGBA and
alpha pixel, image IDs and feedback. They also check committed tool chains,
unchanged source images and errors, stale-preview rejection, undo, full-size
input loading and resource-limit rejection. This tests the service boundary;
it does not claim cross-version Pillow/OpenCV reproducibility.

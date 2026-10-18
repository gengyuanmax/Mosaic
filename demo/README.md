# MosaicGym web service

`server.py` hosts the website and a session API that calls this checkout's
`mosaic_gym.toolbox.Toolbox.run` directly. It does not implement image operations.
Model inference and GPU hardware are not needed.

```sh
python -m venv .venv-demo
.venv-demo/bin/pip install -r demo/requirements.txt
.venv-demo/bin/python demo/server.py --port 8767
```

Open `http://127.0.0.1:8767/#alpha-canvas`. The local page uses the local API;
the public page connects to the [MosaicGym CPU Space](https://huggingface.co/spaces/gengyuanmax/MosaicGym).
Its API endpoint is configured through the page’s `data-toolbox-api` attribute.
The website contains 796 recorded trajectories loaded individually; these
remain available independently of the Python service.

## Deploy the CPU Space

Generate the minimal Docker build context from this checkout:

```sh
python demo/build_space.py /tmp/mosaicgym-space
hf upload gengyuanmax/MosaicGym /tmp/mosaicgym-space --type space
```

The destination must be empty. The explicit package contains only the unchanged
Toolbox Python sources, service, playground frontend, six example images, font
files and their licenses. It does not include model weights or trajectory logs.
The generated `SHA256SUMS.json` records every copied file. The Space targets
`cpu-basic` hardware, Docker port 7860 and Python 3.12. Hugging Face may require
a PRO subscription to host Docker Spaces; confirm account eligibility before
creating one. Package
versions are pinned in `requirements.txt`. `demo/Dockerfile` also works from
the full repository root with `docker build -f demo/Dockerfile .`.

GitHub Pages is allowed as an API origin. No credentials or tokens are placed in
the website or container. Sessions use unguessable IDs, expire after inactivity,
and contain only the selected public example images and generated results. The
service has no upload, arbitrary-file, shell or model-inference endpoint. Request
threads, JSON size, output size and workspace memory are bounded; tool validation
and error messages remain the real Toolbox's responsibility.

CPU Spaces may sleep. The UI reports service errors and provides Retry during
startup and Reset for workspace recovery. If deployment fails, inspect the Space build/runtime logs
and `GET /api/gym/config`, which returns the Toolbox source digest. Do not replace
the Python implementation with browser-side approximations. The page's
`data-toolbox-api` attribute identifies the public endpoint; localhost always
uses `/api/gym` on its own origin.

## Validation

`python demo/test_server.py` compares all ten tools on source and RGBA inputs
with direct calls, including alpha pixels and chained operations. Use the same
input files and pinned dependency versions for comparisons across hosts.
Frontend replay and navigation checks are independent of the service.

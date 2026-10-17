#!/usr/bin/env bash
# Thin wrapper to run a command inside the verl v0.7.1 Apptainer container.
#
#   rl/scripts/verl.sh python -c 'import torch; print(torch.__version__)'
#   rl/scripts/verl.sh bash            # interactive shell in the container
#
# Build the container image first with rl/scripts/build_container.sh, or point
# SIF at your own image. On clusters whose compute nodes mount only selected
# filesystems, keep the repo, the .sif, the HF cache and the datasets on a
# mounted path and bind it via WS.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WS="${WS:-$PROJECT_ROOT}"                                # extra bind root (data / caches)
SIF="${SIF:-$PROJECT_ROOT/verl-v0.7.1.sif}"              # training container image

# Put the vendored verl clone AND our src/ (mosaic_gym) on PYTHONPATH so both
# the editable verl and our tool/reward/agent-loop adapters import without
# reinstall.
export APPTAINERENV_PYTHONPATH="${PROJECT_ROOT}/rl/verl:${PROJECT_ROOT}/src:${PROJECT_ROOT}/rl:${PYTHONPATH:-}"

exec apptainer exec --nv \
  --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --bind "${WS}:${WS}" \
  "$SIF" "$@"

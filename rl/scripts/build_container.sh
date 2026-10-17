#!/usr/bin/env bash
# Build the verl v0.7.1 training container as an Apptainer .sif.
#
# Two engine images (cut alongside verl v0.7.1); pick with the 1st arg:
#   sglang (DEFAULT, what training uses): verlai/verl:sgl059.latest  — SGLang 0.5.9
#   vllm:                                 verlai/verl:vllm017.latest — vLLM 0.17.0
# Both ship torch 2.10+cu129, transformers 4.57.6, flash_attn 2.8.3, ray 2.54.0.
#
# Cluster tips:
#   * If your $HOME is inode-limited, point APPTAINER_TMPDIR/CACHEDIR (via WS)
#     at a scratch filesystem with plenty of inodes and store the .sif there.
#   * A docker:// pull-and-convert usually needs no fakeroot on login nodes.
#
# Usage:
#   bash rl/scripts/build_container.sh          # sglang (default)
#   bash rl/scripts/build_container.sh vllm     # vllm
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WS="${WS:-$PROJECT_ROOT}"

# Engine: sglang (default, used for training) or vllm.
ENGINE="${1:-sglang}"
case "$ENGINE" in
  sglang) _IMG="docker://verlai/verl:sgl059.latest";  _SIF="verl-v0.7.1-sgl059.sif" ;;
  vllm)   _IMG="docker://verlai/verl:vllm017.latest"; _SIF="verl-v0.7.1-vllm017.sif" ;;
  *) echo "usage: $0 [sglang|vllm]" >&2; exit 1 ;;
esac
IMAGE="${IMAGE:-$_IMG}"
# .sif lives inside the repo tree (gitignored) so all RL infra sits together.
SIF="${SIF:-$PROJECT_ROOT/rl/containers/$_SIF}"

export APPTAINER_CACHEDIR="$WS/tmp/apptainer/cache"
export APPTAINER_TMPDIR="$WS/tmp/apptainer/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$SIF")"

echo "[build] image=$IMAGE"
echo "[build] sif=$SIF"
apptainer build "$SIF" "$IMAGE"
ls -lah "$SIF"

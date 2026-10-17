#!/usr/bin/env bash
# Run from the repo root:  ./scripts/run.sh [--dataset blink --env vanilla ...]
exec python -m mosaic_gym.runner.inference --config configs/default.yaml "$@"

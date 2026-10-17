"""Runner config — HOW this run executes and WHERE output goes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunnerConfig:
    env: str = "tool"                  # "tool" = full agent loop with tools; "vanilla" = single call, no tools exposed
    max_steps: int = 10                # tool-call turns allowed per episode (loop owns this)
    out_dir: str = "outputs/mmiu"      # run dir: record/<task>.jsonl (+ messages/ with save_messages)
    save_messages: bool = False        # also write the DeepEyes-style SFT conversation + images
    resume: bool = True                # skip sample_ids already in record/<task>.jsonl and append
    price_in_per_m: float = 0.15       # USD / 1M input tokens (cost estimate; default gpt-4o-mini)
    price_out_per_m: float = 0.60      # USD / 1M output tokens

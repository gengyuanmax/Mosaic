"""Data config — WHAT to run on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    dataset_dir: str = "data/mmiu/datasets"          # dir holding the canonical <task>.json files
    tasks: List[str] = field(default_factory=list)   # subset names to run ([] -> nothing)
    base_dir: str = "data/mmiu"                       # prefix for relative image paths in a sample
    max_samples: Optional[int] = None                # per task; None -> all

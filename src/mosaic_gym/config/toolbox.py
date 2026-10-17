"""Toolbox config — how the tools are presented to the model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ToolboxConfig:
    # Which description tier the model sees in the <tools> block:
    #   baseline   -> 1 sentence per tool (what it does); the model figures out usage
    #   prompting  -> baseline + a short "Use when … ; avoid when …" steer
    description_mode: str = "prompting"
    # Which tools the model can call — for tool-ablation experiments.
    #   None / omitted / [] -> the full default toolbox (all 10 tools)
    #   a list of names     -> only those tools, e.g. ["crop", "make_collage"]
    # Valid names: crop, rotate, resize, flip, apply_affine_transformation, draw_normalized_coordinate_grid,
    #   compare_per_pixel_image_difference, apply_homography_transformation,
    #   make_collage, overlay_images.
    # An unknown name raises at startup so a typo fails loudly.
    tools: Optional[List[str]] = field(default=None)

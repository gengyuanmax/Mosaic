"""Tool implementations — one tool per file.

Each file defines exactly one ``ToolBase`` subclass plus any helpers that
tool alone needs. Shared helpers live in ``mosaic_gym.toolbox.utils``.
This 1:1 layout maps cleanly onto verl's per-``BaseTool`` files for the
eventual RL migration.
"""

from mosaic_gym.toolbox.tools.apply_affine_transformation import (
    ApplyAffineTransformationTool,
)
from mosaic_gym.toolbox.tools.apply_homography_transformation import (
    ApplyHomographyTransformationTool,
)
from mosaic_gym.toolbox.tools.compare_per_pixel_image_difference import (
    ComparePerPixelImageDifferenceTool,
)
from mosaic_gym.toolbox.tools.crop import CropTool
from mosaic_gym.toolbox.tools.draw_normalized_coordinate_grid import (
    DrawNormalizedCoordinateGridTool,
)
from mosaic_gym.toolbox.tools.flip import FlipTool
from mosaic_gym.toolbox.tools.make_collage import MakeCollageTool
from mosaic_gym.toolbox.tools.overlay_images import OverlayImagesTool
from mosaic_gym.toolbox.tools.resize import ResizeTool
from mosaic_gym.toolbox.tools.rotate import RotateTool

# Single-image transforms.
ATOMIC_TOOL_CLASSES = [
    CropTool,
    RotateTool,
    ResizeTool,
    FlipTool,
    ApplyAffineTransformationTool,
    DrawNormalizedCoordinateGridTool,
]
# Multi-image composition / comparison / judgment.
COMPOSITION_TOOL_CLASSES = [
    ComparePerPixelImageDifferenceTool,
    ApplyHomographyTransformationTool,
    MakeCollageTool,
    OverlayImagesTool,
]
ALL_TOOL_CLASSES = ATOMIC_TOOL_CLASSES + COMPOSITION_TOOL_CLASSES

# name -> class, the single lookup for "give me the tool called X".
TOOL_CLASSES_BY_NAME = {cls.name: cls for cls in ALL_TOOL_CLASSES}


def build_atomic_tools():
    return [cls() for cls in ATOMIC_TOOL_CLASSES]


def build_composition_tools():
    return [cls() for cls in COMPOSITION_TOOL_CLASSES]


def build_default_tools():
    """Instantiate every tool in the default toolbox."""
    return [cls() for cls in ALL_TOOL_CLASSES]


def build_tools(names=None):
    """Instantiate the named tools (default: all). Used for tool ablations.

    ``names`` falsy (None / empty) -> the full default toolbox. Otherwise a list
    of tool names (e.g. ``["crop", "make_collage"]``); order is preserved and
    duplicates dropped. Raises ``ValueError`` on any unknown name so a typo in a
    config fails loudly instead of silently dropping a tool.
    """
    if not names:
        return build_default_tools()
    unknown = [n for n in names if n not in TOOL_CLASSES_BY_NAME]
    if unknown:
        raise ValueError(
            "Unknown tool name(s): %s. Available: %s"
            % (", ".join(unknown), ", ".join(sorted(TOOL_CLASSES_BY_NAME)))
        )
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(TOOL_CLASSES_BY_NAME[n]())
    return out


__all__ = [
    "ApplyAffineTransformationTool",
    "ApplyHomographyTransformationTool",
    "ComparePerPixelImageDifferenceTool",
    "CropTool",
    "DrawNormalizedCoordinateGridTool",
    "FlipTool",
    "MakeCollageTool",
    "OverlayImagesTool",
    "ResizeTool",
    "RotateTool",
    "ATOMIC_TOOL_CLASSES",
    "COMPOSITION_TOOL_CLASSES",
    "ALL_TOOL_CLASSES",
    "TOOL_CLASSES_BY_NAME",
    "build_atomic_tools",
    "build_composition_tools",
    "build_default_tools",
    "build_tools",
]

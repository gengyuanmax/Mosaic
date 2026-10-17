"""Toolbox package.

The stateful per-trajectory tool executor (``Toolbox``) plus the image store
(``ImageState``), the registry builder, and the tool classes. Tools live
one-per-file under ``toolbox/tools/``.
"""

from mosaic_gym.toolbox.base import ToolRegistry
from mosaic_gym.toolbox.core import Toolbox
from mosaic_gym.toolbox.registry import build_registry
from mosaic_gym.toolbox.state import ImageState, coerce_image
from mosaic_gym.toolbox.tools import (
    ALL_TOOL_CLASSES,
    ApplyAffineTransformationTool,
    ApplyHomographyTransformationTool,
    ComparePerPixelImageDifferenceTool,
    CropTool,
    DrawNormalizedCoordinateGridTool,
    FlipTool,
    MakeCollageTool,
    OverlayImagesTool,
    ResizeTool,
    RotateTool,
    build_atomic_tools,
    build_composition_tools,
    build_default_tools,
)

__all__ = [
    "Toolbox",
    "ImageState",
    "coerce_image",
    "ToolRegistry",
    "build_registry",
    "build_default_tools",
    "build_atomic_tools",
    "build_composition_tools",
    "ALL_TOOL_CLASSES",
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
]

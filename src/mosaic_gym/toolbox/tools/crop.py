"""Crop tool — extract a rectangular region from one image.

Coordinates are a NORMALIZED bounding box (corners in [0, 1]), following
Anthropic's multimodal crop cookbook. This makes the tool dimension-agnostic:
the model never needs to know the image's pixel size (0.5 is always the center),
and an out-of-bounds / empty box is impossible by construction.
"""

from __future__ import annotations

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import build_tool_feedback, format_size, normalize_to_rgba


def _frac(value, name):
    """Coerce one normalized coordinate to a float clamped to [0, 1]."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number between 0 and 1" % name)
    return min(1.0, max(0.0, f))


class CropTool(ToolBase):
    name = "crop"
    description_baseline = (
        "Crops a rectangular region from one image. It accepts an input "
        "target image and the crop box corner coordinates x1, y1, x2, y2 "
        "as inputs.\n"
        "\n"
        "It returns: a new image containing only the cropped region."
    )
    description_prompting = 'Crop a rectangular region from one image, given as a normalized bounding box: corners x1,y1 (top-left) and x2,y2 (bottom-right), each in 0-1 (0.5 = image center). Use it to zoom into a region you have already visually located; avoid it when the region is not yet localized and the box would be guesswork.'

    @classmethod
    def schema(cls):
        return {
            "required": ["target_image", "x1", "y1", "x2", "y2"],
            "properties": {
                "x1": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "Left edge of the crop box.",
                },
                "y1": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "Top edge of the crop box.",
                },
                "x2": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "Right edge of the crop box. Must be greater than x1.",
                },
                "y2": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": "Bottom edge of the crop box. Must be greater than y1.",
                },
                "target_image": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the image to crop, "
                        "such as img1."
                    ),
                },
            },
        }

    def execute(self, state, arguments, target_images):
        src_id = target_images[0]
        src = normalize_to_rgba(state.get_image(src_id))
        w, h = src.size
        x1 = _frac(arguments["x1"], "x1")
        y1 = _frac(arguments["y1"], "y1")
        x2 = _frac(arguments["x2"], "x2")
        y2 = _frac(arguments["y2"], "y2")
        left, top, right, bottom = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        if right <= left or bottom <= top:
            raise ValueError("crop box is empty (need x2 > x1 and y2 > y1)")
        out = src.crop((left, top, right, bottom))
        return ToolOutput(
            produced_images=[out],
            tool_feedback_text=build_tool_feedback(
                "crop",
                size=format_size(out),
                box=(left, top, right, bottom),
            ),
            error_code=None,
            error_message=None,
        )

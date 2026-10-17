"""Flip tool — mirror one image horizontally or vertically."""

from __future__ import annotations

from PIL import ImageOps

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import (
    build_tool_feedback,
    format_size,
    normalize_to_rgba,
)


class FlipTool(ToolBase):
    name = "flip"
    description_baseline = (
        "Flips one image horizontally or vertically. It accepts an input "
        "target image and the mode as inputs.\n"
        "\n"
        "It returns: a new image flipped by the given mode."
    )
    description_prompting = 'Flip one image horizontally or vertically. Use to mirror an image for easier comparison; avoid when you need an arbitrary rotation rather than a mirror.'

    @classmethod
    def schema(cls):
        return {
            "required": ["target_image", "mode"],
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["horizontal", "vertical"],
                    "description": (
                        "Flip axis. 'horizontal' flips left-to-right; "
                        "'vertical' flips top-to-bottom."
                    ),
                },
                "target_image": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the image to flip, "
                        "such as img1."
                    ),
                },
            },
        }

    def execute(self, state, arguments, target_images):
        src_id = target_images[0]
        src = normalize_to_rgba(state.get_image(src_id))
        mode = arguments["mode"]
        if mode == "horizontal":
            out = ImageOps.mirror(src)
        elif mode == "vertical":
            out = ImageOps.flip(src)
        else:
            raise ValueError("flip mode must be 'horizontal' or 'vertical', got %r" % mode)
        return ToolOutput(
            produced_images=[out],
            tool_feedback_text=build_tool_feedback(
                "flip",
                mode=mode,
                size=format_size(out),
            ),
            error_code=None,
            error_message=None,
        )

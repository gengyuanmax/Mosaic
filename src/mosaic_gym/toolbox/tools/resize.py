"""Resize tool — resize one image to an exact target size."""

from __future__ import annotations

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import (
    build_tool_feedback,
    coerce_positive_int,
    format_size,
    get_resample_filter,
    normalize_to_rgba,
)


class ResizeTool(ToolBase):
    name = "resize"
    description_baseline = (
        "Resizes one image to an exact width and height in pixels. It "
        "accepts an input target image and the width and height as "
        "inputs.\n"
        "\n"
        "It returns: a new image at exactly width x height."
    )
    description_prompting = 'Resize one image to an exact width and height in pixels. Use for a size-constrained or normalized view; avoid when you want a zoomed crop rather than a global resize (use crop).'

    @classmethod
    def schema(cls):
        return {
            "required": ["target_image", "width", "height"],
            "properties": {
                "width": {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "maximum": 4096,
                    "description": (
                        "Output width in pixels. Resize scales the whole image; do not treat width "
                        "as a crop-box span or use an arbitrarily tiny size."
                    ),
                },
                "height": {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "maximum": 4096,
                    "description": (
                        "Output height in pixels. Together with width it sets the exact output size; "
                        "the target_image is stretched or squashed if the aspect ratio changes."
                    ),
                },
                # `resample` removed from the model-facing
                # schema; the tool always uses bilinear (the previous
                # default).
                # `preserve_aspect_ratio` removed from the
                # model-facing schema; the tool always produces an
                # exact width x height canvas (the previous default).
                "target_image": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the image to resize, "
                        "such as img1."
                    ),
                },
            },
        }

    def execute(self, state, arguments, target_images):
        src_id = target_images[0]
        src = normalize_to_rgba(state.get_image(src_id))
        target_width = coerce_positive_int(arguments["width"], "width")
        target_height = coerce_positive_int(arguments["height"], "height")
        # hardcoded bilinear; uses the fixed default.
        resample_filter = get_resample_filter("bilinear")

        # always exact width x height; uses the fixed default.
        output_size = (target_width, target_height)
        out = src.resize(output_size, resample=resample_filter)
        return ToolOutput(
            produced_images=[out],
            tool_feedback_text=build_tool_feedback(
                "resize",
                requested="%dx%d" % (target_width, target_height),
                size=format_size(out),
            ),
            error_code=None,
            error_message=None,
        )

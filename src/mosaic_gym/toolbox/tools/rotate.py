"""Rotate tool — rotate one image by an arbitrary angle."""

from __future__ import annotations

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import (
    build_tool_feedback,
    format_size,
    get_resample_filter,
    normalize_to_rgba,
)


class RotateTool(ToolBase):
    name = "rotate"
    description_baseline = (
        "Rotates one image by an arbitrary angle in degrees. It accepts "
        "an input target image and the angle as inputs.\n"
        "\n"
        "It returns: a new image rotated by the angle."
    )
    description_prompting = 'Rotate one image by an arbitrary angle in degrees. Use when orientation is the main uncertainty or to test a candidate angle; avoid when a simple flip is enough.'

    @classmethod
    def schema(cls):
        return {
            "required": ["target_image", "angle"],
            "properties": {
                "angle": {
                    "type": "number",
                    "minimum": -360,
                    "maximum": 360,
                    "description": (
                        "Rotation angle in degrees for the target_image. Allowed range is "
                        "[-360, 360]. Use rotate mainly to test a "
                        "concrete orientation hypothesis or a candidate angle from the task, rather than "
                        "blindly trying arbitrary large angles."
                    ),
                },
                # `resample` removed from the model-facing
                # schema; the tool always uses bilinear (the previous
                # default).
                # `expand` removed; the tool always grows the
                # output canvas to keep all rotated pixels visible
                # (formerly the default ``expand=True``). If the model
                # needs a same-canvas-size rotation, compose ``rotate`` +
                # ``crop`` to centre-crop back to the source dimensions.
                "target_image": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the image to rotate, "
                        "such as img1."
                    ),
                },
            },
        }

    def execute(self, state, arguments, target_images):
        src_id = target_images[0]
        # RGBA so the corners exposed by the expanded canvas stay transparent
        # (alpha=0) instead of a baked-in black fill.
        src = normalize_to_rgba(state.get_image(src_id))
        angle = float(arguments["angle"])
        # hardcoded bilinear; uses the fixed default.
        # hardcoded ``expand=True``; uses the fixed
        # default and keeps all rotated pixels visible.
        out = src.rotate(
            angle,
            expand=True,
            resample=get_resample_filter("bilinear"),
        )
        return ToolOutput(
            produced_images=[out],
            tool_feedback_text=build_tool_feedback(
                "rotate",
                angle=angle,
                size=format_size(out),
            ),
            error_code=None,
            error_message=None,
        )

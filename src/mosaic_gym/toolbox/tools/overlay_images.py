"""Overlay-images tool — composite one image onto another at a normalized position."""

from __future__ import annotations

from PIL import Image

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


class OverlayImagesTool(ToolBase):
    name = "overlay_images"
    description_baseline = (
        "Overlays the foreground image onto the background image at a "
        "normalized position. It accepts the background and foreground "
        "images and the x and y coordinates in the background image as "
        "inputs.\n"
        "\n"
        "It returns: a new image with the foreground image composited "
        "onto the background image."
    )
    description_prompting = 'Paste the foreground image onto the background at normalized position (x, y) and return the merged image. Use to composite or superimpose one image onto another at a known position; avoid when you only want to view two images side by side (use make_collage).'
    min_target_images = 2
    max_target_images = 2

    @classmethod
    def schema(cls):
        return {
            "required": ["background", "foreground", "x", "y"],
            "properties": {
                "background": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the base image, "
                        "such as img1."
                    ),
                },
                "foreground": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the image placed "
                        "onto the background at (x, y), such as img2."
                    ),
                },
                "x": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": (
                        "X coordinate in the background image where the "
                        "foreground image's left edge is placed."
                    ),
                },
                "y": {
                    "type": "number", "minimum": 0, "maximum": 1,
                    "description": (
                        "Y coordinate in the background image where the "
                        "foreground image's top edge is placed."
                    ),
                },
            },
        }

    def execute(self, state, arguments, target_images):
        targets = list(target_images or [])
        if len(targets) != 2:
            raise ValueError("overlay_images requires exactly two images")
        args = arguments or {}
        base = normalize_to_rgba(state.get_image(targets[0]))
        fill = normalize_to_rgba(state.get_image(targets[1]))
        w, h = base.size
        x = int(_frac(args["x"], "x") * w)
        y = int(_frac(args["y"], "y") * h)
        # the fill keeps its native size (no scaling). The canvas
        # is the union of the base frame and the fill's rectangle: x/y are
        # normalized to [0,1] so the fill never starts left of / above the
        # base, and the canvas only grows right/down to keep any overflow.
        # alpha_composite respects transparency in both inputs — the fill's
        # transparent areas let the base show through, and areas neither
        # image covers stay transparent.
        out_w = max(w, x + fill.width)
        out_h = max(h, y + fill.height)
        out = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        out.alpha_composite(base)
        out.alpha_composite(fill, dest=(x, y))

        return ToolOutput(
            produced_images=[out],
            tool_feedback_text=build_tool_feedback("overlay_images", pos=(x, y), size=format_size(out)),
        )

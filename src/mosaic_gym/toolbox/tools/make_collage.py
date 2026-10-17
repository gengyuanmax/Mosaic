"""Make-collage tool — arrange multiple images in an M×N grid canvas."""

from __future__ import annotations

from PIL import Image

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import build_tool_feedback, format_size, normalize_to_rgba


def _ensure_targets(target_images, tool_name: str, minimum: int):
    targets = list(target_images or [])
    if len(targets) < minimum:
        raise ValueError("%s requires at least %d target images" % (tool_name, minimum))
    return targets


class MakeCollageTool(ToolBase):
    name = "make_collage"
    description_baseline = (
        "Tiles N input images into a single output image arranged as a "
        "`rows` x `cols` grid, filled in row-major order. It accepts the "
        "input target images and the rows and cols as inputs.\n"
        "\n"
        "It returns: a new image with the input images arranged in a "
        "rows x cols grid."
    )
    description_prompting = 'Arrange several images into a single image as a rows x cols grid. Use to view multiple images side by side for comparison; avoid when a single image is enough.'
    min_target_images = 2
    max_target_images = None
    # feedback_form intentionally left at the ToolBase default (image_only).

    @classmethod
    def schema(cls):
        return {
            "required": ["target_images", "rows", "cols"],
            "properties": {
                "rows": {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "maximum": 8,
                    "description": (
                        "Number of rows in the M×N layout. Set to 1 for "
                        "a horizontal strip, to len(target_images) for "
                        "a vertical strip, or to any value such that "
                        "rows * cols >= len(target_images) for a "
                        "rectangular grid."
                    ),
                },
                "cols": {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "maximum": 8,
                    "description": (
                        "Number of columns in the M×N layout. Set to "
                        "len(target_images) for a horizontal strip, to "
                        "1 for a vertical strip, or to any value such "
                        "that rows * cols >= len(target_images) for a "
                        "rectangular grid."
                    ),
                },
                # layout / align / padding /
                # preserve_aspect_ratio / show_image_id / padding_color
                # removed; grid_rows / grid_cols renamed to rows / cols.
                "target_images": {
                    "type": "array",
                    "image_ref": True,
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": (
                        "Stable visible image tokens such as img1, img2. "
                        "Select 2 or more images to lay out; they fill "
                        "the M×N grid in row-major order (left-to-right "
                        "then top-to-bottom). Read the user prompt "
                        "carefully and order target_images deliberately "
                        "— do not blindly reuse the original input order."
                    ),
                },
            },
        }

    def execute(self, state, arguments, target_images):
        targets = _ensure_targets(target_images, self.name, self.min_target_images)
        images = [normalize_to_rgba(state.get_image(image_id)) for image_id in targets]

        rows = int(arguments["rows"])
        cols = int(arguments["cols"])
        if rows * cols < len(targets):
            raise ValueError(
                "make_collage needs rows*cols >= the number of images, but got "
                "rows=%d cols=%d (%d cells) for %d images. Increase rows or cols so "
                "every image has a cell." % (rows, cols, rows * cols, len(targets))
            )

        canvas = self._render_grid(images, rows, cols)
        feedback_kwargs = {
            "rows": rows,
            "cols": cols,
            "count": len(targets),
            "size": format_size(canvas),
        }

        return ToolOutput(
            produced_images=[
                canvas,
            ],
            tool_feedback_text=build_tool_feedback("make_collage", **feedback_kwargs),
        )

    @classmethod
    def _render_grid(cls, images, rows: int, cols: int) -> Image.Image:
        # Uniform cells sized to the largest image across all inputs; each
        # image keeps its native size, centered in its cell. Cell margins
        # around smaller images, leftover trailing cells, and any transparency
        # carried by the inputs stay transparent on the RGBA canvas
        # (alpha_composite, not an opaque fill).
        cell_w = max(img.width for img in images)
        cell_h = max(img.height for img in images)
        canvas_w = cols * cell_w
        canvas_h = rows * cell_h
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        for idx, image in enumerate(images[: rows * cols]):
            row = idx // cols
            col = idx % cols
            origin_x = col * cell_w + (cell_w - image.width) // 2
            origin_y = row * cell_h + (cell_h - image.height) // 2
            canvas.alpha_composite(image, dest=(origin_x, origin_y))
        return canvas

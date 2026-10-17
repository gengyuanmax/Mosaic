"""Draw-normalized-coordinate-grid tool — draw a normalized coordinate grid on one image."""

from __future__ import annotations

from PIL import Image, ImageDraw

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import (
    build_tool_feedback,
    coerce_positive_int,
    format_size,
    normalize_to_rgba,
)


class DrawNormalizedCoordinateGridTool(ToolBase):
    name = "draw_normalized_coordinate_grid"
    description_baseline = (
        "Draws a normalized coordinate grid over one image as a spatial "
        "reference. It accepts an input target image and the grid row "
        "and column numbers as inputs.\n"
        "\n"
        "It returns: a new image with red gridlines drawn over the input "
        "and normalized 0-1 tick labels along the top and left edges."
    )
    description_prompting = "Draw a coordinate grid over one image as a spatial reference. Use for 'where in the image' questions about position, movement, or alignment; avoid when exact positions do not matter."
    min_target_images = 1
    max_target_images = 1

    # Margin band reserved on the top and left edges for the normalized
    # axis tick labels. Kept large enough for the default PIL bitmap
    # font's 4-character ``0.00`` labels and consistent regardless of
    # image size.
    _MARGIN_PX = 34

    @classmethod
    def schema(cls):
        return {
            "required": ["target_image", "grid_rows", "grid_cols"],
            "properties": {
                "grid_rows": {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "maximum": 8,
                    "description": (
                        "Number of grid rows. Horizontal gridlines are "
                        "labeled top-to-bottom with normalized y values "
                        "0.00, ..., 1.00 along the left edge. Choose "
                        "grid_rows to match the spatial granularity the "
                        "task needs — 4 for coarse regions, up to 8 for "
                        "finer discrimination (denser grids reduce label "
                        "readability)."
                    ),
                },
                "grid_cols": {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "maximum": 8,
                    "description": (
                        "Number of grid columns. Vertical gridlines are "
                        "labeled left-to-right with normalized x values "
                        "0.00, ..., 1.00 along the top edge. Choose "
                        "grid_cols to match the spatial granularity the "
                        "task needs; see grid_rows for sizing guidance."
                    ),
                },
                # `show_coordinates` and `label_cells` removed;
                # the normalized [0, 1] axis ruler is always drawn and
                # there are no in-cell labels.
                "target_image": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the image to "
                        "overlay the grid on, such as img1."
                    ),
                },
            },
        }

    def execute(self, state, arguments, target_images):
        src_id = target_images[0]
        src = normalize_to_rgba(state.get_image(src_id))
        grid_rows = coerce_positive_int(arguments["grid_rows"], "grid_rows")
        grid_cols = coerce_positive_int(arguments["grid_cols"], "grid_cols")

        out = self._render_grid(src, grid_rows, grid_cols)
        return ToolOutput(
            produced_images=[
                out,
            ],
            tool_feedback_text=build_tool_feedback(
                "draw_normalized_coordinate_grid",
                rows=grid_rows,
                cols=grid_cols,
                size=format_size(out),
            ),
            error_code=None,
            error_message=None,
        )

    def _render_grid(self, src: Image.Image, grid_rows: int, grid_cols: int) -> Image.Image:
        # the normalized [0, 1] axis ruler is always drawn,
        # so the top/left margin band is always reserved.
        src_w, src_h = src.size
        margin = self._MARGIN_PX
        canvas_w = src_w + margin
        canvas_h = src_h + margin
        # RGBA so transparency carried by the source survives: ``paste``
        # copies the source's alpha into the image region, while the margin
        # band and the ruler/grid lines stay opaque.
        canvas = Image.new("RGBA", (canvas_w, canvas_h), color=(255, 255, 255, 255))
        canvas.paste(src, (margin, margin))
        draw = ImageDraw.Draw(canvas)

        # The image lives within [left, right) x [top, bottom).
        left = margin
        top = margin
        right = left + src_w
        bottom = top + src_h

        # Cell boundary positions in canvas coordinates. We include the outer
        # border so the model sees a closed grid.
        x_lines = [left + int(round(i * src_w / grid_cols)) for i in range(grid_cols + 1)]
        y_lines = [top + int(round(j * src_h / grid_rows)) for j in range(grid_rows + 1)]

        # White outline drawn first so the red line lands on top cleanly.
        for x in x_lines:
            for dx in (-1, 1):
                xo = max(left, min(right - 1, x + dx))
                draw.line([(xo, top), (xo, bottom - 1)], fill="white", width=1)
        for y in y_lines:
            for dy in (-1, 1):
                yo = max(top, min(bottom - 1, y + dy))
                draw.line([(left, yo), (right - 1, yo)], fill="white", width=1)
        for x in x_lines:
            xc = min(right - 1, x)
            draw.line([(xc, top), (xc, bottom - 1)], fill="red", width=1)
        for y in y_lines:
            yc = min(bottom - 1, y)
            draw.line([(left, yc), (right - 1, yc)], fill="red", width=1)

        self._draw_normalized_axis_labels(draw, x_lines, y_lines, margin, canvas_w, canvas_h)
        return canvas

    def _draw_normalized_axis_labels(
        self, draw, x_lines, y_lines, margin: int, canvas_w: int, canvas_h: int,
    ) -> None:
        # normalized [0, 1] tick labels at each gridline.
        # x = i / grid_cols along the top edge, y = j / grid_rows along
        # the left edge; origin top-left, bare 2-decimal numbers. The
        # first / last tick on each axis is clamped inward so 0.00 / 1.00
        # stay inside the canvas.
        grid_cols = len(x_lines) - 1
        grid_rows = len(y_lines) - 1

        band_cy = margin // 2
        for i, x in enumerate(x_lines):
            label = "%.2f" % (i / grid_cols)
            tw, th = self._measure_text(draw, label)
            tx = max(0, min(canvas_w - tw, x - tw // 2))
            ty = max(0, band_cy - th // 2)
            draw.text((tx, ty), label, fill="red")

        band_cx = margin // 2
        for j, y in enumerate(y_lines):
            label = "%.2f" % (j / grid_rows)
            tw, th = self._measure_text(draw, label)
            tx = max(0, min(margin - tw, band_cx - tw // 2))
            ty = max(0, min(canvas_h - th, y - th // 2))
            draw.text((tx, ty), label, fill="red")

    @staticmethod
    def _measure_text(draw, text: str):
        # ``textbbox`` exists on modern Pillow; fall back to ``textsize`` on
        # older versions. The default bitmap font's metrics are coarse but
        # accurate enough for centring.
        if hasattr(draw, "textbbox"):
            x0, y0, x1, y1 = draw.textbbox((0, 0), text)
            return x1 - x0, y1 - y0
        return draw.textsize(text)

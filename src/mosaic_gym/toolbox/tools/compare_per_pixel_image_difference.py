"""Compare-per-pixel-image-difference tool — visualize per-pixel intensity differences as a heatmap."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import build_tool_feedback, format_size, normalize_to_rgba


# Strict per-pixel gray absdiff mapped straight onto the JET colormap — no
# blur, no tolerance band: every nonzero intensity difference shows at its
# true magnitude.
#
# Transparency semantics: a transparent pixel holds NO content, so it must not
# be compared (its RGB is an arbitrary fill — flattening it to black used to
# make "no pixel here" compare equal to genuinely black content). The per-pixel
# diff is weighted by W = (alpha_a/255)·(alpha_b/255), and W becomes the
# heatmap's own alpha channel — where either image has no content the heatmap
# is transparent ("not compared").


class ComparePerPixelImageDifferenceTool(ToolBase):
    name = "compare_per_pixel_image_difference"
    description_baseline = (
        "Compares two images pixel by pixel. It accepts two input images "
        "of the same size as inputs.\n"
        "\n"
        "It returns: a heatmap image where warmer colors mark larger "
        "per-pixel differences and cool colors mark identical or "
        "near-identical pixels."
    )
    description_prompting = 'Return a heatmap of the exact per-pixel difference between two images (warmer = differ more, cool = identical or nearly so). Use for before/after or spot-the-difference between two related images; avoid when the two images are unrelated scenes.'
    min_target_images = 2
    max_target_images = 2
    # ComparePerPixelImageDifferenceTool is weakened to a pure heatmap visualizer.
    # The previous ``image_and_text`` form was used so that bbox / metric
    # text could be surfaced into the next user message; with bboxes and
    # metrics deleted there is no text to add, so the tool falls back to
    # the ToolBase default (``image_only``).

    @classmethod
    def schema(cls):
        return {
            "required": ["image1", "image2"],
            "properties": {
                "image1": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the first image to "
                        "compare, such as img1."
                    ),
                },
                "image2": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the second image to "
                        "compare, such as img2. It must have the same pixel "
                        "size as image1."
                    ),
                },
            },
        }

    def execute(self, state, arguments, target_images):
        _ = arguments  # no tool-level arguments
        targets = list(target_images or [])
        if len(targets) != 2:
            raise ValueError("diff requires exactly two images")

        a = normalize_to_rgba(state.get_image(targets[0]))
        b = normalize_to_rgba(state.get_image(targets[1]))
        if a.size != b.size:
            raise ValueError(
                "compare_per_pixel_image_difference needs two images of the same size, but got "
                "%dx%d and %dx%d. Images of different sizes cannot be compared."
                % (a.size[0], a.size[1], b.size[0], b.size[1])
            )

        a_arr = np.array(a)
        b_arr = np.array(b)

        # Strict per-pixel gray difference — no blur, no tolerance band.
        a_gray = cv2.cvtColor(a_arr[..., :3], cv2.COLOR_RGB2GRAY)
        b_gray = cv2.cvtColor(b_arr[..., :3], cv2.COLOR_RGB2GRAY)
        diff_gray = cv2.absdiff(a_gray, b_gray)

        # Per-pixel comparison weight: the product of both alphas. Where either
        # image holds no content the weighted diff is zeroed AND the heatmap
        # itself is transparent there ("not compared"), so it cannot be read
        # as a genuine zero-difference.
        weight = (a_arr[..., 3] / 255.0) * (b_arr[..., 3] / 255.0)
        diff_weighted = np.round(diff_gray * weight).astype(np.uint8)

        heat = cv2.cvtColor(cv2.applyColorMap(diff_weighted, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        alpha = np.round(weight * 255).astype(np.uint8)
        out = Image.fromarray(np.dstack([heat, alpha]), mode="RGBA")

        return ToolOutput(
            produced_images=[out],
            tool_feedback_text=build_tool_feedback(
                "diff",
                size=format_size(out),
                compared="%d%%" % round(float(weight.mean()) * 100),
            ),
        )

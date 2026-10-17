"""apply_affine_transformation — apply a single parameterized 2D affine transform to one image."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import build_tool_feedback, format_size, normalize_to_rgba


class ApplyAffineTransformationTool(ToolBase):
    name = "apply_affine_transformation"
    description_baseline = (
        "Applies a 2D affine transform — rotation, uniform scaling, and "
        "translation — to one image. It accepts an input target image "
        "and the rotation_angle, translation_dx, translation_dy, and "
        "scale as inputs.\n"
        "\n"
        "It returns: a new image with the affine transform applied."
    )
    description_prompting = "Apply a 2D affine transform (rotation_angle, scale, translation_dx, translation_dy) to one image, given the four parameters. Use when the task provides concrete affine parameters to verify on an image; avoid when no such parameters exist, or when a 3x3 projective transform is needed (use apply_homography_transformation)."
    PARAM_KEYS = ("rotation_angle", "translation_dx", "translation_dy", "scale")
    min_target_images = 1
    max_target_images = 1
    _EPSILON = 1e-9
    # Extreme (but schema-legal) scale/translation values could otherwise
    # demand a multi-gigabyte canvas now that the output grows with the
    # transformed content.
    _MAX_OUTPUT_SIDE = 8192

    @classmethod
    def schema(cls):
        return {
            "required": ["target_image", "rotation_angle", "translation_dx", "translation_dy", "scale"],
            "properties": {
                "rotation_angle": {
                    "type": "number", "minimum": -1e4, "maximum": 1e4,
                    "description": "Rotation angle in degrees.",
                },
                "translation_dx": {
                    "type": "number", "minimum": -1e5, "maximum": 1e5,
                    "description": "Horizontal translation in pixels.",
                },
                "translation_dy": {
                    "type": "number", "minimum": -1e5, "maximum": 1e5,
                    "description": "Vertical translation in pixels.",
                },
                "scale": {
                    "type": "number", "minimum": -1e4, "maximum": 1e4,
                    "description": "Uniform scale factor. Must be non-zero.",
                },
                "target_image": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the image to "
                        "transform, such as img1."
                    ),
                },
            },
        }

    def validate_arguments(self, arguments):
        try:
            scale = float(arguments["scale"])
        except (KeyError, TypeError, ValueError):
            return []  # registry validator already reported the missing/type error
        if abs(scale) <= self._EPSILON:
            return ["apply_affine_transformation scale must be non-zero (the transform must be invertible)"]
        return []

    def execute(self, state, arguments, target_images):
        targets = list(target_images or [])
        if len(targets) != 1:
            raise ValueError("apply_affine_transformation requires exactly one image")

        params = {key: float(arguments[key]) for key in self.PARAM_KEYS}

        src_id = targets[0]
        # RGBA so output areas no source pixel maps to stay transparent
        # (alpha=0) instead of a baked-in black border; transparency already
        # present in the source is warped along with it.
        src_rgba = np.array(normalize_to_rgba(state.get_image(src_id)))
        height, width = src_rgba.shape[:2]

        angle = float(params["rotation_angle"])
        scale = float(params["scale"])
        dx = float(params["translation_dx"])
        dy = float(params["translation_dy"])

        # Build the transform in cv2's getRotationMatrix2D convention: rotate +
        # scale about the image center, then add the translation.
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        matrix[0, 2] += dx
        matrix[1, 2] += dy

        # Render on the UNION of the source frame and the transformed content's
        # bounding box: nothing is clipped, and — unlike a content-only bounding
        # box — the translation stays visible because the source frame anchors
        # the origin. Canvas areas no source pixel maps to stay transparent.
        corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
        )
        mapped = corners @ matrix[:, :2].T + matrix[:, 2]
        min_x = min(0.0, float(mapped[:, 0].min()))
        min_y = min(0.0, float(mapped[:, 1].min()))
        max_x = max(float(width), float(mapped[:, 0].max()))
        max_y = max(float(height), float(mapped[:, 1].max()))
        out_width = int(np.ceil(max_x - min_x))
        out_height = int(np.ceil(max_y - min_y))
        if max(out_width, out_height) > self._MAX_OUTPUT_SIDE:
            raise ValueError(
                "apply_affine_transformation output canvas %dx%d exceeds the %d px side limit "
                "— use a smaller scale/translation"
                % (out_width, out_height, self._MAX_OUTPUT_SIDE)
            )
        matrix[0, 2] -= min_x
        matrix[1, 2] -= min_y
        warped = cv2.warpAffine(
            src_rgba,
            matrix,
            (out_width, out_height),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0, 0),
        )
        out = Image.fromarray(warped.astype(np.uint8), mode="RGBA")

        return ToolOutput(
            produced_images=[out],
            tool_feedback_text=build_tool_feedback(
                "apply_affine_transformation",
                rotation="%g" % angle,
                scale="%g" % scale,
                dx="%g" % dx,
                dy="%g" % dy,
                size=format_size(out),
            ),
        )

"""apply_homography_transformation — apply a single 3x3 projective transform to one image."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from mosaic_gym.contracts import ToolOutput
from mosaic_gym.toolbox.base import ToolBase
from mosaic_gym.toolbox.utils import build_tool_feedback, format_size, normalize_to_rgba


def _coerce_homography_matrix(values):
    matrix_values = list(values or [])
    if len(matrix_values) != 9:
        raise ValueError("homography matrix must contain exactly 9 numeric values")
    try:
        return np.array([float(value) for value in matrix_values], dtype=np.float64).reshape(3, 3)
    except Exception as exc:
        raise ValueError("homography matrix must contain exactly 9 numeric values") from exc


class ApplyHomographyTransformationTool(ToolBase):
    name = "apply_homography_transformation"
    description_baseline = (
        "Applies a 3x3 projective homography transform to one image. It "
        "accepts an input target image and a 3x3 homography matrix as "
        "inputs.\n"
        "\n"
        "It returns: a new image with the homography applied."
    )
    description_prompting = "Apply a 3x3 projective (homography) transform, given as 9 row-major numbers, to one image. Use to test whether a candidate homography matrix produces the expected result on an image, then compare the warped output with a reference; avoid when no candidate matrix is available, or when only rotation/scale/translation parameters are given (use apply_affine_transformation)."
    min_target_images = 1
    max_target_images = 1
    _EPSILON = 1e-12
    # Extreme (but schema-legal) matrices could otherwise demand a
    # multi-gigabyte canvas now that the output grows with the warped content.
    _MAX_OUTPUT_SIDE = 8192

    @classmethod
    def schema(cls):
        return {
            "required": ["target_image", "matrix"],
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {"type": "number", "minimum": -1e6, "maximum": 1e6},
                    "minItems": 9,
                    "maxItems": 9,
                    "description": (
                        "Row-major 3x3 projective transform as 9 numbers. It "
                        "maps target_image coordinates to output coordinates: "
                        "``p_out = matrix · p_in``. Copy the values verbatim "
                        "from the chosen candidate, including "
                        "scientific-notation terms like -2.0912e-05."
                    ),
                },
                "target_image": {
                    "type": "string",
                    "image_ref": True,
                    "description": (
                        "The stable visible image token of the image to warp, "
                        "such as img1."
                    ),
                },
            },
        }

    def validate_arguments(self, arguments):
        matrix = arguments.get("matrix")
        if not isinstance(matrix, list) or len(matrix) != 9:
            return []  # registry validator already produced the right error
        try:
            matrix_3x3 = _coerce_homography_matrix(matrix)
        except ValueError:
            return []  # registry validator already reported per-element errors
        det = float(np.linalg.det(matrix_3x3))
        if not np.isfinite(det) or abs(det) <= self._EPSILON:
            return ["homography matrix must be invertible (non-singular)"]
        return []

    def execute(self, state, arguments, target_images):
        targets = list(target_images or [])
        if len(targets) != 1:
            raise ValueError("apply_homography_transformation requires exactly one image")

        matrix_3x3 = _coerce_homography_matrix(arguments.get("matrix"))

        # ``matrix`` is the standard FORWARD mapping from source to output
        # coordinates (``p_out = matrix · p_src``).
        # RGBA so canvas areas no source pixel maps to stay transparent
        # (alpha=0) instead of a baked-in black fill; transparency already
        # present in the source is warped along with it.
        src_id = targets[0]
        src_rgba = np.array(normalize_to_rgba(state.get_image(src_id)))
        src_h, src_w = src_rgba.shape[:2]

        # Project the source corners to bound the transformed content. The
        # horizon line (homogeneous w = 0) must not cross the source quad:
        # corners on opposite sides mean the warped image is unbounded.
        corners = np.array(
            [[0, 0, 1], [src_w, 0, 1], [src_w, src_h, 1], [0, src_h, 1]],
            dtype=np.float64,
        )
        projected = corners @ matrix_3x3.T
        w_component = projected[:, 2]
        if (
            not np.all(np.isfinite(projected))
            or np.any(np.abs(w_component) <= self._EPSILON)
            or (np.any(w_component > 0) and np.any(w_component < 0))
        ):
            raise ValueError(
                "homography maps the target image across the projective "
                "horizon, so the output is unbounded; use a different matrix"
            )
        mapped = projected[:, :2] / w_component[:, None]

        # Render on the UNION of the source frame and the warped content's
        # bounding box: nothing is clipped, and — unlike a content-only
        # bounding box — the translation stays visible because the source
        # frame anchors the origin. tool feedback reports where that frame
        # sits on the canvas (it moves only when content lands left/above it).
        min_x = min(0.0, float(mapped[:, 0].min()))
        min_y = min(0.0, float(mapped[:, 1].min()))
        max_x = max(float(src_w), float(mapped[:, 0].max()))
        max_y = max(float(src_h), float(mapped[:, 1].max()))
        out_width = int(np.ceil(max_x - min_x))
        out_height = int(np.ceil(max_y - min_y))
        if max(out_width, out_height) > self._MAX_OUTPUT_SIDE:
            raise ValueError(
                "apply_homography_transformation output canvas %dx%d exceeds the %d px side "
                "limit — the matrix moves the target image too far"
                % (out_width, out_height, self._MAX_OUTPUT_SIDE)
            )
        shift = np.array(
            [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64
        )
        warped = cv2.warpPerspective(
            src_rgba,
            shift @ matrix_3x3,
            (out_width, out_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        out = Image.fromarray(warped.astype(np.uint8), mode="RGBA")

        return ToolOutput(
            produced_images=[out],
            tool_feedback_text=build_tool_feedback(
                "apply_homography_transformation",
                size=format_size(out),
                target_frame_at="(%d,%d)" % (round(-min_x), round(-min_y)),
            ),
        )

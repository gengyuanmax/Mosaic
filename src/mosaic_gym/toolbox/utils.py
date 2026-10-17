"""Shared toolbox helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from PIL import Image


def build_tool_feedback(tool_name: str, **fields) -> str:
    parts = [str(tool_name or "").strip() or "tool"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, float):
            rendered = ("%.4f" % value).rstrip("0").rstrip(".") or "0"
        elif isinstance(value, (tuple, list)):
            rendered = ",".join(str(item) for item in value)
        else:
            rendered = str(value)
        parts.append("%s=%s" % (key, rendered))
    return " ".join(parts)


def format_size(value) -> str:
    if hasattr(value, "size"):
        width, height = value.size
    else:
        width, height = value
    return "%dx%d" % (int(width), int(height))


def fit_size_within_box(src_size, target_size) -> Tuple[int, int]:
    src_w, src_h = src_size
    dst_w, dst_h = target_size
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise ValueError("sizes must be positive")

    scale = min(float(dst_w) / float(src_w), float(dst_h) / float(src_h))
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    return out_w, out_h


def get_resample_filter(name: str):
    resampling = getattr(Image, "Resampling", Image)
    resample_map = {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
        "lanczos": getattr(resampling, "LANCZOS", resampling.BICUBIC),
    }
    key = str(name or "bilinear").strip().lower()
    if key not in resample_map:
        raise ValueError("unsupported resample mode: %s" % name)
    return resample_map[key]


def clamp_box(left: int, top: int, right: int, bottom: int, image_size) -> Tuple[int, int, int, int]:
    width, height = image_size
    clamped_left = max(0, min(int(left), int(width)))
    clamped_top = max(0, min(int(top), int(height)))
    clamped_right = max(clamped_left, min(int(right), int(width)))
    clamped_bottom = max(clamped_top, min(int(bottom), int(height)))
    return clamped_left, clamped_top, clamped_right, clamped_bottom


def centered_offset(container_size, content_size) -> Tuple[int, int]:
    container_w, container_h = container_size
    content_w, content_h = content_size
    return max(0, (container_w - content_w) // 2), max(0, (container_h - content_h) // 2)


def coerce_positive_int(value, field_name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError("%s must be > 0" % field_name)
    return out


def coerce_non_negative_int(value, field_name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError("%s must be >= 0" % field_name)
    return out


def format_target_count(min_target_images: int, max_target_images) -> str:
    minimum = int(min_target_images)
    if max_target_images is None:
        return "at least %d image%s" % (minimum, "" if minimum == 1 else "s")
    maximum = int(max_target_images)
    if minimum == maximum:
        return "exactly %d image%s" % (minimum, "" if minimum == 1 else "s")
    return "%d-%d images" % (minimum, maximum)


def ensure_non_empty_targets(target_images: Iterable[str], tool_name: str):
    targets = list(target_images or [])
    if not targets:
        raise ValueError("%s requires at least one target image" % tool_name)
    return targets


def ensure_min_targets(target_images: Iterable[str], tool_name: str, minimum: int):
    """Return target ids as a list, erroring if fewer than ``minimum`` given.

    Shared by multi-image tools (concatenate, collage). Single-image tools
    just read ``target_images[0]`` directly.
    """
    targets = list(target_images or [])
    if len(targets) < int(minimum):
        raise ValueError("%s requires at least %d target images" % (tool_name, int(minimum)))
    return targets


def to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an H x W x 3 uint8 RGB numpy array."""
    return np.array(image.convert("RGB"))


def _draw_outlined_text(draw, xy, text, fg, bg) -> None:
    """Draw ``text`` at ``xy`` with a 1px outline so it stays readable on
    arbitrary image content.

    Draws the text 4 times shifted by 1px in each cardinal direction in the
    ``bg`` colour, then once at the centre in ``fg`` — works for both the
    default bitmap font and TrueType fonts.
    """
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=bg)
    draw.text((x, y), text, fill=fg)


def normalize_to_rgba(image: Image.Image) -> Image.Image:
    """Normalize a PIL image to RGBA mode — the toolbox-wide canonical mode.

    Every tool's entry normalization: working in RGBA keeps transparency
    (alpha=0 = no image content, e.g. the corners of an expanded rotation)
    alive between tools instead of flattening it to a black fill. Returns
    the original object when already RGBA, a converted copy otherwise.

    """
    if image.mode == "RGBA":
        return image
    return image.convert("RGBA")

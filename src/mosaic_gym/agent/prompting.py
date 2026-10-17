"""Prompt & message construction — everything we SEND to the model.

The system prompt + tool schemas, the per-turn format reminder, the user-message
builder (leading text + image blocks), the tool-result note, and the image
primitives (``data_url`` / ``image_token``). All pure functions — no Agent
state — so they can be unit-tested and reused for offline SFT-data generation.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List

import numpy as np
from PIL import Image

# Appended to the FIRST user question AND re-appended after EVERY tool result —
# this mirrors DeepEyes, which appends USER_PROMPT_V2 to the question and re-uses
# the same string on every tool-response turn. Restating the strict format each
# turn keeps instruct models from drifting on later turns.
# DeepEyes-style <think> variant, kept for a possible Thinking-model or RL run:
# Qwen3-VL-*-Instruct never emits <think> tags (they are a Thinking-variant /
# RL-format-reward convention), so demanding them just weakens the reminder.
# _FORMAT_REMINDER = (
#     "\n\nYou must put your final answer inside <answer> </answer> tags, e.g. <answer> C </answer>. "
#     "Reason step by step, and call a tool to inspect the images whenever it helps. "
#     "Format strictly as <think> </think> <tool_call> </tool_call> (if a tool is needed) "
#     "or <think> </think> <answer> </answer> (when you are sure). "
#     "For a multiple-choice question, put ONLY the option letter inside <answer>."
# )
# ONE table drives every per-format string in both reminders: the <answer>
# example, the per-turn instruction line, and the final-turn commit phrase.
# "mc" is the historic default; every canonical sample carries target.format
# selecting a row (see prepare_mosaic_raw.py). Add a format = add a row.
_ANSWER_FORMATS = {
    "mc": {
        "example": "C",
        "line": "For a multiple-choice question, put ONLY the option letter inside <answer>.",
        "commit": "commit to the single best option letter",
    },
    "yesno": {
        "example": "Yes",
        "line": "Put ONLY Yes or No inside <answer>.",
        "commit": "commit to Yes or No",
    },
    "img": {
        "example": "Image 2",
        "line": "Put ONLY the image reference (like Image 2) inside <answer>.",
        "commit": "commit to the single best image reference",
    },
    "num": {
        "example": "2",
        "line": "Put ONLY the number inside <answer>.",
        "commit": "commit to the single best number",
    },
    "text": {
        "example": "apple",
        "line": "Put ONLY a short answer (a word or short phrase) inside <answer>.",
        "commit": "commit to the single best short answer",
    },
}


def _answer_spec(fmt: str) -> dict:
    return _ANSWER_FORMATS.get(fmt, _ANSWER_FORMATS["mc"])


def format_reminder(fmt: str = "mc") -> str:
    spec = _answer_spec(fmt)
    return (
        "\n\nYou must put your final answer inside <answer> </answer> tags, e.g. <answer> %s </answer>. "
        "Reason step by step in plain text, and call a tool to inspect the images whenever it helps. "
        "End your reply with a single <tool_call> </tool_call> (if a tool is needed) "
        "or with <answer> </answer> (when you are sure). %s"
    ) % (spec["example"], spec["line"])


_FORMAT_REMINDER = format_reminder("mc")

# Replaces _FORMAT_REMINDER on the FINAL allowed turn (step == max_steps). No
# further tool call can run (the loop ends after this turn), so the model is told
# to stop inspecting and commit to an answer NOW. The <answer> contract is
# unchanged, so the scorer's extraction still works.
# <think> variant kept for a possible Thinking-model or RL run (see above):
# _ANSWER_REMINDER = (
#     "\n\nYou are at the step limit — give your FINAL answer NOW, from the evidence you "
#     "already have. Reply ONLY as <think> </think> <answer> </answer> (no tool call). "
#     "The question always has a correct option, so never leave <answer> empty or say it "
#     "cannot be determined — commit to the single best option letter, e.g. <answer> C </answer>."
# )
def answer_reminder(fmt: str = "mc") -> str:
    spec = _answer_spec(fmt)
    return (
        "\n\nYou are at the step limit — give your FINAL answer NOW, from the evidence you "
        "already have. Reply with your step-by-step reasoning followed by <answer> </answer> — "
        "no tool call. The question always has a correct answer, so never leave <answer> empty "
        "or say it cannot be determined — %s, e.g. <answer> %s </answer>."
    ) % (spec["commit"], spec["example"])


_ANSWER_REMINDER = answer_reminder("mc")

# The system prompt = instruction body + # Tools tail. `%s` in the tail is where
# the <tools> signatures get injected. Triple-quoted so the JSON quotes and
# newlines stay literal (no \" / \n noise). The body and tail are kept as
# separate constants because the RL data prep needs the body ALONE: verl's chat
# template appends its own # Tools tail (byte-aligned with ours by design), so
# baking the tail into the dataset system prompt would duplicate it.
SYSTEM_PROMPT_BODY = """You are an agent for multi-image visual reasoning in an environment that provides a suite of tools to support reasoning and task-solving through information processing, visual evidence gathering, transformation, comparison, and verification. Keep working on the user's question until it is fully resolved, and only end your turn once you are sure of the answer. The question always has a correct answer, so never give up or claim it cannot be determined — keep reasoning and using tools until you find it.

Solve the problem step by step. Look closely at the actual content of the images and reason about what they show. When the answer cannot be determined from the images alone, use the provided tools to gather more visual evidence — for example zooming in by cropping a region, rotating or flipping an image, comparing two images, or warping one image by a candidate transform.

You MUST use a tool WHENEVER it can improve your understanding of the images to solve the task. You MUST plan extensively before each tool call, and reflect extensively on the tool result of the previous step.

The provided images are labelled in order as [img1], [img2], ..., each followed by its pixel size — e.g. [img1] (640x480). To pick which image(s) a tool operates on, pass only the imgN token (never the size) in that tool's image argument(s), as declared in its schema. Any image a tool produces becomes the next [imgN], labelled the same way.

Environment:
Image tools are implemented to operate on a shared transparent canvas, similar to a Photoshop document. They let you transform and combine images — including rotating, flipping, cropping, collaging (arranging images into a grid), compositing, and overlaying one image onto another. Each tool call renders its result as a new image, appended as the next [imgN], which you can inspect and operate on further. The canvas expands automatically to fit results that don't fill a plain rectangle, leaving transparent padding in the empty areas (see Conventions).

Conventions:
- Coordinates: the origin (0, 0) is at the top-left corner of the rendered canvas — the full image you see, transparent checkerboard regions included — NOT the top-left of the opaque content within it. Coordinates (points and boxes) refer to that canvas; all tools except apply_affine_transformation (whose translation_dx / translation_dy are in pixels) take them normalized to 0-1 of the canvas size. The x-axis increases to the right and the y-axis increases downward.
- Rotation: angles are measured in degrees. A positive angle rotates the image counter-clockwise; a negative angle rotates it clockwise.
- Transparency: the canvas auto-expands only when an operation produces content that does not fill a plain rectangle — e.g. rotating an image (its corners no longer align to the frame), collaging images of differing sizes, or overlaying/compositing with an offset. The resulting empty areas are shown as a gray-and-white checkerboard (the standard Photoshop "transparency" indicator). Operations whose result is already a full rectangle — such as flipping or cropping — produce no transparent region, and the canvas matches the image content exactly. The checkerboard is NEVER part of the image content: it only marks empty pixels and does not exist in the real image. Ignore it and reason only about the actual opaque content."""

_TOOLS_TAIL = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
%s
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

_SYSTEM_PROMPT = SYSTEM_PROMPT_BODY + _TOOLS_TAIL


# Transparency rendering for the model-facing view. Alpha never survives the
# RGB flatten below, so we choose what transparent pixels look like instead of
# inheriting PIL's default (black). White/light-gray checkerboard is the
# Photoshop/GIMP convention for "no pixel here".
_CHECKER_LIGHT = 255  # white squares
_CHECKER_DARK = 204   # light-gray squares (#CCCCCC)


def _checkerboard(size) -> Image.Image:
    """An opaque RGBA checkerboard at ``size``. Cell size scales with the image
    so the pattern stays visible after any model-side downsampling."""
    w, h = size
    cell = max(8, min(w, h) // 40)
    yy, xx = np.indices((h, w))
    board = np.where((yy // cell + xx // cell) % 2 == 0, _CHECKER_LIGHT, _CHECKER_DARK)
    rgba = np.dstack([board, board, board, np.full((h, w), 255)]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def render_model_view(image) -> Image.Image:
    """The exact pixels the model sees for ``image``: transparency composited
    onto the checkerboard, flattened to RGB.

    Shared by the eval path (``data_url`` below) and the RL adapter
    (``mosaic_gym.rl.tools``) — verl feeds tool-produced PILs straight to the
    processor, whose RGB conversion would otherwise silently render
    transparent regions black instead of the checkerboard the system prompt
    describes. Fully opaque images pass through pixel-unchanged."""
    if image.mode in ("RGBA", "LA", "PA"):
        image = Image.alpha_composite(_checkerboard(image.size), image.convert("RGBA"))
    return image.convert("RGB")


def data_url(image) -> str:
    """Encode a PIL image as a base64 PNG data URL (one image -> one URL).

    An image with an alpha channel is first composited onto the checkerboard,
    so the model sees a deterministic "this area is transparent" rendering.
    Fully opaque pixels are unchanged by the composite, so images without
    transparency encode byte-identically to the pre-checkerboard behavior."""
    buf = io.BytesIO()
    render_model_view(image).save(buf, format="PNG")
    return "data:image/png;base64,%s" % base64.b64encode(buf.getvalue()).decode("utf-8")


def image_token(index: int) -> str:
    """Stable model-facing token for the (0-based) image position."""
    return "img%d" % (index + 1)


def image_label(alias: str, image, include_sizes: bool = True) -> str:
    """The text label shown immediately before an image: ``[img3] (640x480)``.

    Single source of the label format — used by ``build_user_message`` (eval)
    and by the RL agent loop when it re-injects tool-produced images."""
    if include_sizes:
        return "[%s] (%dx%d)" % (alias, image.size[0], image.size[1])
    return "[%s]" % alias


def new_images_note(aliases: List[str]) -> str:
    """The ``New image(s): [img5], [img6].`` sentence appended to a successful
    tool result that produced images (with a leading space, ready to
    concatenate). Empty string for no aliases. Shared with the RL agent loop,
    which assigns aliases at the turn level."""
    if not aliases:
        return ""
    return " New image(s): %s." % ", ".join("[%s]" % a for a in aliases)


def build_tool_schemas(available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn env tool specs into OpenAI-style function schemas (for <tools>)."""
    schemas = []
    for spec in available_tools or []:
        schema = spec.get("schema") or spec.get("model_arg_schema") or {}
        # ``image_ref`` is an internal pipeline marker, not part of
        # the model-facing contract — drop it. Rebuild each property dict
        # rather than popping: the specs share the tool's schema dicts.
        properties = {
            name: {k: v for k, v in prop.items() if k != "image_ref"} if isinstance(prop, dict) else prop
            for name, prop in (schema.get("properties", {}) or {}).items()
        }
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.get("name", ""),
                    "description": spec.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(schema.get("required", []) or []),
                    },
                },
            }
        )
    return schemas


def build_system_prompt(available_tools: List[Dict[str, Any]]) -> str:
    """Render the system prompt: inject the tool signatures into _SYSTEM_PROMPT."""
    # ensure_ascii=False so non-ASCII chars (—, ×, ≥) stay literal, matching what
    # Qwen's own chat template renders for tools (it does not \u-escape them).
    tool_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in build_tool_schemas(available_tools))
    return _SYSTEM_PROMPT % tool_lines


def _result_body(result, id_to_alias: Dict[str, str]) -> str:
    """One tool's result as the text inside a single <tool_response>."""
    tool = result.tool_name or "tool"
    if result.error:
        return (
            "Tool `%s` failed: %s. Try a different call or give your final answer."
            % (tool, result.error.get("message", ""))
        )
    body = ("Tool `%s` result: %s" % (tool, result.tool_feedback or "")).strip()
    body += new_images_note([id_to_alias.get(i, i) for i in (result.new_image_ids or [])])
    return body


def tool_result_note(results, id_to_alias: Dict[str, str], reminder: str = _FORMAT_REMINDER) -> str:
    """A turn's ``ToolResult`` list as the next user message: ONE <tool_response>
    per tool call (in order — Qwen pairs them positionally), then ``reminder``
    re-appended (DeepEyes re-states it every turn). ``reminder`` defaults to
    ``_FORMAT_REMINDER``; the runner passes ``_ANSWER_REMINDER`` on the final turn."""
    blocks = "".join(
        "<tool_response>\n%s\n</tool_response>" % _result_body(r, id_to_alias)
        for r in results
    )
    return "%s%s" % (blocks, reminder)


def build_user_message(
    lead_text: str,
    images_by_id: Dict[str, Any],
    id_to_alias: Dict[str, str],
    already_sent: set,
    include_sizes: bool = True,
) -> Dict[str, Any]:
    """Build one user message: a leading text block, then every image not yet
    shown to the model (each as a ``[imgN] (WxH)`` label + a base64 image_url
    block). The label carries the image's pixel size because the model otherwise
    has no textual source for it — tool feedback is gated to image-only, so a
    tool's own ``size=WxH`` never reaches the model. The label sits immediately
    before its image_url block, which the chat template flattens to
    ``[imgN] (WxH)<|vision_start|>…``, so size and image stay unambiguously paired.

    ``already_sent`` is the set of image_ids the model has already seen; this
    function MUTATES it (adds the newly attached ids) so the caller's dedup state
    stays current across turns — each image is therefore labelled exactly once.
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": lead_text}]
    for image_id, alias in id_to_alias.items():
        if image_id in already_sent:
            continue
        image = images_by_id.get(image_id)
        if image is None:
            continue
        content.append({"type": "text", "text": image_label(alias, image, include_sizes)})
        content.append({"type": "image_url", "image_url": {"url": data_url(image)}})
        already_sent.add(image_id)
    return {"role": "user", "content": content}

# --------------------------------------------------------------------------
# Vanilla (no-tool) mode — the tool sentences stripped, image-size labels
# withheld (deliberate parity choice), and the per-turn format contract merged
# with the final-turn commit mandate (a vanilla call is both first and last
# turn). Texts moved verbatim from evaluation/evaluation_vanilla/run_vanilla.py
# when the two runners were unified behind runner.env = tool | vanilla.
VANILLA_SYSTEM_PROMPT = """You are an agent for multi-image visual reasoning. Keep working on the user's question until it is fully resolved, and only end your turn once you are sure of the answer. The question always has a correct answer, so never give up or claim it cannot be determined — keep reasoning until you find it.

Solve the problem step by step. Look closely at the actual content of the images and reason about what they show.

The provided images are labelled in order as [img1], [img2], .... To point to a specific image, use exactly that imgN token."""


def vanilla_format_reminder(fmt: str = "mc") -> str:
    """The tool pipeline's format contract + final-turn commit mandate merged
    into one (a vanilla call is both first and last turn — see the block
    comment above), with every format-specific bit drawn from the SAME
    _ANSWER_FORMATS table the tool pipeline uses. No more hand-stripped copy."""
    spec = _ANSWER_FORMATS.get(fmt, _ANSWER_FORMATS["mc"])
    return (
        "\n\nYou must put your final answer inside <answer> </answer> tags, e.g. <answer> %s </answer>. "
        "Reason step by step in plain text, then end your reply with <answer> </answer>. "
        "The question always has a correct answer, so never leave <answer> empty or say it "
        "cannot be determined — %s. %s"
    ) % (spec["example"], spec["commit"], spec["line"])

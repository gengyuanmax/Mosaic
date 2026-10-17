"""Run ONE tool call and turn its raw ``ToolOutput`` into a ``ToolResult``.

    run_tool_call(tool_call, state, registry) -> ToolResult
        validate + execute ONE tool call; register its produced images (PIL ->
        image_ids) while keeping the PIL images on the result, gate its feedback,
        attach its name. Any failure (rejected before execute, or an exception
        during execute) comes back as a ToolResult with ``error`` set instead of
        raising — so one bad call in a turn does not abort the others.

The toolbox calls this once per ``ToolCall`` in a turn; the runner pairs the
returned ``ToolResult`` list with the calls by order.
"""

from __future__ import annotations

from typing import List

from mosaic_gym.contracts import ErrorCode
from mosaic_gym.contracts import ToolCall, ToolResult
from mosaic_gym.toolbox.base import FEEDBACK_FORM_IMAGE_ONLY
from mosaic_gym.toolbox.validator import validate_tool_call


def register_images(state, images) -> List[str]:
    """Add produced PIL images to the store; return their newly assigned ids."""
    return [state.add_image(image) for image in images]


def run_tool_call(tool_call: ToolCall, state, registry) -> ToolResult:
    """Validate + execute one tool call and return its ToolResult.

    Errors never propagate — a rejected or crashing call returns a ToolResult
    whose ``error`` is set and which produced no images."""
    valid, error_code, errors = validate_tool_call(tool_call, state, registry)
    if not valid:
        return ToolResult(
            tool_name=tool_call.name,
            error={"code": error_code.value if error_code else None, "message": "; ".join(errors)},
        )

    tool = registry.get(tool_call.name)
    try:
        output = tool.execute(state, tool_call.arguments, tool_call.target_images)
    except Exception as exc:
        return ToolResult(
            tool_name=tool_call.name,
            error={"code": ErrorCode.TOOL_EXECUTION_ERROR.value, "message": str(exc)},
        )

    new_image_ids = register_images(state, output.produced_images)
    # feedback_form gates what the MODEL sees: image_only tools surface only the
    # produced image (no text); text_only / image_and_text show the text.
    model_feedback = "" if tool.feedback_form == FEEDBACK_FORM_IMAGE_ONLY else output.tool_feedback_text
    error = {"code": output.error_code, "message": output.error_message} if output.error_code else None
    return ToolResult(
        tool_name=tool_call.name,
        produced_images=list(output.produced_images),   # kept for base64 render / verl ToolResponse.image
        new_image_ids=new_image_ids,
        tool_feedback=model_feedback,
        error=error,
    )

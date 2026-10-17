"""Tool-call validation — SEMANTIC checks against the toolbox + state.

The ``ToolCall`` shape is already guaranteed by its dataclass, so this only checks
what a free model can still get wrong: the tool exists, the image refs resolve,
the target-image count fits, and the arguments satisfy the tool's own arg schema.
Called once per tool call (a turn may have several).
"""

from typing import List, Tuple

from mosaic_gym.contracts import ErrorCode
from mosaic_gym.contracts import ToolCall
from mosaic_gym.toolbox.base import validate_tool_arguments


def validate_tool_call(tool_call: ToolCall, state, registry) -> Tuple[bool, ErrorCode, List[str]]:
    tool_name = tool_call.name
    if not registry.has(tool_name):
        return False, ErrorCode.TOOL_NOT_FOUND, ["unknown tool: %s" % tool_name]

    missing_images = [image_id for image_id in tool_call.target_images if image_id not in state.images]
    if missing_images:
        return False, ErrorCode.IMAGE_NOT_FOUND, ["missing images: %s" % missing_images]

    tool = registry.get(tool_name)
    target_count = len(tool_call.target_images)
    min_target_images = int(getattr(tool, "min_target_images", 1))
    max_target_images = getattr(tool, "max_target_images", 1)
    if target_count < min_target_images:
        return (
            False,
            ErrorCode.INVALID_TOOL_ARGUMENTS,
            ["tool %s requires at least %d target images" % (tool_name, min_target_images)],
        )
    if max_target_images is not None and target_count > int(max_target_images):
        if min_target_images == int(max_target_images):
            message = "tool %s requires exactly %d target images" % (tool_name, int(max_target_images))
        else:
            message = "tool %s allows at most %d target images" % (tool_name, int(max_target_images))
        return False, ErrorCode.INVALID_TOOL_ARGUMENTS, [message]

    arg_ok, arg_errors = validate_tool_arguments(tool.arg_schema(), tool_call.arguments)
    if not arg_ok:
        return False, ErrorCode.INVALID_TOOL_ARGUMENTS, arg_errors

    custom_errors = list(tool.validate_arguments(tool_call.arguments) or [])
    if custom_errors:
        return False, ErrorCode.INVALID_TOOL_ARGUMENTS, custom_errors

    return True, None, []

"""Boundary contracts: the error taxonomy + the typed schemas that cross the
agent / runner / toolbox / tool boundaries.

Only three dataclasses cross the toolbox boundary (the rollout loop owns parsing
+ termination, so there is no ActionPayload / Observation boundary type):

    ToolCall    runner   -> toolbox  (one tool invocation: name, args, images)
    ToolOutput  tool     -> toolbox  (what ONE tool's execute() returns; raw PIL)
    ToolResult  toolbox  -> runner   (one processed tool result; = verl ToolResponse)

A turn may carry several ``ToolCall`` (the model emits several ``<tool_call>``
blocks); the toolbox runs them in order and returns the matching list of
``ToolResult`` — paired by ORDER (Qwen has no tool_call_id; it pairs the k-th
``<tool_response>`` with the k-th ``<tool_call>`` positionally).

The agent's parse of a model reply (tool calls vs. final answer) is represented
by ``agent.parsing.ParsedReply`` — internal to the agent/runner side, not a
toolbox boundary, so it does not live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from PIL import Image


class ErrorCode(str, Enum):
    INVALID_ACTION_SCHEMA = "INVALID_ACTION_SCHEMA"
    TOOL_PARSING_FAIL = "TOOL_PARSING_FAIL"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    INVALID_TOOL_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    SNAPSHOT_NOT_FOUND = "SNAPSHOT_NOT_FOUND"
    STATE_RESTORE_ERROR = "STATE_RESTORE_ERROR"


# --- runner -> toolbox ------------------------------------------------------
@dataclass
class ToolCall:
    """One tool invocation: which tool, its arguments, and which images."""
    name: str
    arguments: Dict[str, object] = field(default_factory=dict)
    target_images: List[str] = field(default_factory=list)


# --- tool -> toolbox --------------------------------------------------------
@dataclass
class ToolOutput:
    """What ONE tool's ``execute()`` returns — the raw output, before the toolbox
    registers it.

    ``produced_images`` are plain PIL images (no ids yet; the toolbox assigns
    them). ``tool_feedback_text`` is the full summary (surfaced to the model only
    when the tool's ``feedback_form`` keeps text).
    ``error_*`` are set when the tool itself reports a failure. This object stays
    inside the toolbox — ``run_tool_call`` turns it into a ``ToolResult``.
    """
    produced_images: List[Image.Image] = field(default_factory=list)
    tool_feedback_text: str = ""
    error_code: Optional[str] = None
    error_message: Optional[str] = None


# --- toolbox -> runner ------------------------------------------------------
@dataclass
class ToolResult:
    """The toolbox's processed result for ONE tool call — what the runner/agent
    sees, and what maps onto verl's ``ToolResponse``.

    It is the ``ToolOutput`` (or a pre-execute validation error) after the toolbox
    has registered the produced images (-> ``new_image_ids``), gated the feedback
    by ``feedback_form`` (-> model-facing ``tool_feedback``) and attached the tool
    name. ``produced_images`` carries the PIL images themselves so the agent can
    render them (base64) locally and verl can put them in ``ToolResponse.image``;
    ``new_image_ids`` are the stable ids the same images got in the image store.
    ``error`` is None on success, else ``{"code","message"}`` (covers both
    rejected-before-execute and tool-reported failures).
    """
    tool_name: str
    produced_images: List[Image.Image] = field(default_factory=list)
    new_image_ids: List[str] = field(default_factory=list)
    tool_feedback: str = ""                   # model-facing ("" for image-only tools)
    error: Optional[Dict[str, str]] = None

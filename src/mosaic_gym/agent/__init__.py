"""Agent package: a thin stateful ``Agent`` plus pure, reusable helpers.

- ``agent.py``      — the Agent state machine (conversation/image state + the model call)
- ``prompting.py``  — what we SEND (system prompt, tool schemas, user message, image primitives)
- ``parsing.py``— what we read BACK (parse_reply -> ParsedReply, resolve_token)

Public names are re-exported here so ``from mosaic_gym.agent import Agent`` (and
the helpers) keeps working.
"""

from mosaic_gym.agent.agent import Agent
from mosaic_gym.agent.parsing import (
    ParsedReply,
    build_alias_maps,
    image_ref_args_by_tool,
    parse_reply,
    resolve_token,
)
from mosaic_gym.agent.prompting import (
    _ANSWER_REMINDER,
    _FORMAT_REMINDER,
    _SYSTEM_PROMPT,
    build_system_prompt,
    build_tool_schemas,
    build_user_message,
    data_url,
    image_token,
    tool_result_note,
)

__all__ = [
    "Agent",
    "build_system_prompt",
    "build_tool_schemas",
    "build_user_message",
    "tool_result_note",
    "data_url",
    "image_token",
    "parse_reply",
    "ParsedReply",
    "build_alias_maps",
    "image_ref_args_by_tool",
    "resolve_token",
    "_SYSTEM_PROMPT",
    "_FORMAT_REMINDER",
    "_ANSWER_REMINDER",
]

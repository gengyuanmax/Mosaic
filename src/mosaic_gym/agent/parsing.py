"""Agent-side parsing — turn a model reply into a typed ``ParsedReply``, and map
image-ids to the model-facing [imgN] aliases. Pure functions, no Agent state.

    parse_reply(reply, alias_to_id, image_ref_args=None) -> ParsedReply
        parse ALL <tool_call> blocks (in order) into tool_calls=[...], or an
        <answer> into final_answer; a malformed block -> parse_error.
    build_alias_maps(image_ids) -> (alias_to_id, id_to_alias)
        the two-way map between store image_ids and the model's [imgN] tokens.
    resolve_token(token, alias_to_id) -> image_id
        map one model image token (img1 / [img1] / 1) back to an image_id.
    image_ref_args_by_tool(available_tools) -> {tool_name: [(arg, is_array)]}
        which arguments of each tool hold image tokens, from the schema's
        internal ``image_ref`` marker.

``ParsedReply`` is the agent's internal representation of a reply — NOT a toolbox
boundary type. The runner reads it to decide tool-vs-final and what to run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mosaic_gym.agent.prompting import image_token
from mosaic_gym.contracts import ToolCall

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class ParsedReply:
    """What ``parse_reply`` extracted from one model reply.

    Exactly one of three states holds: tool calls (``tool_calls`` non-empty), a
    final answer (``final_answer`` set), or a parse failure (``parse_error`` set).
    The runner decides what to do: run ``tool_calls`` through the toolbox, or
    finish with ``final_answer``, or end the episode on ``parse_error``.
    """
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: Optional[str] = None
    parse_error: Optional[str] = None


def build_alias_maps(image_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Two-way map between store image_ids (img_0001) and model tokens (img1).

    ``alias_to_id`` (token -> id) resolves the model's tool targets;
    ``id_to_alias`` (id -> token) labels images sent to the model."""
    alias_to_id = {image_token(i): iid for i, iid in enumerate(image_ids)}
    id_to_alias = {iid: image_token(i) for i, iid in enumerate(image_ids)}
    return alias_to_id, id_to_alias


def resolve_token(token, alias_to_id: Dict[str, str]) -> str:
    """Map a model image token (img1 / [img1] / 1) to an image_id."""
    text = str(token).strip().lower()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text in alias_to_id:
        return alias_to_id[text]
    if text.isdigit() and ("img%d" % int(text)) in alias_to_id:
        return alias_to_id["img%d" % int(text)]
    return str(token)


def image_ref_args_by_tool(available_tools) -> Dict[str, List[Tuple[str, bool]]]:
    """Per-tool image-argument order: ``{tool_name: [(arg_name, is_array), ...]}``.

    Derived from each tool spec's canonical schema — the properties carrying
    the internal ``"image_ref": True`` marker, in declaration order. ``parse_reply`` uses this to know which arguments of a call hold
    image tokens and in what order they fill ``ToolCall.target_images``.
    """
    result: Dict[str, List[Tuple[str, bool]]] = {}
    for spec in available_tools or []:
        schema = spec.get("schema") or spec.get("model_arg_schema") or {}
        args = [
            (name, prop.get("type") == "array")
            for name, prop in (schema.get("properties") or {}).items()
            if isinstance(prop, dict) and prop.get("image_ref")
        ]
        if args:
            result[str(spec.get("name") or "")] = args
    return result


def _pop_image_tokens(name: str, arguments: Dict, image_ref_args) -> list:
    """Pop one call's raw image tokens out of ``arguments``.

    With a declaration for ``name``, pop each declared image argument in
    schema order — a string argument contributes one token, an array argument
    all of its items. If the model supplied NONE of the declared arguments but
    did send a legacy ``target_images`` array, fall back to it (in the
    declared order, which is what the array already encodes) so the old
    global convention keeps parsing. Without a declaration (unknown tool, or
    no map passed), behave exactly as before: pop ``target_images``.
    """
    spec = (image_ref_args or {}).get(name)
    if not spec:
        return arguments.pop("target_images", []) or []
    tokens = []
    found = False
    for arg_name, is_array in spec:
        if arg_name not in arguments:
            continue
        found = True
        value = arguments.pop(arg_name)
        if value is None:
            continue
        if is_array:
            tokens.extend(value if isinstance(value, list) else [value])
        else:
            tokens.append(value)
    if not found:
        return arguments.pop("target_images", []) or []
    return tokens


def parse_reply(
    reply: str,
    alias_to_id: Dict[str, str],
    image_ref_args: Optional[Dict[str, List[Tuple[str, bool]]]] = None,
) -> ParsedReply:
    """Turn the model reply into a ``ParsedReply``.

    A turn may emit SEVERAL ``<tool_call>`` blocks — all are parsed, in order, into
    ``tool_calls``. On a hard parse failure — any ``<tool_call>`` whose JSON won't
    decode or is missing ``name``, or no action tag at all — ``parse_error`` is set
    (and ``tool_calls`` is empty, ``final_answer`` is None). An unresolved image
    token is NOT a parse error — it is a valid call with a bad ref that the toolbox
    rejects with IMAGE_NOT_FOUND.

    ``image_ref_args`` (from :func:`image_ref_args_by_tool`) tells the parser
    which arguments of each tool hold image tokens; without it — or for a tool
    it does not name — the legacy ``target_images`` array is popped instead.

    The model's reasoning is NOT extracted into a field — it stays in the raw reply
    (recorded as ``model_raw_response``); nothing downstream needs it.
    """
    matches = list(_TOOL_CALL_RE.finditer(reply))
    if matches:
        tool_calls: List[ToolCall] = []
        for match in matches:
            raw = match.group(1).strip()
            try:
                call = json.loads(raw)
            except json.JSONDecodeError as exc:
                return ParsedReply(parse_error="tool_call JSON decode failed: %s | raw=%s" % (exc, raw[:300]))
            if not isinstance(call, dict) or not call.get("name"):
                return ParsedReply(parse_error="tool_call missing 'name': %s" % (str(call)[:300]))
            arguments = dict(call.get("arguments", {}) or {})
            name = str(call["name"])
            raw_targets = _pop_image_tokens(name, arguments, image_ref_args)
            targets = [resolve_token(t, alias_to_id) for t in raw_targets]
            tool_calls.append(ToolCall(name=name, arguments=arguments, target_images=targets))
        return ParsedReply(tool_calls=tool_calls)

    # No <tool_call>. An <answer> is the final answer.
    answer = _ANSWER_RE.search(reply)
    if answer:
        return ParsedReply(final_answer=answer.group(1).strip())
    # No <tool_call> AND no <answer>: the model reasoned but never acted or
    # answered — a format failure. Record it rather than silently passing the
    # whole reply off as the final answer.
    return ParsedReply(
        parse_error="no <tool_call> or <answer> in reply (model neither acted nor answered) | raw tail=%r"
        % (reply[-200:],)
    )

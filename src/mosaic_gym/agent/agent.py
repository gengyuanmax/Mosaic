"""Minimal multimodal tool-calling agent — the LLM layer.

The Agent holds the conversation + image-dedup state and does the LLM-side work
of one turn: build the next user message, call the model, parse the reply into a
``ParsedReply``. It does NOT own the loop or decide when to stop — the runner
(``runner/inference.py``) reads the ``ParsedReply`` and decides tool-vs-final.
All prompt/message construction lives in ``prompting``; reply parsing in
``parsing`` — this file is only the per-turn state + the one model call.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from mosaic_gym.agent.parsing import ParsedReply, build_alias_maps, image_ref_args_by_tool, parse_reply
from mosaic_gym.agent.prompting import (
    answer_reminder,
    format_reminder,
    vanilla_format_reminder,
    build_system_prompt,
    build_user_message,
    tool_result_note,
)
from mosaic_gym.contracts import ToolResult

# Generic fallback sampling, used only when an Agent is built without a profile.
# Per-model defaults live in ``config/agent.py`` (AgentConfig) — the Agent
# itself is model-agnostic: it sends whatever ``sampling`` it is given and routes
# the keys named in ``extra_body_keys`` through ``extra_body`` (the OpenAI SDK's
# escape hatch for non-OpenAI fields such as vLLM's top_k / repetition_penalty).
_FALLBACK_SAMPLING = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096}


class Agent:
    """Drives one episode against any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        available_tools: List[Dict[str, Any]],
        api_key: str = "",
        sampling: Optional[Dict[str, Any]] = None,   # full generation knobs (the values)
        extra_body_keys: tuple = (),                 # which of those route to extra_body
        timeout_sec: Optional[float] = None,         # None = no per-request timeout
    ):
        self.endpoint = str(endpoint or "").rstrip("/")
        self.model = str(model or "")
        self.api_key = str(api_key or "")
        self.sampling = dict(sampling) if sampling else dict(_FALLBACK_SAMPLING)
        self.extra_body_keys = tuple(extra_body_keys)
        # None -> no timeout: large-context retrieval prompts can take minutes to
        # process; a fixed cap just turns them into APITimeoutError. The SLURM
        # walltime is the real backstop.
        self.timeout_sec = float(timeout_sec) if timeout_sec is not None else None
        self._client = None  # lazy OpenAI client, created on first _chat()
        self.system_prompt = build_system_prompt(available_tools)
        # Which arguments of each tool hold image tokens (schema image_ref
        # marker) — parse_reply resolves them into ToolCall.target_images.
        self._image_ref_args = image_ref_args_by_tool(available_tools)
        self._conversation: List[Dict[str, Any]] = []
        self._sent_image_ids: set = set()
        self._question = ""
        # Token usage for the current episode (summed over its turns; zeroed by
        # reset() since the agent is reused episode-to-episode). For cost.
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset(self, question: str, answer_format: str = "mc", env_mode: str = "tool") -> None:
        """Start a fresh episode. The Agent (and its OpenAI client) is reused
        across episodes; only the per-episode state — conversation, sent-image
        set, and token counters — is cleared."""
        self._conversation = [{"role": "system", "content": self.system_prompt}]
        self._sent_image_ids = set()
        self._question = str(question or "")
        self._answer_format = str(answer_format or "mc")
        self._env_mode = str(env_mode or "tool")
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- main entry ----------------------------------------------------
    def respond(
        self,
        image_ids: Sequence[str],
        images_by_id: Dict[str, Any],
        tool_results: Optional[Sequence[ToolResult]] = None,
        is_last_step: bool = False,
    ):
        """Produce the next turn: build the user message from the current images
        (the question on the first turn, the ``<tool_response>`` note for the
        previous turn's ``tool_results`` afterwards), call the model, parse the
        reply -> ``(ParsedReply, raw_reply)``.

        The runner reads the ``ParsedReply`` to decide tool-vs-final; the raw
        reply is recorded as ``model_raw_response``. ``image_ids`` / ``images_by_id``
        come from the toolbox (its current image store). ``is_last_step`` swaps the
        per-turn format reminder for the answer reminder on the final allowed turn."""
        alias_to_id, id_to_alias = build_alias_maps(list(image_ids or []))

        self._conversation.append(self._build_user_message(tool_results, images_by_id, id_to_alias, is_last_step))
        reply = self._chat()
        self._conversation.append({"role": "assistant", "content": reply})
        return parse_reply(reply, alias_to_id, self._image_ref_args), reply

    def _build_user_message(self, tool_results, images_by_id, id_to_alias, is_last_step=False) -> Dict[str, Any]:
        """First turn (no tool_results) = the question; later turns = the
        ``<tool_response>`` note for those results. Attaches any not-yet-sent
        images (updating self._sent_image_ids). On the final turn (``is_last_step``)
        the answer reminder replaces the format reminder."""
        fmt = getattr(self, "_answer_format", "mc")
        if getattr(self, "_env_mode", "tool") == "vanilla":
            # single-call baseline: merged format+commit contract, no tool branch
            reminder = vanilla_format_reminder(fmt)
        else:
            reminder = answer_reminder(fmt) if is_last_step else format_reminder(fmt)
        if tool_results is None:
            lead = self._question + reminder
        else:
            lead = tool_result_note(tool_results, id_to_alias, reminder)
        include_sizes = getattr(self, "_env_mode", "tool") != "vanilla"  # vanilla withholds (WxH) — parity choice
        return build_user_message(lead, images_by_id, id_to_alias, self._sent_image_ids, include_sizes=include_sizes)

    # -- the one model call (OpenAI SDK) -------------------------------
    def _chat(self) -> str:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.endpoint or None,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
                timeout=self.timeout_sec,
            )
        # One model call; the model generates its whole reply itself (no prefill).
        # Native OpenAI fields go top-level; the keys named in extra_body_keys go
        # in extra_body (the SDK's escape hatch for non-OpenAI fields, e.g. vLLM's
        # top_k / repetition_penalty). A provider that declares none sends none.
        native = {k: v for k, v in self.sampling.items() if k not in self.extra_body_keys}
        create_kwargs = {"model": self.model, "messages": self._conversation, **native}
        extra_body = {k: self.sampling[k] for k in self.extra_body_keys if k in self.sampling}
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        response = self._client.chat.completions.create(**create_kwargs)
        reply = response.choices[0].message.content or ""

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return reply

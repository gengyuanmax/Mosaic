"""Custom verl agent loop — the RL twin of the gym runner's conversation shape.

Extends verl's stock ToolAgentLoop with everything the mosaic eval pipeline
does around tool calls, WITHOUT touching the vendored verl tree:

1. **Full-fidelity tool schemas.** verl's YAML→pydantic parse keeps only
   type/description/enum per property (numeric bounds and array ``items`` are
   dropped), so the ``<tools>`` block the policy would see differs from eval.
   We rebuild ``self.tool_schemas`` from the gym registry via the same
   ``prompting.build_tool_schemas`` the eval agent uses.

2. **Gym-shaped tool-response turns.** The stock loop emits ``role: tool``
   messages (image before text, no labels). We instead assemble ONE
   ``role: user`` message per turn, byte-shaped like the gym runner's:
   ``<tool_response>`` blocks (result note incl. ``New image(s): [imgN].``),
   the per-turn format reminder AFTER the blocks, then each produced image
   preceded by its ``[imgN] (WxH)`` label. [imgN] aliases are assigned HERE,
   at the turn level — parallel calls in one turn share the pre-turn pool
   snapshot, so a tool cannot know its own output's index.

3. **RGBA shadow pool.** The model must see transparency composited onto the
   checkerboard (the adapter returns that flattened view), but later tool
   calls must read the RGBA originals (overlay/collage blend by alpha). The
   originals arrive via the adapter's metrics (``rgba_images``) and are kept
   in ``agent_data.mosaic_rgba_pool``, index-aligned with the display pool.

4. **Per-sample, format-aware reminders.** ``extra_info.answer_format``
   (mc/yesno/img/num/text) selects the same ``format_reminder`` /
   ``answer_reminder`` strings the eval runner sends; the answer reminder
   replaces the format reminder on the final allowed turn.

Registered WITHOUT editing verl — loaded via
``actor_rollout_ref.rollout.agent.agent_loop_config_path`` (see
rl/configs/agent_loop/mosaic_agent_loop.yaml), selected with
``actor_rollout_ref.rollout.agent.default_agent_loop=mosaic_tool_agent``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer

from mosaic_gym.agent.prompting import (
    answer_reminder,
    build_tool_schemas,
    format_reminder,
    image_label,
    image_token,
    new_images_note,
)
from mosaic_gym.toolbox.registry import build_registry


@register("mosaic_tool_agent")
class MosaicToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        # The agent_loop_config_path YAML registers this loop by name and passes
        # the whole config (incl. `name`) to hydra.instantiate; swallow it so it
        # doesn't reach the base __init__.
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        # Full-fidelity prompt schemas from the gym registry (point 1 above),
        # restricted to the tools the tool_config actually wired up, in the
        # registry's (name-sorted) order — the same order eval renders.
        specs = [s for s in build_registry().list_tools() if s["name"] in self.tools]
        if specs:
            self.tool_schemas = build_tool_schemas(specs)

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        # Per-sample answer format for the reminders. One loop instance serves
        # one request, so stashing it on self is safe.
        self._answer_format = (kwargs.get("extra_info") or {}).get("answer_format", "mc")
        # Env-verified tool-call ledger (name, success) for reward v3.1 — the
        # reward must not trust tool mentions in model-generated text (fakeable),
        # so the loop records what actually executed and ships it alongside
        # tool_rewards through extra_fields -> extra_info.
        self._tool_names: list[str] = []
        self._tool_success: list[bool] = []
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields["mosaic_tool_names"] = list(self._tool_names)
        output.extra_fields["mosaic_tool_success"] = list(self._tool_success)
        return output

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Execute the turn's tool calls and feed the results back gym-style.

        Reimplements the stock method (verl tool_agent_loop) with the message
        assembly swapped out; the surrounding bookkeeping — parallel-call cap,
        truncation inside ``_call_tool``, response budget check, pool append
        AFTER the budget check, mask/turn counters — mirrors the original.
        """
        executed_calls = agent_data.tool_calls[: self.max_parallel_calls]
        tasks = []
        for tool_call in executed_calls:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Ledger for reward v3.1: env-verified (name, success) per executed call.
        for tool_call, (_resp, tool_reward, call_metrics) in zip(executed_calls, responses):
            ok = bool((call_metrics or {}).get("success", (tool_reward or 0) >= 0))
            self._tool_names.append(str(tool_call.name))
            self._tool_success.append(ok)

        # Lazily start the RGBA shadow pool from the initial (opaque) frames.
        rgba_pool = getattr(agent_data, "mosaic_rgba_pool", None)
        if rgba_pool is None:
            rgba_pool = list(agent_data.image_data or [])
            agent_data.mosaic_rgba_pool = rgba_pool

        # Turn-level [imgN] alias assignment: this turn's products are appended
        # to the pool in response order, so alias indices continue from the
        # current pool size.
        pool_len = len(agent_data.image_data or [])
        blocks = []                      # one <tool_response> body per call
        labelled_images = []             # [(label, display PIL), ...]
        new_display, new_rgba = [], []
        for tool_response, tool_reward, metrics in responses:
            body = tool_response.text or ""
            display = list(tool_response.image or [])
            rgba = list((metrics or {}).get("rgba_images") or []) if isinstance(metrics, dict) else []
            # A non-mosaic failure path (e.g. verl's own "Error when executing
            # tool") carries no rgba twin — fall back to the display view so
            # the two pools stay index-aligned.
            if len(rgba) != len(display):
                rgba = display
            aliases = [image_token(pool_len + len(new_display) + k) for k in range(len(display))]
            body += new_images_note(aliases)
            blocks.append(body)
            for alias, img in zip(aliases, display):
                labelled_images.append((image_label(alias, img), img))
            new_display.extend(display)
            new_rgba.extend(rgba)
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # The reminder rides AFTER the <tool_response> blocks (gym byte shape);
        # on the last allowed turn it becomes the commit-now answer reminder.
        fmt = getattr(self, "_answer_format", "mc")
        next_is_last = (
            self.max_assistant_turns is not None
            and agent_data.assistant_turns >= self.max_assistant_turns - 1
        )
        reminder = answer_reminder(fmt) if next_is_last else format_reminder(fmt)
        lead = "".join("<tool_response>\n%s\n</tool_response>" % b for b in blocks) + reminder

        content: list[dict[str, Any]] = [{"type": "text", "text": lead}]
        for label, _img in labelled_images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image"})
        message = {"role": "user", "content": content if labelled_images else lead}
        agent_data.messages.append(message)

        response_ids = await self.apply_chat_template(
            [message],
            images=new_display if new_display else None,
            videos=None,
            remove_system_prompt=True,
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        if new_display:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            agent_data.image_data.extend(new_display)
            rgba_pool.extend(new_rgba)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

"""The Toolbox — a stateful, per-trajectory tool executor.

One ``Toolbox`` instance == one rollout. It holds the image store (``ImageState``)
and does ONE thing: run tool calls. It does NOT parse model output, decide
tool-vs-final, count steps, terminate, or compute reward — those are the
runner's job (see ``runner/inference.py``). This is the local-eval counterpart of
a verl tool: ``run(tool_calls) -> list[ToolResult]`` maps onto
``BaseTool.execute(parameters) -> ToolResponse``.

A turn may carry several tool calls (the model emits several ``<tool_call>``
blocks); they run in order and the results come back in the same order.
"""

from __future__ import annotations

from typing import Dict, List

from PIL import Image

from mosaic_gym.contracts import ToolCall, ToolResult
from mosaic_gym.toolbox.execution import run_tool_call
from mosaic_gym.toolbox.registry import build_registry
from mosaic_gym.toolbox.state import ImageState, coerce_image


class Toolbox:
    def __init__(self, tool_names=None):
        # tool_names falsy -> the full default toolbox; a list restricts it to a
        # subset (tool-ablation experiments, set via toolbox.tools in the config).
        self.registry = build_registry(tool_names)
        self.state = ImageState()

    def reset(self, images=(), prompt: str = "") -> None:
        """Start a fresh trajectory: a new image store seeded with the input
        images, and the prompt the labelled-candidate tools read from."""
        self.state = ImageState(prompt=prompt)
        for image in images:
            self.state.add_image(coerce_image(image))

    def run(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Run each ToolCall in order; return the matching list of ToolResult
        (paired with the calls by position)."""
        return [run_tool_call(tc, self.state, self.registry) for tc in tool_calls]

    def image_ids(self) -> List[str]:
        return self.state.image_ids()

    def images_by_id(self) -> Dict[str, Image.Image]:
        return self.state.images_by_id()

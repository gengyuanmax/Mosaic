"""verl BaseTool adapters for the full MosaicGym toolbox.

One generic bridge (``MosaicVerlTool``) wraps any ``mosaic_gym.toolbox`` tool —
it does NOT reimplement any image logic. The per-tool classes at the bottom
exist only because verl's tool_config instantiates tools by importable
``class_name``; they are two-liners. The tool_config YAML itself is generated
from the gym registry by ``rl/scripts/gen_tool_config.py`` — do not hand-edit.

Image-pool design: verl passes ``agent_data`` to ``execute`` and
``agent_data.image_data`` is the LIVE image pool for this trajectory (initial
frames + tool-produced images, maintained by our agent loop). Pool position i
is the model-facing token ``img{i+1}`` — the same [imgN] labelling the gym
runner uses, so the finalized string image arguments (``target_image: "img2"``,
``background``/``foreground``, …) work unchanged.

Two views of every produced image (mirrors the gym's ImageState-vs-data_url
split):
  * the MODEL sees the flattened view — transparency composited onto the
    checkerboard (``prompting.render_model_view``), exactly like eval; this is
    what ``ToolResponse.image`` carries and what enters the display pool.
  * later TOOL CALLS must read the RGBA original (overlay/collage blend by
    alpha; baking the checkerboard in would corrupt them). The originals ride
    in the metrics dict (``rgba_images``) and our agent loop keeps them in an
    index-aligned shadow pool (``agent_data.mosaic_rgba_pool``) that this
    adapter prefers as the tool-facing source.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from mosaic_gym.agent.parsing import (
    _pop_image_tokens,
    build_alias_maps,
    image_ref_args_by_tool,
    resolve_token,
)
from mosaic_gym.agent.prompting import render_model_view
from mosaic_gym.contracts import ToolCall
from mosaic_gym.toolbox.base import ToolRegistry
from mosaic_gym.toolbox.execution import run_tool_call
from mosaic_gym.toolbox.state import ImageState
from mosaic_gym.toolbox.tools import (
    ApplyAffineTransformationTool as _Affine,
    ApplyHomographyTransformationTool as _Homography,
    ComparePerPixelImageDifferenceTool as _Diff,
    CropTool as _Crop,
    DrawNormalizedCoordinateGridTool as _Grid,
    FlipTool as _Flip,
    MakeCollageTool as _Collage,
    OverlayImagesTool as _Overlay,
    ResizeTool as _Resize,
    RotateTool as _Rotate,
)

# Tool-call failure penalty, read by mosaic_gym.rl.reward.tool_reward (sign only).
_FAIL_REWARD = -0.05


def _failure(tool_name: str, message: str) -> tuple[ToolResponse, float, dict]:
    """A failed call, phrased exactly like the gym's tool_result_note error body
    (prompting._result_body) so the model sees the same feedback in RL and eval."""
    text = "Tool `%s` failed: %s. Try a different call or give your final answer." % (tool_name, message)
    return ToolResponse(text=text), _FAIL_REWARD, {"success": False}


class MosaicVerlTool(BaseTool):
    """Generic verl adapter around one gym ``ToolBase`` (``_mosaic_cls``)."""

    _mosaic_cls = None  # set by each subclass

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._tool = self._mosaic_cls()
        # Single-tool registry: run_tool_call gives us the gym's validate +
        # execute + feedback-gating pipeline, identical to the eval path.
        self._registry = ToolRegistry()
        self._registry.register(self._tool)
        # Which arguments hold image tokens, in schema order.
        self._image_ref_args = image_ref_args_by_tool([self._tool.spec()])

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        # Stateless per call — images come from agent_data at execute time.
        return instance_id or str(uuid4()), ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        name = self._tool.name
        agent_data = kwargs.get("agent_data")
        # Tool-facing source: the RGBA shadow pool when the agent loop maintains
        # one (ours does), else the display pool (initial images are opaque, so
        # before the first tool call the two are identical).
        pool = list(
            getattr(agent_data, "mosaic_rgba_pool", None)
            or getattr(agent_data, "image_data", None)
            or []
        )
        if not pool:
            return _failure(name, "no images available")

        # Rebuild the gym view of the trajectory: pool order == [imgN] order.
        state = ImageState()
        ids = [state.add_image(img) for img in pool]
        alias_to_id, _ = build_alias_maps(ids)

        # Extract the image-token arguments exactly like the eval parser does.
        arguments = dict(parameters or {})
        raw_targets = _pop_image_tokens(name, arguments, self._image_ref_args)
        targets = [resolve_token(t, alias_to_id) for t in raw_targets]

        call = ToolCall(name=name, arguments=arguments, target_images=targets)
        result = run_tool_call(call, state, self._registry)  # never raises

        if result.error:
            return _failure(name, result.error.get("message", ""))

        # Output guard: Qwen's processor hard-rejects images with aspect ratio
        # > 200 or sides below one patch (28px) — an extreme crop (e.g. a
        # 640x2 sliver) fed back to the model would crash the whole
        # AgentLoopWorker (killed job 621249). Such calls become ordinary
        # failures the model can correct, instead of poison pills.
        for img in result.produced_images:
            w, h = img.size
            if min(w, h) < 28 or max(w, h) / max(1, min(w, h)) > 150:
                return _failure(
                    name,
                    "produced image is %dx%d, too thin or too small for the model to view; "
                    "choose a larger, less extreme region" % (w, h),
                )

        # Model-facing text: the gym note body WITHOUT the "New image(s)" tail —
        # [imgN] aliases are assigned at the TURN level by the agent loop
        # (parallel calls in one turn share the pool snapshot, so a tool cannot
        # know its own output's index).
        text = ("Tool `%s` result: %s" % (name, result.tool_feedback or "")).strip()
        display = [render_model_view(img) for img in result.produced_images]
        metrics = {"success": True, "rgba_images": list(result.produced_images)}
        response = ToolResponse(text=text, image=display) if display else ToolResponse(text=text)
        return response, 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        return None


class CropTool(MosaicVerlTool):
    _mosaic_cls = _Crop


class RotateTool(MosaicVerlTool):
    _mosaic_cls = _Rotate


class ResizeTool(MosaicVerlTool):
    _mosaic_cls = _Resize


class FlipTool(MosaicVerlTool):
    _mosaic_cls = _Flip


class ApplyAffineTransformationTool(MosaicVerlTool):
    _mosaic_cls = _Affine


class DrawNormalizedCoordinateGridTool(MosaicVerlTool):
    _mosaic_cls = _Grid


class ComparePerPixelImageDifferenceTool(MosaicVerlTool):
    _mosaic_cls = _Diff


class ApplyHomographyTransformationTool(MosaicVerlTool):
    _mosaic_cls = _Homography


class MakeCollageTool(MosaicVerlTool):
    _mosaic_cls = _Collage


class OverlayImagesTool(MosaicVerlTool):
    _mosaic_cls = _Overlay

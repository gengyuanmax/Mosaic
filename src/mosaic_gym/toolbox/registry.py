"""Build the tool registry (the tools the model can call).

By default this is the full 12-tool toolbox; pass ``tool_names`` to restrict it
to a subset (tool-ablation experiments — see ``toolbox.tools`` in the config).
"""

from mosaic_gym.toolbox.base import ToolRegistry
from mosaic_gym.toolbox.tools import build_tools


def build_registry(tool_names=None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(build_tools(tool_names))
    return registry

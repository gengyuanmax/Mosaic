"""Run configuration — one YAML, four sections (data / model / agent / runner).

Pure data: these dataclasses hold the knobs we maintain, and ``load_config``
parses a YAML into them. Nothing here imports the agent / toolbox — the runner
wires a config into the live objects. Each section has one clear owner:

    data    -> what to run on            (runner/data.py loads it)
    model   -> which model + how served  (build the Agent's model side; sbatch reads serving)
    agent   -> the policy's behaviour     (build the Agent's policy side)
    toolbox -> how tools are presented    (which description tier the model sees)
    runner  -> how the run executes        (runner/inference.py: loop + output)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict

import yaml

from mosaic_gym.config.agent import AgentConfig
from mosaic_gym.config.data import DataConfig
from mosaic_gym.config.model import ModelConfig, ServingConfig
from mosaic_gym.config.runner import RunnerConfig
from mosaic_gym.config.toolbox import ToolboxConfig

__all__ = ["RunConfig", "DataConfig", "ModelConfig", "ServingConfig", "AgentConfig",
           "ToolboxConfig", "RunnerConfig", "load_config"]


@dataclass
class RunConfig:
    data: DataConfig
    model: ModelConfig
    agent: AgentConfig
    runner: RunnerConfig
    toolbox: ToolboxConfig = field(default_factory=ToolboxConfig)


def _pick(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keys that are fields of ``cls`` (ignore unknown YAML keys)."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in valid}


def _model(raw: Dict[str, Any]) -> ModelConfig:
    raw = dict(raw or {})
    serving = ServingConfig(**_pick(ServingConfig, raw.pop("serving", {})))
    return ModelConfig(serving=serving, **_pick(ModelConfig, raw))


def load_config(path: str) -> RunConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return RunConfig(
        data=DataConfig(**_pick(DataConfig, raw.get("data"))),
        model=_model(raw.get("model")),
        agent=AgentConfig(**_pick(AgentConfig, raw.get("agent"))),
        runner=RunnerConfig(**_pick(RunnerConfig, raw.get("runner"))),
        toolbox=ToolboxConfig(**_pick(ToolboxConfig, raw.get("toolbox"))),
    )

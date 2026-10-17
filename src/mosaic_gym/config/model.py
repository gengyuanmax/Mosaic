"""Model config — WHICH model and HOW it is served."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServingConfig:
    """vLLM launch knobs. Read by the serving sbatch (via ``python -m mosaic_gym.config get``)."""
    max_model_len: int = 65536
    gpu_memory_utilization: float = 0.90
    port: int = 8000


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-VL-8B-Instruct"          # served + requested model id
    endpoint: str = "http://127.0.0.1:8000/v1"        # OpenAI-compatible base URL
    api_key_env: str = "OPENAI_API_KEY"               # env var holding the API key
    serving: ServingConfig = field(default_factory=ServingConfig)

"""Agent config — the POLICY's behaviour (how it samples / calls the model)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Qwen3-VL-8B-Instruct model-card sampling. top_k / repetition_penalty / *_penalty
# are NOT native OpenAI fields, so they ride in extra_body (vLLM reads them there).
_DEFAULT_SAMPLING = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "repetition_penalty": 1.0, "presence_penalty": 1.5,
    "frequency_penalty": 0.3, "max_tokens": 4096,
}
_DEFAULT_EXTRA_BODY_KEYS = ["top_k", "repetition_penalty", "presence_penalty", "frequency_penalty"]


@dataclass
class AgentConfig:
    sampling: Dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_SAMPLING))
    extra_body_keys: List[str] = field(default_factory=lambda: list(_DEFAULT_EXTRA_BODY_KEYS))
    timeout_sec: Optional[float] = None              # None -> no per-request timeout

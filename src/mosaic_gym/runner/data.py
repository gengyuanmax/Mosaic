"""Load a canonical dataset into per-sample ``CaseInput`` — the thin runtime
loader (the heavy convert/validate tooling is one-time, in scripts/data_prep/).

    load_samples(path)            -> iterate raw sample dicts from .jsonl / .json
    sample_to_case(sample, base)  -> a CaseInput (sample_id, prompt, image_paths, target)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class CaseInput:
    sample_id: str
    prompt: str
    image_paths: List[str]
    target: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def sample_to_case(sample: Dict[str, Any], base_dir: str = "") -> CaseInput:
    """One canonical sample -> a CaseInput, resolving relative image paths against
    ``base_dir``."""
    image_paths = []
    for image in sample.get("images", []):
        path = str(image.get("uri_or_path", ""))
        if base_dir and path and not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        image_paths.append(path)
    return CaseInput(
        sample_id=str(sample.get("sample_id", "")),
        prompt=str(sample.get("prompt", "")),
        image_paths=image_paths,
        target=sample.get("target"),
        metadata=dict(sample.get("metadata", {})),
    )


def load_samples(path: str) -> Iterable[Dict[str, Any]]:
    """Yield sample dicts from a .jsonl (one object/line) or .json (a list, or a
    ``{"samples": [...]}`` object, or a single object)."""
    if path.endswith(".jsonl"):
        with open(path, "r") as fh:
            for idx, line in enumerate(fh, start=1):
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                if not isinstance(item, dict):
                    raise ValueError("line %d must be an object" % idx)
                yield item
        return

    with open(path, "r") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("dataset list items must be objects")
            yield item
        return
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            for item in payload["samples"]:
                if not isinstance(item, dict):
                    raise ValueError("dataset samples items must be objects")
                yield item
            return
        yield payload
        return
    raise ValueError("unsupported dataset format: expected object/list/jsonl")

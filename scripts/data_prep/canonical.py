"""Canonical dataset schema + sample builder (PREPROCESSING, not runtime).

This defines the canonical sample shape that runtime loading
(``mosaic_gym.runner.data``) expects. It is used when CONVERTING a raw dataset
into the standardized ``data/mmiu/datasets/<task>.json`` files. Nothing in the
runtime package imports this — it is one-time data-prep tooling.
"""

REQUIRED_FIELDS = ["sample_id", "dataset_name", "prompt", "images", "task_type"]


def build_canonical_sample(sample_id, dataset_name, prompt, images, task_type, target=None, metadata=None):
    return {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "prompt": prompt,
        "images": images,
        "task_type": task_type,
        "target": target,
        "metadata": metadata or {},
    }


import re as _re


def infer_answer_format(value) -> str:
    """Infer the answer format label (target.format) from a ground-truth value.

    mc = single option letter; yesno = Yes/No; num = numeric; img = image
    reference; text = anything else (exact-match fallback). Every canonical
    generator MUST stamp this into target["format"] — the runner selects the
    per-format answer instruction and scorer with it."""
    v = str(value).strip()
    if _re.fullmatch(r"[A-Ja-j]", v):
        return "mc"
    if v.lower() in ("yes", "no", "y", "n"):
        return "yesno"
    if _re.fullmatch(r"-?\d+(?:\.\d+)?", v):
        return "num"
    if _re.fullmatch(r"(img|image)\s*\d+", v, _re.IGNORECASE):
        return "img"
    return "text"

"""verl custom reward for MosaicGym MCQ tasks.

One reward: ``compute_score_accfmt`` — score = accuracy + format, both binary.
The environment still exposes the image tools; the reward is agnostic to
whether they were used (tool-call counts are returned as metrics only, so the
training dashboard can track tool behaviour without shaping it).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Reuse the single source of truth for option-letter extraction (A-H).
from mosaic_gym.runner.scoring import extract_answer, answer_reward as _fmt_reward
from mosaic_gym.agent.prompting import _ANSWER_FORMATS, answer_reminder, format_reminder

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TOOL_RESPONSE_BLOCK = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)

_MIN_REASONING_CHARS = 20


def _model_text(text: str) -> str:
    """The multi-turn response with environment-authored text removed.

    verl's ``solution_str`` is the decode of the WHOLE response — assistant turns
    AND the observation turns the agent loop feeds back. Those observations
    contain ``<tool_response>`` blocks and per-turn reminders, and every
    reminder carries literal ``<answer> </answer>`` / ``<answer> C </answer>``
    examples: without stripping, tag extraction can read the reminder's empty
    example instead of the model's real answer, zeroing acc/format for every
    tool-using rollout. Reminder strings are imported from ``prompting`` so the
    strip stays byte-identical to what the agent loop injects."""
    text = _TOOL_RESPONSE_BLOCK.sub("", text or "")
    for fmt in _ANSWER_FORMATS:
        for reminder in (format_reminder(fmt), answer_reminder(fmt)):
            text = text.replace(reminder, "")
    return text


def _answer_span(text: str) -> Optional[str]:
    """The text inside the LAST model-authored <answer>...</answer>, or None.

    Last, not first: earlier turns may mention answer tags while reasoning; the
    final tag is the model's committed answer (mirrors the gym runner, which
    scores only the final message)."""
    matches = _ANSWER_TAG.findall(_model_text(text))
    return matches[-1].strip() if matches else None


def accuracy_reward(solution_str: str, ground_truth: str, answer_format: str = "mc") -> float:
    """1.0 iff the reply contains an <answer> tag whose extracted content matches
    the ground truth; 0.0 otherwise — including tag-less replies (STRICT by
    design: no fallback extraction from the raw text)."""
    span = _answer_span(solution_str)
    if span is None:
        return 0.0
    pred = extract_answer(span, answer_format)
    if pred is None:
        return 0.0
    return _fmt_reward(pred, ground_truth, answer_format)


def format_reward(solution_str: str, answer_format: str = "mc") -> float:
    """1.0 only if the reply has an <answer>..</answer> tag with a clean
    parseable answer inside AND visible reasoning text BEFORE that tag.

    The reasoning requirement targets no-think direct answers: a rollout whose
    model-authored text (env text stripped) before the scored tag is shorter
    than ``_MIN_REASONING_CHARS`` non-whitespace characters gets 0.0. Earlier
    assistant turns count as reasoning — a multi-turn rollout that reasons,
    calls tools, and then commits a bare final answer still passes."""
    text = _model_text(solution_str)
    matches = list(_ANSWER_TAG.finditer(text))
    if not matches:
        return 0.0
    last = matches[-1]
    span = last.group(1).strip()
    if not span or extract_answer(span, answer_format) is None:
        return 0.0
    reasoning = re.sub(r"\s+", "", text[: last.start()])
    return 1.0 if len(reasoning) >= _MIN_REASONING_CHARS else 0.0


def compute_score_accfmt(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """score = 1.0*acc + 1.0*fmt  (each binary; score in {0, 1, 2}).

    No tool term: the environment offers tools, and the accuracy gradient alone
    decides whether to use them. tool_S (successful calls) and tool_U (distinct
    tools used successfully) are returned as METRICS only — they never enter
    the score — so the dashboard keeps tracking tool behaviour."""
    extra_info = extra_info or {}
    answer_format = extra_info.get("answer_format", "mc")
    acc = accuracy_reward(solution_str, ground_truth, answer_format)
    fmt = format_reward(solution_str, answer_format)

    names = list(extra_info.get("mosaic_tool_names") or [])
    success = list(extra_info.get("mosaic_tool_success") or [])
    ok_names = [str(n) for n, s in zip(names, success) if s]

    return {
        "score": 1.0 * acc + 1.0 * fmt,
        "acc": acc,
        "format": fmt,
        "tool_S": float(len(ok_names)),
        "tool_U": float(len(set(ok_names))),
    }

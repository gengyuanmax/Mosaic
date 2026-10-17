"""Answer scoring — the only "reward" there is: 1.0 if the final answer's choice
matches the ground truth, else 0.0.

No reward engine / per-step reward. The runner records this single outcome signal
so the trajectory is self-contained and verl can use the same notion of
correctness. The authoritative eval scoring lives in ``evaluation/mmiu_score.py``
(this is its in-rollout twin)."""

from __future__ import annotations

import re
from typing import Optional

# An option letter A-H, as the model tends to phrase it: "answer is B", "(C)",
# "**D**", or a lone letter. First match wins.
_CHOICE_PATTERNS = [
    re.compile(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-H])\)?\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-H])\)?\s*[.):]?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\(([A-H])\)"),
    re.compile(r"\b([A-H])\b"),
]


def extract_choice(text: Optional[str]) -> Optional[str]:
    """Pull the chosen option letter (A-H, upper-cased) out of free text, or None."""
    if not text:
        return None
    for pattern in _CHOICE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


# ---- non-letter answer formats (canonical target.format) --------------------
_YESNO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_IMG_REF = re.compile(r"\bimg\s*(\d+)\b|\bimage\s*(\d+)\b", re.IGNORECASE)
_ORDINAL_REF = re.compile(r"\b(first|second|third|fourth|fifth)\b(\s+image)?", re.IGNORECASE)
_ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5"}
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
              "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def extract_answer(text: Optional[str], fmt: str = "mc") -> Optional[str]:
    """Format-aware extraction. Returns the canonical value string or None.

    mc -> option letter "A"; yesno -> "Yes"/"No"; img -> "imgN";
    num -> the number as a string (word numbers normalised)."""
    if not text:
        return None
    if fmt == "yesno":
        m = _YESNO.search(text)
        return m.group(1).capitalize() if m else None
    if fmt == "img":
        m = _IMG_REF.search(text)
        if m:
            return "img" + (m.group(1) or m.group(2))
        m = _ORDINAL_REF.search(text)
        return "img" + _ORDINALS[m.group(1).lower()] if m else None
    if fmt == "text":
        t = text.strip()
        m = re.search(r"<answer>(.*?)</answer>", t, re.DOTALL)
        return (m.group(1) if m else t).strip()
    if fmt == "num":
        m = _NUMBER.search(text)
        if m:
            return m.group(0)
        for w, d in _NUM_WORDS.items():
            if re.search(r"\b%s\b" % w, text, re.IGNORECASE):
                return d
        return None
    return extract_choice(text)


def _norm_gt(ground_truth: str, fmt: str) -> str:
    g = str(ground_truth).strip()
    if fmt == "mc":
        return g.upper()
    if fmt == "yesno":
        return g.capitalize()
    if fmt == "img":
        m = _IMG_REF.search(g)
        if m:
            return "img" + (m.group(1) or m.group(2))
        m = _ORDINAL_REF.search(g)
        return "img" + _ORDINALS[m.group(1).lower()] if m else g.lower()
    if fmt == "num":
        return _NUM_WORDS.get(g.lower(), g)
    if fmt == "text":
        return g
    return g


def answer_reward(final_answer: Optional[str], ground_truth: Optional[str], fmt: str = "mc") -> float:
    """1.0 if the format-aware extracted answer matches the ground truth, else 0.0."""
    if ground_truth is None:
        return 0.0
    got = extract_answer(final_answer, fmt)
    if got is None:
        return 0.0
    gt = _norm_gt(ground_truth, fmt)
    if fmt == "text":
        return 1.0 if got.strip().lower() == gt.strip().lower() else 0.0
    if fmt == "num":
        # numeric equality, not string equality: "3.0" == "3", "100" == "100".
        # (If tolerance-scored numeric tasks are ever added — e.g. VSI absolute
        # distances — thread a tolerance through here instead of exact isclose.)
        try:
            import math
            return 1.0 if math.isclose(float(got), float(gt), rel_tol=1e-9) else 0.0
        except ValueError:
            return 0.0
    return 1.0 if got == gt else 0.0

#!/usr/bin/env python3
"""Convert Mantis-Instruct **nextqa** (multiple-choice, multi-image) into the same
MosaicGym canonical JSON used for MMIU — so the runner + scoring need no changes.

nextqa raw (downloaded from HF, one parquet + extracted frames) has columns
``id / images / conversation / source``. Each row is one 8-frame video with a
MULTI-TURN conversation of A–E multiple-choice Q&A (answers like ``"D: two"``).
We SPLIT each user/assistant pair into ONE canonical sample (all 8 frames shared),
extract the option letter as the target, and drop the raw conversation.

    canonical sample (matches scripts/data_prep/prepare_mmiu_canonical.py):
      sample_id / dataset_name / prompt / images[{image_id,uri_or_path,...}]
      / task_type / target{type:classification, value:<letter>} / metadata

Image paths are stored RELATIVE to --image-base (the extracted-frames dir); the
runner joins them against its ``data.base_dir`` at load time.

Usage:
  python scripts/data_prep/prepare_mantis_canonical.py \
    --parquet data/mantis_instruct/nextqa/train-00000-of-00001.parquet \
    --image-base data/mantis_instruct/nextqa \
    --out data/mantis_instruct/nextqa/canonical/nextqa_debug.json \
    --limit 100
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Leading option letter, as nextqa phrases answers: "D: two", "D", "(D)".
_ANSWER_LETTER = re.compile(r"^\s*\(?([A-Ea-e])\)?\s*[:.\)]?", re.IGNORECASE)
# An option line in the question, e.g. "A: one" / "B) three".
_OPTION_LINE = re.compile(r"^\s*\(?([A-Ea-e])\)?\s*[:.\)]", re.MULTILINE)
_IMAGE_TAG = re.compile(r"<image>")


def extract_letter(answer: str):
    """Pull the chosen option letter (A–E, upper-cased) from an assistant answer."""
    m = _ANSWER_LETTER.match(str(answer or ""))
    return m.group(1).upper() if m else None


def clean_question(text: str) -> str:
    """Strip the raw ``<image>`` placeholder(s) — the 8 frames are attached by the
    runner as [img1]..[img8]; the question refers to them collectively ("the video")."""
    return _IMAGE_TAG.sub("", str(text or "")).strip()


def iter_qa_pairs(conversation):
    """Yield (question, answer) for each consecutive user→assistant turn."""
    turns = list(conversation)
    for i in range(len(turns) - 1):
        a, b = turns[i], turns[i + 1]
        if a.get("role") == "user" and b.get("role") == "assistant":
            yield str(a.get("content", "")), str(b.get("content", ""))


def build_prompt(question: str) -> str:
    """Question (options already inline) + the answer-format directive (MMIU-style)."""
    return "%s\n\nAnswer with only the correct option letter." % question.strip()


def to_canonical(row, qidx: int, question: str, answer: str, letter: str, data_source: str) -> dict:
    vid = str(row["id"])
    images = [
        {
            "image_id": "img_%d" % (i + 1),
            "uri_or_path": str(e["path"]),
            "modality": "image",
            "source": data_source,
        }
        for i, e in enumerate(row["images"])
    ]
    return {
        "sample_id": "nextqa_%s_q%d" % (vid, qidx),
        "dataset_name": data_source,
        "prompt": build_prompt(question),
        "images": images,
        "task_type": "nextqa",
        "target": {"type": "classification", "format": "mc", "value": letter},
        "metadata": {
            "source": "nextqa",
            "video_id": vid,
            "turn": qidx,
            "answer_text": answer.strip(),
            "num_images": len(images),
        },
    }


def convert(parquet: Path, image_base: Path, limit: int, data_source: str, max_per_video: int):
    df = pd.read_parquet(parquet)
    records = []
    n_skipped_noletter = 0
    n_missing_img = 0
    for _, row in df.iterrows():
        if limit and len(records) >= limit:
            break
        per_video = 0
        for qidx, (question, answer) in enumerate(iter_qa_pairs(row["conversation"])):
            if limit and len(records) >= limit:
                break
            if max_per_video and per_video >= max_per_video:
                break
            letter = extract_letter(answer)
            # Keep only well-formed MCQ answers whose letter is actually an option.
            options = {m.upper() for m in _OPTION_LINE.findall(question)}
            if letter is None or (options and letter not in options):
                n_skipped_noletter += 1
                continue
            rec = to_canonical(row, qidx, clean_question(question), answer, letter, data_source)
            # Warn (don't fail) if a referenced frame is missing on disk.
            for img in rec["images"]:
                if not (image_base / img["uri_or_path"]).exists():
                    n_missing_img += 1
                    break
            records.append(rec)
            per_video += 1
    return records, n_skipped_noletter, n_missing_img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default="data/mantis_instruct/nextqa/train-00000-of-00001.parquet")
    ap.add_argument("--image-base", default="data/mantis_instruct/nextqa",
                    help="Dir the extracted frames live in (uri_or_path is relative to this).")
    ap.add_argument("--out", default="data/mantis_instruct/nextqa/canonical/nextqa_debug.json")
    ap.add_argument("--limit", type=int, default=100, help="Max canonical samples (0 = all).")
    ap.add_argument("--max-per-video", type=int, default=0,
                    help="Cap Q&A pairs taken per video for diversity (0 = all turns).")
    ap.add_argument("--data-source", default="nextqa", help="dataset_name / verl data_source tag.")
    args = ap.parse_args(argv)

    records, n_noletter, n_missing = convert(
        Path(args.parquet), Path(args.image_base), args.limit, args.data_source, args.max_per_video)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print("wrote %d canonical samples -> %s" % (len(records), out))
    print("  skipped (no/invalid answer letter): %d" % n_noletter)
    print("  samples with a missing frame on disk: %d" % n_missing)
    if records:
        from collections import Counter
        dist = Counter(r["target"]["value"] for r in records)
        print("  answer-letter distribution:", dict(sorted(dist.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

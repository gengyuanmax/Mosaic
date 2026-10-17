#!/usr/bin/env python3
"""Convert downloaded V*Bench (craigwu/vstar_bench) into MosaicGym canonical JSON.

V*Bench ships images as files on disk (under data/vstar/raw/<category>/*.jpg) plus
a master test_questions.jsonl (image / text / category / question_id / label). It is
single-image, MCQ — so the converter just references the on-disk images (no dump),
splitting by category into one <category>.json each. base_dir = data/vstar/raw.

  python scripts/data_prep/prepare_vstar_canonical.py --raw data/vstar/raw --out-dir data/vstar/datasets
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/vstar/raw")
    ap.add_argument("--out-dir", default="data/vstar/datasets")
    args = ap.parse_args()
    raw = Path(args.raw)
    rows = [json.loads(l) for l in (raw / "test_questions.jsonl").open() if l.strip()]
    by_task = defaultdict(list)
    for r in rows:
        cat = str(r.get("category") or "vstar")
        rec = {
            "sample_id": "vstar_%s_%s" % (cat, r.get("question_id")),
            "dataset_name": "vstar",
            "prompt": str(r["text"]).strip(),  # already has options + "answer with the option's letter"
            "images": [{"image_id": "img_1", "uri_or_path": r["image"], "modality": "image", "source": "vstar"}],
            "task_type": "vstar_" + cat,
            "target": {"type": "classification", "format": "mc", "value": str(r["label"]).strip()},
            "metadata": {"category": cat, "question_id": r.get("question_id"), "image": r["image"]},
        }
        by_task[rec["task_type"]].append(rec)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for task, recs in sorted(by_task.items()):
        (out / (task + ".json")).write_text(json.dumps(recs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("  %s: %d" % (task, len(recs)))
    print("total %d samples -> %s" % (len(rows), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

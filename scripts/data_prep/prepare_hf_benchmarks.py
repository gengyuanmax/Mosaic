#!/usr/bin/env python3
"""Convert HF multi-image benchmarks (BLINK / MuirBench / Mantis-Eval) into the
same MosaicGym canonical JSON used for MMIU.

Each source is a HuggingFace dataset whose images are embedded PIL objects (not
path-referenced files like MMIU's extracted zips). For every sample we:
  1. dump its images to disk as PNG  -> data/<name>/images/<sample_id>/img_N.png
  2. emit one canonical record       -> data/<name>/datasets/<task>.json

Canonical schema matches scripts/data_prep/prepare_mmiu_canonical.py exactly, so
the runner + scoring need no changes. Inline ``<image>`` placeholders in the
question/options are replaced, IN READING ORDER, with the agent's ``[imgN]``
tokens (the agent appends every image after the text labelled [img1], [img2], …).

Usage (on a login node, after the HF cache is populated):
  python scripts/data_prep/prepare_hf_benchmarks.py --dataset all --out-root data
"""

from __future__ import annotations

import argparse
import json
import os
import re

from canonical import infer_answer_format
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the already-downloaded cache

from datasets import get_dataset_config_names, load_dataset  # noqa: E402

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_IMAGE_TAG = re.compile(r"<image>")


def safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value or "").strip())
    return name.strip("._") or "task"


def letter_only(value: str) -> str:
    """'(A)' / 'A.' / 'A' -> 'A'."""
    m = re.search(r"[A-Za-z]", str(value or ""))
    return m.group(0).upper() if m else str(value or "").strip()


class ImageTokenizer:
    """Replace each ``<image>`` with the next [imgN] token, in reading order."""

    def __init__(self):
        self.n = 0

    def sub(self, text: str) -> str:
        def repl(_m):
            self.n += 1
            return "[img%d]" % self.n
        return _IMAGE_TAG.sub(repl, str(text or ""))


def dump_images(images, sample_id: str, images_root: Path) -> list[str]:
    """Save PIL images as PNG; return their paths relative to the dataset base."""
    out_dir = images_root / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rel = []
    for i, img in enumerate(images):
        fname = "img_%d.png" % (i + 1)
        img.convert("RGB").save(out_dir / fname)
        rel.append("images/%s/%s" % (sample_id, fname))
    return rel


def canonical(sample_id, dataset_name, prompt, rel_paths, task, answer, metadata) -> dict:
    return {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "prompt": prompt,
        "images": [
            {"image_id": "img_%d" % (i + 1), "uri_or_path": p, "modality": "image", "source": dataset_name}
            for i, p in enumerate(rel_paths)
        ],
        "task_type": task,
        "target": {"type": "classification", "format": infer_answer_format(answer), "value": answer},
        "metadata": metadata,
    }


def write_by_task(records, out_dir: Path, dataset_name: str) -> None:
    by_task: dict[str, list] = {}
    for r in records:
        by_task.setdefault(str(r["task_type"]), []).append(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": dataset_name, "num_samples": len(records), "num_tasks": len(by_task), "tasks": {}}
    for task, recs in sorted(by_task.items()):
        path = out_dir / (safe_filename(task) + ".json")
        path.write_text(json.dumps(recs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["tasks"][task] = {"path": str(path), "num_samples": len(recs)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("  wrote %d samples across %d tasks -> %s" % (len(records), len(by_task), out_dir))


# --------------------------- per-dataset builders ---------------------------

def build_mantis(base: Path) -> None:
    print("[Mantis-Eval]")
    ds = load_dataset("TIGER-Lab/Mantis-Eval", "mantis_eval")["test"]
    images_root = base / "images"
    records = []
    for i, r in enumerate(ds):
        sid = "mantis_%04d" % i
        tok = ImageTokenizer()
        q = tok.sub(r["question"])
        if r.get("question_type") == "multi-choice" and r["options"]:
            opts = "\n".join(tok.sub(o) for o in r["options"])
            prompt = "%s\n\nChoices:\n%s\n\nAnswer with only the correct option letter." % (q, opts)
            answer = letter_only(r["answer"])
        else:
            # short-answer items (17 of 217): no options; keep the original
            # answer verbatim so infer_answer_format picks yesno/text/num and
            # the runner scores with the matching extractor.
            prompt = "%s\n\nGive a short direct answer." % q
            answer = str(r["answer"]).strip()
        rel = dump_images(list(r["images"]), sid, images_root)
        records.append(canonical(
            sid, "mantis_eval", prompt, rel, "mantis_eval", answer,
            {"orig_id": r["id"], "category": r.get("category", ""), "data_source": r.get("data_source", ""),
             "question": r["question"], "options": r["options"]},
        ))
        if (i + 1) % 50 == 0:
            print("    %d/%d" % (i + 1, len(ds)))
    write_by_task(records, base / "datasets", "mantis_eval")


def build_muirbench(base: Path) -> None:
    print("[MuirBench]")
    ds = load_dataset("MUIRBENCH/MUIRBench", "default")["test"]
    images_root = base / "images"
    records = []
    for i, r in enumerate(ds):
        sid = "muir_%05d" % i
        tok = ImageTokenizer()
        q = tok.sub(r["question"])
        lines = []
        for j, opt in enumerate(r["options"]):
            lines.append("%s: %s" % (_LETTERS[j], tok.sub(opt)))
        prompt = "%s\n\nChoices:\n%s\n\nAnswer with only the correct option letter." % (q, "\n".join(lines))
        imgs = list(r["image_list"])
        if tok.n != len(imgs):
            print("    !! sample %s: %d <image> tags but %d images (kept)" % (sid, tok.n, len(imgs)))
        rel = dump_images(imgs, sid, images_root)
        records.append(canonical(
            sid, "muirbench", prompt, rel, str(r["task"]), letter_only(r["answer"]),
            {"orig_idx": r["idx"], "image_relation": r.get("image_relation", ""),
             "image_type": r.get("image_type", ""), "counterpart_idx": r.get("counterpart_idx", ""),
             "question": r["question"], "options": r["options"]},
        ))
        if (i + 1) % 200 == 0:
            print("    %d/%d" % (i + 1, len(ds)))
    write_by_task(records, base / "datasets", "muirbench")


_BLINK_CONFIGS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]


def build_blink(base: Path) -> None:
    print("[BLINK]")
    # Hardcode the 14 subtask configs: get_dataset_config_names() degrades to
    # ['default'] under HF_HUB_OFFLINE, which the cache doesn't have.
    cfgs = _BLINK_CONFIGS
    images_root = base / "images"
    records = []
    for cfg in cfgs:
        d = load_dataset("BLINK-Benchmark/BLINK", cfg)
        split = "val" if "val" in d else list(d.keys())[0]  # val carries the answer
        ds = d[split]
        task = "blink_" + safe_filename(cfg).lower()
        for i, r in enumerate(ds):
            sid = "%s_%04d" % (task, i)
            imgs = [r[k] for k in ("image_1", "image_2", "image_3", "image_4") if r.get(k) is not None]
            # BLINK's `prompt` field is the ready, ordinal-referenced prompt.
            prompt = "%s\n\nAnswer with only the correct option letter." % str(r["prompt"]).strip()
            rel = dump_images(imgs, sid, images_root)
            records.append(canonical(
                sid, "blink", prompt, rel, task, letter_only(r["answer"]),
                {"orig_idx": r["idx"], "sub_task": r["sub_task"], "split": split,
                 "question": r["question"], "choices": r["choices"]},
            ))
        print("    %s: %d (%s split)" % (task, len(ds), split))
    write_by_task(records, base / "datasets", "blink")


def _to_pil(value):
    """Coerce an embedded image (PIL / {'bytes':...} / base64 str) to a PIL image."""
    import base64, io
    from PIL import Image
    if hasattr(value, "convert"):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, str):
        return Image.open(io.BytesIO(base64.b64decode(value)))
    raise ValueError("unsupported image type: %r" % type(value))


def build_hrbench(base: Path) -> None:
    print("[HR-Bench]")
    images_root = base / "images"
    records = []
    for split in ("hrbench_4k", "hrbench_8k"):
        ds = load_dataset("DreamMr/HR-Bench", "hrbench_version_split", split=split)
        task = split  # hrbench_4k / hrbench_8k
        for i, r in enumerate(ds):
            sid = "%s_%04d" % (task, i)
            opts = "\n".join("%s. %s" % (L, r[L]) for L in ("A", "B", "C", "D") if r.get(L) is not None)
            prompt = "%s\n\n%s\n\nAnswer with only the correct option letter." % (str(r["question"]).strip(), opts)
            rel = dump_images([_to_pil(r["image"])], sid, images_root)
            records.append(canonical(
                sid, "hrbench", prompt, rel, task, letter_only(r["answer"]),
                {"orig_index": r.get("index"), "category": r.get("category", ""),
                 "cycle_category": r.get("cycle_category"), "question": r["question"]},
            ))
            if (i + 1) % 100 == 0:
                print("    %s %d/%d" % (task, i + 1, len(ds)))
    write_by_task(records, base / "datasets", "hrbench")


def build_mme(base: Path) -> None:
    print("[MME]")
    ds = load_dataset("lmms-lab/MME", "default", split="test")
    images_root = base / "images"
    records = []
    for i, r in enumerate(ds):
        sid = "mme_%05d" % i
        q = str(r["question"]).replace("Please answer yes or no.", "").strip()
        prompt = "%s\n\n(A) Yes\n(B) No\n\nAnswer with only the correct option letter." % q
        ans = "A" if str(r["answer"]).strip().lower() == "yes" else "B"
        rel = dump_images([_to_pil(r["image"])], sid, images_root)
        records.append(canonical(
            sid, "mme", prompt, rel, "mme_" + str(r.get("category", "mme")), ans,
            {"question_id": r.get("question_id"), "category": r.get("category", ""),
             "orig_answer": r["answer"], "question": r["question"]},
        ))
        if (i + 1) % 300 == 0:
            print("    %d/%d" % (i + 1, len(ds)))
    write_by_task(records, base / "datasets", "mme")


BUILDERS = {"mantis": ("mantis_eval", build_mantis), "muirbench": ("muirbench", build_muirbench),
            "blink": ("blink", build_blink), "hrbench": ("hrbench", build_hrbench),
            "mme": ("mme", build_mme)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(BUILDERS) + ["all"], default="all")
    ap.add_argument("--out-root", default="data")
    args = ap.parse_args()
    targets = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    out_root = Path(args.out_root).resolve()
    for t in targets:
        subdir, fn = BUILDERS[t]
        fn(out_root / subdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

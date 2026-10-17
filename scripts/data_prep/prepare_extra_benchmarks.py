#!/usr/bin/env python3
"""Convert the second wave of multi-image benchmarks into the MosaicGym
canonical JSON: MMRA, Q-Bench2 (dev), MIRB, VisChainBench.

Unlike prepare_hf_benchmarks.py this never extracts images to the filesystem:
every image is streamed straight into data/<name>/images.tar (PIL -> PNG bytes,
or zip member -> tar member), and the canonical records reference paths inside
that tar.  The eval runner auto-extracts images.tar on first use
(mosaic_gym.runner.inference._ensure_images_extracted), so the extracted tree
only exists while a benchmark is actually being evaluated — this keeps us under
cluster file-count quotas.

  MMRA          m-a-p/MMRA            1,024 two-image MCQ, 11 sub-tasks
  Q-Bench2      q-future/Q-Bench2-HF  dev split only (test answers are hidden),
                                      image-pair quality MCQ, split by question
                                      type (yes/no | what | how)
  MIRB          VLLMs/MIRB            short-answer (num / yesno / text) over
                                      2-42 images; *_text tasks skipped; images
                                      referenced from a shared pool
  VisChainBench eyehole/VisChainBench CS/Daily/Eng sections, teacher-forced
                                      flattening: each chain turn becomes one
                                      MCQ whose choices are 4 candidate images
                                      and whose context is the correct images
                                      of all previous turns (IC/PI variants not
                                      converted)

Usage (inside the verl container, network on):
  HF_HUB_OFFLINE=0 python scripts/data_prep/prepare_extra_benchmarks.py --dataset all
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import tarfile
import zipfile

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TarImageSink:
    """Append-only images.tar writer; dedupes by member name."""

    def __init__(self, bench_dir):
        os.makedirs(bench_dir, exist_ok=True)
        self.path = os.path.join(bench_dir, "images.tar")
        self.tar = tarfile.open(self.path, "w")
        self.seen = set()

    def add_bytes(self, name, data):
        if name in self.seen:
            return name
        info = tarfile.TarInfo(name)
        info.size = len(data)
        self.tar.addfile(info, io.BytesIO(data))
        self.seen.add(name)
        return name

    def add_pil(self, name, img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return self.add_bytes(name, buf.getvalue())

    def close(self):
        self.tar.close()
        print("  images.tar: %d members, %.1f MB" % (
            len(self.seen), os.path.getsize(self.path) / 2**20))


def write_tasks(bench_dir, name, by_task):
    os.makedirs(os.path.join(bench_dir, "datasets"), exist_ok=True)
    total = 0
    for task, records in sorted(by_task.items()):
        p = os.path.join(bench_dir, "datasets", task + ".json")
        with open(p, "w") as fh:
            json.dump(records, fh, ensure_ascii=False)
        total += len(records)
        print("  %-40s %5d" % (task, len(records)))
    print("  TOTAL %s: %d records" % (name, total))


def record(sample_id, name, prompt, images, task, fmt, value, meta):
    return {"sample_id": sample_id, "dataset_name": name, "prompt": prompt,
            "images": images, "task_type": task,
            "target": {"type": "classification", "format": fmt, "value": str(value)},
            "metadata": meta}


def refs(paths, name):
    return [{"image_id": "img_%d" % (i + 1), "uri_or_path": p, "modality": "image", "source": name}
            for i, p in enumerate(paths)]


# ---------------------------------------------------------------------------
def prepare_mmra(out_root):
    from datasets import load_dataset
    name = "mmra"
    bench = os.path.join(out_root, name)
    sink = TarImageSink(bench)
    ds = load_dataset("m-a-p/MMRA")["train"]
    by_task = {}
    for i, r in enumerate(ds):
        is_choice = "choice" in str(r.get("QA_type", "")).lower()
        sid = "mmra_%s_%05d" % (re.sub(r"\W+", "", r["Task"]).lower(), i)
        paths = [sink.add_pil("images/%s/img_%d.png" % (sid, j + 1), r["image%d" % (j + 1)])
                 for j in range(2)]
        base = r["question"].strip() + "\nFigure 1 is [img1] and Figure 2 is [img2]."
        task = "mmra_" + re.sub(r"\W+", "_", r["Task"]).strip("_").lower()
        if is_choice:
            opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            letters = sorted(opts)
            prompt = (base + "\nSelect from the following choices.\n"
                      + "\n".join("(%s) %s" % (l, opts[l]) for l in letters)
                      + "\n\nAnswer with only the correct option letter.")
            fmt, gt = "mc", str(r["answer"]).strip().upper()
            choices = [opts[l] for l in letters]
        else:  # True or false QA -> yes/no
            prompt = base + "\n\nAnswer with only yes or no."
            fmt, gt = "yesno", str(r["answer"]).strip().capitalize()
            choices = ["Yes", "No"]
        by_task.setdefault(task, []).append(record(
            sid, name, prompt, refs(paths, name), task, fmt, gt,
            {"orig_idx": i, "sub_task": r["Task"], "qa_type": r["QA_type"],
             "question": r["question"], "choices": choices}))
    write_tasks(bench, name, by_task)
    sink.close()


# ---------------------------------------------------------------------------
_QB2_TYPE = {0: "yes_no", 1: "what", 2: "how"}


def prepare_qbench2(out_root):
    from datasets import load_dataset
    name = "qbench2"
    bench = os.path.join(out_root, name)
    sink = TarImageSink(bench)
    ds = load_dataset("q-future/Q-Bench2-HF")["dev"]
    by_task = {}
    for i, r in enumerate(ds):
        opts = [r["option%d" % k] for k in range(4)]
        # drop trailing placeholder options; never drop before the correct letter
        while opts and str(opts[-1]).strip().upper() in ("N/A", "NA", ""):
            opts.pop()
        gt = str(r["correct_choice"]).strip().upper()
        if not opts or gt not in _LETTERS[: len(opts)]:
            continue
        sid = "qbench2_dev_%05d" % i
        paths = [sink.add_pil("images/%s/img_%d.png" % (sid, j + 1), r["image%d" % (j + 1)])
                 for j in range(2)]
        prompt = (r["question"].strip()
                  + "\nThe first image is [img1] and the second image is [img2]."
                  + "\nSelect from the following choices.\n"
                  + "\n".join("(%s) %s" % (_LETTERS[k], o) for k, o in enumerate(opts))
                  + "\n\nAnswer with only the correct option letter.")
        task = "qbench2_" + _QB2_TYPE.get(r["question_type"], "other")
        by_task.setdefault(task, []).append(record(
            sid, name, prompt, refs(paths, name), task, "mc", gt,
            {"orig_idx": i, "question_type": r["question_type"],
             "question_concern": r["question_concern"], "question": r["question"], "choices": opts}))
    write_tasks(bench, name, by_task)
    sink.close()


# ---------------------------------------------------------------------------
_MIRB_SKIP = {"arxiv_text", "codeu_text", "plot_text"}
_YESNO_RE = re.compile(r"^(yes|no)$", re.IGNORECASE)


def _mirb_fmt(ans):
    if isinstance(ans, (int, float)) or re.match(r"^-?\d+(\.\d+)?$", str(ans).strip()):
        return "num", "Answer with only the number."
    if _YESNO_RE.match(str(ans).strip()):
        return "yesno", "Answer with only yes or no."
    if len(str(ans).strip()) == 1 and str(ans).strip().upper() in _LETTERS:
        return "mc", "Answer with only the correct option letter."
    return "text", "Answer with only the exact short answer, wrapped as <answer>...</answer>."


def prepare_mirb(out_root):
    from huggingface_hub import snapshot_download
    name = "mirb"
    bench = os.path.join(out_root, name)
    sink = TarImageSink(bench)
    repo = snapshot_download("VLLMs/MIRB", repo_type="dataset")
    zf = zipfile.ZipFile(os.path.join(repo, "images.zip"))
    members = {n for n in zf.namelist() if not n.endswith("/")}
    by_task = {}
    for fn in sorted(os.listdir(repo)):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        if stem in _MIRB_SKIP:
            continue
        rows = json.load(open(os.path.join(repo, fn)))
        task = "mirb_" + stem
        for i, r in enumerate(rows):
            imgs = r.get("images") or []
            if not imgs or any(p not in members for p in imgs):
                continue
            paths = [sink.add_bytes(p, zf.read(p)) for p in imgs]
            fmt, instr = _mirb_fmt(r["answers"])
            prompt = str(r["questions"]).strip() + "\n\n" + instr
            sid = "mirb_%s_%04d" % (stem, i)
            by_task.setdefault(task, []).append(record(
                sid, name, prompt, refs(paths, name), task, fmt, r["answers"],
                {"orig_idx": i, "sub_task": stem}))
    write_tasks(bench, name, by_task)
    sink.close()


# ---------------------------------------------------------------------------
_VCB_SECTIONS = ("CSTask", "DailyTask", "EngTask")


def prepare_vischainbench(out_root):
    from huggingface_hub import snapshot_download
    name = "vischainbench"
    bench = os.path.join(out_root, name)
    sink = TarImageSink(bench)
    repo = snapshot_download("eyehole/VisChainBench", repo_type="dataset")
    by_task = {}
    for section in _VCB_SECTIONS:
        zf = zipfile.ZipFile(os.path.join(repo, "VisChainBench", section + ".zip"))
        members = {n for n in zf.namelist() if not n.endswith("/")}
        tasks = sorted({n.split("/")[1] for n in members if n.count("/") >= 2},
                       key=lambda t: (len(t), t))
        task_name = "vischain_" + section.replace("Task", "").lower()
        for tid in tasks:
            tj = "%s/%s/task.json" % (section, tid)
            if tj not in members:
                continue
            steps = json.loads(zf.read(tj))
            scene = str(steps.pop("initial_scene_description", "")).strip()
            steps = {k: v for k, v in steps.items() if isinstance(v, dict)}
            keys = sorted(steps, key=lambda k: int(re.sub(r"\D", "", k) or 0))
            context = []  # (step_question, pool_path) of previous correct picks
            for t, k in enumerate(keys):
                q = steps[k]
                choices = q.get("choices") or []
                correct = [ci for ci, c in enumerate(choices) if c.get("correct")]
                if len(correct) != 1 or not (2 <= len(choices) <= len(_LETTERS)):
                    continue
                ok = True
                pool = []
                for c in choices:
                    m = "%s/%s/%s" % (section, tid, c["image"])
                    if m not in members:
                        ok = False
                        break
                    pool.append(sink.add_bytes("images/" + m, zf.read(m)))
                if not ok:
                    continue
                ctx_paths = [p for _, p in context]
                n_ctx = len(ctx_paths)
                lines = [
                    "You are completing a multi-step task, one step at a time.",
                ]
                if scene:
                    lines.append("Scenario: %s" % scene)
                if context:
                    lines.append("Steps already completed, in order:")
                    for ci, (cq, _) in enumerate(context):
                        lines.append("  Step %d: %s  (done, shown in [img%d])" % (ci + 1, cq, ci + 1))
                lines.append("Current step: %s" % q["question"].strip())
                lines.append("Which image shows the correct choice for this step?")
                lines.append("\n".join("(%s) [img%d]" % (_LETTERS[ci], n_ctx + ci + 1)
                                       for ci in range(len(choices))))
                lines.append("\nAnswer with only the correct option letter.")
                sid = "vcb_%s_%s_q%d" % (section.lower(), tid, t + 1)
                by_task.setdefault(task_name, []).append(record(
                    sid, name, "\n".join(lines), refs(ctx_paths + pool, name), task_name,
                    "mc", _LETTERS[correct[0]],
                    {"section": section, "task_id": tid, "turn": t + 1, "n_turns": len(keys),
                     "task_description": q.get("task_description", "")}))
                context.append((q["question"].strip(), pool[correct[0]]))
    write_tasks(bench, name, by_task)
    sink.close()


# ---------------------------------------------------------------------------
PREPARERS = {"mmra": prepare_mmra, "qbench2": prepare_qbench2,
             "mirb": prepare_mirb, "vischainbench": prepare_vischainbench}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all"] + sorted(PREPARERS))
    ap.add_argument("--out-root", default="data")
    a = ap.parse_args()
    for name in (sorted(PREPARERS) if a.dataset == "all" else [a.dataset]):
        print("==== %s ====" % name)
        PREPARERS[name](a.out_root)


if __name__ == "__main__":
    main()

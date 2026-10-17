#!/usr/bin/env python3
"""MosaicGym canonical JSON -> verl RLHF JSONL (tool-less GRPO smoke set).

Reads a canonical dataset (scripts/data_prep/prepare_mantis_canonical.py output)
and emits verl-ready JSONL, one training row per line:

    {
      "data_source": "nextqa",                 # selects the reward fn in verl
      "prompt": [ {role:system,...}, {role:user, content:"<image>..xN  <question>"} ],
      "images": [ {"image": "/abs/frame_0.jpg"}, ... ],   # paths, NOT bytes (readable)
      "ability": "nextqa",
      "reward_model": {"style":"rule", "ground_truth": "D"},
      "extra_info": {"index":i, "question":..., "answer":"D", "video_id":..., "num_images":N}
    }

Why JSONL + path dicts: verl's RLHFDataset reads .jsonl, and qwen_vl_utils
.fetch_image resolves a local path in {"image": "<path>"} — so we get a
human-readable dataset with images referenced on disk (already extracted).

verl's RLHFDataset._build_messages asserts (#"<image>" placeholders == #images),
so we prepend EXACTLY len(images) placeholders. --max-frames subsamples frames
(8 nextqa frames blow up image tokens on small GPUs); placeholders stay in sync.

Default (no --with-tools) is the tool-LESS smoke set. --with-tools emits the
full-toolbox variant: gym system-prompt body, [imgN] (WxH) labels, per-format
reminder, and tools_kwargs for all 10 tools.

Usage:
  python rl/data_prep/canonical_to_verl.py \
    --canonical data/mantis_instruct/nextqa/canonical/nextqa_debug.json \
    --image-base data/mantis_instruct/nextqa \
    --out-dir data/mantis_instruct/nextqa/verl \
    --max-frames 4 --val-size 16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

# Single source of truth for the tool variant: the SAME system-prompt body,
# [imgN] (WxH) label format, and per-format reminder the eval runner sends.
# The # Tools tail is deliberately absent — verl's chat template injects the
# tool signatures itself (byte-aligned with the gym's manual rendering; the
# full-fidelity schemas come from MosaicToolAgentLoop).
from mosaic_gym.agent.prompting import SYSTEM_PROMPT_BODY, format_reminder, image_token
from mosaic_gym.toolbox.tools import TOOL_CLASSES_BY_NAME

# Requires format contract B — must match src/mosaic_gym/rl/reward.py::format_reward
# (the whole reply is <think>..</think><answer><letter></answer>), else the model
# never produces the structure and the format reward is always 0.
_FORMAT_RULE = (
    "Format your reply EXACTLY as:\n"
    "<think> your step-by-step reasoning </think>\n"
    "<answer> X </answer>\n"
    "where X is the single correct option letter (A, B, C, D or E) and nothing else "
    "goes inside the <answer> tags."
)

_SYSTEM_PROMPT = (
    "You are a careful visual assistant. You are shown a sequence of video frames "
    "in order, followed by a multiple-choice question. Look closely at all the frames "
    "and reason about what happens.\n\n" + _FORMAT_RULE
)


def capped_size(w: int, h: int, max_pixels: int):
    """The (w, h) qwen_vl_utils' smart_resize will produce under ``max_pixels``
    (sqrt downscale, dims rounded to multiples of 32). Used so the [imgN] (WxH)
    label states the size the model ACTUALLY sees, not the on-disk size."""
    if max_pixels <= 0 or w * h <= max_pixels:
        return w, h
    scale = (max_pixels / (w * h)) ** 0.5
    rw = max(32, int(round(w * scale / 32)) * 32)
    rh = max(32, int(round(h * scale / 32)) * 32)
    return rw, rh


def stride_indices(n: int, k: int):
    """K indices spanning [0, n) evenly (first and last frame always kept).

    Replaces the old prefix truncation (paths[:k]), which silently dropped the
    tail of the clip and made temporal questions unanswerable."""
    if k <= 0 or k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def canonical_to_rows(canonical, image_base: Path, data_source: str, max_frames: int, with_tools: bool = False,
                      image_max_pixels: int = 0, image_token_budget: int = 0):
    """Yield one verl row per canonical sample."""
    system_prompt = SYSTEM_PROMPT_BODY if with_tools else _SYSTEM_PROMPT
    for idx, s in enumerate(canonical):
        # Resolve frame paths to ABSOLUTE (verl opens them at rollout time from an
        # unknown cwd), optionally stride-subsampling to ``max_frames``.
        paths = [str((image_base / img["uri_or_path"]).resolve()) for img in s["images"]]
        if max_frames:
            paths = [paths[i] for i in stride_indices(len(paths), max_frames)]
        n = len(paths)

        target = s.get("target") or {}
        gt = target.get("value") if isinstance(target, dict) else target
        answer_format = (target.get("format", "mc") if isinstance(target, dict) else "mc")
        gt = str(gt or "").strip()
        if answer_format == "mc":
            gt = gt.upper()

        question = str(s.get("prompt", "")).strip()
        image_dicts = [{"image": p} for p in paths]
        if with_tools:
            # Mirror the gym's first-turn user message byte-for-byte: the
            # question + per-format reminder as the lead text, then every frame
            # as a `[imgN] (WxH)` label immediately followed by its image (the
            # chat template concatenates adjacent segments with no separator,
            # exactly like the gym's separate content blocks flatten).
            lead = question + format_reminder(answer_format)
            sizes = []
            for p in paths:
                with Image.open(p) as im:
                    sizes.append(im.size)
            # Per-image cap: the global --image-max-pixels, tightened per ROW when
            # the row's total image tokens (~pixels/32^2 for Qwen3-VL) would blow
            # --image-token-budget — e.g. VSI rows with 32 frames of 1408x1408.
            cap = image_max_pixels
            if image_token_budget:
                est = sum(min(w * h, cap) if cap else w * h for w, h in sizes) // 1024
                if est > image_token_budget:
                    row_cap = image_token_budget * 1024 // n
                    cap = min(cap, row_cap) if cap else row_cap
            labels = []
            for i, (w, h) in enumerate(sizes):
                if cap and w * h > cap:
                    # qwen_vl_utils downsizes this image at load time; label the
                    # size the model will actually see and tell fetch_image the cap.
                    image_dicts[i]["max_pixels"] = cap
                    w, h = capped_size(w, h, cap)
                labels.append("[%s] (%dx%d)" % (image_token(i), w, h))
            user_content = lead + "".join("%s<image>" % lab for lab in labels)
        else:
            user_content = "%s\n%s" % ("<image>" * n, question)

        extra_info = {
            "index": idx,
            "question": question,
            "answer": gt,
            "answer_format": answer_format,   # mosaic_gym.rl.reward dispatches on this
            "video_id": (s.get("metadata") or {}).get("video_id"),
            "num_images": n,
        }
        if with_tools:
            # Declare the full toolbox. No images in create_kwargs — tools read
            # frames from the live pool (agent_data.image_data); verl does not
            # gate calls on these entries, they only supply create_kwargs.
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {name: {"create_kwargs": {}} for name in sorted(TOOL_CLASSES_BY_NAME)}

        yield {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "images": image_dicts,
            "ability": data_source,
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": extra_info,
        }


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", default="data/mantis_instruct/nextqa/canonical/nextqa_debug.json")
    ap.add_argument("--image-base", default="data/mantis_instruct/nextqa",
                    help="Dir the canonical uri_or_path is relative to.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: .../verl, or .../verl_tools with --with-tools).")
    ap.add_argument("--data-source", default="nextqa")
    ap.add_argument("--max-frames", type=int, default=0, help="Stride-subsample to K frames spanning the clip (0 = all).")
    ap.add_argument("--val-size", type=int, default=16, help="Rows carved off for val.jsonl.")
    ap.add_argument("--image-token-budget", type=int, default=0,
                    help="Per-row image-token budget (~pixels/1024 each); rows over it get a tighter per-image max_pixels. 0 = off.")
    ap.add_argument("--image-max-pixels", type=int, default=0,
                    help="Per-image pixel cap written into the image dicts (qwen_vl_utils downsizes at load); 0 = off.")
    ap.add_argument("--with-tools", action="store_true",
                    help="Emit the full-toolbox variant (gym prompts, [imgN] labels, all-tools tools_kwargs).")
    args = ap.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = "data/mantis_instruct/nextqa/" + ("verl_tools" if args.with_tools else "verl")

    canonical = json.load(open(args.canonical, encoding="utf-8"))
    rows = list(canonical_to_rows(canonical, Path(args.image_base), args.data_source, args.max_frames, args.with_tools, args.image_max_pixels, args.image_token_budget))

    # Sanity: every row's placeholder count must equal its image count (verl asserts this).
    for r in rows:
        n_ph = r["prompt"][-1]["content"].count("<image>")
        assert n_ph == len(r["images"]), "placeholder/image mismatch: %d vs %d" % (n_ph, len(r["images"]))

    out = Path(args.out_dir)
    val = rows[: args.val_size]
    train = rows[args.val_size:] if args.val_size else rows
    n_tr = write_jsonl(out / "train.jsonl", train)
    n_va = write_jsonl(out / "val.jsonl", val) if val else 0

    print("wrote %d train + %d val -> %s" % (n_tr, n_va, out))
    print("  frames/sample: %s" % (args.max_frames or "all(8)"))
    print("  image paths are absolute; verl reads via {\"image\": path}")
    if train:
        ex = train[0]
        print("  example user content head:", repr(ex["prompt"][-1]["content"][:80]))
        print("  example ground_truth:", ex["reward_model"]["ground_truth"], "| #images:", len(ex["images"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

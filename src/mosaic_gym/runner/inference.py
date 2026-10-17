"""Inference runner — drive the agent over a dataset, config-first.

ONE entry: read a YAML config (data / model / agent / runner sections), build the
Toolbox + Agent once, loop the data config's tasks, and for each run every sample
through ``run_episode`` (the rollout loop), writing ``record/<task>.jsonl`` (+ the
SFT ``messages/`` with ``save_messages``).

    python -m mosaic_gym.runner.inference --config configs/qwen3vl_mmiu.yaml

The rollout loop owns what the toolbox does NOT: it calls the agent, decides
tool-vs-final from the parsed reply, runs the chosen tool calls through the
toolbox, loops until a final answer / parse error / ``max_steps``, and scores the
answer into a reward.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from mosaic_gym.agent import Agent
from mosaic_gym.agent.parsing import ParsedReply
from mosaic_gym.config import RunConfig, load_config
from mosaic_gym.contracts import ToolCall, ToolResult
from mosaic_gym.runner.data import load_samples, sample_to_case
from mosaic_gym.runner.scoring import answer_reward
from mosaic_gym.agent.prompting import VANILLA_SYSTEM_PROMPT
from mosaic_gym.toolbox import Toolbox


# ---- build the agent from config -----------------------------------------
def build_agent(model_cfg, agent_cfg, tools, endpoint: str = "", api_key: str = "") -> Agent:
    """Wire the model + agent config into a live Agent (config stays pure data)."""
    return Agent(
        endpoint=endpoint or model_cfg.endpoint,
        model=model_cfg.name,
        available_tools=tools,
        api_key=api_key or os.environ.get(model_cfg.api_key_env, ""),
        sampling=dict(agent_cfg.sampling),
        extra_body_keys=tuple(agent_cfg.extra_body_keys),
        timeout_sec=agent_cfg.timeout_sec,
    )


# ---- one episode ----------------------------------------------------------
def _dump_calls(tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
    return [
        {"name": tc.name, "arguments": tc.arguments, "target_images": tc.target_images}
        for tc in tool_calls
    ]


def _dump_results(results: List[ToolResult]) -> List[Dict[str, Any]]:
    """One entry per tool call this turn (``error`` is None on success, else
    {code, message}). PIL images are not serialised — ``new_image_ids`` refer to
    them in the image store."""
    return [
        {
            "tool_name": r.tool_name,
            "tool_feedback": r.tool_feedback,
            "new_image_ids": r.new_image_ids,
            "error": r.error,
        }
        for r in results
    ]


def run_episode(
    toolbox: Toolbox,
    agent: Agent,
    question: str,
    images: List[Any],
    ground_truth: Optional[str] = None,
    answer_format: str = "mc",
    env_mode: str = "tool",
    max_steps: int = 16,
) -> Dict[str, Any]:
    """Drive one episode and return its record (one ``round`` per turn). Each round
    records the model's raw reply, the parsed tool calls (or the final answer /
    parse error), and the per-call tool results. The full conversation for SFT
    export stays on ``agent._conversation``."""
    agent.reset(question, answer_format=answer_format, env_mode=env_mode)
    toolbox.reset(images=images, prompt=question)

    rounds: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None
    parse_error: Optional[str] = None
    run_error: Optional[str] = None
    tool_results: Optional[List[ToolResult]] = None  # None on the first turn (sends the question)
    done = False
    step = 0

    while not done and step < max_steps:
        step += 1
        # The LAST allowed step (step == max_steps) is an answer-only turn: tools
        # are off and it carries the answer reminder. Earlier steps allow tools and
        # use the normal format reminder. So total turns never exceed max_steps and
        # the answer reminder is shown exactly once.
        is_last_step = step == max_steps
        parsed: ParsedReply
        try:
            parsed, raw_reply = agent.respond(
                toolbox.image_ids(), toolbox.images_by_id(), tool_results, is_last_step=is_last_step)
        except Exception as exc:
            # The model call failed (e.g. a 400 context-overflow). Record WHICH step
            # it died on — the bare run_error record otherwise has no rounds — then stop.
            run_error = "%s: %s" % (type(exc).__name__, exc)
            rounds.append({
                "step": step, "model_raw_response": "", "tool_calls": None,
                "final_answer": None, "parse_error": None, "tool_results": [],
                "run_error": run_error,
            })
            break

        round_rec: Dict[str, Any] = {
            "step": step,
            "model_raw_response": raw_reply,
            "tool_calls": None,
            "final_answer": None,
            "parse_error": None,
            "tool_results": [],
        }

        if parsed.parse_error:
            parse_error = parsed.parse_error
            round_rec["parse_error"] = parse_error
            rounds.append(round_rec)
            break

        if parsed.final_answer is not None or not parsed.tool_calls:
            final_answer = parsed.final_answer if parsed.final_answer is not None else ""
            round_rec["final_answer"] = final_answer
            rounds.append(round_rec)
            done = True
            break

        if is_last_step:
            # Answer-only last turn but the model still called a tool: IGNORE it
            # (do not run it) and end at max_steps with no answer. tool_calls stays
            # None on the record; the raw reply still shows what it tried.
            rounds.append(round_rec)
            break

        tool_results = toolbox.run(parsed.tool_calls)
        round_rec["tool_calls"] = _dump_calls(parsed.tool_calls)
        round_rec["tool_results"] = _dump_results(tool_results)
        rounds.append(round_rec)

    reason = (
        "run_error" if run_error is not None
        else "parse_error" if parse_error is not None
        else "final_answer" if final_answer is not None
        else "max_steps"
    )
    # No answer (final_answer None/"") scores 0.0 — failing to answer is wrong, not
    # unscorable. reward is None only when the sample has no ground truth.
    reward = answer_reward(final_answer, ground_truth, answer_format) if ground_truth is not None else None
    tool_use_count = sum(len(r["tool_calls"]) for r in rounds if r["tool_calls"])
    record: Dict[str, Any] = {
        "reward": reward,
        "tool_use_count": tool_use_count,        # total tool calls across the episode
        "question": question,
        "num_input_images": len(images),
        "ground_truth": ground_truth,
        "final_answer": final_answer if final_answer is not None else "",
        "termination_reason": reason,
        "num_steps": len(rounds),
        "usage": {
            "prompt_tokens": getattr(agent, "prompt_tokens", 0),
            "completion_tokens": getattr(agent, "completion_tokens", 0),
        },
        "rounds": rounds,
    }
    if run_error is not None:
        record["run_error"] = run_error          # also flags it for the eval scorers
    return record


# ---- SFT messages export --------------------------------------------------
def build_sft_messages(conversation, sample_id, messages_dir):
    """Convert the live conversation (base64 image_url blocks) into a DeepEyes-style
    messages list — content strings with ``<image>`` placeholders — and dump each
    image to ``messages_dir/images/<sample_id>/imgN.png``. Returns (messages,
    image_paths) with paths relative to messages_dir (a portable SFT set).

    The PNGs are the exact bytes the model was sent — the model-facing DISPLAY
    rendering (transparency already flattened onto the checkerboard by
    ``data_url``), not the RGBA source images in ``ImageState``. Deliberate:
    SFT inputs must match what the model actually saw. If an RGBA-source
    export is ever needed, add a separate channel; do not repoint this one."""
    img_dir = os.path.join(messages_dir, "images", sample_id)
    os.makedirs(img_dir, exist_ok=True)
    messages, image_paths, n = [], [], 0
    for msg in conversation:
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append({"role": msg["role"], "content": content})
            continue
        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "image_url":
                n += 1
                fname = "img%d.png" % n
                b64 = (block.get("image_url") or {}).get("url", "").split(",", 1)[-1]
                with open(os.path.join(img_dir, fname), "wb") as fh:
                    fh.write(base64.b64decode(b64))
                image_paths.append("images/%s/%s" % (sample_id, fname))
                parts.append("<image>")
        messages.append({"role": msg["role"], "content": "".join(parts)})
    return messages, image_paths


def _sample_has_error(result: Dict[str, Any]) -> bool:
    for r in result.get("rounds", []):
        if r.get("parse_error"):
            return True
        if any(tr.get("error") for tr in (r.get("tool_results") or [])):
            return True
    return False


# ---- one task -------------------------------------------------------------
def run_task(task: str, dataset_path: str, toolbox: Toolbox, agent: Agent, cfg: RunConfig) -> dict:
    """Run every (up to max_samples) sample of one task, writing record/<task>.jsonl
    (+ messages/<task>.jsonl when save_messages). Returns a per-task summary."""
    rc, dc = cfg.runner, cfg.data
    record_dir = os.path.join(rc.out_dir, "record")
    os.makedirs(record_dir, exist_ok=True)
    record_path = os.path.join(record_dir, task + ".jsonl")

    done_ids = set()
    if rc.resume and os.path.exists(record_path):
        with open(record_path) as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass  # half-written line from an earlier crash — ignore
        print("[resume] %s: %d samples already done, skipping them" % (task, len(done_ids)))

    mode = "a" if rc.resume else "w"
    record_fh = open(record_path, mode)
    messages_fh = None
    messages_dir = ""
    if rc.save_messages:
        messages_dir = os.path.join(rc.out_dir, "messages")
        os.makedirs(messages_dir, exist_ok=True)
        messages_fh = open(os.path.join(messages_dir, task + ".jsonl"), mode)

    n = n_err = tot_in = tot_out = 0
    try:
        for sample in load_samples(dataset_path):
            if dc.max_samples is not None and n >= dc.max_samples:
                break
            n += 1
            sample_id = str(sample.get("sample_id", "?"))
            if sample_id in done_ids:
                continue
            messages_line = None
            try:
                case = sample_to_case(sample, base_dir=dc.base_dir)
                sample_id = case.sample_id
                images = [_open_image(p, dc.base_dir).convert("RGBA") for p in case.image_paths]
                ground_truth = case.target.get("value") if isinstance(case.target, dict) else case.target
                answer_format = (case.target.get("format", "mc") if isinstance(case.target, dict) else "mc")
                result = run_episode(toolbox, agent, case.prompt, images,
                                     ground_truth=ground_truth,
                                     max_steps=1 if rc.env == "vanilla" else rc.max_steps,
                                     answer_format=answer_format, env_mode=rc.env)
                if _sample_has_error(result):
                    n_err += 1
                record = {"sample_id": case.sample_id, "task": task, **result}
                usage = result.get("usage") or {}
                tot_in += int(usage.get("prompt_tokens", 0) or 0)
                tot_out += int(usage.get("completion_tokens", 0) or 0)
                if messages_fh is not None:
                    msgs, image_paths = build_sft_messages(agent._conversation, case.sample_id, messages_dir)
                    messages_line = {"sample_id": case.sample_id, "task": task, "messages": msgs, "images": image_paths}
                print("[%s %d] %s rounds=%d reward=%s -> %r" % (
                    task, n, case.sample_id, result["num_steps"], result["reward"],
                    (result["final_answer"] or "")[:50]))
            except Exception as exc:
                n_err += 1
                record = {"sample_id": sample_id, "task": task, "run_error": "%s: %s" % (type(exc).__name__, exc)}
                print("[%s %d] %s RUN ERROR: %s" % (task, n, sample_id, str(exc)[:200]))
            record_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            record_fh.flush()
            if messages_fh is not None and messages_line is not None:
                messages_fh.write(json.dumps(messages_line, ensure_ascii=False) + "\n")
                messages_fh.flush()
    finally:
        record_fh.close()
        if messages_fh is not None:
            messages_fh.close()

    cost = tot_in / 1e6 * rc.price_in_per_m + tot_out / 1e6 * rc.price_out_per_m
    return {"task": task, "num_samples": n, "samples_with_errors": n_err,
            "prompt_tokens": tot_in, "completion_tokens": tot_out, "est_cost_usd": round(cost, 4)}


# ---- the whole run --------------------------------------------------------
_NON_TASK_FILES = {"manifest"}  # data/mmiu/datasets/manifest.json is metadata, not a task


def _ensure_images_extracted(base_dir: str) -> None:
    """Benchmark images ship as <base_dir>/images.tar to stay under the cluster
    file-count quota; extract on first use if images/ is absent.  A benchmark
    whose extracted tree would itself blow the quota (MMIU: ~77k files) ships a
    <base_dir>/.tar_direct marker instead: nothing is extracted and images are
    read straight out of the tar (see _open_image)."""
    if not base_dir:
        return
    if os.path.isfile(os.path.join(base_dir, ".tar_direct")):
        return
    img_dir = os.path.join(base_dir, "images")
    tarball = os.path.join(base_dir, "images.tar")
    if not os.path.isdir(img_dir) and os.path.isfile(tarball):
        import tarfile
        print("[data] extracting %s -> %s" % (tarball, img_dir))
        with tarfile.open(tarball) as tf:
            tf.extractall(base_dir)


_TAR_POOLS: Dict[str, tuple] = {}


def _open_image(path: str, base_dir: str = "") -> Image.Image:
    """Image.open with an images.tar fallback: if the file is not on disk but
    <base_dir>/images.tar holds a member at the base_dir-relative path, read it
    from the tar (index built once per process). Keeps huge benchmarks fully
    tar-resident — no extraction, no file-count quota pressure."""
    if os.path.exists(path) or not base_dir:
        return Image.open(path)
    tarball = os.path.join(base_dir, "images.tar")
    if os.path.isfile(tarball):
        pool = _TAR_POOLS.get(tarball)
        if pool is None:
            import tarfile
            tf = tarfile.open(tarball)
            pool = (tf, {m.name: m for m in tf.getmembers() if m.isfile()})
            _TAR_POOLS[tarball] = pool
            print("[data] tar-direct: indexed %d members in %s" % (len(pool[1]), tarball))
        tf, idx = pool
        rel = os.path.relpath(path, base_dir)
        member = idx.get(rel) or idx.get(rel.replace(os.sep, "/"))
        if member is not None:
            import io as _io
            return Image.open(_io.BytesIO(tf.extractfile(member).read()))
    return Image.open(path)  # raises the usual FileNotFoundError


def run_inference(cfg: RunConfig, endpoint: str = "", api_key: str = "") -> List[dict]:
    """Build the Toolbox + Agent once, then run the data config's tasks. (Sharding
    across jobs is the sbatch's job — it passes this shard's tasks via --tasks.)"""
    _ensure_images_extracted(cfg.data.base_dir)
    toolbox = Toolbox(tool_names=cfg.toolbox.tools)   # None -> all 10 tools; a list -> ablation subset
    tools = toolbox.registry.list_tools(cfg.toolbox.description_mode)   # baseline | prompting
    print("[tools] %d active: %s" % (len(tools), ", ".join(t["name"] for t in tools)))
    agent = build_agent(cfg.model, cfg.agent, tools, endpoint=endpoint, api_key=api_key)
    if cfg.runner.env == "vanilla":
        # no-tool baseline: same loop, single answer-only turn, tool-free prompt
        agent.system_prompt = VANILLA_SYSTEM_PROMPT

    # Empty data.tasks -> run EVERY <task>.json in dataset_dir (minus metadata files).
    tasks = list(cfg.data.tasks)
    if not tasks:
        tasks = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(cfg.data.dataset_dir, "*.json")))
        tasks = [t for t in tasks if t not in _NON_TASK_FILES]
        print("[tasks] none listed -> %d tasks in %s" % (len(tasks), cfg.data.dataset_dir))

    summaries = []
    for task in tasks:
        dataset_path = os.path.join(cfg.data.dataset_dir, task + ".json")
        if not os.path.exists(dataset_path):
            print("!! missing %s — skipping" % dataset_path)
            continue
        print("==================== %s (n=%s) ====================" % (task, cfg.data.max_samples))
        summaries.append(run_task(task, dataset_path, toolbox, agent, cfg))
    print("\n=== run summary ===")
    print(json.dumps({"out_dir": cfg.runner.out_dir, "tasks": summaries}, ensure_ascii=False, indent=2))
    return summaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the agent over MMIU tasks from a config.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the run config YAML")
    parser.add_argument("--env", default=None, choices=["tool", "vanilla"],
                        help="Override runner.env: 'tool' = agent loop with tools, 'vanilla' = single call, no tools")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name; sets data.dataset_dir=data/<NAME>/datasets, "
                             "data.base_dir=data/<NAME> and (unless --out-dir) runner.out_dir=outputs/<NAME>")
    # Optional overrides (the sbatch passes the dynamic vLLM endpoint; everything
    # else lives in the config). None -> use the config's value.
    parser.add_argument("--endpoint", default="", help="Override model.endpoint (vLLM's node is dynamic)")
    parser.add_argument("--model-name", default=None,
                        help="Override model.name (the vLLM --served-model-name); lets one config serve every checkpoint")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--tasks", nargs="*", default=None, help="Override data.tasks")
    parser.add_argument("--max-samples", type=int, default=None, help="Override data.max_samples")
    parser.add_argument("--out-dir", default=None, help="Override runner.out_dir")
    parser.add_argument("--max-steps", type=int, default=None, help="Override runner.max_steps")
    parser.add_argument("--description-mode", default=None, help="Override toolbox.description_mode (baseline|prompting)")
    parser.add_argument("--tools", nargs="*", default=None,
                        help="Override toolbox.tools (tool-ablation subset; omit for all). "
                             "e.g. --tools crop make_collage")
    parser.add_argument("--save-messages", action="store_true", help="Override runner.save_messages -> True")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.env:
        cfg.runner.env = args.env
    if args.dataset:
        cfg.data.dataset_dir = "data/%s/datasets" % args.dataset
        cfg.data.base_dir = "data/%s" % args.dataset
        if args.out_dir is None:
            cfg.runner.out_dir = "outputs/%s" % args.dataset
    if not cfg.data.dataset_dir:
        parser.error("no dataset selected: pass --dataset <name> or set data.dataset_dir in the config")
    if args.tasks is not None:
        cfg.data.tasks = args.tasks
    if args.model_name:
        cfg.model.name = args.model_name
    if args.max_samples is not None:
        cfg.data.max_samples = args.max_samples
    if args.out_dir is not None:
        cfg.runner.out_dir = args.out_dir
    if args.max_steps is not None:
        cfg.runner.max_steps = args.max_steps
    if args.description_mode:
        cfg.toolbox.description_mode = args.description_mode
    if args.tools is not None:
        cfg.toolbox.tools = args.tools
    if args.save_messages:
        cfg.runner.save_messages = True
    run_inference(cfg, endpoint=args.endpoint, api_key=args.api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

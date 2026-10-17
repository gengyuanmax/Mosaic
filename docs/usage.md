# Mosaic usage guide

MosaicGym trains and evaluates vision-language agents that reason by
**re-representing images**: the model works inside an image *workspace* with ten
deterministic tools (crop, rotate, flip, resize, collage, overlay, per-pixel
difference, homography / affine transforms, coordinate grid), interleaving
chain-of-thought with tool calls and committing a final `<answer>`.

The repo contains three things:

| Part | Where | What |
|---|---|---|
| **Environment** | `src/mosaic_gym/` | the toolbox (validators, deterministic execution, image registry), the agent protocol (prompting / parsing), and the evaluation runner |
| **Training** | `rl/` | GRPO training on top of a vendored [verl](https://github.com/volcengine/verl) v0.7.1 (our tool/agent-loop/reward adapters live in `src/mosaic_gym/rl/`, verl itself is unmodified vendor code) |
| **Data prep** | `scripts/data_prep/` | converters that turn each public benchmark (BLINK, MMIU, Mantis-Eval, M4Bench, MuirBench, …) into one canonical format |

Reward is deliberately minimal: `score = accuracy + format` (both binary; see
`src/mosaic_gym/rl/reward.py`). No tool-use term — the accuracy gradient alone
decides whether the agent uses the workspace.

---

## 1. Environment setup

### Option A: pip / uv (evaluation only)

```bash
# Python >= 3.10
pip install -e .            # or: uv sync
```

This installs the evaluation side (toolbox + runner). Training additionally
needs the verl container (Option B/C).

### Option B: Docker (evaluation service)

```bash
docker build -t mosaic-gym .
```

### Option C: Apptainer container (training)

Training runs inside the official verl v0.7.1 image (torch 2.10 + cu129,
SGLang 0.5.9 engine):

```bash
bash rl/scripts/build_container.sh          # builds rl/containers/verl-v0.7.1-sgl059.sif
rl/scripts/verl.sh python -c "import verl"  # smoke-check the wrapper
```

`rl/scripts/verl.sh` binds the repo into the container and puts the vendored
`rl/verl` plus `src/` on `PYTHONPATH`. Override `SIF=` / `WS=` if your image or
data live elsewhere.

## 2. Data preparation

Every dataset is converted once into the same canonical layout:

```
data/<name>/
├── datasets/<task>.json     # one file per sub-task: sample_id, prompt, images[], target
└── images.tar               # all images in one tar (the runner auto-extracts
                             #   to local /tmp — friendly to file-count quotas)
```

Converters (see each file's header for the exact source it downloads):

```bash
python scripts/data_prep/prepare_hf_benchmarks.py      # BLINK, M4Bench, ... from HF hub
python scripts/data_prep/prepare_mantis_canonical.py   # Mantis-Eval
python scripts/data_prep/prepare_mmiu_canonical.py     # MMIU (json)
python scripts/data_prep/prepare_mmiu_tar.py           # MMIU (images.tar)
python scripts/data_prep/prepare_extra_benchmarks.py   # MuirBench, QBench2, ...
```

For RL training, additionally convert the canonical training split into verl's
chat-jsonl format:

```bash
python rl/data_prep/canonical_to_verl.py --help
```

## 3. Running the main experiment (evaluation)

Serve the model with any OpenAI-compatible server, e.g. vLLM:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 \
  --max-model-len 98304 --limit-mm-per-prompt '{"image": 64}'
```

Then run the agent over a benchmark:

```bash
# Tool workspace (Visual CoT) — the main setting
./scripts/run.sh --dataset blink --env tool

# Text-only chain-of-thought baseline (no tools)
python -m mosaic_gym.runner.inference \
  --config configs/default_vanilla.yaml --dataset blink --env vanilla
```

Outputs land in `outputs/<run>/record/<task>.jsonl`, one line per sample with
the full multi-turn trace (raw responses, tool calls, tool results, reward).
Accuracy = mean of `reward >= 1`. Runs are resumable: re-running the same
command skips already-recorded samples.

Sampling defaults (temperature 0.7 / top_p 0.8 / top_k 20, model-card values)
and the serving profile live in `configs/default.yaml` /
`configs/default_vanilla.yaml`; the toolset can be restricted per run
(e.g. `--tools crop`).

## 4. Training

One script, one reward:

```bash
# inside the container; 1 node x 4 GPUs (H100-class)
rl/scripts/verl.sh bash rl/example/train.sh
```

Key hyper-parameters (all in `rl/example/train.sh`): GRPO with group
size n=8, no KL/entropy terms, lr 1e-6, batch 32 / mini-batch 16 /
micro-batch 1, ppo_epochs 2, async vLLM rollout with the multi-turn tool agent
loop (max 10 assistant turns), prompt/response budgets 16384/20480 within a
36864-token context, up to 64 images per rollout. Checkpoints are written to
`rl/train/outputs/<experiment>/global_step_N/` every 10 steps; merge FSDP
shards to a HF checkpoint with verl's `verl.model_merger`.

The agent loop and toolbox bridge that verl loads are configured in
`rl/configs/agent_loop/mosaic_agent_loop.yaml` and
`rl/configs/tool_config/mosaic_tools.yaml` (regenerate the latter with
`rl/scripts/gen_tool_config.py` after changing the toolset).

## 5. Repository layout

```
src/mosaic_gym/
├── toolbox/          the ten image tools + validator + deterministic executor
├── agent/            system prompts, reply parsing (<tool_call>/<answer>)
├── runner/           evaluation runner, dataset loading, scoring
├── rl/               verl adapters: agent loop, tool bridge, reward (acc+fmt)
├── config/           typed run configuration
└── contracts.py      shared dataclasses (ToolCall, ToolResult, ...)
rl/
├── verl/             vendored verl v0.7.1 (unmodified)
├── example/train.sh
├── scripts/          container build + run wrappers, tool-config generator
├── configs/          agent-loop / tool-config YAML
└── data_prep/        canonical -> verl jsonl
scripts/
├── run.sh            one-line eval entry point
└── data_prep/        per-benchmark converters -> canonical format
configs/              default.yaml (tool) / default_vanilla.yaml (no-tool)
```

## License

MIT — see [LICENSE](../LICENSE).

## MosaicBench on Hugging Face

[Dataset and source-specific licenses](https://huggingface.co/datasets/gengyuanmax/MosaicBench) · [Paper](https://arxiv.org/abs/2609.39363)

```bash
pip install huggingface_hub
hf download gengyuanmax/MosaicBench --repo-type dataset --local-dir data/MosaicBench
python scripts/data_prep/prepare_mosaicbench.py data/MosaicBench data/mosaicbench
mosaic-run --dataset mosaicbench --max-samples 1 --endpoint http://localhost:8000/v1
```

The converter keeps the original prompts, ordered choices and image references. It prepares 540 samples with distributed images and writes a coverage report listing the 20 COCO samples without images. These 540 samples are a subset, not the complete 560-sample benchmark. COCO sources and acquisition information remain in the dataset repository. No images are downloaded by the converter.

Both released checkpoint candidates passed BF16 loading and two-image generation on an RTX A6000. MosaicAgent also completed an actual crop → generated-image feedback → final-answer episode. These smoke checks do not reproduce the paper’s benchmark results. GPU training and full paper-result reproduction have not been revalidated for this release.

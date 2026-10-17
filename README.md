# Mosaic

### Rethinking Multi-Image Re-Representation in Multi-Image Understanding

**Gengyuan Zhang · Xiao Han · Xinyu Xie · Tong Liu · Volker Tresp**

[Project page](https://gengyuanmax.github.io/Mosaic/) · [Paper](https://arxiv.org/abs/2609.39363) · [MosaicAgent-8B](https://huggingface.co/gengyuanmax/MosaicAgent-8B) · [MosaicBench](https://huggingface.co/datasets/gengyuanmax/MosaicBench) · [Usage guide](docs/usage.md)

Mosaic gives vision-language models an image workspace: they can crop, rotate, align, combine and compare images, then use the resulting visual evidence to answer a question. MosaicAgent-8B learns to use this workspace through reinforcement learning with accuracy and format rewards.

## Release

| Resource | Description | Access |
| --- | --- | --- |
| MosaicGym | Image workspace, ten tools, agent protocol and evaluation runner | This repository |
| MosaicAgent-8B | Qwen3-VL-based model trained with image tools | [Hugging Face](https://huggingface.co/gengyuanmax/MosaicAgent-8B) |
| MosaicLLM-8B | No-tool, single-turn comparison model | Upload in progress |
| MosaicBench | 32 task types described in the paper; public download includes 560 evaluation samples with questions, ordered candidate answers and ground truth | [Hugging Face](https://huggingface.co/datasets/gengyuanmax/MosaicBench) |

MosaicBench distributes 780 images covering 540 samples. The 20 COCO-derived samples retain source links and filenames but do not include images. The complete benchmark definition and the subset runnable with bundled images are distinct. See the dataset card for source-specific licenses and acquisition details.

Training adapters are included; training data is not part of this release.

## Quick start

### 1. Install

```bash
git clone https://github.com/gengyuanmax/Mosaic.git
cd Mosaic
python -m pip install -e .
python -m pip install huggingface_hub
```

Python 3.10 or newer is required. Model serving needs a separate Qwen3-VL-compatible environment and sufficient GPU memory.

### 2. Prepare MosaicBench

```bash
hf download gengyuanmax/MosaicBench --repo-type dataset --local-dir data/MosaicBench
python scripts/data_prep/prepare_mosaicbench.py data/MosaicBench data/mosaicbench
```

This preserves prompts, choices, answers and image order, prepares the 540 samples with bundled images, and writes a coverage report for the 20 excluded samples. It does not download COCO images or change the benchmark split.

### 3. Serve the model and run an example

In a compatible vLLM environment:

```bash
vllm serve gengyuanmax/MosaicAgent-8B --dtype bfloat16 --port 8000
```

Then, from this repository:

```bash
mosaic-run --dataset mosaicbench --env tool \
  --model-name gengyuanmax/MosaicAgent-8B \
  --endpoint http://localhost:8000/v1 --max-samples 1
```

Set the serving context length and image limits for your hardware and workload. The explicit BF16 setting matches the stored checkpoint tensors. The runner executes model-generated tool calls and returns the resulting images to the model; plain text generation alone does not execute the workspace.

Outputs are written to `outputs/mosaicbench/record/`. Remove `--max-samples 1` to run the prepared subset; report its coverage when presenting results.

## What is included

- `src/mosaic_gym/toolbox/`: ten image operations and image-state management.
- `src/mosaic_gym/agent/`: prompting, tool-call parsing and conversation management.
- `src/mosaic_gym/runner/`: dataset loading, multi-turn evaluation and scoring.
- `src/mosaic_gym/rl/` and `rl/`: training adapters and vendored verl.
- `scripts/data_prep/`: benchmark converters, including the HF MosaicBench bridge.

See the [usage guide](docs/usage.md) for configuration, other benchmarks, containers and training-code setup.

## Validation

Both checkpoint candidates passed real BF16 loading and two-image generation on an NVIDIA RTX A6000. MosaicAgent completed an actual crop call, received the generated image and returned a final answer. These limited functional checks are not a reproduction of the paper’s evaluation results. Hugging Face download verification is tracked separately from the initial checkpoint checks.

For CPU protocol checks:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests
```

## Citation

```bibtex
@misc{zhang2026mosaic,
  title={Rethinking Multi-Image Re-Representation in Multi-Image Understanding},
  author={Gengyuan Zhang and Xiao Han and Xinyu Xie and Tong Liu and Volker Tresp},
  year={2026},
  eprint={2609.39363},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.39363}
}
```

## License

Mosaic code uses the [MIT license](LICENSE). Vendored verl retains its [Apache 2.0 license](rl/verl/LICENSE) and attribution. Model and dataset licenses are documented in their respective Hugging Face repositories.

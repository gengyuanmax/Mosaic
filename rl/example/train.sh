#!/bin/bash
# GRPO training of the Mosaic agent inside the image workspace.
#
# Reward: acc + format (compute_score_accfmt in src/mosaic_gym/rl/reward.py).
# Geometry: 1 node x 4 GPUs (H100-class, >=80GB). Run inside the training
# container via rl/scripts/verl.sh, e.g.
#
#   rl/scripts/verl.sh bash rl/example/train.sh
#
# Data: produce data/mosaic_train/verl/train.jsonl (and the
# validation file) with rl/data_prep/canonical_to_verl.py first.

set -euo pipefail

# Some SLURM sites export ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES
# and verl refuses to start with both set.
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WANDB_ENTITY="${WANDB_ENTITY:-YOUR_WANDB_ENTITY}"
# Allocator-only knob (no effect on numerics): reclaims reserved-but-fragmented
# memory that can push update_actor over the edge on long multi-image sequences.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="$PROJECT_ROOT/data/mosaic_train/verl"
REWARD="${REWARD:-compute_score_accfmt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mosaic_${REWARD#compute_score_}}"
# ONE dir per run: verl writes checkpoints (global_step_N/) + rollout/val dumps here.
OUTPUT_DIR="$PROJECT_ROOT/rl/train/outputs/$EXPERIMENT_NAME"

echo "[train] data=$DATA_DIR out=$OUTPUT_DIR"

# Memory notes for the 4x H100 (94GB) single-node geometry:
#   * tensor_model_parallel_size=1 — the 8B model fits a single GPU; four
#     independent rollout engines beat TP2 on throughput.
#   * optimizer_offload=True + actor param_offload=True — a 36k-token
#     multi-image sequence needs ~10GB of update-phase headroom that sharded
#     Adam states would otherwise occupy.
#   * max_prompt_length=16384 — samples carry up to 32 images under an 11k
#     image-token budget; 8192 would filter them all out.
#   * ulysses_sequence_parallel_size=4 halves activation peaks on the 4-GPU
#     FSDP geometry.
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_DIR/train.jsonl" \
  data.val_files="$PROJECT_ROOT/data/mosaic_validation/verl/val.jsonl" \
  data.val_batch_size=128 \
  data.train_batch_size=32 \
  data.max_prompt_length=16384 \
  data.max_response_length=20480 \
  data.filter_overlong_prompts=True \
  data.return_multi_modal_inputs=False \
  actor_rollout_ref.model.path=Qwen/Qwen3-VL-8B-Instruct \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_epochs=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  +actor_rollout_ref.rollout.limit_images=64 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.rollout.enable_sleep_mode=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_model_len=36864 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.36 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_ROOT/rl/configs/tool_config/mosaic_tools.yaml" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=10 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_ROOT/rl/configs/agent_loop/mosaic_agent_loop.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=mosaic_tool_agent \
  reward.custom_reward_function.path="$PROJECT_ROOT/src/mosaic_gym/rl/reward.py" \
  reward.custom_reward_function.name="$REWARD" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=mosaic_rl \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=3 \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.rollout_data_dir="$OUTPUT_DIR/rollout_data" \
  trainer.validation_data_dir="$OUTPUT_DIR/val_rollout_data" \
  "$@"

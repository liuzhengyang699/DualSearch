#!/usr/bin/env bash

set -xeuo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

DATA_DIR=${DATA_DIR:-data/evqa_search}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/test.parquet}

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
WAND_PROJECT=${WAND_PROJECT:-DualSearch}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-dual-search-grpo-qwen3-vl-4b}

TEXT_RETRIEVER_URL=${TEXT_RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
VISION_RETRIEVER_URL=${VISION_RETRIEVER_URL:-http://127.0.0.1:8001/vision_retrieve}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.image_key=images \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.agent.default_agent_loop=dual_search_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=dual_search/llm_agent/dual_search_agent_loop.yaml \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WAND_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="verl_checkpoints/${EXPERIMENT_NAME}" \
    max_turns=2 \
    retriever.url="${TEXT_RETRIEVER_URL}" \
    retriever.vision_search_url="${VISION_RETRIEVER_URL}" \
    retriever.topk=3 \
    retriever.max_obs_length=500 \
    "$@"

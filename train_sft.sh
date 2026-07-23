#!/usr/bin/env bash
set -euo pipefail

# This launcher consumes SFT Parquet files that already exist locally. It never
# downloads EVQA, iNaturalist, Wikipedia, or model data.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Qwen checkpoint directory}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${PROJECT_ROOT}/data/evqa_search}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-${SFT_DATA_DIR}/sft_train.parquet}"
SFT_VAL_FILE="${SFT_VAL_FILE:-${SFT_DATA_DIR}/sft_val.parquet}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/dual-search-sft}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"

if [[ ! -f "${SFT_TRAIN_FILE}" ]]; then
  echo "Missing SFT train data: ${SFT_TRAIN_FILE}" >&2
  exit 2
fi
if [[ ! -f "${SFT_VAL_FILE}" ]]; then
  echo "Missing SFT validation data: ${SFT_VAL_FILE}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must be an existing local directory: ${MODEL_PATH}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
torchrun \
  --standalone \
  --nnodes="${NNODES}" \
  --nproc-per-node="${N_GPUS_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  model.path="${MODEL_PATH}" \
  model.tokenizer_path="${MODEL_PATH}" \
  data.train_files="${SFT_TRAIN_FILE}" \
  data.val_files="${SFT_VAL_FILE}" \
  data.max_length=8192 \
  data.max_token_len_per_gpu=8192 \
  data.pad_mode=no_padding \
  data.truncation=error \
  trainer.default_local_dir="${SFT_OUTPUT_DIR}" \
  trainer.project_name=dual-search-sft \
  trainer.experiment_name=qwen-native-tools \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.test_freq=after_each_epoch \
  trainer.save_freq=after_each_epoch \
  checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  checkpoint.load_contents='[model,optimizer,extra]' \
  "$@"

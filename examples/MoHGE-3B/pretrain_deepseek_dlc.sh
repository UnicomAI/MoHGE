#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Set these variables from the shell or edit the defaults below before launching.
# Example:
#   TRAIN_DATA="/path/to/mmap_text_document" PRETRAINED_MODEL="/path/to/llama3-tokenizer" bash pretrain_deepseek_dlc.sh
DSW="${DSW:-dlc}"
MODEL="${MODEL:-A3B}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BATCH_TOTAL="${BATCH_TOTAL:-512}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-3e-5}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
PADDING_LENGTH="${PADDING_LENGTH:-4096}"
PRECISION="${PRECISION:-bf16}"
MODEL_PARALLEL="${MODEL_PARALLEL:-1}"
PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-1}"
CONTEXT_PARALLEL="${CONTEXT_PARALLEL:-1}"
EXPERT_PARALLEL="${EXPERT_PARALLEL:-4}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
USE_MEGA_ZERO="${USE_MEGA_ZERO:-true}"
USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-true}"
FINE_TUNE="${FINE_TUNE:-false}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-sel}"
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-false}"
CKPT_INTERVAL="${CKPT_INTERVAL:-50000}"

# Path prefix of Megatron idxmap data, without the .bin/.idx suffix.
# Weighted blends are also supported, for example:
#   TRAIN_DATA="0.7 /path/to/data_a_text_document 0.3 /path/to/data_b_text_document"
TRAIN_DATA="${TRAIN_DATA:-${SCRIPT_DIR}/data/mmap_train_text_document}"
VALID_DATA="${VALID_DATA:-${TRAIN_DATA}}"

# LLama3Tokenizer reads tokenizer files from Megatron's --load path in this codebase.
# Use a local HuggingFace tokenizer/model directory or a converted Megatron checkpoint.
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${TOKENIZER_PATH:-none}}"
if [ "${PRETRAINED_MODEL}" = "none" ]; then
    echo "Please set PRETRAINED_MODEL or TOKENIZER_PATH to a local LLaMA3 tokenizer/model directory." >&2
    exit 1
fi

TOTAL_TOKENS="${TOTAL_TOKENS:-1066793903690}"
WARMUP_TOKENS="${WARMUP_TOKENS:-10000}"
OUTPUT_LOG="${OUTPUT_LOG:-${SCRIPT_DIR}/workspace/output_mohge_3b_pretrain}"

bash "${SCRIPT_DIR}/run_mcore_deepseek_dlc.sh" \
  "${DSW}" \
  "${MODEL}" \
  "${BATCH_SIZE}" \
  "${BATCH_TOTAL}" \
  "${LEARNING_RATE}" \
  "${MIN_LEARNING_RATE}" \
  "${SEQ_LENGTH}" \
  "${PADDING_LENGTH}" \
  "${PRECISION}" \
  "${MODEL_PARALLEL}" \
  "${PIPELINE_PARALLEL}" \
  "${CONTEXT_PARALLEL}" \
  "${EXPERT_PARALLEL}" \
  "${SEQUENCE_PARALLEL}" \
  "${USE_MEGA_ZERO}" \
  "${USE_FLASH_ATTENTION}" \
  "${FINE_TUNE}" \
  "${CHECKPOINT_MODE}" \
  "${OFFLOAD_OPTIMIZER}" \
  "${CKPT_INTERVAL}" \
  "${TRAIN_DATA}" \
  "${VALID_DATA}" \
  "${PRETRAINED_MODEL}" \
  "${TOTAL_TOKENS}" \
  "${WARMUP_TOKENS}" \
  "${OUTPUT_LOG}"

#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 HF_CKPT_PATH OUTPUT_MCORE_PATH [MODEL_SIZE]" >&2
    exit 2
fi

MODEL_SIZE="${3:-8B}"
HF_CKPT_PATH="$1"
OUTPUT_MCORE_PATH="$2"

bash hf2mcore_convertor_llama3_1.sh \
  "${MODEL_SIZE}" \
  "${HF_CKPT_PATH}" \
  "${OUTPUT_MCORE_PATH}" \
  1 \
  1 \
  false \
  true \
  false \
  bf16

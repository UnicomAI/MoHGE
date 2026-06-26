#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 MCORE_CKPT_PATH OUTPUT_HF_PATH TOKENIZER_OR_MODEL_DIR [MODEL_SIZE]" >&2
    exit 2
fi

MODEL_SIZE="${4:-8B}"
MCORE_CKPT_PATH="$1"
OUTPUT_HF_PATH="$2"
TOKENIZER_OR_MODEL_DIR="$3"

bash hf2mcore_convertor_llama3_1.sh \
  "${MODEL_SIZE}" \
  "${MCORE_CKPT_PATH}" \
  "${OUTPUT_HF_PATH}" \
  1 \
  1 \
  true \
  true \
  false \
  bf16 \
  "${TOKENIZER_OR_MODEL_DIR}"

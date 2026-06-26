#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 INPUT_JSON OUTPUT_PREFIX TOKENIZER_OR_MODEL_DIR" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "${SCRIPT_DIR}/run_make_pretraining_dataset_megatron.sh" \
  "$1" \
  DeepSeekV2Tokenizer \
  text \
  "$2" \
  "$3"

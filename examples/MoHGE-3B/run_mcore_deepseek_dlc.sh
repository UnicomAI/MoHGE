#!/bin/bash
set -euo pipefail

if [ "$#" -ne 26 ]; then
    echo "Usage: $0 ENV MODEL_SIZE BATCH_SIZE GLOBAL_BATCH_SIZE LR MIN_LR SEQ_LEN PAD_LEN PRECISION TP PP CP EP SP DO FLASH SFT AC OPTIMIZER_OFFLOAD SAVE_INTERVAL DATASET_PATH VALID_DATASET_PATH LOAD_PATH TRAIN_TOKENS WARMUP_TOKENS OUTPUT_BASEPATH" >&2
    exit 2
fi

ENV="$1"
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_PATH="$(dirname "$(dirname "${CURRENT_DIR}")")"
export PYTHONPATH="${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

MP_DATASET_TYPE="${MP_DATASET_TYPE:-idxmap}"
MP_AC_LAYERS="${MP_AC_LAYERS:-1}"
MP_SFT_PACKING="${MP_SFT_PACKING:-false}"

case "${ENV}" in
    dsw)
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
        MASTER_ADDR="${MASTER_ADDR:-localhost}"
        MASTER_PORT="${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}"
        NNODES="${NNODES:-1}"
        NODE_RANK="${NODE_RANK:-0}"
        GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
        ;;
    dlc)
        NNODES="${WORLD_SIZE:?WORLD_SIZE must be set for dlc mode}"
        NODE_RANK="${RANK:?RANK must be set for dlc mode}"
        GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
        MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set for dlc mode}"
        MASTER_PORT="${MASTER_PORT:-6000}"
        ;;
    *)
        echo "Unknown ENV '${ENV}'. Use 'dsw' or 'dlc'." >&2
        exit 2
        ;;
esac

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

MODEL_SIZE="$2"
BATCH_SIZE="$3"
GLOBAL_BATCH_SIZE="$4"
LR="$5"
MIN_LR="$6"
SEQ_LEN="$7"
PAD_LEN="$8"
PR="$9"
TP="${10}"
PP="${11}"
CP="${12}"
EP="${13}"
SP="${14}"
DO="${15}"
FL="${16}"
SFT="${17}"
AC="${18}"
OPTIMIZER_OFFLOAD="${19}"
SAVE_INTERVAL="${20}"
DATASET_PATH="${21}"
VALID_DATASET_PATH="${22}"
PRETRAIN_CHECKPOINT_PATH="${23}"
TRAIN_TOKENS="${24}"
WARMUP_TOKENS="${25}"
OUTPUT_BASEPATH="${26}"

vp_options=""
if [ -n "${MP_VP:-}" ]; then
    vp_options=" --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ "${FL}" = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
elif [ "${FL}" = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
else
    echo "USE_FLASH_ATTENTION must be true or false." >&2
    exit 2
fi

case "${MODEL_SIZE}" in
    A2.4B)
        HIDDEN_SIZE=2048
        NUM_ATTN_HEADS=16
        NUM_LAYERS=9
        INTERMEDIATE_SIZE=4096
        MOE_INTERMEDIATE_SIZE=896
        MAX_POSITION_EMBEDDINGS="${SEQ_LEN}"
        EXTRA_VOCAB_SIZE=2400
        KV_LORA_RANK=512
        QK_NOPE_HEAD_DIM=128
        QK_ROPE_HEAD_DIM=64
        V_HEAD_DIM=128
        ROPE_THETA=10000
        SCALE_FACTOR=40
        NUM_EXPERTS=64
        ROUTER_TOPK=6
        NUM_SHARED_EXPERTS=2
        MOE_LAYER_FREQ=1
        RMS_NORM_EPS=1e-6
        MOE_AUX_LOSS_COEFF=3e-3
        ;;
    A1B)
        HIDDEN_SIZE=1024
        NUM_ATTN_HEADS=16
        NUM_LAYERS=9
        INTERMEDIATE_SIZE=4096
        MOE_INTERMEDIATE_SIZE=896
        MAX_POSITION_EMBEDDINGS="${SEQ_LEN}"
        EXTRA_VOCAB_SIZE=2400
        KV_LORA_RANK=512
        QK_NOPE_HEAD_DIM=128
        QK_ROPE_HEAD_DIM=64
        V_HEAD_DIM=128
        ROPE_THETA=10000
        SCALE_FACTOR=40
        NUM_EXPERTS=32
        ROUTER_TOPK=6
        NUM_SHARED_EXPERTS=2
        MOE_LAYER_FREQ=1
        RMS_NORM_EPS=1e-6
        MOE_AUX_LOSS_COEFF=3e-3
        ;;
    A3B)
        HIDDEN_SIZE=1024
        NUM_ATTN_HEADS=16
        NUM_LAYERS=15
        INTERMEDIATE_SIZE=6144
        MOE_INTERMEDIATE_SIZE=1024
        MAX_POSITION_EMBEDDINGS="${SEQ_LEN}"
        EXTRA_VOCAB_SIZE=2400
        KV_LORA_RANK=512
        QK_NOPE_HEAD_DIM=128
        QK_ROPE_HEAD_DIM=64
        V_HEAD_DIM=128
        ROPE_THETA=10000
        SCALE_FACTOR=40
        NUM_EXPERTS=64
        ROUTER_TOPK=6
        NUM_SHARED_EXPERTS=2
        MOE_LAYER_FREQ=1
        RMS_NORM_EPS=1e-6
        MOE_AUX_LOSS_COEFF=5e-3
        ;;
    *)
        echo "Unsupported MODEL_SIZE '${MODEL_SIZE}'." >&2
        exit 2
        ;;
esac

moe_options=" \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --moe-shared-expert-intermediate-size $((MOE_INTERMEDIATE_SIZE * NUM_SHARED_EXPERTS)) \
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"

TP_COMM_OVERLAP=$((TP > 1 ? 1 : 0))
comm_overlap_option=" --overlap-grad-reduce --overlap-param-gather"
if [ "${TP_COMM_OVERLAP}" -eq 1 ]; then
    comm_overlap_option=" --tp-comm-overlap --overlap-grad-reduce --overlap-param-gather"
fi

case "${AC}" in
    full)
        if [ $(((NUM_LAYERS / PP) % MP_AC_LAYERS)) -ne 0 ]; then
            echo "The number of layers per pipeline rank must be a multiple of MP_AC_LAYERS." >&2
            exit 2
        fi
        activation_checkpoint_options=" --recompute-method uniform --recompute-num-layers ${MP_AC_LAYERS} --recompute-granularity full"
        ;;
    sel)
        activation_checkpoint_options=" --recompute-activations"
        ;;
    none|false)
        activation_checkpoint_options=""
        ;;
    offload)
        activation_checkpoint_options=" --cpu-offloading --cpu-offloading-num-layers ${MP_AC_LAYERS}"
        if [ "${TP_COMM_OVERLAP}" -eq 1 ]; then
            comm_overlap_option=" --tp-comm-overlap"
        else
            comm_overlap_option=""
        fi
        ;;
    *)
        echo "Unsupported CHECKPOINT_MODE '${AC}'." >&2
        exit 2
        ;;
esac

case "${PR}" in
    fp16)
        pr_options=" --fp16 --apply-query-key-layer-scaling"
        export NVTE_APPLY_QK_LAYER_SCALING=1
        ;;
    bf16)
        pr_options=" --bf16"
        ;;
    fp8)
        pr_options=" --bf16 --fp8-format hybrid --fp8-amax-compute-algo max --fp8-amax-history-len 1024"
        ;;
    *)
        echo "Unsupported precision '${PR}'." >&2
        exit 2
        ;;
esac

if [ "${OPTIMIZER_OFFLOAD}" != false ] && [ "${DO}" = false ]; then
    echo "Optimizer offload requires distributed optimizer; enabling it."
    DO=true
fi

do_options=""
if [ "${DO}" = true ]; then
    do_options=" --use-distributed-optimizer"
fi

sp_options=""
if [ "${SP}" = true ] && [ "${TP}" -gt 1 ]; then
    sp_options=" --sequence-parallel"
fi

uneven_split_option=""
if [ -n "${MP_PP0_LAYERS:-}" ]; then
    if [ "${PP}" -le 1 ]; then
        echo "MP_PP0_LAYERS can only be used when PP > 1." >&2
        exit 2
    fi
    if [ $(((NUM_LAYERS - MP_PP0_LAYERS) % (PP - 1))) -ne 0 ]; then
        echo "With uneven pipeline split, remaining layers must divide the remaining stages." >&2
        exit 2
    fi
    uneven_split_option=" --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}"
fi

load_options=""
if [ "${PRETRAIN_CHECKPOINT_PATH}" != none ]; then
    load_options=" --load ${PRETRAIN_CHECKPOINT_PATH}"
fi

case "${OPTIMIZER_OFFLOAD}" in
    static)
        offload_option=" --optimizer hybridadam --optimizer-offload-policy static --optimizer-offload-fraction 1.0"
        ;;
    auto)
        offload_option=" --optimizer hybridadam --optimizer-offload-policy auto"
        ;;
    false)
        offload_option=""
        ;;
    *)
        echo "Unsupported OFFLOAD_OPTIMIZER '${OPTIMIZER_OFFLOAD}'." >&2
        exit 2
        ;;
esac

if [ "${SFT}" = true ]; then
    TRAIN_ITERS="${TRAIN_TOKENS}"
    LR_WARMUP_ITERS="${WARMUP_TOKENS}"
    LR_DECAY_ITERS=$((TRAIN_ITERS - LR_WARMUP_ITERS))
    PREFIX="finetune-mcore-deepseek-v2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_option=" --eod-mask-loss --train-mode finetune"
else
    TRAIN_ITERS=$((TRAIN_TOKENS / GLOBAL_BATCH_SIZE / SEQ_LEN))
    LR_WARMUP_ITERS=$((WARMUP_TOKENS / GLOBAL_BATCH_SIZE / SEQ_LEN))
    LR_DECAY_ITERS="${TRAIN_ITERS}"
    PREFIX="pretrain-mcore-deepseek-v2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_option=" --train-mode pretrain"
fi

if [ "${MP_DATASET_TYPE}" != idxmap ]; then
    echo "Only idxmap datasets are enabled in this MoHGE example." >&2
    exit 2
fi

dataset_option=" \
    --data-path ${DATASET_PATH} \
    --split 99,1,0 \
    --dataset LLama-Pretrain-Idxmap"

packing_options=""
if [ "${MP_SFT_PACKING}" = true ]; then
    packing_options=" --reset-position-ids --no-create-attention-mask-in-dataloader"
fi

NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard" "${OUTPUT_BASEPATH}/checkpoint" "${OUTPUT_BASEPATH}/log"
current_time="$(date "+%Y.%m.%d-%H.%M.%S")"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"
mkdir -p "${TENSORBOARD_DIR}" "${SAVED_PRETRAIN_CHECKPOINT_PATH}"

if [ -d "${PRETRAIN_CHECKPOINT_PATH}" ]; then
    find -L "${PRETRAIN_CHECKPOINT_PATH}" -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 -r cp -t "${SAVED_PRETRAIN_CHECKPOINT_PATH}"
fi

megatron_options=" \
    --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --init-method-std 0.008 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --lr-decay-iters ${LR_DECAY_ITERS} \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --train-iters ${TRAIN_ITERS} \
    --micro-batch-size ${BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --max-padding-length ${PAD_LEN} \
    --log-interval 1 \
    --log-throughput \
    --eval-interval 10000 \
    --eval-iters 10 \
    --save-interval ${SAVE_INTERVAL} \
    --tensorboard-queue-size 1 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --virtual-pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --num-workers 16 \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --patch-tokenizer-type LLama3Tokenizer \
    --swiglu \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --use-rotary-position-embeddings \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --position-embedding-type rope \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --rotary-base ${ROPE_THETA} \
    --rotary-scaling-factor ${SCALE_FACTOR} \
    --kv-channels ${V_HEAD_DIM} \
    --qk-layernorm \
    --multi-latent-attention \
    --ckpt-format torch"

te_options=" --transformer-impl transformer_engine"
LOG_FILE="${OUTPUT_BASEPATH}/log/${NAME}_${current_time}.log"
run_cmd="torchrun ${DISTRIBUTED_ARGS} ${CURRENT_DIR}/pretrain_deepseek.py ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} ${packing_options} ${uneven_split_option} ${comm_overlap_option}"

echo "${run_cmd}"
eval "${run_cmd}" 2>&1 | tee "${LOG_FILE}"

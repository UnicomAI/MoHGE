# MoHGE

MoHGE: Mixture of Heterogeneous Grouped Experts for Language Modeling.

Paper: http://arxiv.org/abs/2604.23108

Accepted to ACL 2026 Industry Track.

This repository contains the MoHGE implementation built on top of [Alibaba Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch). The MoHGE model code is under `megatron_patch/model/DeepSeek_MoHGE`, and the 3B pretraining example is under `examples/MoHGE-3B`.

## Environment

Use a CUDA environment that can run Megatron-LM, Transformer Engine, and FlashAttention. The easiest path is to start from the same PAI/Megatron-compatible image used by Pai-Megatron-Patch, then install the Python packages required by Megatron-LM and this patch repository.

Example setup:

```bash
conda create -n mohge python=3.10 -y
conda activate mohge

# Install PyTorch, Transformer Engine, FlashAttention, and Megatron-LM dependencies
# according to your CUDA/NCCL environment.
pip install transformers datasets einops sentencepiece tiktoken
```

For distributed training, make sure `torchrun`, NCCL, CUDA, and the NVIDIA driver are visible on every node.

## Download Megatron-LM

Place the Megatron-LM snapshot in the root of this repository and name the directory `Megatron-LM`:

```bash
cd /path/to/Pai-Megatron-Patch-MoHGE

# Use the exact Megatron-LM snapshot used for your experiments.
# If you have a tarball:
tar -xf /path/to/Megatron-LM.tar.gz

# Or clone Megatron-LM into the expected directory and checkout your target commit.
git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
```

The training scripts add both this repository and `./Megatron-LM` to `PYTHONPATH`.

## Prepare Data and Tokenizer

The example expects Megatron idxmap pretraining data. `TRAIN_DATA` should be the dataset prefix without the `.bin` or `.idx` suffix:

```bash
TRAIN_DATA=/path/to/mmap_train_text_document
```

For a weighted blend, pass the standard Megatron format:

```bash
TRAIN_DATA="0.7 /path/to/data_a_text_document 0.3 /path/to/data_b_text_document"
```

`PRETRAINED_MODEL` is used as Megatron's `--load` path and is also where `LLama3Tokenizer` reads tokenizer files. Set it to a local HuggingFace tokenizer/model directory or a converted Megatron checkpoint that contains the tokenizer files.

## Train MoHGE-3B

Run the pretraining entry script from `examples/MoHGE-3B`:

```bash
cd examples/MoHGE-3B

DSW=dlc \
TRAIN_DATA=/path/to/mmap_train_text_document \
VALID_DATA=/path/to/mmap_valid_text_document \
PRETRAINED_MODEL=/path/to/llama3_tokenizer_or_checkpoint \
bash pretrain_deepseek_dlc.sh
```

For a single-node DSW-style run, use:

```bash
DSW=dsw \
GPUS_PER_NODE=4 \
TRAIN_DATA=/path/to/mmap_train_text_document \
PRETRAINED_MODEL=/path/to/llama3_tokenizer_or_checkpoint \
bash pretrain_deepseek_dlc.sh
```

Common overrides:

```bash
MODEL=A3B
BATCH_SIZE=4
BATCH_TOTAL=512
SEQ_LENGTH=4096
EXPERT_PARALLEL=4
TOTAL_TOKENS=1066793903690
OUTPUT_LOG=./workspace/output_mohge_3b_pretrain
```

Training outputs are written to `examples/MoHGE-3B/workspace/` by default. This directory, datasets, checkpoints, logs, local papers, and the local `Megatron-LM/` snapshot are ignored by Git.

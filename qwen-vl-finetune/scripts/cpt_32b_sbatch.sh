#!/bin/bash
# Slurm launcher for Qwen3-VL-32B continued pretraining (runs scripts/cpt_32b.sh
# on every node inside the training container).
#
# Submit (mode is required: "full" trains the whole model S1-style, "llm"
# trains the language model only):
#   sbatch scripts/cpt_32b_sbatch.sh full
#   sbatch scripts/cpt_32b_sbatch.sh llm
# Overrides (forwarded to scripts/cpt_32b.sh via the environment):
#   MODEL_PATH=... MINT1T_DATA_DIR=... DATASETS=... LLM_LAST_N=8 REPORT_TO=wandb \
#     CONTAINER_IMAGE=<image> sbatch --nodes=2 --partition=<p> scripts/cpt_32b_sbatch.sh full

#SBATCH --account=coreai_mlperf_training
#SBATCH --exclusive
#SBATCH --job-name=coreai_mlperf_training-training.qwen3vl_32b_cpt
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fsw/coreai_mlperf_training/users/dfridman/Qwen3-VL/slurm_logs/slurm_%j.out
#SBATCH --partition=batch
#SBATCH --time=01:00:00

set -eux

# Training mode: "full" (whole model, S1-style) or "llm" (LLM only)
mode=${1:?usage: sbatch scripts/cpt_32b_sbatch.sh <full|llm>}
case "$mode" in
  full)
    # Whole model trainable; text-only samples ride along safely because every
    # batch is anchored by an image sample
    export TUNE_MM_VISION=True
    export TUNE_MM_MLP=True
    export TUNE_MM_LLM=True
    export ALLOW_TEXT_ONLY=False
    export REQUIRE_IMAGE_PER_BATCH=True
    ;;
  llm)
    # Vision tower and projector frozen; text-only batches handled by the
    # zero-weighted dummy vision forward
    export TUNE_MM_VISION=False
    export TUNE_MM_MLP=False
    export TUNE_MM_LLM=True
    export ALLOW_TEXT_ONLY=True
    export REQUIRE_IMAGE_PER_BATCH=False
    ;;
  *)
    echo "unknown mode '$mode' (expected: full | llm)" >&2
    exit 1
    ;;
esac

LUSTRE_DIR=/lustre/fsw/coreai_mlperf_training/users/dfridman
REPO_DIR=${LUSTRE_DIR}/Qwen3-VL/qwen-vl-finetune
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"gitlab-master.nvidia.com/dl/mlperf/optimized:deepseekv3_671b.pytorch.65028332"}

# Rendezvous on the first node; cpt_32b.sh picks these up (NODE_RANK comes from
# SLURM_NODEID inside each srun task)
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=$SLURM_NNODES

# Local HF checkpoint (inside the LUSTRE_DIR mount); picked up by cpt_32b.sh
export MODEL_PATH=${MODEL_PATH:-${LUSTRE_DIR}/checkpoints/hf/Qwen3-VL-32B-Instruct}

# LLM_LAST_N>0 trains only the last N LLM decoder layers (development)
export LLM_LAST_N=${LLM_LAST_N:--1}

# Peak learning rate (linear warmup to this, then cosine decay to 0)
export LR=${LR:-2e-5}

# Variable-length packed sequences allocate many differently-sized buffers,
# which fragments the CUDA caching allocator; expandable segments avoids
# fragmentation-induced OOM ("reserved but unallocated" in the OOM message)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Per-image pixel budget at training time (256*28*28 -> <=196 tokens/image,
# reduced so micro batch 2 fits in memory; keep in sync with the converter's
# --image-word-cost)
export MAX_PIXELS=${MAX_PIXELS:-200704}

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "${LUSTRE_DIR}:${LUSTRE_DIR}" \
     --container-workdir "$REPO_DIR" \
     --no-container-mount-home \
     --kill-on-bad-exit=1 \
     bash scripts/cpt_32b.sh

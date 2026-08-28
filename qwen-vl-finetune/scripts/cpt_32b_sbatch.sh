#!/bin/bash
# Slurm launcher for Qwen3-VL-32B continued pretraining (runs scripts/cpt_32b.sh
# on every node inside the training container).
#
# Submit:
#   sbatch scripts/cpt_32b_sbatch.sh
# Overrides (forwarded to scripts/cpt_32b.sh via the environment):
#   MODEL_PATH=... MINT1T_DATA_DIR=... DATASETS=... LLM_LAST_N=8 REPORT_TO=wandb \
#     CONTAINER_IMAGE=<image> sbatch --nodes=2 --partition=<p> scripts/cpt_32b_sbatch.sh

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

# Which components to train (defaults: LLM only). Enable more by exporting
# True, e.g. TUNE_MM_VISION=True TUNE_MM_MLP=True; LLM_LAST_N>0 trains only
# the last N LLM layers (development). NOTE: TUNE_MM_VISION=True requires
# ALLOW_TEXT_ONLY=False and a dataset with no text-only samples.
export TUNE_MM_VISION=${TUNE_MM_VISION:-False}
export TUNE_MM_MLP=${TUNE_MM_MLP:-False}
export TUNE_MM_LLM=${TUNE_MM_LLM:-True}
export LLM_LAST_N=${LLM_LAST_N:--1}
export ALLOW_TEXT_ONLY=${ALLOW_TEXT_ONLY:-True}

# Peak learning rate (linear warmup to this, then cosine decay to 0)
export LR=${LR:-2e-6}

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "${LUSTRE_DIR}:${LUSTRE_DIR}" \
     --container-workdir "$REPO_DIR" \
     --no-container-mount-home \
     --kill-on-bad-exit=1 \
     bash scripts/cpt_32b.sh

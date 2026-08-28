#!/bin/bash
# Slurm launcher for Qwen3-VL-32B continued pretraining (runs scripts/cpt_32b.sh
# on every node inside the training container).
#
# Submit:
#   CONTAINER_IMAGE=<image> sbatch scripts/cpt_32b_sbatch.sh
# Overrides (forwarded to scripts/cpt_32b.sh via the environment):
#   MODEL_PATH=... MINT1T_DATA_DIR=... DATASETS=... LLM_LAST_N=8 REPORT_TO=wandb \
#     CONTAINER_IMAGE=<image> sbatch --nodes=2 --partition=<p> scripts/cpt_32b_sbatch.sh

#SBATCH --account=coreai_mlperf_training
#SBATCH --exclusive
#SBATCH --job-name=coreai_mlperf_training-training.qwen3vl_32b_cpt
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fsw/coreai_mlperf_training/users/dfridman/checkpoints/slurm_logs/slurm_%j.out
#SBATCH --partition=gb200
#SBATCH --time=04:00:00

set -eux

LUSTRE_DIR=/lustre/fsw/coreai_mlperf_training
REPO_DIR=${LUSTRE_DIR}/users/dfridman/Qwen3-VL/qwen-vl-finetune
CONTAINER_IMAGE=${CONTAINER_IMAGE:?set CONTAINER_IMAGE to the training container image}

# Rendezvous on the first node; cpt_32b.sh picks these up (NODE_RANK comes from
# SLURM_NODEID inside each srun task)
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=$SLURM_NNODES

export HF_HOME=${HF_HOME:-${LUSTRE_DIR}/users/dfridman/hf_home}
export NCCL_MNNVL_ENABLE=0

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "${LUSTRE_DIR}:${LUSTRE_DIR}" \
     --container-workdir "$REPO_DIR" \
     --no-container-mount-home \
     --kill-on-bad-exit=1 \
     bash scripts/cpt_32b.sh

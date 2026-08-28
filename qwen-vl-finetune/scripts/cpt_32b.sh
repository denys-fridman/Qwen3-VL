#!/bin/bash
# Continued pretraining (full-sequence next-token loss) of Qwen3-VL-32B
# from an existing checkpoint. Requires ~8x80G GPUs with ZeRO-3; switch to
# scripts/zero3_offload.json if you hit OOM.

# Distributed training configuration (multi-node: set MASTER_ADDR/NNODES and
# per-node NODE_RANK, or launch via scripts/cpt_32b_sbatch.sh under Slurm)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# Usage: bash scripts/cpt_32b.sh [MODEL_PATH] [MINT1T_DATA_DIR]
# Positional argument > env var > default.
# HuggingFace ID or a local checkpoint path. NOTE: train_qwen.py picks the model
# class from the path name — it must contain "qwen3", and for the dense model the
# last path component must NOT contain the letter "a" (that selects the MoE class).
llm=${1:-${MODEL_PATH:-"Qwen/Qwen3-VL-32B-Instruct"}}

# Training hyperparameters
# Typical continued-pretraining LR for this scale is 1e-6 to 1e-5 depending on
# data volume; start low if your corpus is small.
# NOTE: flags target transformers v5 (warmup_steps <1 means warmup ratio);
# on transformers 4.x use --warmup_ratio instead.
lr=${LR:-2e-6}
batch_size=2
grad_accum_steps=8
seed=${SEED:-42}
# Samples held out from the training data for per-epoch eval loss (0 disables)
eval_samples=${EVAL_SAMPLES:-128}

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (register your corpus in qwenvl/data/__init__.py;
# mint1t is produced by tools/preprocess_mint1t.py). MINT1T_DATA_DIR is the
# output directory of that script and is read by qwenvl/data/__init__.py;
# exported so the torchrun workers inherit it.
export MINT1T_DATA_DIR=${2:-${MINT1T_DATA_DIR:-"/lustre/fsw/coreai_mlperf_training/users/dfridman/datasets/mint1t/processed"}}
datasets=${DATASETS:-"mint1t%100"}

# Output configuration
run_name="qwen3vl-32b-cpt"
# /results is container-local (not mounted): each run starts with a clean
# output dir, and checkpoints are discarded when the job ends
output_dir=${OUTPUT_DIR:-"/results"}
# Metrics reporting: "none" by default; e.g. REPORT_TO=wandb to enable
report_to=${REPORT_TO:-"none"}

# Text-only samples would make their rank skip the vision tower and miss its
# ZeRO-3 parameter all-gathers, deadlocking NCCL. allow_text_only handles this
# by running a zero-weighted dummy vision forward on text-only batches
# (option 1, see enable_dummy_vision_forward in qwenvl/train/trainer.py).
# TODO: implement and compare the alternatives:
#   (2) batch construction that guarantees >=1 image per packed sequence
#   (3) non-parameter-sharded parallelism (e.g. ZeRO-2), where text-only
#       batches need no workaround
allow_text_only=True

# Which components to train (default: the whole model). NOTE: the repo README
# advises freezing the vision tower (TUNE_MM_VISION=False) when training on
# mixed image+video data.
tune_mm_vision=${TUNE_MM_VISION:-True}
tune_mm_mlp=${TUNE_MM_MLP:-True}
tune_mm_llm=${TUNE_MM_LLM:-True}

# Development knob: train only the last N LLM decoder layers (plus final norm
# and lm_head), freezing the rest. Cuts optimizer-state memory roughly in
# proportion, e.g. LLM_LAST_N=8 fits a single node; -1 trains the full LLM.
llm_last_n=${LLM_LAST_N:--1}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --train_on_all_tokens True \
    --data_flatten True \
    --allow_text_only ${allow_text_only} \
    --tune_mm_vision ${tune_mm_vision} \
    --tune_mm_mlp ${tune_mm_mlp} \
    --tune_mm_llm ${tune_mm_llm} \
    --tune_llm_last_n_layers ${llm_last_n} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 10 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "epoch" \
    --eval_samples ${eval_samples} \
    --save_strategy "no" \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_steps 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --seed ${seed} \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to ${report_to}"

# Launch training
torchrun --nnodes=${NNODES} \
         --node_rank=${NODE_RANK} \
         --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

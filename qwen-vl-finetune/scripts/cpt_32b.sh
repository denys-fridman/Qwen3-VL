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
lr=${LR:-2e-5}
# batch_size 2 is needed in full mode: with REQUIRE_IMAGE_PER_BATCH the
# second slot is what lets text-only samples ride along. The memory for it
# comes from the reduced MAX_PIXELS below (batch 2 at 576*28*28 was OOM,
# since data_flatten packs the micro-batch into one sequence).
batch_size=2
grad_accum_steps=8
seed=${SEED:-42}
# Per-image pixel budget, applied by the image processor at training time
# (stored images keep native resolution). 200704 = 256*28*28 -> up to 196
# vision tokens per image for Qwen3-VL (one token per 32x32 pixels): reduced
# from 576*28*28 (441 tokens) to fit micro batch 2 in memory — batch 2 at the
# larger budget OOMs. Data chunked with the converter's default
# --image-word-cost 352 stays safely within context at either setting;
# raising MAX_PIXELS beyond 576*28*28 requires re-chunking to match.
max_pixels=${MAX_PIXELS:-200704}
min_pixels=${MIN_PIXELS:-784}
# Samples held out from the training data for eval loss (0 disables)
eval_samples=${EVAL_SAMPLES:-1024}
# Evaluate every N optimizer steps (a float <1 works as a ratio of total steps)
eval_steps=${EVAL_STEPS:-10}

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (register your corpus in qwenvl/data/__init__.py;
# mint1t is produced by tools/preprocess_mint1t.py). MINT1T_DATA_DIR is the
# output directory of that script and is read by qwenvl/data/__init__.py;
# exported so the torchrun workers inherit it.
export MINT1T_DATA_DIR=${2:-${MINT1T_DATA_DIR:-"/lustre/fsw/coreai_mlperf_training/users/dfridman/datasets/MINT-1T-HTML/processed"}}
datasets=${DATASETS:-"mint1t%100"}

# Output configuration
run_name="qwen3vl-32b-cpt"
# /results is container-local (not mounted): each run starts with a clean
# output dir, and checkpoints are discarded when the job ends
output_dir=${OUTPUT_DIR:-"/results"}
# Metrics reporting: "none" by default; e.g. REPORT_TO=wandb to enable
report_to=${REPORT_TO:-"none"}

# Text-only samples would make their rank skip the vision tower and miss its
# ZeRO-3 parameter all-gathers, deadlocking NCCL. Two implemented remedies:
#   (1) ALLOW_TEXT_ONLY: zero-weighted dummy vision forward on text-only
#       batches (enable_dummy_vision_forward in qwenvl/train/trainer.py).
#       Only valid with a FROZEN vision tower — with TUNE_MM_VISION=True the
#       gradient reduce-scatter order diverges across ranks and hangs.
#   (2) REQUIRE_IMAGE_PER_BATCH: every train batch gets >=1 image anchor and
#       the eval split is image-only (ImageGuaranteedBatchSampler), so the
#       vision tower runs on real data everywhere — safe with a trainable
#       tower; excess text-only samples are dropped per epoch. Default, for
#       S1-style training.
# TODO: implement and compare (3) non-parameter-sharded parallelism
# (e.g. ZeRO-2), where text-only batches need no workaround.
allow_text_only=${ALLOW_TEXT_ONLY:-False}
require_image_per_batch=${REQUIRE_IMAGE_PER_BATCH:-True}

# Which components to train. Default mimics Qwen3-VL S1 (Multimodal
# Pre-Training): all components trainable, seq len 8192, interleaved VL data
# with text-only samples mixed in via image-guaranteed batches.
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
    --require_image_per_batch ${require_image_per_batch} \
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
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "steps" \
    --eval_steps ${eval_steps} \
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

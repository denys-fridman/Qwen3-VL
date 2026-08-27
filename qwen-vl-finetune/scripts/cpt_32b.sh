#!/bin/bash
# Continued pretraining (full-sequence next-token loss) of Qwen3-VL-32B
# from an existing checkpoint. Requires ~8x80G GPUs with ZeRO-3; switch to
# scripts/zero3_offload.json if you hit OOM.

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# HuggingFace ID or a local checkpoint path. NOTE: train_qwen.py picks the model
# class from the path name — it must contain "qwen3", and for the dense model the
# last path component must NOT contain the letter "a" (that selects the MoE class).
llm=${MODEL_PATH:-"Qwen/Qwen3-VL-32B-Instruct"}

# Training hyperparameters
# Typical continued-pretraining LR for this scale is 1e-6 to 1e-5 depending on
# data volume; start low if your corpus is small.
lr=2e-6
batch_size=2
grad_accum_steps=8

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (register your corpus in qwenvl/data/__init__.py)
datasets=${DATASETS:-"your_cpt_dataset%100"}

# Output configuration
run_name="qwen3vl-32b-cpt"
output_dir=./output_cpt_32b

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --train_on_all_tokens True \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

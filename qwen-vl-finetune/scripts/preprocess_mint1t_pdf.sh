#!/bin/bash
# Convert MINT-1T PDF-subset tar shards into the qwen-vl-finetune training
# format (images extracted from the per-document TIFFs; see DATA.md).
#
# Usage:
#   bash scripts/preprocess_mint1t_pdf.sh <data_dir>
#
# Reads shards from <data_dir>/*.tar (override the glob with DATA_FILES) and
# writes the processed dataset (images/ + annotations.jsonl) to
# <data_dir>/processed. CPU-bound (TIFF decoding); no network needed.
set -euo pipefail

data_dir=${1:?usage: bash scripts/preprocess_mint1t_pdf.sh <data_dir>}
data_files=${DATA_FILES:-"${data_dir}/*.tar"}
num_workers=${NUM_WORKERS:-$(nproc)}
# Tokenizer for exact token verification (requires transformers; run inside
# the training container). TOKENIZER=none disables it.
tokenizer=${TOKENIZER:-"/lustre/fsw/coreai_mlperf_training/users/dfridman/checkpoints/hf/Qwen3-VL-32B-Instruct"}
tokenizer_flag=()
if [ "${tokenizer}" != "none" ]; then
    tokenizer_flag=(--tokenizer "${tokenizer}")
fi
# Optional English-score filter, e.g. MIN_EN_SCORE=0.5 (0 = keep all languages)
min_en_score=${MIN_EN_SCORE:-0}

script_dir=$(dirname "$(readlink -f "$0")")

python "${script_dir}/../tools/preprocess_mint1t_pdf.py" \
    --data-files "${data_files}" \
    --output-dir "${data_dir}/processed" \
    --num-workers "${num_workers}" \
    --min-en-score "${min_en_score}" \
    --keep-text-only \
    ${tokenizer_flag[@]+"${tokenizer_flag[@]}"}

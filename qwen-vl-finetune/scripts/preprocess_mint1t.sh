#!/bin/bash
# Convert raw MINT-1T parquet shards into the qwen-vl-finetune training format
# (downloads and validates images; see DATA.md for the data rationale).
#
# Usage:
#   bash scripts/preprocess_mint1t.sh <data_dir>
#
# Reads shards from <data_dir>/data_v1_1/*.parquet (override the glob with
# DATA_FILES) and writes the processed dataset (images/ + annotations.jsonl)
# to <data_dir>/processed. Run on a node with internet access; re-runs skip
# already-downloaded images.
set -euo pipefail

data_dir=${1:?usage: bash scripts/preprocess_mint1t.sh <data_dir>}
data_files=${DATA_FILES:-"${data_dir}/data_v1_1/*.parquet"}
num_workers=${NUM_WORKERS:-32}
timeout=${TIMEOUT:-3}

script_dir=$(dirname "$(readlink -f "$0")")

python "${script_dir}/../tools/preprocess_mint1t.py" \
    --data-files "${data_files}" \
    --output-dir "${data_dir}/processed" \
    --num-workers "${num_workers}" \
    --timeout "${timeout}" \
    --keep-text-only

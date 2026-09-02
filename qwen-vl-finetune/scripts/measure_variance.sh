#!/bin/bash
# Launch N identical training runs with seeds 1..N to measure run-to-run
# variance. Slurm logs land in ROOT/<timestamp>/<mode>/seed_<seed>.out, which
# tools/plot_losses.py can plot directly. A follow-up analysis job
# (scripts/cov_analysis_sbatch.sh) is queued with a dependency on all N runs
# and writes the coefficient-of-variance sweep results into the same folder.
#
# Usage:
#   bash scripts/measure_variance.sh <full|llm> [N=10]
# Training knobs pass through the environment as usual, e.g.
#   LR=1e-5 MAX_STEPS=150 bash scripts/measure_variance.sh full 5
set -euo pipefail

mode=${1:?usage: bash scripts/measure_variance.sh <full|llm> [N=10]}
num_runs=${2:-10}
case "$mode" in
  full|llm) ;;
  *) echo "unknown mode '$mode' (expected: full | llm)" >&2; exit 1 ;;
esac
if ! [[ "$num_runs" =~ ^[1-9][0-9]*$ ]]; then
    echo "N must be a positive integer, got '$num_runs'" >&2
    exit 1
fi

ROOT=${MEASURE_VARIANCE_ROOT:-/lustre/fsw/coreai_mlperf_training/users/dfridman/logs/multimodal/qwen/measure_variance}
out_dir=${ROOT}/$(date +%Y%m%d_%H%M%S)/${mode}
mkdir -p "$out_dir"   # Slurm does not create log directories

script_dir=$(dirname "$(readlink -f "$0")")

echo "logs: $out_dir"
job_ids=()
for seed in $(seq 1 "$num_runs"); do
    job_id=$(SEED=$seed sbatch --parsable \
        --job-name="qwen3vl_cpt_${mode}_seed${seed}" \
        --output="${out_dir}/seed_${seed}.out" \
        "${script_dir}/cpt_32b_sbatch.sh" "$mode")
    job_id=${job_id%%;*}   # --parsable may append ";cluster"
    job_ids+=("$job_id")
    echo "seed=${seed} job=${job_id}" | tee -a "${out_dir}/jobs.txt"
done

# Analysis runs after every training job has terminated (any exit status);
# its log is analysis.log (not *.out) so the analysis doesn't parse itself.
dependency=$(IFS=:; echo "${job_ids[*]}")
analysis_id=$(sbatch --parsable \
    --dependency="afterany:${dependency}" \
    --job-name="qwen3vl_cov_${mode}" \
    --output="${out_dir}/analysis.log" \
    "${script_dir}/cov_analysis_sbatch.sh" "$out_dir")
echo "analysis job=${analysis_id%%;*} (afterany:${dependency})" | tee -a "${out_dir}/jobs.txt"

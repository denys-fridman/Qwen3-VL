#!/bin/bash
# Coefficient-of-variance analysis over a directory of seed-run logs.
# Submitted by scripts/measure_variance.sh with --dependency=afterany:<jobs>,
# so it runs once all training jobs have terminated (whatever their status).
#
# Usage:
#   sbatch --output=<logs_dir>/analysis.log scripts/cov_analysis_sbatch.sh <logs_dir>
# Writes training_curves.png (tools/plot_losses.py), coefficient_of_variance.png,
# cov_sweep.csv, cov_pivot.csv and cov_detailed.csv into <logs_dir>.

#SBATCH --account=coreai_mlperf_training
#SBATCH --job-name=coreai_mlperf_training-analysis.qwen3vl_cov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=36x2-a01r
#SBATCH --time=00:20:00

set -eux

logs_dir=${1:?usage: sbatch scripts/cov_analysis_sbatch.sh <logs_dir>}

LUSTRE_DIR=/lustre/fsw/coreai_mlperf_training/users/dfridman
REPO_DIR=${LUSTRE_DIR}/Qwen3-VL/qwen-vl-finetune
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"gitlab-master.nvidia.com/dl/mlperf/optimized:deepseekv3_671b.pytorch.65028332"}

# Sweep target losses 2.0 .. 2.8 in steps of 0.025
SWEEP_MIN=${SWEEP_MIN:-2.0}
SWEEP_MAX=${SWEEP_MAX:-2.8}
SWEEP_STEP=${SWEEP_STEP:-0.025}

srun --container-image "$CONTAINER_IMAGE" \
     --container-mounts "${LUSTRE_DIR}:${LUSTRE_DIR}" \
     --container-workdir "$REPO_DIR" \
     --no-container-mount-home \
     bash -c "
        python -c 'import pandas, matplotlib' 2>/dev/null || pip install --quiet pandas matplotlib
        python tools/plot_losses.py '${logs_dir}'
        python tools/measure_coefficient_of_variance.py '${logs_dir}' \
            --sweep-range ${SWEEP_MIN} ${SWEEP_MAX} ${SWEEP_STEP} \
            --pivot-table --pivot-output cov_pivot.csv \
            --plot --plot-output coefficient_of_variance.png \
            --output cov_sweep.csv \
            --detailed-table cov_detailed.csv \
            --general-analysis
     "

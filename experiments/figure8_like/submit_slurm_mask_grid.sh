#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

TASKS=${TASKS:-"SST-2 cr SNLI MNLI QNLI QQP MRPC"}
SEEDS=${SEEDS:-"100 13 21 42 87"}
SBATCH_ARGS=${SBATCH_ARGS:-"--gres=gpu:1 --cpus-per-task=4 --mem=48G --time=24:00:00"}

mkdir -p log_files/slurm

read -r -a TASK_ARRAY <<< "$TASKS"
read -r -a SEED_ARRAY <<< "$SEEDS"

for task in "${TASK_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    job_name="fig8-${task}-${seed}"
    echo "Submitting $job_name"
    # shellcheck disable=SC2086
    sbatch \
      --job-name="$job_name" \
      --output="log_files/slurm/%x-%j.out" \
      --error="log_files/slurm/%x-%j.err" \
      --export=ALL,TASKS="$task",SEEDS="$seed" \
      $SBATCH_ARGS \
      "$SCRIPT_DIR/slurm_task_seed.sbatch"
  done
done

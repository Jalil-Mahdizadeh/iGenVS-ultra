#!/usr/bin/env bash
#SBATCH --job-name=rl-bench-frozen
#SBATCH --partition=gpu
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=10:00:00
#SBATCH --array=0-4%5
#SBATCH --output=phase-10-rl-bench/logs/%A_%a.out
#SBATCH --error=phase-10-rl-bench/logs/%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR}"
phase_dir="${repo_root}/phase-10-rl-bench"
mapfile -t targets < "${phase_dir}/targets.txt"
target="${targets[${SLURM_ARRAY_TASK_ID}]}"

srun --ntasks=1 --gpus-per-task=4 --cpu-bind=cores \
  python3 "${phase_dir}/scripts/run_target.py" --target "${target}" --gpus 4

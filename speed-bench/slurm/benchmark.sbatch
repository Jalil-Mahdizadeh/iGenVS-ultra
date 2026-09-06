#!/usr/bin/env bash
# One immutable entry point for every cold full-speed benchmark case.
set -Eeuo pipefail

readonly ROOT="/nobackup/proj/disk/theo-storage/personal/jalil/iGenVS-ultra"
cd "$ROOT"

: "${BENCH_KIND:?BENCH_KIND is required}"
: "${BENCH_GPUS:?BENCH_GPUS is required}"

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

if [[ "$BENCH_KIND" == "docking" ]]; then
    : "${BENCH_ENGINE:?BENCH_ENGINE is required for docking}"
    : "${BENCH_MODE:?BENCH_MODE is required for docking}"
    exec python speed-bench/scripts/run_case.py docking \
        --engine "$BENCH_ENGINE" \
        --mode "$BENCH_MODE" \
        --gpus "$BENCH_GPUS"
fi

if [[ "$BENCH_KIND" == "screening" ]]; then
    exec python speed-bench/scripts/run_case.py screening --gpus "$BENCH_GPUS"
fi

echo "unsupported BENCH_KIND=$BENCH_KIND" >&2
exit 2

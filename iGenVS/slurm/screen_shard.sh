#!/usr/bin/env bash
set -euo pipefail

: "${IGENVS_IMAGE:?set IGENVS_IMAGE}"
: "${IGENVS_INPUT:?set IGENVS_INPUT}"
: "${IGENVS_OUTPUT:?set IGENVS_OUTPUT}"

shard_index="${SLURM_PROCID:-${SLURM_ARRAY_TASK_ID:-0}}"
num_shards="${IGENVS_NUM_SHARDS:-1}"
size_x="${IGENVS_SIZE_X:-22.5}"
size_y="${IGENVS_SIZE_Y:-22.5}"
size_z="${IGENVS_SIZE_Z:-22.5}"
search_mode="${IGENVS_SEARCH_MODE:-balance}"
engine="${IGENVS_ENGINE:-unidock}"
scoring="${IGENVS_SCORING:-auto}"
pocket_args=()
if [[ -n "${IGENVS_TARGET:-}" ]]; then
  pocket_args=(--target "${IGENVS_TARGET}")
else
  : "${IGENVS_RECEPTOR:?set IGENVS_TARGET or IGENVS_RECEPTOR}"
  : "${IGENVS_CENTER_X:?set IGENVS_TARGET or IGENVS_CENTER_X}"
  : "${IGENVS_CENTER_Y:?set IGENVS_TARGET or IGENVS_CENTER_Y}"
  : "${IGENVS_CENTER_Z:?set IGENVS_TARGET or IGENVS_CENTER_Z}"
  pocket_args=(
    --receptor "${IGENVS_RECEPTOR}"
    --center "${IGENVS_CENTER_X}" "${IGENVS_CENTER_Y}" "${IGENVS_CENTER_Z}"
    --size "${size_x}" "${size_y}" "${size_z}"
  )
  if [[ "$engine" == "autodock-gpu" ]]; then
    : "${IGENVS_ADGPU_GRID:?set IGENVS_TARGET or IGENVS_ADGPU_GRID}"
    pocket_args+=(--adgpu-grid "${IGENVS_ADGPU_GRID}")
  fi
fi
batch_size="${IGENVS_BATCH_SIZE:-auto}"
prep_workers="${IGENVS_PREP_WORKERS:-auto}"
validation_workers="${IGENVS_VALIDATION_WORKERS:-auto}"
prep_mode="${IGENVS_PREP_MODE:-standard}"
pose_output="${IGENVS_POSE_OUTPUT:-none}"
profile_args=()
if [[ -n "${IGENVS_BATCH_PROFILE:-}" ]]; then
  profile_args=(--batch-profile "${IGENVS_BATCH_PROFILE}")
fi

engine_args=(--engine "$engine" --scoring "$scoring")
mps_started=0
if [[ "$engine" == "autodock-gpu" ]]; then
  engine_args+=(
    --adgpu-cpu-threads "${IGENVS_ADGPU_CPU_THREADS:-4}"
    --adgpu-workers "${IGENVS_ADGPU_WORKERS:-auto}"
    --adgpu-executable "${IGENVS_ADGPU_EXECUTABLE:-autodock_gpu}"
    --adgpu-local-search "${IGENVS_ADGPU_LOCAL_SEARCH:-ad}"
  )
  if [[ -n "${IGENVS_ADGPU_RUNS:-}" ]]; then
    engine_args+=(--adgpu-runs "${IGENVS_ADGPU_RUNS}")
  fi
  if [[ -n "${IGENVS_ADGPU_EVALUATIONS:-}" ]]; then
    engine_args+=(--adgpu-evaluations "${IGENVS_ADGPU_EVALUATIONS}")
  fi
  if [[ "${IGENVS_ADGPU_NO_HEURISTICS:-0}" == "1" ]]; then
    engine_args+=(--adgpu-no-heuristics)
  fi
  if [[ "${IGENVS_ADGPU_NO_AUTOSTOP:-0}" == "1" ]]; then
    engine_args+=(--adgpu-no-autostop)
  fi
  if [[ "${IGENVS_ADGPU_MPS:-0}" == "1" ]]; then
    mps_root="${SLURM_TMPDIR:-/tmp}/igenvs-mps-${SLURM_JOB_ID:-manual}-${shard_index}"
    mkdir -p "$mps_root/pipe" "$mps_root/log"
    export CUDA_MPS_PIPE_DIRECTORY="$mps_root/pipe"
    export CUDA_MPS_LOG_DIRECTORY="$mps_root/log"
    nvidia-cuda-mps-control -d
    mps_started=1
  fi
else
  engine_args+=(
    --refine-step "${IGENVS_REFINE_STEP:-3}"
    --unidock-verbosity "${IGENVS_UNIDOCK_VERBOSITY:-0}"
  )
  if [[ "${IGENVS_NO_REFINE:-0}" == "1" ]]; then
    engine_args+=(--no-refine)
  fi
fi

cleanup_mps() {
  if [[ "$mps_started" == "1" ]]; then
    printf 'quit\n' | nvidia-cuda-mps-control >/dev/null || true
  fi
}
if [[ "$mps_started" == "1" ]]; then
  trap cleanup_mps EXIT
fi

apptainer exec --nv "${IGENVS_IMAGE}" igenvs screen \
  --input "${IGENVS_INPUT}" \
  "${engine_args[@]}" \
  "${pocket_args[@]}" \
  --search-mode "${search_mode}" \
  --batch-size "${batch_size}" \
  "${profile_args[@]}" \
  --prep-workers "${prep_workers}" \
  --validation-workers "${validation_workers}" \
  --prep-mode "${prep_mode}" \
  --pose-output "${pose_output}" \
  --num-shards "${num_shards}" \
  --shard-index "${shard_index}" \
  --output-dir "${IGENVS_OUTPUT}/shard-${shard_index}"

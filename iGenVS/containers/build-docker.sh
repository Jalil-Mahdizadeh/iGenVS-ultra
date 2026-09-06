#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${1:-${IGENVS_DOCKER_TAG:-igenvs-ultra/igenvs:latest}}"
cuda_architectures="${IGENVS_CUDA_ARCHITECTURES:-80;86;89;90;100;120}"
adgpu_targets="${IGENVS_ADGPU_TARGETS:-80 86 89 90 100 120}"

if [[ "$#" -gt 1 ]]; then
    printf 'usage: %s [IMAGE_TAG]\n' "$0" >&2
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    printf 'docker is required to build the AMD64 image\n' >&2
    exit 1
fi
if [[ ! "${cuda_architectures}" =~ ^[0-9]+(;[0-9]+)*$ ]]; then
    printf 'IGENVS_CUDA_ARCHITECTURES must be a semicolon-separated numeric list\n' >&2
    exit 2
fi
if [[ ! "${adgpu_targets}" =~ ^[0-9]+([[:space:]]+[0-9]+)*$ ]]; then
    printf 'IGENVS_ADGPU_TARGETS must be a whitespace-separated numeric list\n' >&2
    exit 2
fi

# shellcheck source=containers/verify-sources.sh
source "${project_root}/containers/verify-sources.sh"
verify_igenvs_sources "${project_root}"

cd "${project_root}"
docker build \
    --platform linux/amd64 \
    --build-arg "CUDA_ARCHITECTURES=${cuda_architectures}" \
    --build-arg "ADGPU_TARGETS=${adgpu_targets}" \
    --tag "${image_tag}" \
    .

printf 'built %s for linux/amd64 (Uni-Dock=%s; AutoDock-GPU=%s)\n' \
    "${image_tag}" "${cuda_architectures}" "${adgpu_targets}"
printf 'verify with: docker run --rm --gpus all %q doctor\n' "${image_tag}"

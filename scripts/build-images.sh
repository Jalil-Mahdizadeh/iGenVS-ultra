#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
igenvs_tag="${IGENVS_DOCKER_IMAGE:-igenvs-ultra/igenvs:latest}"
gmolai_tag="${GMOLAI_DOCKER_IMAGE:-igenvs-ultra/gmolai:latest}"
decoder="${project_root}/gMolAI-v2.0/inference/models/decoder_inference.pt"

if ! command -v docker >/dev/null 2>&1; then
    printf 'error: Docker Engine with BuildKit is required\n' >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    printf 'error: the Docker daemon is unavailable to this user\n' >&2
    exit 1
fi
if [[ ! -f "${decoder}" ]]; then
    printf 'error: missing gMolAI decoder: %s\n' "${decoder}" >&2
    printf 'run git lfs pull and retry\n' >&2
    exit 1
fi
if head -n 1 "${decoder}" | grep -q '^version https://git-lfs.github.com/spec/'; then
    printf 'error: gMolAI model files are Git LFS pointers, not model data\n' >&2
    printf 'run git lfs pull and retry\n' >&2
    exit 1
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

printf 'Building iGenVS image: %s\n' "${igenvs_tag}"
IGENVS_DOCKER_TAG="${igenvs_tag}" \
    "${project_root}/iGenVS/containers/build-docker.sh" "${igenvs_tag}"

printf 'Building gMolAI image: %s\n' "${gmolai_tag}"
docker build \
    --platform linux/amd64 \
    --tag "${gmolai_tag}" \
    "${project_root}/gMolAI-v2.0"

docker run --rm "${igenvs_tag}" --version
docker run --rm "${gmolai_tag}" --version

printf '\nImages are ready:\n  %s\n  %s\n' "${igenvs_tag}" "${gmolai_tag}"
printf 'Run ./igenvs-ultra doctor to validate GPU access and release assets.\n'

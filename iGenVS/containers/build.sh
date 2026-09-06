#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_path="${IGENVS_IMAGE_PATH:-${project_root}/containers/iGenVS.SIF}"
definition_path="${project_root}/containers/iGenVS.def"
build_tmp="${APPTAINER_TMPDIR:-/tmp}"

# shellcheck source=containers/verify-sources.sh
source "${project_root}/containers/verify-sources.sh"
verify_igenvs_sources "${project_root}"

cd "${project_root}"
APPTAINER_TMPDIR="${build_tmp}" apptainer build --fakeroot --force \
    "${image_path}" "${definition_path}"

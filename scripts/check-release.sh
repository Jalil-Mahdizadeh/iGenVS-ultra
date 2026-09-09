#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
required=(
    "iGenVS/Dockerfile"
    "gMolAI-v2.0/Dockerfile"
    "gMolAI-v2.0/.dockerignore"
    "gMolAI-v2.0/inference/models/SHA256SUMS"
    "phase-5-head-selection/artifacts/input-standardizer.npz"
    "user-pipeline/igenvs-ultra"
    "user-pipeline/src/igenvs_ultra/rl_runtime.py"
    "user-pipeline/src/igenvs_ultra/core_runtime.py"
    "user-pipeline/src/igenvs_ultra/rl_workflow.py"
    "phase-10-rl-dev/freeze.json"
    "phase-10-rl-dev/maintenance.json"
    "phase-10-rl-dev/protocol.json"
    "phase-10-rl-dev/scripts/validate_target.py"
    "phase-10-rl-dev/src/igenvs_rl/cli.py"
    "examples/4ag8-screen/models/final.json"
)

for relative in "${required[@]}"; do
    if [[ ! -e "${project_root}/${relative}" ]]; then
        printf 'missing release file: %s\n' "${relative}" >&2
        exit 1
    fi
done

bash -n \
    "${project_root}/igenvs-ultra" \
    "${project_root}/scripts/build-images.sh" \
    "${project_root}/scripts/check-release.sh" \
    "${project_root}/iGenVS/containers/build-docker.sh"

(
    cd "${project_root}/gMolAI-v2.0/inference/models"
    sha256sum --check SHA256SUMS
)

(
    cd "${project_root}/phase-10-rl-dev"
    sha256sum --check freeze.sha256 protocol.sha256
)

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="${project_root}/user-pipeline/src" \
    python3 -m unittest discover \
        -s "${project_root}/user-pipeline/tests" \
        -p 'test_*.py'

python3 - "${project_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in (
    root / "examples/4ag8-screen/fit-config.json",
    root / "examples/4ag8-screen/fit-summary.json",
    root / "examples/4ag8-screen/models/final.json",
    root / "examples/4ag8-screen/models/round-5/ensemble-manifest.json",
):
    json.loads(path.read_text(encoding="utf-8"))
print("release metadata: valid JSON")
PY

if git -C "${project_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${project_root}" diff --check
    if git -C "${project_root}" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "${project_root}" show --check --format= HEAD
    else
        git -C "${project_root}" diff --cached --check
    fi
fi

printf 'release checks passed\n'

#!/usr/bin/env bash

# Shared source-integrity checks for the Apptainer and Docker builds.

verify_checkout() {
    local checkout_path="$1"
    local expected_commit="$2"
    if [[ ! -d "${checkout_path}" ]]; then
        printf 'missing required source tree: %s\n' "${checkout_path}" >&2
        return 1
    fi

    # Normal release clones contain vendored files without nested Git metadata.
    # Their exact contents are pinned by the parent iGenVS commit.
    if [[ ! -e "${checkout_path}/.git" ]]; then
        if ! git -C "${checkout_path}" rev-parse --is-inside-work-tree \
            >/dev/null 2>&1; then
            printf 'vendored source tree is outside a Git checkout: %s\n' \
                "${checkout_path}" >&2
            return 1
        fi
        local tracked
        tracked="$(git -C "${checkout_path}" ls-files -- .)"
        if [[ -z "${tracked}" ]]; then
            printf 'vendored source tree is not tracked: %s\n' \
                "${checkout_path}" >&2
            return 1
        fi
        return
    fi

    local actual_commit
    actual_commit="$(git -C "${checkout_path}" rev-parse --verify HEAD)"
    if [[ "${actual_commit}" != "${expected_commit}" ]]; then
        printf '%s is at %s; expected %s\n' \
            "${checkout_path}" \
            "${actual_commit}" \
            "${expected_commit}" >&2
        return 1
    fi
}

verify_clean_paths() {
    local checkout_path="$1"
    shift
    local required_path
    for required_path in "$@"; do
        if [[ "${required_path}" != "." && \
              ! -e "${checkout_path}/${required_path}" ]]; then
            printf 'missing required source path: %s/%s\n' \
                "${checkout_path}" "${required_path}" >&2
            return 1
        fi
    done

    local diff_base=(HEAD)
    if ! git -C "${checkout_path}" rev-parse --verify HEAD >/dev/null 2>&1; then
        # A newly assembled source release may be staged in an unborn parent
        # repository before its first commit. In that case the index is the
        # intended snapshot, so compare the worktree with it.
        diff_base=()
    fi
    if ! git -C "${checkout_path}" diff --quiet "${diff_base[@]}" -- "$@"; then
        printf 'copied source paths have tracked changes: %s\n' \
            "${checkout_path}" >&2
        return 1
    fi
    local untracked
    untracked="$(git -C "${checkout_path}" ls-files \
        --others --exclude-standard -- "$@")"
    if [[ -n "${untracked}" ]]; then
        printf 'copied source paths contain untracked files: %s\n%s\n' \
            "${checkout_path}" "${untracked}" >&2
        return 1
    fi
}

verify_igenvs_sources() {
    local project_root="$1"

    verify_checkout "${project_root}/iGen3" \
        9fc8fb4e3337712dda77e7614215f765e56f998b
    verify_checkout "${project_root}/third_party/Uni-Dock" \
        95e409172b15dec0989aea70b0f2328e8ca52025
    verify_checkout "${project_root}/third_party/AutoDock-GPU" \
        e63e6f6280ebfad18caa3e8f48afdc269e79e063
    verify_checkout "${project_root}/third_party/AutoGrid" \
        6d2847beaeac8ff43ca99094707fd74e3ca1ff37

    verify_clean_paths "${project_root}/iGen3" \
        pyproject.toml README.md requirements.txt src models examples
    verify_clean_paths "${project_root}/third_party/Uni-Dock" .
    verify_clean_paths "${project_root}/third_party/AutoDock-GPU" .
    verify_clean_paths "${project_root}/third_party/AutoGrid" .
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    verify_igenvs_sources "${project_root}"
fi

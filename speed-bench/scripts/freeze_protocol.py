#!/usr/bin/env python3
"""Freeze code, assets, containers, fixtures, and benchmark semantics."""

from __future__ import annotations

import json

from common import (
    GMOLAI_IMAGE,
    IGENVS_IMAGE,
    PROTOCOL,
    ROOT,
    SPEED,
    atomic_json,
    sha256,
    tree_sha256,
    utc_now,
)


def main() -> None:
    if PROTOCOL.exists():
        raise FileExistsError(f"refusing to replace locked protocol: {PROTOCOL}")
    fixture = SPEED / "inputs/fixed-libraries.manifest.json"
    if not fixture.is_file():
        raise FileNotFoundError("prepare the fixed docking library before locking the protocol")
    locked_files = {
        "speed-bench/plan.md": sha256(SPEED / "plan.md"),
        "user-pipeline/igenvs-ultra": sha256(ROOT / "user-pipeline/igenvs-ultra"),
        "speed-bench/inputs/fixed-libraries.manifest.json": sha256(fixture),
        "phase-7-benchmark-docking/targets/4ag8/manifest.json": sha256(
            ROOT / "phase-7-benchmark-docking/targets/4ag8/manifest.json"
        ),
        "phase-9-benchmark-active-learning/rounds/round-5/models/ensemble-manifest.json": sha256(
            ROOT / "phase-9-benchmark-active-learning/rounds/round-5/models/ensemble-manifest.json"
        ),
        "phase-5-head-selection/artifacts/input-standardizer.npz": sha256(
            ROOT / "phase-5-head-selection/artifacts/input-standardizer.npz"
        ),
    }
    for size in (20, 40, 80):
        relative = f"speed-bench/inputs/fixed-{size}k.csv"
        locked_files[relative] = sha256(ROOT / relative)
    containers = {}
    for label, path in (("igenvs", IGENVS_IMAGE), ("gmolai", GMOLAI_IMAGE)):
        stat = path.stat()
        containers[label] = {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path),
        }
    protocol = {
        "schema_version": 3,
        "status": "locked_before_execution",
        "locked_at": utc_now(),
        "hardware": "Arrhenius NVIDIA GH200 120GB",
        "fixture_sampling": {
            "temperature": 1.0,
            "top_k": 64,
            "seed": 2026090601,
        },
        "cold_policy": "fresh output and empty run-private performance-profile cache; one sample per case",
        "gpu_counts": [1, 2, 4],
        "docking": {
            "rows_per_gpu": 20_000,
            "target": "phase-7-benchmark-docking/targets/4ag8",
            "engine_modes": {
                "unidock": ["fast", "balance", "detail"],
                "autodock-gpu": ["fast"],
            },
            "pose_output": "none",
            "performance_settings": "all auto",
        },
        "screening": {
            "committed_finite_scores": 10_000_000,
            "model": "4ag8 AL round-5",
            "generator": "iGen3 base-isomeric de-novo",
            "performance_settings": "all auto through screen-fast count-only interface",
            "cross_batch_overlap": True,
            "save_policy": "all",
            "sampling": "public screen-fast defaults; molecule count is the only argument",
        },
        "primary_metrics": {
            "docking_input_per_hour": "all input rows / public-wrapper complete wall",
            "docking_successful_per_hour": "finite successful rows / public-wrapper complete wall",
            "docking_engine_per_hour": "finite successful rows / slowest concurrent engine wall",
            "screening_per_hour": "exact finite committed and saved scores / count-only invocation complete wall",
        },
        "locked_files": locked_files,
        "locked_source_trees": {
            "user-pipeline/src/igenvs_ultra": tree_sha256(
                ROOT / "user-pipeline/src/igenvs_ultra"
            ),
            "iGenVS/src/igenvs": tree_sha256(ROOT / "iGenVS/src/igenvs"),
            "iGenVS/iGen3/src/igen3": tree_sha256(ROOT / "iGenVS/iGen3/src/igen3"),
            "gMolAI-v2.0": tree_sha256(ROOT / "gMolAI-v2.0"),
            "speed-bench/scripts": tree_sha256(SPEED / "scripts"),
            "speed-bench/slurm": tree_sha256(SPEED / "slurm", ("*.sbatch",)),
        },
        "locked_asset_trees": {
            "phase-7-benchmark-docking/targets/4ag8": tree_sha256(
                ROOT / "phase-7-benchmark-docking/targets/4ag8", ("*",)
            ),
            "phase-9-benchmark-active-learning/rounds/round-5/models": tree_sha256(
                ROOT / "phase-9-benchmark-active-learning/rounds/round-5/models",
                ("*",),
            ),
            "iGenVS/iGen3/models": tree_sha256(ROOT / "iGenVS/iGen3/models", ("*",)),
            "gMolAI-v2.0/inference/models": tree_sha256(
                ROOT / "gMolAI-v2.0/inference/models", ("*",)
            ),
        },
        "containers": containers,
    }
    atomic_json(PROTOCOL, protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

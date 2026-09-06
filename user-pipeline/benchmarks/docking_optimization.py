#!/usr/bin/env python3
"""Reproducible one-GPU smoke/medium check for optimized regular docking.

This is deliberately not the locked 1/2/4-GPU speed benchmark.  It exercises
the user-facing ``igenvs-ultra dock`` path and records invocation-to-durable-
result wall time against a deterministic prefix of the locked 20k library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "user-pipeline/benchmarks/results"
FIXED_LIBRARY = ROOT / "speed-bench/inputs/fixed-20k.csv"
TARGET = ROOT / "phase-7-benchmark-docking/targets/4ag8"
IGENVS_IMAGE = ROOT.parent / "iGenVS/containers/iGenVS.SIF"

LOCKED_ONE_GPU = {
    ("unidock", "fast"): {
        "wall_seconds": 432.427,
        "input_per_hour": 166_502,
        "successful_engine_per_hour": 246_377,
    },
    ("unidock", "balance"): {
        "wall_seconds": 1_138.114,
        "input_per_hour": 63_263,
        "successful_engine_per_hour": 71_363,
    },
    ("unidock", "detail"): {
        "wall_seconds": 1_402.366,
        "input_per_hour": 51_342,
        "successful_engine_per_hour": 56_206,
    },
    ("autodock-gpu", "fast"): {
        "wall_seconds": 2_156.219,
        "input_per_hour": 33_392,
        "successful_engine_per_hour": 34_885,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stage_prefix(destination: Path, count: int) -> int:
    with FIXED_LIBRARY.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        with destination.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            written = 0
            for row in reader:
                if written == count:
                    break
                writer.writerow(row)
                written += 1
            target.flush()
            os.fsync(target.fileno())
    return written


def hardware() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "affinity_cpus": (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        ),
        "nvidia_smi": query.stdout.strip().splitlines() if query.returncode == 0 else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("unidock", "autodock-gpu"), required=True)
    parser.add_argument("--mode", choices=("fast", "balance", "detail"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.count < 1 or args.count > 20_000:
        parser.error("--count must be between 1 and 20000")

    case = (args.engine, args.mode)
    if case not in LOCKED_ONE_GPU:
        parser.error("the comparison suite covers Uni-Dock all modes and AutoDock-GPU fast")
    for required in (FIXED_LIBRARY, TARGET / "manifest.json", IGENVS_IMAGE):
        if not required.exists():
            raise FileNotFoundError(required)

    result_dir = RESULT_ROOT / args.label
    if result_dir.exists():
        raise FileExistsError(result_dir)
    result_dir.mkdir(parents=True)
    source = result_dir / "input.csv"
    if stage_prefix(source, args.count) != args.count:
        raise RuntimeError("locked library was shorter than requested benchmark prefix")

    job = result_dir / "job"
    command = [
        str(ROOT / "user-pipeline/igenvs-ultra"),
        "dock",
        "--execution", "apptainer",
        "--assets-dir", str(ROOT),
        "--igenvs-image", str(IGENVS_IMAGE),
        "--target", str(TARGET),
        "--input", str(source),
        "--input-format", "csv",
        "--smiles-column", "smiles",
        "--id-column", "molecule_id",
        "--output-dir", str(job),
        "--engine", args.engine,
        "--search-mode", args.mode,
        "--scoring", "vina" if args.engine == "unidock" else "ad4",
        "--pose-output", "none",
        "--docking-gpus", "1",
        "--batch-size", "auto",
        "--prep-workers", "auto",
        "--validation-workers", "auto",
        "--embed-max-attempts", "auto",
        "--embed-timeout", "auto",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "APPTAINERENV_PYTHONPATH": str(ROOT / "iGenVS/src"),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = result_dir / "pipeline.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    wall = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"docking failed with exit {completed.returncode}; see {log_path}"
        )

    manifest_path = job / "docking/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest["counts"]
    timings = manifest["timings"]
    consumed = int(manifest["validation"]["input"])
    if consumed != args.count or manifest.get("status") != "complete":
        raise RuntimeError(
            f"run did not certify the requested input: {consumed}/{args.count}"
        )
    successful = int(counts["docked"])
    engine_wall = float(timings["docking_wall_seconds"])
    baseline = LOCKED_ONE_GPU[case]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "label": args.label,
        "engine": args.engine,
        "mode": args.mode,
        "input_rows": args.count,
        "prepared_rows": int(counts["prepared"]),
        "successful_rows": successful,
        "preparation_failed": int(counts["preparation_failed"]),
        "preparation_timed_out": int(counts.get("preparation_timed_out", 0)),
        "launcher_wall_seconds": wall,
        "engine_wall_seconds": engine_wall,
        "input_per_hour_e2e": args.count * 3600.0 / wall,
        "successful_per_hour_e2e": successful * 3600.0 / wall,
        "successful_per_hour_engine_wall": successful * 3600.0 / engine_wall,
        "locked_20k_one_gpu_baseline": baseline,
        "relative_input_rate_vs_locked_baseline": (
            (args.count * 3600.0 / wall) / baseline["input_per_hour"]
        ),
        "configuration": manifest["runtime"],
        "timings": timings,
        "hardware": hardware(),
        "command": command,
        "input_sha256": sha256(source),
        "target_manifest_sha256": sha256(TARGET / "manifest.json"),
        "results_sha256": sha256(job / "docking/results.csv"),
        "igenvs_source_sha256": sha256(ROOT / "iGenVS/src/igenvs/pipeline.py"),
        "preparation_source_sha256": sha256(
            ROOT / "iGenVS/src/igenvs/preparation.py"
        ),
        "autodock_gpu_source_sha256": sha256(
            ROOT / "iGenVS/src/igenvs/autodock_gpu.py"
        ),
        "workflow_source_sha256": sha256(
            ROOT / "user-pipeline/src/igenvs_ultra/workflow.py"
        ),
    }
    atomic_json(result_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

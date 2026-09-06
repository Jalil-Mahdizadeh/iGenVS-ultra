#!/usr/bin/env python3
"""Reproducible smoke/medium benchmark for the persistent ultra screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "user-pipeline/benchmarks/results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stage_model_job(job: Path) -> Path:
    release_root = ROOT / "phase-9-benchmark-active-learning/rounds/round-5"
    released_path = release_root / "models/ensemble-manifest.json"
    released = json.loads(released_path.read_text(encoding="utf-8"))
    target = released["targets"]["4ag8"]
    members = []
    for member in target["members"]:
        checkpoint = release_root / member["checkpoint"]
        if sha256(checkpoint) != member["checkpoint_sha256"]:
            raise RuntimeError(f"released checkpoint changed: {checkpoint}")
        members.append({**member, "checkpoint": str(checkpoint)})
    model_dir = job / "models/round-5"
    model_dir.mkdir(parents=True)
    manifest = model_dir / "ensemble-manifest.json"
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "target": "4ag8",
            "stage": "round-5",
            "architecture": released["architecture"],
            "round": 5,
            "standardizer_sha256": released["standardizer_sha256"],
            "members": members,
        },
    )
    atomic_json(job / "fit-config.json", {"target_name": "4ag8", "benchmark_stub": True})
    atomic_json(
        job / "models/final.json",
        {
            "schema_version": 1,
            "status": "complete",
            "stage": "round-5",
            "ensemble_manifest": str(manifest.relative_to(job)),
            "ensemble_manifest_sha256": sha256(manifest),
        },
    )
    return manifest


def hardware() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
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
        "cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "nvidia_smi": result.stdout.strip().splitlines() if result.returncode == 0 else [],
    }


def collect(screen: Path) -> dict[str, Any]:
    stages = {
        "generation_seconds": 0.0,
        "generated_rows": 0,
        "validation_seconds": 0.0,
        "admission_seconds": 0.0,
        "score_seconds": 0.0,
        "policy_seconds": 0.0,
        "encoding_seconds": 0.0,
        "head_seconds": 0.0,
        "encoder_batch_sizes": [],
        "encoder_calibrations": [],
    }
    batches = sorted(screen.glob("batches/batch-*"))
    for batch in batches:
        generation = json.loads((batch / "generation.json").read_text(encoding="utf-8"))
        validation = json.loads((batch / "validation/validation.json").read_text(encoding="utf-8"))
        admission = json.loads((batch / "admission.json").read_text(encoding="utf-8"))
        score = json.loads((batch / "scores.manifest.json").read_text(encoding="utf-8"))
        stages["generation_seconds"] += float(generation["parallel_elapsed_seconds"])
        stages["generated_rows"] += int(generation["produced"])
        stages["validation_seconds"] += float(validation.get("elapsed_seconds", 0.0))
        stages["admission_seconds"] += float(admission.get("elapsed_seconds", 0.0))
        stages["score_seconds"] += float(
            score.get("parallel_score_wall_seconds", score.get("score_wall_seconds", 0.0))
        )
        stages["policy_seconds"] += float(score["encoder"].get("policy_seconds", 0.0))
        stages["encoding_seconds"] += float(score["encoder"].get("encoding_seconds", 0.0))
        stages["head_seconds"] += float(score.get("inference_seconds_member_sum", 0.0))
        selected_batch = score["encoder"].get("batch_size")
        if (
            selected_batch is not None
            and selected_batch not in stages["encoder_batch_sizes"]
        ):
            stages["encoder_batch_sizes"].append(selected_batch)
        calibration = score["encoder"].get("batch_calibration")
        if (
            calibration is not None
            and calibration not in stages["encoder_calibrations"]
        ):
            stages["encoder_calibrations"].append(calibration)
    return {**stages, "batches": len(batches)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--stream-batch-size", default="auto")
    parser.add_argument("--generator-batch-size", default="auto")
    parser.add_argument("--encoder-batch-size", default="auto")
    parser.add_argument("--compile-generator", action="store_true")
    parser.add_argument("--ephemeral", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")

    result_dir = RESULT_ROOT / args.label
    if result_dir.exists():
        raise RuntimeError(f"benchmark label already exists: {result_dir}")
    result_dir.mkdir(parents=True)
    scratch_parent = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    scratch = Path(tempfile.mkdtemp(prefix=f"igenvs-speed-opt-{args.label}-", dir=scratch_parent))
    job = scratch / "job"
    job.mkdir()
    stage_model_job(job)
    command = [
        str(ROOT / "user-pipeline/igenvs-ultra"),
        "screen",
        "--execution",
        "apptainer",
        "--assets-dir",
        str(ROOT),
        "--igenvs-image",
        str(ROOT.parent / "iGenVS/containers/iGenVS.SIF"),
        "--gmolai-image",
        str(ROOT.parent / "gMolAI/containers/gmolai-pyg-25.09-arm64.sif"),
        "--job-dir",
        str(job),
        "--generate-count",
        str(args.count),
        "--model",
        "base-isomeric",
        "--generation-mode",
        "de-novo",
        "--temperature",
        "1.0",
        "--top-k",
        "64",
        "--generator-seed",
        "2026090601",
        "--generator-batch-size",
        str(args.generator_batch_size),
        "--stream-batch-size",
        str(args.stream_batch_size),
        "--encoder-batch-size",
        str(args.encoder_batch_size),
        "--encoder-backend",
        "optimized",
        "--encoder-workers",
        "auto",
        "--encoder-threads",
        "4",
        "--screen-name",
        "speed-opt",
        "--save-policy",
        "all",
    ]
    if args.compile_generator:
        command.append("--compile-generator")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    if args.ephemeral:
        environment["IGENVS_ULTRA_EPHEMERAL_WORKERS"] = "1"
    log_path = result_dir / "pipeline.log"
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8", newline="") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment
            )
        wall = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(f"screen failed with {completed.returncode}; see {log_path}")
        screen = job / "screens/speed-opt"
        manifest = json.loads((screen / "manifest.json").read_text(encoding="utf-8"))
        stages = collect(screen)
        if int(manifest["encoded_rows"]) != args.count:
            raise RuntimeError(
                f"screen committed {manifest['encoded_rows']:,}/{args.count:,} scores"
            )
        if int(manifest["saved_rows"]) != args.count:
            raise RuntimeError("all-score benchmark did not save every committed score")
        if not bool(manifest.get("exact_requested_score_count")):
            raise RuntimeError("exact requested-score completion was not certified")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "label": args.label,
            "count": args.count,
            "wall_seconds": wall,
            "molecules_per_second": args.count / wall,
            "molecules_per_hour": args.count * 3600.0 / wall,
            "configuration": {
                "stream_batch_size": args.stream_batch_size,
                "generator_batch_size": args.generator_batch_size,
                "encoder_batch_size": args.encoder_batch_size,
                "compile_generator": args.compile_generator,
                "execution_engine": manifest["execution_engine"],
            },
            "stages": stages,
            "workers": manifest.get("workers"),
            "screen_manifest": manifest,
            "result_sha256": manifest["results_sha256"],
            "pipeline_workflow_sha256": sha256(
                ROOT / "user-pipeline/src/igenvs_ultra/workflow.py"
            ),
            "generation_worker_sha256": sha256(
                ROOT / "user-pipeline/src/igenvs_ultra/generation_worker.py"
            ),
            "model_ops_sha256": sha256(
                ROOT / "user-pipeline/src/igenvs_ultra/model_ops.py"
            ),
            "hardware": hardware(),
        }
        atomic_json(result_dir / "summary.json", summary)
        for source_name in ("config.json", "workers.json", "manifest.json"):
            source = screen / source_name
            if source.is_file():
                shutil.copy2(source, result_dir / source_name)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()

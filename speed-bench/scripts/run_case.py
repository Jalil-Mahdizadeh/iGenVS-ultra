#!/usr/bin/env python3
"""Run one cold docking or ultra-screening benchmark case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

from common import (
    IGENVS_IMAGE,
    PROTOCOL,
    ROOT,
    SPEED,
    TARGET,
    atomic_json,
    benchmark_environment,
    copy_debug_logs,
    hardware_snapshot,
    load_and_verify_protocol,
    load_json,
    require_gpu_allocation,
    sha256,
    stage_model_job,
    utc_now,
)


def run_logged(
    command: list[str], log_path: Path, environment: dict[str, str], *, cwd: Path
) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode, time.perf_counter() - started


def assert_new_result(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"cold benchmark result already exists: {path}")
    path.mkdir(parents=True)


def aggregate_docking_stages(
    manifest: dict[str, Any], job: Path, wrapper: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shard_paths = sorted((job / "docking/runs").glob("shard-*/manifest.json"))
    shards = [load_json(path) for path in shard_paths] if shard_paths else [manifest]
    timings = [item.get("timings", {}) for item in shards]

    def total(key: str) -> float:
        return sum(float(item.get(key, 0.0)) for item in timings)

    def critical(key: str) -> float:
        return max((float(item.get(key, 0.0)) for item in timings), default=0.0)

    shared_validation = manifest.get("validation", {})
    validation_seconds = float(shared_validation.get("elapsed_seconds", 0.0))
    if not validation_seconds:
        validation_seconds = critical("validation_seconds")
    stages = {
        "target_setup_seconds": float(
            wrapper.get("timings", {}).get("target_setup_seconds", 0.0)
        ),
        "validation_seconds": validation_seconds,
        "preparation_cpu_seconds_sum": total("preparation_cpu_seconds_sum"),
        "preparation_wait_seconds_critical": critical("preparation_wait_seconds"),
        "engine_wall_seconds_critical": critical("docking_wall_seconds"),
        "engine_invocation_seconds_critical": critical("docking_invocation_seconds"),
        "engine_worker_seconds_sum": total("docking_invocation_seconds"),
        "result_processing_seconds_critical": critical("result_processing_seconds"),
        "result_processing_seconds_sum": total("result_processing_seconds"),
        "cleanup_seconds_critical": critical("cleanup_seconds"),
        "cleanup_seconds_sum": total("cleanup_seconds"),
        "shard_screening_seconds_critical": critical("screening_seconds"),
        "wrapper_docking_stage_seconds": float(
            wrapper.get("timings", {}).get("docking_stage_seconds", 0.0)
        ),
    }
    return stages, shards


def validate_docking_results(path: Path) -> tuple[int, int]:
    rows = 0
    successful = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            if row.get("status") == "success":
                value = float(row["docking_score"])
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite docking score at output row {rows}")
                successful += 1
    return rows, successful


def run_docking(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    total_rows = 20_000 * args.gpus
    case = f"{args.engine}-{args.mode}-{args.gpus}gpu-cold"
    result = SPEED / "results/docking" / case
    assert_new_result(result)
    scratch_parent = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    scratch = Path(tempfile.mkdtemp(prefix=f"igenvs-{case}-", dir=scratch_parent))
    profile = scratch / "profile"
    runtime_scratch = scratch / "runtime"
    profile.mkdir()
    runtime_scratch.mkdir()
    job = result / "job"
    log = result / "pipeline.log"
    input_path = SPEED / f"inputs/fixed-{total_rows // 1000}k.csv"
    command = [
        str(ROOT / "user-pipeline/igenvs-ultra"),
        "dock",
        "--execution",
        "apptainer",
        "--assets-dir",
        str(ROOT),
        "--igenvs-image",
        str(IGENVS_IMAGE),
        "--target",
        str(TARGET),
        "--input",
        str(input_path),
        "--input-format",
        "csv",
        "--smiles-column",
        "smiles",
        "--id-column",
        "molecule_id",
        "--output-dir",
        str(job),
        "--engine",
        args.engine,
        "--search-mode",
        args.mode,
        "--pose-output",
        "none",
    ]
    environment = benchmark_environment(profile, runtime_scratch)
    atomic_json(
        result / "state.json",
        {
            "status": "running",
            "started_at": utc_now(),
            "case": case,
            "command": command,
            "cold_profile": str(profile),
            "hardware": hardware_snapshot(),
        },
    )
    try:
        returncode, wall = run_logged(command, log, environment, cwd=ROOT)
        if returncode:
            raise RuntimeError(f"public docking wrapper exited {returncode}; see {log}")
        wrapper = load_json(job / "regular-summary.json")
        manifest_path = job / "docking/manifest.json"
        manifest = load_json(manifest_path)
        config = load_json(job / "regular-config.json")
        automatic = config["docking"]
        expected_auto = {
            "batch_size": "auto",
            "prep_workers": "auto",
            "validation_workers": "auto",
            "embed_max_attempts": "auto",
            "embed_timeout": "auto",
            "docking_gpus": "auto",
            "adgpu_workers": "auto",
        }
        for key, expected in expected_auto.items():
            if automatic.get(key) != expected:
                raise RuntimeError(f"{case} did not leave {key} automatic: {automatic.get(key)}")
        results_path = Path(manifest["outputs"]["results"])
        output_rows, finite_successful = validate_docking_results(results_path)
        if output_rows != total_rows:
            raise RuntimeError(f"{case} emitted {output_rows:,}/{total_rows:,} result rows")
        successful = int(manifest.get("counts", {}).get("docked", 0))
        if finite_successful != successful:
            raise RuntimeError(
                f"{case} finite success count {finite_successful:,} != manifest {successful:,}"
            )
        stages, shards = aggregate_docking_stages(manifest, job, wrapper)
        empty_engine_shards = [
            index
            for index, shard in enumerate(shards)
            if int(shard.get("counts", {}).get("prepared", 0)) > 0
            and int(shard.get("counts", {}).get("docked", 0)) == 0
        ]
        if empty_engine_shards:
            raise RuntimeError(
                f"{case} had prepared molecules but zero successful docks on shards "
                f"{empty_engine_shards}; treating this as an engine/GPU failure"
            )
        engine_wall = stages["engine_wall_seconds_critical"]
        if wall <= 0 or engine_wall <= 0 or successful <= 0:
            raise RuntimeError(f"{case} has invalid timing or successful count")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "case": case,
            "sample": "single cold run",
            "engine": args.engine,
            "mode": args.mode,
            "gpus": args.gpus,
            "input_rows": total_rows,
            "output_rows": output_rows,
            "prepared_rows": int(manifest.get("counts", {}).get("prepared", 0)),
            "successful_rows": successful,
            "yield": successful / total_rows,
            "complete_wall_seconds": wall,
            "input_per_hour_complete_wall": total_rows * 3600.0 / wall,
            "successful_per_hour_complete_wall": successful * 3600.0 / wall,
            "successful_per_hour_engine_wall": successful * 3600.0 / engine_wall,
            "stage_timings": stages,
            "automatic_configuration": automatic,
            "resolved_shard_runtime": [item.get("runtime", {}) for item in shards],
            "counts": manifest.get("counts", {}),
            "result_sha256": sha256(results_path),
            "input_sha256": sha256(input_path),
            "target_manifest_sha256": sha256(TARGET / "manifest.json"),
            "protocol_sha256": sha256(PROTOCOL),
            "command": command,
            "hardware": hardware_snapshot(),
        }
        atomic_json(result / "summary.json", summary)
        atomic_json(result / "state.json", {"status": "complete", "completed_at": utc_now()})
        print(json.dumps(summary, indent=2, sort_keys=True))
    except BaseException as exc:
        if job.exists():
            copy_debug_logs(job, result)
        atomic_json(
            result / "failure.json",
            {"status": "failed", "failed_at": utc_now(), "error": repr(exc)},
        )
        raise
    finally:
        shutil.rmtree(scratch)


def score_stage_units(batch: Path, score: dict[str, Any]) -> list[dict[str, Any]]:
    paths = sorted(batch.glob("score-shards/shard-*/scores.manifest.json"))
    return [load_json(path) for path in paths] if paths else [score]


def collect_screen_stages(screen: Path) -> dict[str, Any]:
    stages: dict[str, Any] = {
        "stream_batches": 0,
        "generated_valid_rows": 0,
        "candidate_slots": 0,
        "generation_seconds_critical_sum": 0.0,
        "validation_seconds_sum": 0.0,
        "admission_seconds_sum": 0.0,
        "score_stage_seconds_critical_sum": 0.0,
        "policy_seconds_critical_sum": 0.0,
        "encoding_seconds_critical_sum": 0.0,
        "head_inference_seconds_critical_sum": 0.0,
        "encoder_batch_sizes": [],
        "generator_batch_sizes": [],
    }
    for batch in sorted(screen.glob("batches/batch-*")):
        generation_path = batch / "generation.json"
        score_path = batch / "scores.manifest.json"
        if not generation_path.is_file() or not score_path.is_file():
            raise RuntimeError(f"incomplete terminal screen batch: {batch}")
        generation = load_json(generation_path)
        validation = load_json(batch / "validation/validation.json")
        admission = load_json(batch / "admission.json")
        score = load_json(score_path)
        units = score_stage_units(batch, score)
        stages["stream_batches"] += 1
        stages["generated_valid_rows"] += int(generation["produced"])
        stages["generation_seconds_critical_sum"] += float(
            generation["parallel_elapsed_seconds"]
        )
        for shard in generation.get("shards", []):
            generator = shard.get("generator", {})
            stages["candidate_slots"] += int(
                generator.get("candidates_generated", shard.get("produced", 0))
            )
            selected = generator.get("batch_size")
            if selected and selected not in stages["generator_batch_sizes"]:
                stages["generator_batch_sizes"].append(selected)
        stages["validation_seconds_sum"] += float(validation.get("elapsed_seconds", 0.0))
        stages["admission_seconds_sum"] += float(admission.get("elapsed_seconds", 0.0))
        stages["score_stage_seconds_critical_sum"] += float(
            score.get("parallel_score_wall_seconds", score.get("score_wall_seconds", 0.0))
        )
        stages["policy_seconds_critical_sum"] += max(
            float(item.get("encoder", {}).get("policy_seconds", 0.0)) for item in units
        )
        stages["encoding_seconds_critical_sum"] += max(
            float(item.get("encoder", {}).get("encoding_seconds", 0.0)) for item in units
        )
        stages["head_inference_seconds_critical_sum"] += max(
            float(item.get("inference_seconds_member_sum", 0.0)) for item in units
        )
        for item in units:
            selected = item.get("encoder", {}).get("batch_size")
            if selected and selected not in stages["encoder_batch_sizes"]:
                stages["encoder_batch_sizes"].append(selected)
    if stages["candidate_slots"] == 0:
        stages["candidate_slots"] = stages["generated_valid_rows"]
    return stages


def validate_screen_results(path: Path, expected: int) -> dict[str, Any]:
    numeric = [
        "member_probability_260904",
        "member_probability_260905",
        "member_probability_260906",
        "ensemble_probability",
        "ensemble_mutual_information",
    ]
    rows = 0
    first: list[dict[str, str]] = []
    last: deque[dict[str, str]] = deque(maxlen=3)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(numeric).difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"score output lacks numeric fields: {sorted(missing)}")
        for row in reader:
            rows += 1
            for field in numeric:
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(f"non-finite {field} at score row {rows}")
            if len(first) < 3:
                first.append(row)
            last.append(row)
    if rows != expected:
        raise RuntimeError(f"screen saved {rows:,}/{expected:,} finite score rows")
    return {"rows": rows, "first": first, "last": list(last)}


def run_screening(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    case = f"screening-{args.gpus}gpu-cold"
    result = SPEED / "results/screening" / case
    assert_new_result(result)
    scratch_parent = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    scratch = Path(tempfile.mkdtemp(prefix=f"igenvs-{case}-", dir=scratch_parent))
    profile = scratch / "profile"
    runtime_scratch = scratch / "runtime"
    job = scratch / "job"
    profile.mkdir()
    runtime_scratch.mkdir()
    job.mkdir()
    stage_model_job(job)
    expected = int(protocol["screening"]["committed_finite_scores"])
    command = [str(ROOT / "user-pipeline/igenvs-ultra"), "screen-fast", str(expected)]
    environment = benchmark_environment(profile, runtime_scratch)
    environment["IGENVS_ULTRA_JOB"] = str(job)
    log = result / "pipeline.log"
    atomic_json(
        result / "state.json",
        {
            "status": "running",
            "started_at": utc_now(),
            "case": case,
            "command": command,
            "cold_profile": str(profile),
            "hardware": hardware_snapshot(),
        },
    )
    completed_successfully = False
    try:
        returncode, wall = run_logged(command, log, environment, cwd=job)
        if returncode:
            raise RuntimeError(f"count-only screen exited {returncode}; see {log}")
        screen = job / f"screens/fast-{expected}"
        manifest = load_json(screen / "manifest.json")
        config = load_json(screen / "config.json")
        if not (
            int(manifest["encoded_rows"]) == expected
            and int(manifest["saved_rows"]) == expected
            and bool(manifest.get("exact_requested_score_count"))
        ):
            raise RuntimeError("screen did not commit and save exactly the requested score count")
        if manifest.get("save_policy") != "all":
            raise RuntimeError("count-only benchmark did not retain all scores")
        if not manifest.get("cross_batch_overlap") or not config.get("cross_batch_overlap"):
            raise RuntimeError("cross-batch generation/scoring overlap was not active")
        automatic_checks = {
            "stream_batch": config["stream_batch_decision"].get("requested"),
            "generator_batch": config["source"].get("generator_batch_size"),
            "encoder_batch": config["encoder"].get("batch_size"),
            "validation_workers": config.get("validation_workers"),
        }
        if any(value != "auto" for value in automatic_checks.values()):
            raise RuntimeError(f"screen performance setting was not automatic: {automatic_checks}")
        if int(config.get("screen_gpu_count", 0)) != args.gpus:
            raise RuntimeError(
                f"screen planned {config.get('screen_gpu_count')} scoring GPUs, expected {args.gpus}"
            )
        if int(config.get("generation_logical_shards", 0)) != args.gpus:
            raise RuntimeError(
                "automatic iGen3 fan-out did not match the visible GPU allocation"
            )
        workers = manifest.get("workers", {})
        if len(workers.get("generation", [])) != args.gpus:
            raise RuntimeError("not every allocated GPU received a persistent iGen3 worker")
        if len(workers.get("scoring", [])) != args.gpus:
            raise RuntimeError("not every allocated GPU received a persistent scoring worker")
        results_path = Path(manifest["results"])
        validation = validate_screen_results(results_path, expected)
        database = sqlite3.connect(str(screen / "dedup.sqlite3"))
        try:
            unique_admitted = int(database.execute("SELECT COUNT(*) FROM smiles").fetchone()[0])
        finally:
            database.close()
        if unique_admitted != int(manifest["admitted_unique_rows"]):
            raise RuntimeError("dedup authority count differs from admitted manifest count")
        stages = collect_screen_stages(screen)
        generation_seconds = stages["generation_seconds_critical_sum"]
        encoding_seconds = stages["encoding_seconds_critical_sum"]
        head_seconds = stages["head_inference_seconds_critical_sum"]
        if min(wall, generation_seconds, encoding_seconds, head_seconds) <= 0:
            raise RuntimeError("screen contains a non-positive required timing")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "case": case,
            "sample": "single cold run",
            "gpus": args.gpus,
            "requested_scores": expected,
            "committed_finite_scores": expected,
            "saved_scores": validation["rows"],
            "admitted_unique_rows": int(manifest["admitted_unique_rows"]),
            "candidate_slots": stages["candidate_slots"],
            "end_to_end_candidate_yield": expected / stages["candidate_slots"],
            "encoding_yield": expected / int(manifest["admitted_unique_rows"]),
            "complete_wall_seconds": wall,
            "screening_per_hour_complete_wall": expected * 3600.0 / wall,
            "generation_valid_per_hour": stages["generated_valid_rows"] * 3600.0 / generation_seconds,
            "generation_candidate_slots_per_hour": stages["candidate_slots"] * 3600.0 / generation_seconds,
            "encoding_per_hour": expected * 3600.0 / encoding_seconds,
            "head_inference_per_hour": expected * 3600.0 / head_seconds,
            "stage_timings": {
                **stages,
                **manifest.get("timings", {}),
            },
            "automatic_plan": {
                "stream_batch": config["stream_batch_decision"],
                "generation_logical_shards": config["generation_logical_shards"],
                "screen_gpu_count": config["screen_gpu_count"],
                "generator_batch_sizes": stages["generator_batch_sizes"],
                "encoder_batch_sizes": stages["encoder_batch_sizes"],
                "workers": manifest.get("workers", {}),
            },
            "cross_batch_overlap": True,
            "result_sha256": manifest["results_sha256"],
            "result_samples": {"first": validation["first"], "last": validation["last"]},
            "protocol_sha256": sha256(PROTOCOL),
            "command": command,
            "hardware": hardware_snapshot(),
        }
        atomic_json(result / "summary.json", summary)
        for name in ("config.json", "workers.json", "manifest.json"):
            source = screen / name
            if source.is_file():
                shutil.copy2(source, result / name)
        if profile.is_dir():
            shutil.copytree(profile, result / "cold-profile-records")
        copy_debug_logs(job, result)
        atomic_json(result / "state.json", {"status": "complete", "completed_at": utc_now()})
        completed_successfully = True
        print(json.dumps(summary, indent=2, sort_keys=True))
    except BaseException as exc:
        if job.exists():
            copy_debug_logs(job, result)
        atomic_json(
            result / "failure.json",
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": repr(exc),
                "preserved_scratch": str(scratch),
            },
        )
        raise
    finally:
        if completed_successfully:
            shutil.rmtree(scratch)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="kind", required=True)
    dock = subparsers.add_parser("docking")
    dock.add_argument("--engine", choices=("unidock", "autodock-gpu"), required=True)
    dock.add_argument("--mode", choices=("fast", "balance", "detail"), required=True)
    dock.add_argument("--gpus", type=int, choices=(1, 2, 4), required=True)
    screen = subparsers.add_parser("screening")
    screen.add_argument("--gpus", type=int, choices=(1, 2, 4), required=True)
    args = parser.parse_args()
    if args.kind == "docking" and args.engine == "autodock-gpu" and args.mode != "fast":
        parser.error("AutoDock-GPU is benchmarked only in fast mode")
    protocol = load_and_verify_protocol()
    require_gpu_allocation(args.gpus)
    if args.kind == "docking":
        run_docking(args, protocol)
    else:
        run_screening(args, protocol)


if __name__ == "__main__":
    main()

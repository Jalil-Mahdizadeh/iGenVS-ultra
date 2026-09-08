#!/usr/bin/env python3
"""Run one receptor through the universal-protocol development workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completed_updates(stage_dir: Path) -> int:
    history = stage_dir / "history.csv"
    if not history.is_file():
        return 0
    with history.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["update"]) if rows else 0


def _update_loop_seconds(stage_dir: Path) -> float:
    history = stage_dir / "history.csv"
    if not history.is_file():
        return 0.0
    with history.open("r", encoding="utf-8", newline="") as handle:
        return sum(float(row["seconds"]) for row in csv.DictReader(handle))


def _init_stage(
    *,
    wrapper: Path,
    stage_dir: Path,
    target_dir: Path,
    stage: dict[str, Any],
    protocol: dict[str, Any],
    gpus: int,
    initial_model: Path | None,
) -> None:
    if (stage_dir / "config.json").is_file():
        return
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise RuntimeError(f"incomplete non-empty stage directory: {stage_dir}")

    docking = protocol["docking"]
    sampling = protocol["sampling"]
    command = [
        str(wrapper),
        "init",
        "--job-dir",
        str(stage_dir),
        "--target",
        str(target_dir),
        "--model",
        protocol["base_model"],
        "--engine",
        docking["engine"],
        "--scoring",
        docking["scoring"],
        "--search-mode",
        str(stage.get("search_mode", docking["search_mode"])),
        "--shards",
        str(gpus),
        "--prep-workers",
        "16",
        "--validation-workers",
        "8",
        "--reference-count",
        str(protocol["reference_count"]),
        "--batch-size",
        str(stage["batch_size"]),
        "--evaluation-count",
        str(stage["evaluation_count"]),
        "--evaluation-every",
        str(stage["evaluation_every"]),
        "--temperature",
        str(sampling["temperature"]),
        "--top-k",
        str(sampling["top_k"]),
        "--generator-seed",
        str(stage["generator_seed"]),
        "--docking-seed",
        str(docking["seed"]),
        "--learning-rate",
        str(stage["learning_rate"]),
        "--kl-beta",
        str(stage["kl_beta"]),
        "--target-kl",
        str(stage["target_kl"]),
        "--reward-mode",
        stage["reward_mode"],
        "--tail-fraction",
        str(stage.get("tail_fraction", 0.1)),
        "--tail-weight",
        str(stage.get("tail_weight", 1.0)),
        "--elite-fraction",
        str(stage.get("elite_fraction", 0.01)),
        "--minimum-elite-unique",
        str(stage["minimum_elite_unique"]),
    ]
    if initial_model is not None:
        command.extend(["--initial-model-root", str(initial_model)])
    if stage["reward_seen_molecules"]:
        command.append("--reward-seen-molecules")
    if stage["require_lipinski"]:
        command.append("--require-lipinski")
    if float(stage.get("minimum_qed", 0.0)) > 0.0:
        command.extend(["--minimum-qed", str(stage["minimum_qed"])])
    if stage.get("maximum_absolute_formal_charge") is not None:
        command.extend(
            [
                "--maximum-absolute-formal-charge",
                str(stage["maximum_absolute_formal_charge"]),
            ]
        )
    if float(stage.get("minimum_fraction_csp3", 0.0)) > 0.0:
        command.extend(
            ["--minimum-fraction-csp3", str(stage["minimum_fraction_csp3"])]
        )
    if stage.get("maximum_aromatic_rings") is not None:
        command.extend(
            ["--maximum-aromatic-rings", str(stage["maximum_aromatic_rings"])]
        )
    if stage.get("reward_occurrence_cap") is not None:
        command.extend(
            ["--reward-occurrence-cap", str(stage["reward_occurrence_cap"])]
        )
    if stage.get("fresh_evaluation_docking", False):
        command.append("--fresh-evaluation-docking")
    if float(stage.get("maximum_top_molecule_fraction", 1.0)) < 1.0:
        command.extend(
            [
                "--maximum-top-molecule-fraction",
                str(stage["maximum_top_molecule_fraction"]),
            ]
        )
    subprocess.run(command, check=True)


def _copy_reference(source_stage: Path, destination_stage: Path) -> None:
    source = source_stage / "reference"
    destination = destination_stage / "reference"
    if destination.is_dir():
        if _sha256(source / "scores.json") != _sha256(destination / "scores.json"):
            raise RuntimeError("stage reference does not match the frozen target reference")
        return
    if not (source / "scores.json").is_file():
        raise RuntimeError(f"source reference is incomplete: {source}")
    shutil.copytree(source, destination)


def _run_training_to(
    *,
    wrapper: Path,
    stage_dir: Path,
    requested_total: int,
    stage_name: str,
    timing_records: list[dict[str, Any]],
    gpus: int,
) -> None:
    completed = _completed_updates(stage_dir)
    if completed > requested_total:
        raise RuntimeError(f"{stage_name} has {completed} updates, beyond requested {requested_total}")
    remaining = requested_total - completed
    if remaining == 0:
        print(f"[development] {stage_name}: already at update {completed}", flush=True)
        return

    started_at = _utc_now()
    started = time.perf_counter()
    subprocess.run(
        [str(wrapper), "train", "--job-dir", str(stage_dir), "--updates", str(remaining)],
        check=True,
    )
    elapsed = time.perf_counter() - started
    timing_records.append(
        {
            "stage": stage_name,
            "updates_before": completed,
            "updates_after": requested_total,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "wall_seconds": elapsed,
            "gpu_count": gpus,
            "gpu_hours": elapsed * gpus / 3600.0,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "hostname": socket.gethostname(),
        }
    )


def _evaluation_rows(stage_dir: Path) -> list[dict[str, Any]]:
    path = stage_dir / "evaluations.csv"
    if not path.is_file():
        return []
    parsed: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("label", "")
            if not label.startswith("update-"):
                continue
            try:
                update = int(label.removeprefix("update-"))
                fraction = float(row["qualified_elite_fraction"])
                unique = int(float(row["qualified_elite_unique_count"]))
                chemistry = float(row["chemistry_fraction"])
                top_molecule = float(row["top_molecule_fraction"])
                positive = float(row["positive_score_fraction"])
            except (KeyError, TypeError, ValueError):
                continue
            parsed.append(
                {
                    "update": update,
                    "qualified_elite_fraction": fraction,
                    "qualified_elite_unique_count": unique,
                    "chemistry_fraction": chemistry,
                    "top_molecule_fraction": top_molecule,
                    "positive_score_fraction": positive,
                }
            )
    return sorted(parsed, key=lambda row: int(row["update"]))


def _gate_check(stage_dir: Path, rule: dict[str, Any], update: int) -> dict[str, Any]:
    consecutive = int(rule["consecutive_evaluations"])
    rows = [row for row in _evaluation_rows(stage_dir) if int(row["update"]) <= update]
    selected = rows[-consecutive:]
    expected_updates = list(range(update - consecutive + 1, update + 1))
    observed_updates = [int(row["update"]) for row in selected]
    individual_passes = [
        float(row["qualified_elite_fraction"])
        >= float(rule["minimum_qualified_elite_fraction"])
        and int(row["qualified_elite_unique_count"])
        >= int(rule["minimum_qualified_elite_unique"])
        and float(row["chemistry_fraction"])
        >= float(rule["minimum_chemistry_fraction"])
        and float(row["top_molecule_fraction"])
        <= float(rule["maximum_top_molecule_fraction"])
        and float(row["positive_score_fraction"])
        <= float(rule["maximum_positive_score_fraction"])
        for row in selected
    ]
    passed = observed_updates == expected_updates and len(selected) == consecutive and all(individual_passes)
    return {
        "checked_at_update": update,
        "expected_evaluation_updates": expected_updates,
        "evaluations": selected,
        "individual_passes": individual_passes,
        "passed": passed,
    }


def _write_timing(
    path: Path,
    *,
    target: str,
    gpus: int,
    segments: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    stopping: dict[str, Any] | None,
) -> None:
    _atomic_json(
        path,
        {
            "target": target,
            "gpu_count": gpus,
            "segments": segments,
            "stages": [
                {
                    "stage": stage["name"],
                    "updates": _completed_updates(path.parent / stage["name"]),
                    "update_loop_seconds": _update_loop_seconds(path.parent / stage["name"]),
                }
                for stage in stages
            ],
            "adaptive_stopping": stopping,
            "training_wall_seconds": sum(float(row["wall_seconds"]) for row in segments),
            "training_gpu_hours": sum(float(row["gpu_hours"]) for row in segments),
            "updated_at": _utc_now(),
        },
    )


def _run_adaptive_stage(
    *,
    wrapper: Path,
    stage_dir: Path,
    stage: dict[str, Any],
    timing_records: list[dict[str, Any]],
    gpus: int,
    timing_path: Path,
    target: str,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    rule = stage["adaptive_stopping"]
    minimum = int(rule["minimum_updates"])
    block = int(rule["block_updates"])
    maximum = int(rule["maximum_updates"])
    if int(stage["updates"]) != minimum or minimum <= 0 or block <= 0 or maximum < minimum:
        raise ValueError("invalid adaptive stopping schedule")

    stopping_path = stage_dir / "stopping.json"
    if stopping_path.is_file():
        stopping = json.loads(stopping_path.read_text(encoding="utf-8"))
        if _completed_updates(stage_dir) != int(stopping["stopped_at_update"]):
            raise RuntimeError("completed updates differ from the recorded stopping decision")
        return stopping

    completed = _completed_updates(stage_dir)
    if completed < minimum:
        target_update = minimum
    elif (completed - minimum) % block:
        target_update = min(maximum, minimum + math.ceil((completed - minimum) / block) * block)
    else:
        target_update = completed

    checks: list[dict[str, Any]] = []
    while True:
        if target_update > completed:
            _run_training_to(
                wrapper=wrapper,
                stage_dir=stage_dir,
                requested_total=target_update,
                stage_name=stage["name"],
                timing_records=timing_records,
                gpus=gpus,
            )
            completed = _completed_updates(stage_dir)
            _write_timing(
                timing_path,
                target=target,
                gpus=gpus,
                segments=timing_records,
                stages=stages,
                stopping=None,
            )

        check = _gate_check(stage_dir, rule, completed)
        checks.append(check)
        if check["passed"]:
            status = "gate_met"
            break
        if completed >= maximum:
            status = "maximum_updates_reached"
            break
        target_update = min(maximum, completed + block)

    stopping = {
        "status": status,
        "gate_met": status == "gate_met",
        "stopped_at_update": completed,
        "rule": rule,
        "checks": checks,
        "recorded_at": _utc_now(),
    }
    _atomic_json(stopping_path, stopping)
    return stopping


def main() -> None:
    args = _arguments()
    phase_dir = Path(__file__).resolve().parents[1]
    repo_root = phase_dir.parent
    protocol_path = phase_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    targets = [
        line.strip()
        for line in (phase_dir / "targets.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.target not in targets:
        raise ValueError(f"target must be one of: {', '.join(targets)}")
    if args.gpus < 1:
        raise ValueError("--gpus must be positive")

    prepared_target = (repo_root / protocol["prepared_target_root"] / args.target).resolve()
    if not (prepared_target / "manifest.json").is_file():
        raise FileNotFoundError(f"prepared target is missing: {prepared_target}")
    wrapper = (phase_dir / "scripts" / "igenvs-rl").resolve()
    target_root = (phase_dir / args.target).resolve()
    training_root = target_root / "training"
    target_root.mkdir(parents=True, exist_ok=True)

    protocol_hash = _sha256(protocol_path)
    protocol_migrated = False
    target_manifest = target_root / "run.json"
    if target_manifest.is_file():
        recorded = json.loads(target_manifest.read_text(encoding="utf-8"))
        if recorded["protocol_sha256"] != protocol_hash:
            lineage = protocol.get("development_lineage", {})
            parent_hash = lineage.get("parent_protocol_sha256")
            reused_stages = list(lineage.get("reused_stages", []))
            if recorded["protocol_sha256"] != parent_hash or not reused_stages:
                raise RuntimeError("existing target run uses an unrelated protocol")
            incomplete = [
                name
                for name in reused_stages
                if not (training_root / name / "model-latest").is_dir()
            ]
            if incomplete:
                raise RuntimeError(
                    "cannot reuse incomplete parent stages: " + ", ".join(incomplete)
                )
            recorded.update(
                {
                    "parent_protocol_sha256": parent_hash,
                    "protocol_sha256": protocol_hash,
                    "reused_parent_stages": reused_stages,
                    "protocol_migrated_at": _utc_now(),
                }
            )
            _atomic_json(target_manifest, recorded)
            protocol_migrated = True
    else:
        _atomic_json(
            target_manifest,
            {
                "target": args.target,
                "prepared_target": str(prepared_target),
                "prepared_target_manifest_sha256": _sha256(prepared_target / "manifest.json"),
                "protocol": str(protocol_path.resolve()),
                "protocol_sha256": protocol_hash,
                "created_at": _utc_now(),
            },
        )

    timing_path = training_root / "timing.json"
    timing_records = []
    if timing_path.is_file():
        timing_records = list(json.loads(timing_path.read_text(encoding="utf-8")).get("segments", []))
    if protocol_migrated:
        reused = set(protocol["development_lineage"]["reused_stages"])
        timing_records = [row for row in timing_records if row.get("stage") in reused]

    previous_stage: Path | None = None
    stopping: dict[str, Any] | None = None
    stages = protocol["stages"]
    for stage in stages:
        stage_dir = training_root / stage["name"]
        initial_model = previous_stage / "model-latest" if previous_stage else None
        if initial_model is not None and not initial_model.is_dir():
            raise FileNotFoundError(f"previous-stage latest model is missing: {initial_model}")
        _init_stage(
            wrapper=wrapper,
            stage_dir=stage_dir,
            target_dir=prepared_target,
            stage=stage,
            protocol=protocol,
            gpus=args.gpus,
            initial_model=initial_model,
        )
        first_mode = str(stages[0].get("search_mode", protocol["docking"]["search_mode"]))
        stage_mode = str(stage.get("search_mode", protocol["docking"]["search_mode"]))
        if previous_stage is not None and stage_mode == first_mode:
            _copy_reference(training_root / stages[0]["name"], stage_dir)

        if "adaptive_stopping" in stage:
            stopping = _run_adaptive_stage(
                wrapper=wrapper,
                stage_dir=stage_dir,
                stage=stage,
                timing_records=timing_records,
                gpus=args.gpus,
                timing_path=timing_path,
                target=args.target,
                stages=stages,
            )
        else:
            _run_training_to(
                wrapper=wrapper,
                stage_dir=stage_dir,
                requested_total=int(stage["updates"]),
                stage_name=stage["name"],
                timing_records=timing_records,
                gpus=args.gpus,
            )
        if not (stage_dir / "model-latest").is_dir():
            raise RuntimeError(f"latest model export is missing after {stage['name']}")
        previous_stage = stage_dir
        _write_timing(
            timing_path,
            target=args.target,
            gpus=args.gpus,
            segments=timing_records,
            stages=stages,
            stopping=stopping,
        )

    if args.skip_validation:
        return
    if previous_stage is None:
        raise RuntimeError("the protocol contains no stages")
    image = Path(
        os.environ.get(
            "IGENVS_IMAGE",
            "/nobackup/proj/disk/theo-storage/personal/jalil/iGenVS/containers/iGenVS.SIF",
        )
    )
    python_path = ":".join(
        [
            str(phase_dir / "src"),
            str(repo_root / "iGenVS" / "iGen3" / "src"),
            str(repo_root / "iGenVS" / "src"),
        ]
    )
    validation = protocol["validation"]
    subprocess.run(
        [
            "apptainer",
            "exec",
            "--nv",
            "--bind",
            f"{repo_root}:{repo_root}",
            "--env",
            f"PYTHONPATH={python_path}",
            str(image),
            "python3",
            str(phase_dir / "scripts" / "validate_target.py"),
            "--target-id",
            args.target,
            "--target",
            str(prepared_target),
            "--final-stage",
            str(previous_stage),
            "--output-dir",
            str(target_root / "validation"),
            "--protocol",
            str(protocol_path),
            "--count",
            str(validation["raw_draws_per_arm"]),
            "--seed",
            str(validation["sampling_seed"]),
            "--shards",
            str(args.gpus),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

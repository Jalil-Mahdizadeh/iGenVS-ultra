"""User-facing orchestration for the accepted target-specific iGen3 RL protocol."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any

from .workflow import (
    PipelineError,
    Runtime,
    atomic_json,
    make_target_config,
    prepare_target,
    resolve_assets,
    sha256,
    utc_now,
    visible_gpu_tokens,
)


FROZEN_PROTOCOL_SHA256 = "4b9aa7fad0e563bddb28de5edc96d061ac24b1165b4f324a9711e60416d0c03b"
FROZEN_BUNDLE_SHA256 = "32bb9d74ef4354c9fab44a9f01bfe037e2d889b39043854c27f662040e08e3e2"
RL_MAINTENANCE_SHA256 = "cb1195dc144b997e5bde741b56ef807f494e2316b63e2fd3dd9f236ad1c008ae"
FROZEN_PADDING_ANGSTROM = 5.0
FROZEN_BUNDLE = "phase-10-rl-dev"
RL_MODEL_ID = "base-isomeric"
RL_MODEL_DIRECTORY = "base_isomeric"
RL_WEIGHTS = "iGen3_base_isomeric_256d.pth"
RL_VOCAB = "vocab.pkl"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"required RL file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid JSON in RL file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"RL JSON root must be an object: {path}")
    return payload


def frozen_rl_bundle(assets: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Locate and verify the byte-frozen protocol and implementation bundle."""
    phase = assets / FROZEN_BUNDLE
    freeze_path = phase / "freeze.json"
    protocol_path = phase / "protocol.json"
    maintenance_path = phase / "maintenance.json"
    freeze = _read_json(freeze_path)
    if (
        freeze.get("status") != "accepted_and_frozen"
        or sha256(freeze_path) != FROZEN_BUNDLE_SHA256
    ):
        raise PipelineError(f"RL bundle is not marked accepted and frozen: {freeze_path}")
    recorded = freeze.get("protocol", {}).get("sha256")
    observed = sha256(protocol_path) if protocol_path.is_file() else None
    if recorded != FROZEN_PROTOCOL_SHA256 or observed != FROZEN_PROTOCOL_SHA256:
        raise PipelineError(
            "frozen RL protocol integrity check failed; restore phase-10-rl-dev/protocol.json"
        )
    maintenance = _read_json(maintenance_path)
    if (
        sha256(maintenance_path) != RL_MAINTENANCE_SHA256
        or maintenance.get("status") != "operational_maintenance"
        or maintenance.get("base_freeze", {}).get("sha256") != FROZEN_BUNDLE_SHA256
        or maintenance.get("protocol", {}).get("sha256") != FROZEN_PROTOCOL_SHA256
    ):
        raise PipelineError(f"RL maintenance integrity check failed: {maintenance_path}")
    implementation = dict(freeze.get("implementation_sha256", {}))
    overrides = maintenance.get("implementation_sha256_overrides", {})
    if not isinstance(overrides, dict) or not set(overrides).issubset(implementation):
        raise PipelineError(f"RL maintenance overrides are invalid: {maintenance_path}")
    implementation.update(overrides)
    for relative, expected in implementation.items():
        path = phase / relative
        if not path.is_file() or sha256(path) != expected:
            raise PipelineError(f"frozen RL implementation integrity check failed: {path}")
    return phase, protocol_path, _read_json(protocol_path)


def _runtime_command(assets: Path, *arguments: str) -> list[str]:
    bootstrap = assets / "user-pipeline/src/igenvs_ultra/rl_runtime.py"
    if not bootstrap.is_file():
        raise PipelineError(f"RL runtime bootstrap is missing: {bootstrap}")
    return ["python3", str(bootstrap), *arguments]


def _completed_updates(stage_dir: Path) -> int:
    progress = stage_dir / "progress.json"
    if progress.is_file():
        payload = _read_json(progress)
        completed = int(payload.get("completed_updates", -1))
        if payload.get("status") == "complete" and completed >= 0:
            return completed
    history = stage_dir / "history.csv"
    if not history.is_file():
        return 0
    with history.open("r", encoding="utf-8", newline="") as handle:
        completed = 0
        for row in csv.DictReader(handle):
            completed = int(row["update"])
    return completed


def _update_loop_seconds(stage_dir: Path) -> float:
    history = stage_dir / "history.csv"
    if not history.is_file():
        return 0.0
    with history.open("r", encoding="utf-8", newline="") as handle:
        return sum(float(row["seconds"]) for row in csv.DictReader(handle))


def _make_rl_config(
    args: Any,
    assets: Path,
    protocol: dict[str, Any],
    gpu_count: int,
) -> dict[str, Any]:
    config = make_target_config(args, assets)
    config.update(
        {
            "workflow": "target-specific-iGen3-RL",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": FROZEN_PROTOCOL_SHA256,
            "gpu_count": gpu_count,
        }
    )
    return config


def _ensure_rl_config(job: Path, desired: dict[str, Any]) -> dict[str, Any]:
    path = job / "rl-config.json"
    if path.is_file():
        existing = _read_json(path)
        if existing != desired:
            raise PipelineError(
                f"existing RL job has different target, protocol, or GPU settings: {path}; "
                "use a new --output-dir"
            )
        return existing
    if job.exists() and any(job.iterdir()):
        raise PipelineError(f"output directory is non-empty but is not an RL job: {job}")
    job.mkdir(parents=True, exist_ok=True)
    atomic_json(path, desired)
    return desired


def _validate_prepared_target(target: Path) -> None:
    pocket = _read_json(target / "pocket.json")
    try:
        padding = float(pocket["padding_angstrom"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"prepared target does not record pocket padding: {target}") from exc
    if not math.isclose(padding, FROZEN_PADDING_ANGSTROM, abs_tol=1e-9):
        raise PipelineError(
            f"the frozen RL protocol requires {FROZEN_PADDING_ANGSTROM:g} A target padding; "
            f"the prepared target records {padding:g} A"
        )


def _expected_stage_settings(
    *,
    assets: Path,
    target_dir: Path,
    stage: dict[str, Any],
    protocol: dict[str, Any],
    gpu_count: int,
    initial_model: Path | None,
) -> dict[str, Any]:
    docking = protocol["docking"]
    sampling = protocol["sampling"]
    return {
        "schema_version": 1,
        "target": str(target_dir.resolve()),
        "igenvs_project": str((assets / "iGenVS").resolve()),
        "model_root": str((assets / "iGenVS/iGen3/models").resolve()),
        "model_id": protocol["base_model"],
        "oracle": "igenvs",
        "engine": docking["engine"],
        "scoring": docking["scoring"],
        "search_mode": str(stage.get("search_mode", docking["search_mode"])),
        "shards": gpu_count,
        "prep_workers": 16,
        "validation_workers": 8,
        "batch_size": stage["batch_size"],
        "reference_count": protocol["reference_count"],
        "evaluation_count": stage["evaluation_count"],
        "evaluation_every": stage["evaluation_every"],
        "temperature": sampling["temperature"],
        "top_k": sampling["top_k"],
        "generator_seed": stage["generator_seed"],
        "docking_seed": docking["seed"],
        "learning_rate": stage["learning_rate"],
        "kl_beta": stage["kl_beta"],
        "target_kl": stage["target_kl"],
        "max_grad_norm": 1.0,
        "reward_mode": stage["reward_mode"],
        "tail_fraction": stage.get("tail_fraction", 0.1),
        "tail_weight": stage.get("tail_weight", 1.0),
        "reward_seen_molecules": stage["reward_seen_molecules"],
        "elite_fraction": stage.get("elite_fraction", 0.01),
        "initial_model_root": str(initial_model.resolve()) if initial_model else None,
        "minimum_elite_unique": stage["minimum_elite_unique"],
        "require_lipinski": stage["require_lipinski"],
        "minimum_qed": stage.get("minimum_qed", 0.0),
        "maximum_absolute_formal_charge": stage.get("maximum_absolute_formal_charge"),
        "minimum_fraction_csp3": stage.get("minimum_fraction_csp3", 0.0),
        "maximum_aromatic_rings": stage.get("maximum_aromatic_rings"),
        "reward_occurrence_cap": stage.get("reward_occurrence_cap"),
        "fresh_evaluation_docking": stage.get("fresh_evaluation_docking", False),
        "maximum_top_molecule_fraction": stage.get("maximum_top_molecule_fraction", 1.0),
    }


def _verify_stage_config(path: Path, expected: dict[str, Any]) -> None:
    observed = _read_json(path)
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise PipelineError(
            f"RL stage config differs from the frozen protocol ({', '.join(mismatches)}): {path}"
        )


def _init_stage(
    *,
    runtime: Runtime,
    assets: Path,
    job: Path,
    stage_dir: Path,
    target_dir: Path,
    stage: dict[str, Any],
    protocol: dict[str, Any],
    gpu_count: int,
    initial_model: Path | None,
) -> None:
    config_path = stage_dir / "config.json"
    expected = _expected_stage_settings(
        assets=assets,
        target_dir=target_dir,
        stage=stage,
        protocol=protocol,
        gpu_count=gpu_count,
        initial_model=initial_model,
    )
    if config_path.is_file():
        _verify_stage_config(config_path, expected)
        return
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise PipelineError(f"incomplete non-empty RL stage directory: {stage_dir}")

    docking = protocol["docking"]
    sampling = protocol["sampling"]
    arguments = [
        "init",
        "--job-dir",
        str(stage_dir),
        "--target",
        str(target_dir),
        "--image",
        str(runtime.igenvs_image),
        "--igenvs-project",
        str(assets / "iGenVS"),
        "--model-root",
        str(assets / "iGenVS/iGen3/models"),
        "--model",
        protocol["base_model"],
        "--engine",
        docking["engine"],
        "--scoring",
        docking["scoring"],
        "--search-mode",
        str(stage.get("search_mode", docking["search_mode"])),
        "--shards",
        str(gpu_count),
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
        arguments.extend(["--initial-model-root", str(initial_model)])
    if stage["reward_seen_molecules"]:
        arguments.append("--reward-seen-molecules")
    if stage["require_lipinski"]:
        arguments.append("--require-lipinski")
    if float(stage.get("minimum_qed", 0.0)) > 0.0:
        arguments.extend(["--minimum-qed", str(stage["minimum_qed"])])
    if stage.get("maximum_absolute_formal_charge") is not None:
        arguments.extend(
            ["--maximum-absolute-formal-charge", str(stage["maximum_absolute_formal_charge"])]
        )
    if float(stage.get("minimum_fraction_csp3", 0.0)) > 0.0:
        arguments.extend(["--minimum-fraction-csp3", str(stage["minimum_fraction_csp3"])])
    if stage.get("maximum_aromatic_rings") is not None:
        arguments.extend(["--maximum-aromatic-rings", str(stage["maximum_aromatic_rings"])])
    if stage.get("reward_occurrence_cap") is not None:
        arguments.extend(["--reward-occurrence-cap", str(stage["reward_occurrence_cap"])])
    if stage.get("fresh_evaluation_docking", False):
        arguments.append("--fresh-evaluation-docking")
    if float(stage.get("maximum_top_molecule_fraction", 1.0)) < 1.0:
        arguments.extend(
            ["--maximum-top-molecule-fraction", str(stage["maximum_top_molecule_fraction"])]
        )

    runtime.run_logged(
        "igenvs",
        _runtime_command(assets, *arguments),
        job / f"logs/rl-{stage['name']}-init.log",
        gpu=False,
    )
    _verify_stage_config(config_path, expected)


def _copy_reference(source_stage: Path, destination_stage: Path) -> None:
    source = source_stage / "reference"
    destination = destination_stage / "reference"
    if not (source / "scores.json").is_file():
        raise PipelineError(f"source RL reference is incomplete: {source}")
    if destination.is_dir() and not (destination / "scores.json").is_file():
        destination.rename(destination.with_name(f"reference.incomplete-{time.time_ns()}"))
    if destination.is_dir():
        if not (source / "scores.json").is_file() or sha256(source / "scores.json") != sha256(
            destination / "scores.json"
        ):
            raise PipelineError("RL stage reference does not match the frozen target reference")
        return
    if not (source / "scores.json").is_file():
        raise PipelineError(f"source RL reference is incomplete: {source}")
    temporary = destination.with_name(f"reference.copying-{time.time_ns()}")
    shutil.copytree(source, temporary)
    temporary.rename(destination)


def _recover_stage(runtime: Runtime, assets: Path, job: Path, stage_dir: Path) -> int:
    runtime.run_logged(
        "igenvs",
        _runtime_command(assets, "recover", "--job-dir", str(stage_dir)),
        job / f"logs/rl-{stage_dir.name}-recover.log",
        gpu=False,
    )
    return _completed_updates(stage_dir)


def _run_training_to(
    *,
    runtime: Runtime,
    assets: Path,
    job: Path,
    stage_dir: Path,
    requested_total: int,
    stage_name: str,
    timing_records: list[dict[str, Any]],
    gpu_count: int,
) -> None:
    completed = _recover_stage(runtime, assets, job, stage_dir)
    progress_path = stage_dir / "progress.json"
    has_checkpoint_progress = progress_path.is_file()
    if completed > requested_total and has_checkpoint_progress:
        raise PipelineError(
            f"RL stage {stage_name} has {completed} updates, beyond requested {requested_total}"
        )
    remaining = max(0, requested_total - completed)
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    export_ready = bool(
        progress.get("completed_updates") == requested_total
        and progress.get("model_latest_exported")
    )
    if remaining == 0 and export_ready:
        print(f"[iGenVS-ultra] RL {stage_name}: already at update {completed}", flush=True)
        return

    started_at = utc_now()
    started = time.perf_counter()
    runtime.run_logged(
        "igenvs",
        _runtime_command(
            assets,
            "train",
            "--job-dir",
            str(stage_dir),
            "--target-update",
            str(requested_total),
        ),
        job / f"logs/rl-{stage_name}-{completed + 1}-to-{requested_total}.log",
        gpu=True,
    )
    observed = _completed_updates(stage_dir)
    if observed != requested_total:
        raise PipelineError(
            f"RL {stage_name} stopped at update {observed}; expected {requested_total}"
        )
    elapsed = time.perf_counter() - started
    timing_records.append(
        {
            "stage": stage_name,
            "updates_before": completed,
            "updates_after": requested_total,
            "started_at": started_at,
            "ended_at": utc_now(),
            "wall_seconds": elapsed,
            "gpu_count": gpu_count,
            "gpu_hours": elapsed * gpu_count / 3600.0,
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
                parsed.append(
                    {
                        "update": int(label.removeprefix("update-")),
                        "qualified_elite_fraction": float(row["qualified_elite_fraction"]),
                        "qualified_elite_unique_count": int(float(row["qualified_elite_unique_count"])),
                        "chemistry_fraction": float(row["chemistry_fraction"]),
                        "top_molecule_fraction": float(row["top_molecule_fraction"]),
                        "positive_score_fraction": float(row["positive_score_fraction"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(parsed, key=lambda row: int(row["update"]))


def _gate_check(stage_dir: Path, rule: dict[str, Any], update: int) -> dict[str, Any]:
    consecutive = int(rule["consecutive_evaluations"])
    rows = [row for row in _evaluation_rows(stage_dir) if int(row["update"]) <= update]
    selected = rows[-consecutive:]
    expected = list(range(update - consecutive + 1, update + 1))
    observed = [int(row["update"]) for row in selected]
    individual = [
        float(row["qualified_elite_fraction"])
        >= float(rule["minimum_qualified_elite_fraction"])
        and int(row["qualified_elite_unique_count"])
        >= int(rule["minimum_qualified_elite_unique"])
        and float(row["chemistry_fraction"]) >= float(rule["minimum_chemistry_fraction"])
        and float(row["top_molecule_fraction"])
        <= float(rule["maximum_top_molecule_fraction"])
        and float(row["positive_score_fraction"])
        <= float(rule["maximum_positive_score_fraction"])
        for row in selected
    ]
    return {
        "checked_at_update": update,
        "expected_evaluation_updates": expected,
        "evaluations": selected,
        "individual_passes": individual,
        "passed": observed == expected and len(selected) == consecutive and all(individual),
    }


def _write_training_timing(
    path: Path,
    *,
    target: str,
    gpu_count: int,
    segments: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    stopping: dict[str, Any] | None,
) -> None:
    atomic_json(
        path,
        {
            "target": target,
            "gpu_count": gpu_count,
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
            "updated_at": utc_now(),
        },
    )


def _run_adaptive_stage(
    *,
    runtime: Runtime,
    assets: Path,
    job: Path,
    stage_dir: Path,
    stage: dict[str, Any],
    timing_records: list[dict[str, Any]],
    gpu_count: int,
    timing_path: Path,
    target: str,
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    rule = stage["adaptive_stopping"]
    minimum = int(rule["minimum_updates"])
    block = int(rule["block_updates"])
    maximum = int(rule["maximum_updates"])
    if int(stage["updates"]) != minimum or minimum <= 0 or block <= 0 or maximum < minimum:
        raise PipelineError("frozen RL protocol has an invalid adaptive stopping schedule")

    # Even a recorded stopping decision may precede an interrupted model
    # export. Legacy history is never allowed to choose the next boundary.
    completed = _recover_stage(runtime, assets, job, stage_dir)
    stopping_path = stage_dir / "stopping.json"
    if stopping_path.is_file():
        stopping = _read_json(stopping_path)
        if stopping.get("rule") != rule:
            raise PipelineError(f"recorded RL stopping rule differs from the frozen protocol: {stopping_path}")
        stopped_at = int(stopping["stopped_at_update"])
        if completed == stopped_at:
            return stopping
        if completed > stopped_at:
            raise PipelineError("completed RL updates exceed the recorded stopping decision")
        stopping_path.rename(stopping_path.with_name(f"stopping.incomplete-{time.time_ns()}.json"))

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
                runtime=runtime,
                assets=assets,
                job=job,
                stage_dir=stage_dir,
                requested_total=target_update,
                stage_name=stage["name"],
                timing_records=timing_records,
                gpu_count=gpu_count,
            )
            completed = _completed_updates(stage_dir)
            _write_training_timing(
                timing_path,
                target=target,
                gpu_count=gpu_count,
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
        "recorded_at": utc_now(),
    }
    atomic_json(stopping_path, stopping)
    return stopping


def _publish_model(
    job: Path,
    final_stage: Path,
    target_name: str,
) -> Path:
    source = final_stage / "model"
    source_weights = source / RL_MODEL_DIRECTORY / RL_WEIGHTS
    source_vocab = source / RL_MODEL_DIRECTORY / RL_VOCAB
    if not source_weights.is_file() or not source_vocab.is_file():
        raise PipelineError(f"selected RL model export is incomplete: {source}")
    checkpoint = final_stage / "checkpoints/best.pt"
    if not checkpoint.is_file():
        raise PipelineError(f"final RL stage has no selected best checkpoint: {checkpoint}")

    destination = job / "model"
    if destination.exists() and not destination.is_dir():
        raise PipelineError(f"published RL model path is not a directory: {destination}")
    if destination.is_dir():
        destination_weights = destination / RL_MODEL_DIRECTORY / RL_WEIGHTS
        destination_vocab = destination / RL_MODEL_DIRECTORY / RL_VOCAB
        if (
            not destination_weights.is_file()
            or not destination_vocab.is_file()
            or sha256(destination_weights) != sha256(source_weights)
            or sha256(destination_vocab) != sha256(source_vocab)
        ):
            raise PipelineError(f"published RL model differs from the selected checkpoint: {destination}")
    else:
        temporary = job / "model.partial"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)

    manifest = {
        "schema_version": 1,
        "model_id": RL_MODEL_ID,
        "target": target_name,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "selected_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "selection": "best checkpoint selected by the frozen online evaluator",
        },
        "artifacts": {
            str(Path(RL_MODEL_DIRECTORY) / RL_WEIGHTS): sha256(source_weights),
            str(Path(RL_MODEL_DIRECTORY) / RL_VOCAB): sha256(source_vocab),
        },
        "published_at": utc_now(),
    }
    atomic_json(destination / "manifest.json", manifest)
    return destination


def train_rl(args: Any) -> dict[str, Any]:
    assets = resolve_assets(args)
    _, _, protocol = frozen_rl_bundle(assets)
    job = Path(args.output_dir).expanduser().resolve()
    gpu_tokens = visible_gpu_tokens(getattr(args, "gpu_ids", None))
    gpu_count = len(gpu_tokens)
    config = _make_rl_config(args, assets, protocol, gpu_count)
    runtime = Runtime(args, assets, job, require_gmolai=False)

    if getattr(args, "dry_run", False):
        plan = {
            "command": "rl-train",
            "job": str(job),
            "runtime": runtime.execution,
            "visible_gpu_ids": gpu_tokens,
            "target": config,
            "frozen_protocol": {
                "id": protocol["protocol_id"],
                "sha256": FROZEN_PROTOCOL_SHA256,
                "stages": [
                    {
                        "name": stage["name"],
                        "updates": stage["updates"],
                        "maximum_updates": stage.get("adaptive_stopping", {}).get(
                            "maximum_updates", stage["updates"]
                        ),
                        "search_mode": stage.get("search_mode", protocol["docking"]["search_mode"]),
                    }
                    for stage in protocol["stages"]
                ],
            },
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan

    if gpu_count < 1:
        raise PipelineError("RL training requires at least one visible NVIDIA GPU")
    config = _ensure_rl_config(job, config)
    summary_path = job / "rl-summary.json"
    if summary_path.is_file():
        completed = _read_json(summary_path)
        if (
            completed.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256
            or completed.get("target") != config["target_name"]
        ):
            raise PipelineError(f"completed RL summary does not match this job: {summary_path}")
        _resolve_model_dir(job / "model")
        print(f"[iGenVS-ultra] RL job already complete; model: {job / 'model'}", flush=True)
        return completed
    target_dir = prepare_target(args, job, runtime, config)
    _validate_prepared_target(target_dir)

    training_root = job / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    timing_path = training_root / "timing.json"
    timing_records = (
        list(_read_json(timing_path).get("segments", [])) if timing_path.is_file() else []
    )
    stages = list(protocol["stages"])
    previous_stage: Path | None = None
    stopping: dict[str, Any] | None = None
    for stage in stages:
        stage_dir = training_root / stage["name"]
        initial_model = previous_stage / "model-latest" if previous_stage else None
        if initial_model is not None and not initial_model.is_dir():
            raise PipelineError(f"previous RL stage model is missing: {initial_model}")
        _init_stage(
            runtime=runtime,
            assets=assets,
            job=job,
            stage_dir=stage_dir,
            target_dir=target_dir,
            stage=stage,
            protocol=protocol,
            gpu_count=gpu_count,
            initial_model=initial_model,
        )
        first_mode = str(stages[0].get("search_mode", protocol["docking"]["search_mode"]))
        stage_mode = str(stage.get("search_mode", protocol["docking"]["search_mode"]))
        if previous_stage is not None and stage_mode == first_mode:
            _copy_reference(training_root / stages[0]["name"], stage_dir)

        if "adaptive_stopping" in stage:
            stopping = _run_adaptive_stage(
                runtime=runtime,
                assets=assets,
                job=job,
                stage_dir=stage_dir,
                stage=stage,
                timing_records=timing_records,
                gpu_count=gpu_count,
                timing_path=timing_path,
                target=config["target_name"],
                stages=stages,
            )
        else:
            _run_training_to(
                runtime=runtime,
                assets=assets,
                job=job,
                stage_dir=stage_dir,
                requested_total=int(stage["updates"]),
                stage_name=stage["name"],
                timing_records=timing_records,
                gpu_count=gpu_count,
            )
        if not (stage_dir / "model-latest").is_dir():
            raise PipelineError(f"latest model export is missing after RL stage {stage['name']}")
        previous_stage = stage_dir
        _write_training_timing(
            timing_path,
            target=config["target_name"],
            gpu_count=gpu_count,
            segments=timing_records,
            stages=stages,
            stopping=stopping,
        )

    if previous_stage is None:
        raise PipelineError("frozen RL protocol contains no training stages")
    model_dir = _publish_model(job, previous_stage, config["target_name"])
    timing = _read_json(timing_path)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "target": config["target_name"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "gpu_count": gpu_count,
        "training_wall_seconds": timing.get("training_wall_seconds"),
        "training_gpu_hours": timing.get("training_gpu_hours"),
        "training_timing": str(timing_path),
        "model_dir": str(model_dir),
        "model_manifest": str(model_dir / "manifest.json"),
    }
    atomic_json(job / "rl-summary.json", summary)
    print(f"[iGenVS-ultra] RL training complete; model: {model_dir}", flush=True)
    return summary


def _resolve_model_dir(value: Path) -> tuple[Path, dict[str, Any]]:
    model_dir = value.expanduser().resolve()
    manifest_path = model_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise PipelineError(f"unsupported RL model manifest schema: {manifest_path}")
    if manifest.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256:
        raise PipelineError(f"model was not produced by the frozen RL protocol: {model_dir}")
    if manifest.get("model_id") != RL_MODEL_ID:
        raise PipelineError(f"unsupported target-specific RL model layout: {model_dir}")
    artifacts = manifest.get("artifacts", {})
    for relative in (str(Path(RL_MODEL_DIRECTORY) / RL_WEIGHTS), str(Path(RL_MODEL_DIRECTORY) / RL_VOCAB)):
        path = model_dir / relative
        if not path.is_file() or artifacts.get(relative) != sha256(path):
            raise PipelineError(f"RL model integrity check failed: {path}")
    return model_dir, manifest


def generate_rl(args: Any) -> dict[str, Any]:
    assets = resolve_assets(args)
    frozen_rl_bundle(assets)
    model_dir, model_manifest = _resolve_model_dir(Path(args.model_dir))
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise PipelineError("--output must use a .csv filename")
    runtime = Runtime(args, assets, output.parent, require_gmolai=False)
    plan = {
        "command": "rl-generate",
        "runtime": runtime.execution,
        "model_dir": str(model_dir),
        "target": model_manifest.get("target"),
        "count": int(args.count),
        "seed": int(args.seed),
        "output": str(output),
        "output_contract": "exactly N valid unique isomeric SMILES",
    }
    if getattr(args, "dry_run", False):
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan
    if output.exists():
        raise PipelineError(f"generation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_smiles = output.with_suffix(".smi.partial")
    temporary_csv = output.with_suffix(".csv.partial")
    temporary_smiles.unlink(missing_ok=True)
    temporary_csv.unlink(missing_ok=True)
    try:
        runtime.run_logged(
            "igenvs",
            [
                "igen3",
                "--model-dir",
                str(model_dir),
                "generate",
                "--model",
                RL_MODEL_ID,
                "--mode",
                "de-novo",
                "--output",
                str(temporary_smiles),
                "--count",
                str(args.count),
                "--batch-size",
                "auto",
                "--temperature",
                "1.0",
                "--top-k",
                "64",
                "--seed",
                str(args.seed),
                "--no-progress",
            ],
            output.with_suffix(".generation.log"),
            gpu=True,
            extra_paths=(model_dir, output.parent),
        )
        smiles = [
            line.strip()
            for line in temporary_smiles.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(smiles) != int(args.count):
            raise PipelineError(
                f"iGen3 produced {len(smiles)} valid unique SMILES, fewer than the requested "
                f"{args.count}; no CSV was committed"
            )
        if len(set(smiles)) != len(smiles):
            raise PipelineError("iGen3 returned duplicate SMILES; no CSV was committed")

        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["molecule_id", "smiles"], lineterminator="\n")
            writer.writeheader()
            for index, smiles_value in enumerate(smiles, start=1):
                writer.writerow({"molecule_id": f"rl_{index:08d}", "smiles": smiles_value})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_csv, output)
    finally:
        temporary_smiles.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)

    result = {
        **plan,
        "status": "complete",
        "rows": len(smiles),
        "output_sha256": sha256(output),
        "completed_at": utc_now(),
    }
    atomic_json(output.with_suffix(".manifest.json"), result)
    print(f"[iGenVS-ultra] wrote {len(smiles):,} valid unique RL SMILES to {output}", flush=True)
    return result


def rl_status(job: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "job": str(job),
        "exists": job.is_dir(),
        "workflow": "target-specific-iGen3-RL",
    }
    if not job.is_dir():
        return result
    for name in ("rl-config.json", "training/timing.json", "rl-summary.json"):
        path = job / name
        result[name] = _read_json(path) if path.is_file() else {"status": "not-complete"}
    result["model_ready"] = (job / "model/manifest.json").is_file()
    return result

"""Empirical outer-batch tuning for a receptor, box, and GPU."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from .docking import DockingConfig, dock_batch_resilient
from .errors import InputError
from .hardware import auto_worker_count, default_tuning_sizes, hardware_snapshot, selected_gpu
from .ingress import iter_source_records, iter_validated_records, validate_library
from .preparation import prepare_records
from .target import load_target


@dataclass(frozen=True)
class TuningConfig:
    input_path: Path
    receptor: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    profile_path: Path
    target: Path | None = None
    engine: str = "unidock"
    autodock_gpu_fld: Path | None = None
    batch_sizes: tuple[int, ...] = ()
    input_format: str = "auto"
    smiles_column: str = "smiles"
    id_column: str | None = None
    delimiter: str = "auto"
    prep_workers: str | int = "auto"
    prep_mode: str = "standard"
    fragment_policy: str = "reject"
    search_mode: str = "balance"
    scoring: str = "vina"
    num_modes: int = 1
    energy_range: float = 3.0
    refine_step: int = 3
    no_refine: bool = False
    unidock_verbosity: int = 0
    seed: int = 181129
    device_id: int = 0
    max_gpu_memory: int = 0
    scratch_dir: Path | None = None
    unidock_executable: str = "unidock"
    autodock_gpu_runs: int | None = None
    autodock_gpu_evaluations: int | None = None
    autodock_gpu_heuristics: bool = True
    autodock_gpu_autostop: bool = True
    autodock_gpu_local_search: str = "ad"
    autodock_gpu_cpu_threads: int = 4
    autodock_gpu_workers: int = 1
    autodock_gpu_executable: str = "autodock_gpu"


def _choose_batch(rows: list[dict[str, Any]]) -> int:
    eligible = [row for row in rows if row["success_fraction"] >= 0.99 and row["successful"] > 0]
    if not eligible:
        raise InputError("no tested batch size produced at least 99% finite outputs")
    return int(
        max(
            eligible,
            key=lambda row: (row["successful_ligands_per_second"], -row["batch_size"]),
        )["batch_size"]
    )


def tune_docking(config: TuningConfig) -> dict[str, Any]:
    input_path = config.input_path.expanduser().resolve()
    receptor = config.receptor.expanduser().resolve()
    if not input_path.is_file() or not receptor.is_file():
        raise InputError("tuning input library and receptor must both exist")
    if config.engine not in {"unidock", "autodock-gpu"}:
        raise InputError("engine must be unidock or autodock-gpu")
    if config.engine == "unidock" and config.scoring not in {"vina", "vinardo"}:
        raise InputError("Uni-Dock scoring must be vina or vinardo")
    if config.engine == "autodock-gpu" and config.scoring != "ad4":
        raise InputError("AutoDock-GPU scoring must be ad4")
    autodock_gpu_fld = (
        config.autodock_gpu_fld.expanduser().resolve()
        if config.autodock_gpu_fld is not None
        else None
    )
    if config.engine == "autodock-gpu":
        if autodock_gpu_fld is None or not autodock_gpu_fld.is_file():
            raise InputError("AutoDock-GPU tuning requires an existing .maps.fld grid")
        if config.num_modes != 1:
            raise InputError("AutoDock-GPU currently emits exactly one best pose")
        if not 1 <= config.autodock_gpu_workers <= 64:
            raise InputError("AutoDock-GPU worker count must be between 1 and 64")
    elif config.autodock_gpu_workers != 1:
        raise InputError("AutoDock-GPU workers cannot be used with Uni-Dock")
    prepared_target = None
    if config.target is not None:
        prepared_target = load_target(config.target)
        if receptor != prepared_target.receptor:
            raise InputError("tuning receptor does not match the prepared target")
        if tuple(config.center) != prepared_target.center or tuple(config.size) != prepared_target.size:
            raise InputError("tuning box does not match the prepared target")
        if (
            config.engine == "autodock-gpu"
            and autodock_gpu_fld != prepared_target.autodock_gpu_fld
        ):
            raise InputError("tuning grid does not match the prepared target")
    sizes = sorted(
        set(
            config.batch_sizes
            or tuple(
                default_tuning_sizes(
                    selected_gpu(config.device_id),
                    engine=config.engine,
                )
            )
        )
    )
    if not sizes or sizes[0] <= 0:
        raise InputError("at least one positive batch size is required")
    workers = auto_worker_count(config.prep_workers, cap=64)
    scratch_root = config.scratch_dir or Path(tempfile.gettempdir())
    scratch_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="igenvs-tune-", dir=scratch_root) as temporary:
        work = Path(temporary)
        validation = validate_library(
            iter_source_records(
                input_path,
                input_format=config.input_format,
                smiles_column=config.smiles_column,
                id_column=config.id_column,
                delimiter=config.delimiter,
            ),
            output_dir=work / "validation",
            workers=workers,
            fragment_policy=config.fragment_policy,
            deduplicate=True,
        )
        records = list(
            islice(iter_validated_records(Path(str(validation["validated_path"]))), max(sizes))
        )
        if not records:
            raise InputError("no valid molecules are available for tuning")
        prepared, prep_failures = prepare_records(
            records,
            output_dir=work / "prepared",
            workers=workers,
            seed=config.seed,
            max_atoms=256 if config.engine == "autodock-gpu" else 300,
            max_torsions=57 if config.engine == "autodock-gpu" else 48,
            prep_mode=config.prep_mode,
        )
        if not prepared:
            raise InputError("none of the pilot molecules could be prepared")

        dock_config = DockingConfig(
            receptor=receptor,
            center=config.center,
            size=config.size,
            engine=config.engine,
            search_mode=config.search_mode,
            scoring=config.scoring,
            num_modes=config.num_modes,
            energy_range=config.energy_range,
            seed=config.seed,
            device_id=config.device_id,
            max_gpu_memory=config.max_gpu_memory,
            refine_step=config.refine_step,
            no_refine=config.no_refine,
            verbosity=config.unidock_verbosity,
            scores_only_output=True,
            unidock_executable=config.unidock_executable,
            autodock_gpu_fld=autodock_gpu_fld,
            autodock_gpu_runs=config.autodock_gpu_runs,
            autodock_gpu_evaluations=config.autodock_gpu_evaluations,
            autodock_gpu_heuristics=config.autodock_gpu_heuristics,
            autodock_gpu_autostop=config.autodock_gpu_autostop,
            autodock_gpu_local_search=config.autodock_gpu_local_search,
            autodock_gpu_cpu_threads=config.autodock_gpu_cpu_threads,
            autodock_gpu_workers=config.autodock_gpu_workers,
            autodock_gpu_executable=config.autodock_gpu_executable,
        )
        measurements: list[dict[str, Any]] = []
        for size in sizes:
            attempted = min(size, len(prepared))
            if attempted < size and (size != sizes[-1] or attempted / size < 0.99):
                continue
            print(
                f"[iGenVS] tuning {config.engine} outer batch {size:,}",
                flush=True,
            )
            docking_started = perf_counter()
            results, invocations = dock_batch_resilient(
                prepared[:attempted],
                config=dock_config,
                work_dir=work / f"batch_{size}",
                label=f"tune_{size}",
                retry_missing=False,
            )
            seconds = perf_counter() - docking_started
            worker_seconds = sum(item.seconds for item in invocations)
            successful = sum(result.status == "success" for result in results)
            measurements.append(
                {
                    "batch_size": size,
                    "attempted": attempted,
                    "preparation_yield_fraction": attempted / size,
                    "successful": successful,
                    "success_fraction": successful / attempted,
                    "seconds": seconds,
                    "worker_seconds_sum": worker_seconds,
                    "attempted_ligands_per_second": attempted / max(seconds, 1e-9),
                    "successful_ligands_per_second": successful / max(seconds, 1e-9),
                }
            )
        if not measurements:
            raise InputError(
                f"pilot has only {len(prepared)} prepared ligands; smallest requested batch is {sizes[0]}"
            )
        selected = _choose_batch(measurements)
        profile = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": config.engine,
            "scoring_function": config.scoring,
            "selected_batch_size": selected,
            "selected_autodock_gpu_workers": config.autodock_gpu_workers,
            "selection_rule": "highest measured successful throughput among batches with >=99% success; smallest batch breaks ties",
            "hardware": hardware_snapshot(config.device_id),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            "target": (
                {
                    "path": str(prepared_target.directory),
                    "manifest": str(prepared_target.manifest),
                    "pocket": str(prepared_target.pocket),
                }
                if prepared_target is not None
                else None
            ),
            "validation": validation,
            "preparation_failures": len(prep_failures),
            "measurements": measurements,
        }
    config.profile_path.parent.mkdir(parents=True, exist_ok=True)
    config.profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile

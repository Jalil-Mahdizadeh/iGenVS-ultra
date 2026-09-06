"""End-to-end staged and double-buffered virtual-screening pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Sequence

from .docking import DockInvocation, DockingConfig, dock_batch_resilient
from .errors import InputError
from .generation import GenerationConfig, run_igen3
from .hardware import (
    auto_preparation_worker_count,
    auto_worker_count,
    hardware_snapshot,
    resolve_autodock_gpu_workers,
    resolve_docking_batch_size,
)
from .ingress import (
    chunked,
    inspect_prevalidated_library,
    iter_source_records,
    iter_validated_records,
    validate_library,
)
from .mps import ManagedCudaMPS, start_cuda_mps
from .preparation import (
    DEFAULT_EMBED_MAX_ATTEMPTS,
    DEFAULT_EMBED_TIMEOUT_SECONDS,
    PREPARATION_MODES,
    collect_preparation_batch,
    submit_preparation_batch,
)
from .records import DockingResult, PreparationFailure, ValidatedRecord
from .target import load_target


RESULT_FIELDS = [
    "molecule_id",
    "original_smiles",
    "canonical_smiles",
    "source_row",
    "status",
    "docking_engine",
    "scoring_function",
    "docking_score",
    "num_poses",
    "heavy_atoms",
    "rotatable_bonds",
    "prepared_atoms",
    "prepared_torsions",
    "preparation_seconds",
    "dock_batch_seconds",
    "batch_id",
    "pose_ref",
    "error",
]


@dataclass(frozen=True)
class ScreenConfig:
    output_dir: Path
    receptor: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    target: Path | None = None
    engine: str = "unidock"
    autodock_gpu_fld: Path | None = None
    input_path: Path | None = None
    prevalidated_input: Path | None = None
    input_format: str = "auto"
    smiles_column: str = "smiles"
    id_column: str | None = None
    delimiter: str = "auto"
    generation: GenerationConfig | None = None
    docking_batch_size: str | int = "auto"
    batch_profile: Path | None = None
    prep_workers: str | int = "auto"
    prep_mode: str = "standard"
    embed_max_attempts: str | int = "auto"
    embed_timeout_seconds: str | int = "auto"
    validation_workers: str | int = "auto"
    fragment_policy: str = "reject"
    deduplicate: bool = True
    num_shards: int = 1
    shard_index: int = 0
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
    keep_work: bool = False
    pose_output: str = "merged"
    individual_poses: bool = False
    max_atoms: int = 300
    max_torsions: int = 57
    unidock_executable: str = "unidock"
    autodock_gpu_runs: int | None = None
    autodock_gpu_evaluations: int | None = None
    autodock_gpu_heuristics: bool = True
    autodock_gpu_autostop: bool = True
    autodock_gpu_local_search: str = "ad"
    autodock_gpu_cpu_threads: int = 4
    autodock_gpu_workers: str | int = "auto"
    autodock_gpu_executable: str = "autodock_gpu"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tool_version(command: Sequence[str]) -> str | None:
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (process.stdout or process.stderr).strip()
    return text.splitlines()[0].strip() if text else None


def _prepare_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise InputError(f"output path exists and is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise InputError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _resolve_scratch_root(config: ScreenConfig) -> Path:
    if config.scratch_dir is not None:
        root = config.scratch_dir
    elif os.environ.get("SLURM_TMPDIR"):
        root = Path(os.environ["SLURM_TMPDIR"])
    else:
        root = Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _source_records(config: ScreenConfig, source_path: Path):
    return iter_source_records(
        source_path,
        input_format=config.input_format if config.generation is None else "smi",
        smiles_column=config.smiles_column,
        id_column=config.id_column,
        delimiter=config.delimiter,
        num_shards=config.num_shards,
        shard_index=config.shard_index,
    )


def _failure_row(
    failure: PreparationFailure,
    batch_id: int,
    *,
    engine: str,
    scoring: str,
) -> dict[str, object]:
    record = failure.record
    return {
        "molecule_id": record.molecule_id,
        "original_smiles": record.original_smiles,
        "canonical_smiles": record.canonical_smiles,
        "source_row": record.source_row,
        "status": failure.status,
        "docking_engine": engine,
        "scoring_function": scoring,
        "docking_score": "",
        "num_poses": 0,
        "heavy_atoms": record.heavy_atoms,
        "rotatable_bonds": record.rotatable_bonds,
        "prepared_atoms": "",
        "prepared_torsions": "",
        "preparation_seconds": f"{failure.seconds:.6f}",
        "dock_batch_seconds": "",
        "batch_id": batch_id,
        "pose_ref": "",
        "error": failure.error,
    }


def _result_row(
    result: DockingResult,
    batch_id: int,
    pose_ref: str,
    *,
    engine: str,
    scoring: str,
) -> dict[str, object]:
    record = result.ligand.record
    return {
        "molecule_id": record.molecule_id,
        "original_smiles": record.original_smiles,
        "canonical_smiles": record.canonical_smiles,
        "source_row": record.source_row,
        "status": result.status,
        "docking_engine": engine,
        "scoring_function": scoring,
        "docking_score": f"{result.scores[0]:.6f}" if result.scores else "",
        "num_poses": len(result.scores),
        "heavy_atoms": record.heavy_atoms,
        "rotatable_bonds": record.rotatable_bonds,
        "prepared_atoms": result.ligand.atom_count,
        "prepared_torsions": result.ligand.torsion_count,
        "preparation_seconds": f"{result.ligand.seconds:.6f}",
        "dock_batch_seconds": f"{result.batch_seconds:.6f}",
        "batch_id": batch_id,
        "pose_ref": pose_ref,
        "error": result.error,
    }


def _persist_pose(
    result: DockingResult,
    *,
    output_dir: Path,
    merged_handle,
    pose_output: str,
) -> str:
    if pose_output == "none":
        return ""
    if result.status != "success" or result.pose_path is None:
        return ""
    if pose_output == "individual":
        pose_dir = output_dir / "poses"
        pose_dir.mkdir(parents=True, exist_ok=True)
        target = pose_dir / result.pose_path.name
        shutil.copyfile(result.pose_path, target)
        return str(target.relative_to(output_dir))
    if merged_handle is None:
        raise ValueError("merged pose output requires an open output handle")
    marker = result.ligand.record.molecule_id.replace("\n", " ").replace("\r", " ")
    merged_handle.write(f"REMARK IGENVS MOLECULE_ID {marker}\n")
    with result.pose_path.open("r", encoding="utf-8", errors="replace") as source:
        shutil.copyfileobj(source, merged_handle)
    merged_handle.write("\n")
    return f"poses.pdbqt#{marker}"


def _append_invocation_logs(
    invocations: Sequence[DockInvocation],
    *,
    stdout_handle,
    stderr_handle,
    batch_id: int,
) -> None:
    for index, invocation in enumerate(invocations):
        header = f"\n===== batch={batch_id} invocation={index} exit={invocation.returncode} seconds={invocation.seconds:.6f} =====\n"
        stdout_handle.write(header)
        stderr_handle.write(header)
        with invocation.stdout_path.open("r", encoding="utf-8", errors="replace") as source:
            shutil.copyfileobj(source, stdout_handle)
        with invocation.stderr_path.open("r", encoding="utf-8", errors="replace") as source:
            shutil.copyfileobj(source, stderr_handle)


def _next_batch(
    iterator: Iterator[list[ValidatedRecord]],
    batch_id: int,
    executor: ProcessPoolExecutor,
    work_root: Path,
    config: ScreenConfig,
    embed_max_attempts: int,
    embed_timeout_seconds: int,
) -> tuple[int, Path, list[Future]] | None:
    try:
        records = next(iterator)
    except StopIteration:
        return None
    batch_dir = work_root / f"batch_{batch_id:06d}"
    engine_max_atoms = 256 if config.engine == "autodock-gpu" else 300
    engine_max_torsions = 57 if config.engine == "autodock-gpu" else 48
    futures = submit_preparation_batch(
        executor,
        records,
        output_dir=batch_dir / "prepared",
        seed=config.seed,
        max_atoms=min(config.max_atoms, engine_max_atoms),
        max_torsions=min(config.max_torsions, engine_max_torsions),
        prep_mode=config.prep_mode,
        embed_max_attempts=embed_max_attempts,
        embed_timeout_seconds=embed_timeout_seconds,
    )
    return batch_id, batch_dir, futures


def _preparation_batches(
    records: Iterator[ValidatedRecord],
    *,
    batch_size: int,
    first_batch_size: int,
) -> Iterator[list[ValidatedRecord]]:
    """Emit an optional ramp batch, followed by full steady-state batches."""

    if first_batch_size < 1 or first_batch_size > batch_size:
        raise ValueError("first preparation batch must be within the outer batch size")
    if first_batch_size < batch_size:
        first = list(islice(records, first_batch_size))
        if first:
            yield first
    yield from chunked(records, batch_size)


def _first_preparation_batch_size(
    *,
    engine: str,
    valid_records: int,
    batch_size: int,
    prep_workers: int,
    automatic: bool,
) -> int:
    """Choose a ramp only where overlap can amortize another engine launch."""

    if engine != "unidock" or not automatic:
        return batch_size
    target = max(1_024, min(4_096, prep_workers * 32))
    target = 1 << (target - 1).bit_length()
    ramp = min(batch_size, target)
    return ramp if ramp < batch_size and valid_records >= 3 * ramp else batch_size


def run_screen(config: ScreenConfig) -> dict[str, Any]:
    source_count = sum(
        value is not None
        for value in (config.input_path, config.prevalidated_input, config.generation)
    )
    if source_count != 1:
        raise InputError(
            "select exactly one source: input, trusted prevalidated input, or iGen3 generation"
        )
    if config.engine not in {"unidock", "autodock-gpu"}:
        raise InputError("engine must be unidock or autodock-gpu")
    if config.engine == "unidock" and config.scoring not in {"vina", "vinardo"}:
        raise InputError("Uni-Dock scoring must be vina or vinardo")
    if config.engine == "autodock-gpu" and config.scoring != "ad4":
        raise InputError("AutoDock-GPU scoring must be ad4")
    autodock_gpu_workers, autodock_gpu_workers_source = (
        resolve_autodock_gpu_workers(
            config.autodock_gpu_workers,
            profile_path=config.batch_profile,
            engine=config.engine,
            device_id=config.device_id,
            cpu_threads_per_worker=config.autodock_gpu_cpu_threads,
        )
    )
    receptor = config.receptor.expanduser().resolve()
    if not receptor.is_file():
        raise InputError(f"prepared receptor PDBQT does not exist: {receptor}")
    if receptor.suffix.lower() != ".pdbqt":
        raise InputError("screening requires a prepared .pdbqt receptor; use prepare-receptor first")
    autodock_gpu_fld = (
        config.autodock_gpu_fld.expanduser().resolve()
        if config.autodock_gpu_fld is not None
        else None
    )
    if config.engine == "autodock-gpu":
        if autodock_gpu_fld is None or not autodock_gpu_fld.is_file():
            raise InputError("AutoDock-GPU requires an existing .maps.fld grid")
        if autodock_gpu_fld.suffix.lower() != ".fld":
            raise InputError("AutoDock-GPU grid descriptor must end in .fld")
        if config.num_modes != 1:
            raise InputError("AutoDock-GPU currently emits exactly one best pose")
    prepared_target = None
    if config.target is not None:
        prepared_target = load_target(config.target)
        if receptor != prepared_target.receptor:
            raise InputError("screening receptor does not match the prepared target")
        if tuple(config.center) != prepared_target.center or tuple(config.size) != prepared_target.size:
            raise InputError("screening box does not match the prepared target")
        if (
            config.engine == "autodock-gpu"
            and autodock_gpu_fld != prepared_target.autodock_gpu_fld
        ):
            raise InputError(
                "screening AutoDock-GPU grid does not match the prepared target"
            )
    if any(value <= 0 for value in config.size):
        raise InputError("all docking box dimensions must be positive")
    pose_output = "individual" if config.individual_poses else config.pose_output
    if pose_output not in {"none", "merged", "individual"}:
        raise InputError("pose_output must be none, merged, or individual")
    if config.prep_mode not in PREPARATION_MODES:
        raise InputError(f"prep_mode must be one of {', '.join(PREPARATION_MODES)}")

    output_dir = config.output_dir.expanduser().resolve()
    _prepare_output_directory(output_dir)
    logs_dir = output_dir / "logs"
    input_dir = output_dir / "input"
    validation_dir = output_dir / "validation"
    logs_dir.mkdir()
    input_dir.mkdir()
    manifest_path = output_dir / "manifest.json"
    started = perf_counter()
    timings: dict[str, float] = {
        "generation_seconds": 0.0,
        "validation_seconds": 0.0,
        "preparation_wait_seconds": 0.0,
        "preparation_cpu_seconds_sum": 0.0,
        "docking_wall_seconds": 0.0,
        "docking_invocation_seconds": 0.0,
        "result_processing_seconds": 0.0,
        "cleanup_seconds": 0.0,
        "screening_seconds": 0.0,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "config": _jsonable(config),
        "hardware": hardware_snapshot(config.device_id),
        "timings": timings,
        "receptor": {"path": str(receptor), "sha256": _sha256(receptor)},
        "autodock_gpu_grid": (
            {
                "path": str(autodock_gpu_fld),
                "sha256": _sha256(autodock_gpu_fld),
            }
            if autodock_gpu_fld is not None
            else None
        ),
        "target": (
            {
                "path": str(prepared_target.directory),
                "manifest_sha256": _sha256(prepared_target.manifest),
                "pocket_sha256": _sha256(prepared_target.pocket),
            }
            if prepared_target is not None
            else None
        ),
        "tools": {
            "unidock": (
                _tool_version([config.unidock_executable, "--version"])
                if config.engine == "unidock"
                else None
            ),
            "autodock_gpu": (
                _tool_version([config.autodock_gpu_executable, "--help"])
                if config.engine == "autodock-gpu"
                else None
            ),
            "autogrid4": None,
            "igen3": (
                _tool_version(["igen3", "list-models"])
                if config.generation is not None
                else None
            ),
        },
    }
    _write_manifest(manifest_path, manifest)

    mps_manager: ManagedCudaMPS | None = None

    try:
        if config.generation is not None:
            source_path = input_dir / "generated.smi"
            print(f"[iGenVS] generating {config.generation.count:,} iGen3 molecules", flush=True)
            generation_started = perf_counter()
            manifest["generation"] = run_igen3(config.generation, output_path=source_path, log_dir=logs_dir)
            timings["generation_seconds"] = perf_counter() - generation_started
        elif config.input_path is not None:
            source_path = config.input_path.expanduser().resolve()  # type: ignore[union-attr]
            manifest["source"] = {"path": str(source_path), "format": config.input_format}
        else:
            source_path = config.prevalidated_input.expanduser().resolve()  # type: ignore[union-attr]
            manifest["source"] = {
                "path": str(source_path),
                "format": "validated-csv-v1",
                "trusted_prevalidated": True,
            }

        validation_started = perf_counter()
        if config.prevalidated_input is not None:
            print("[iGenVS] verifying trusted prevalidated shard", flush=True)
            validation = inspect_prevalidated_library(
                source_path,
                num_shards=config.num_shards,
                shard_index=config.shard_index,
            )
        else:
            print("[iGenVS] validating and deduplicating with RDKit", flush=True)
            validation = validate_library(
                _source_records(config, source_path),
                output_dir=validation_dir,
                workers=config.validation_workers,
                fragment_policy=config.fragment_policy,
                deduplicate=config.deduplicate,
            )
        timings["validation_seconds"] = perf_counter() - validation_started
        manifest["validation"] = validation
        if int(validation["valid"]) == 0:
            raise InputError("no valid unique molecules remained after validation")

        batch_size, batch_source = resolve_docking_batch_size(
            config.docking_batch_size,
            profile_path=config.batch_profile,
            device_id=config.device_id,
            engine=config.engine,
        )
        prep_workers = min(
            auto_preparation_worker_count(config.prep_workers, cap=64),
            int(validation["valid"]),
        )
        embed_max_attempts = (
            DEFAULT_EMBED_MAX_ATTEMPTS
            if str(config.embed_max_attempts).lower() == "auto"
            else int(config.embed_max_attempts)
        )
        embed_timeout_seconds = (
            DEFAULT_EMBED_TIMEOUT_SECONDS
            if str(config.embed_timeout_seconds).lower() == "auto"
            else int(config.embed_timeout_seconds)
        )
        if embed_max_attempts < 1 or embed_timeout_seconds < 1:
            raise InputError("embedding attempt and timeout budgets must be positive")
        first_batch_size = _first_preparation_batch_size(
            engine=config.engine,
            valid_records=int(validation["valid"]),
            batch_size=batch_size,
            prep_workers=prep_workers,
            automatic=batch_source == "hardware-heuristic",
        )
        scratch_root = _resolve_scratch_root(config)
        if config.engine == "autodock-gpu":
            mps_manager = start_cuda_mps(
                autodock_gpu_workers,
                scratch_root=scratch_root,
                automatic=str(config.autodock_gpu_workers).lower() == "auto",
            )
            autodock_gpu_workers = mps_manager.workers
            autodock_gpu_workers_source = (
                f"{autodock_gpu_workers_source};{mps_manager.source}"
            )
        manifest["runtime"] = {
            "engine": config.engine,
            "scoring_function": config.scoring,
            "docking_batch_size": batch_size,
            "docking_batch_source": batch_source,
            "preparation_workers": prep_workers,
            "preparation_mode": config.prep_mode,
            "embed_max_attempts": embed_max_attempts,
            "embed_timeout_seconds": embed_timeout_seconds,
            "hard_molecule_strategy": "bounded-attempts-timeout-aware-fallback-v1",
            "double_buffered_preparation": True,
            "first_preparation_batch_size": first_batch_size,
            "adaptive_preparation_ramp": first_batch_size < batch_size,
            "pose_output": pose_output,
            "refine_step": config.refine_step,
            "no_refine": config.no_refine,
            "unidock_verbosity": config.unidock_verbosity,
            "autodock_gpu_runs": config.autodock_gpu_runs,
            "autodock_gpu_evaluations": config.autodock_gpu_evaluations,
            "autodock_gpu_heuristics": config.autodock_gpu_heuristics,
            "autodock_gpu_autostop": config.autodock_gpu_autostop,
            "autodock_gpu_local_search": config.autodock_gpu_local_search,
            "autodock_gpu_cpu_threads": config.autodock_gpu_cpu_threads,
            "autodock_gpu_workers": autodock_gpu_workers,
            "autodock_gpu_workers_source": autodock_gpu_workers_source,
        }
        _write_manifest(manifest_path, manifest)
        print(
            f"[iGenVS] screening {int(validation['valid']):,} molecules with "
            f"{config.engine} in batches of {batch_size:,}; {prep_workers} CPU "
            f"preparation workers ({config.prep_mode}); "
            f"GPU workers={autodock_gpu_workers}; poses={pose_output}",
            flush=True,
        )

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
            scores_only_output=pose_output == "none",
            unidock_executable=config.unidock_executable,
            autodock_gpu_fld=autodock_gpu_fld,
            autodock_gpu_runs=config.autodock_gpu_runs,
            autodock_gpu_evaluations=config.autodock_gpu_evaluations,
            autodock_gpu_heuristics=config.autodock_gpu_heuristics,
            autodock_gpu_autostop=config.autodock_gpu_autostop,
            autodock_gpu_local_search=config.autodock_gpu_local_search,
            autodock_gpu_cpu_threads=config.autodock_gpu_cpu_threads,
            autodock_gpu_workers=autodock_gpu_workers,
            autodock_gpu_executable=config.autodock_gpu_executable,
        )
        counts = {
            "prepared": 0,
            "preparation_failed": 0,
            "preparation_timed_out": 0,
            "docked": 0,
            "docking_failed": 0,
            "batches": 0,
        }

        screening_started = perf_counter()
        with ExitStack() as stack:
            if config.keep_work:
                work_root = output_dir / "work"
                work_root.mkdir()
            else:
                temp_context = tempfile.TemporaryDirectory(prefix="igenvs-", dir=scratch_root)
                work_root = Path(stack.enter_context(temp_context))
            results_handle = stack.enter_context((output_dir / "results.csv").open("w", encoding="utf-8", newline=""))
            pose_handle = None
            if pose_output == "merged":
                pose_handle = stack.enter_context((output_dir / "poses.pdbqt").open("w", encoding="utf-8"))
            dock_stdout = stack.enter_context(
                (logs_dir / f"{config.engine}.stdout.log").open(
                    "w", encoding="utf-8"
                )
            )
            dock_stderr = stack.enter_context(
                (logs_dir / f"{config.engine}.stderr.log").open(
                    "w", encoding="utf-8"
                )
            )
            writer = csv.DictWriter(results_handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()

            validated_records = iter_validated_records(
                Path(str(validation["validated_path"])),
                num_shards=(config.num_shards if config.prevalidated_input is not None else 1),
                shard_index=(config.shard_index if config.prevalidated_input is not None else 0),
            )
            batches = iter(
                _preparation_batches(
                    validated_records,
                    batch_size=batch_size,
                    first_batch_size=first_batch_size,
                )
            )
            with ProcessPoolExecutor(max_workers=prep_workers) as executor:
                current = _next_batch(
                    batches,
                    1,
                    executor,
                    work_root,
                    config,
                    embed_max_attempts,
                    embed_timeout_seconds,
                )
                while current is not None:
                    batch_id, batch_dir, futures = current
                    preparation_wait_started = perf_counter()
                    prepared, prep_failures = collect_preparation_batch(futures)
                    timings["preparation_wait_seconds"] += perf_counter() - preparation_wait_started
                    timings["preparation_cpu_seconds_sum"] += sum(
                        item.seconds for item in [*prepared, *prep_failures]
                    )
                    # Start CPU work for the next batch before occupying the GPU.
                    following = _next_batch(
                        batches,
                        batch_id + 1,
                        executor,
                        work_root,
                        config,
                        embed_max_attempts,
                        embed_timeout_seconds,
                    )
                    counts["prepared"] += len(prepared)
                    counts["preparation_failed"] += len(prep_failures)
                    counts["preparation_timed_out"] += sum(
                        failure.status == "preparation_timeout"
                        for failure in prep_failures
                    )
                    # Preparation failures and engine results are produced by
                    # different asynchronous paths. Buffer only this already
                    # bounded outer batch and restore source order before
                    # writing it. This preserves a streaming result file while
                    # allowing the public multi-GPU wrapper to k-way merge
                    # shard outputs without a costly whole-library sort.
                    batch_rows = []
                    for failure in prep_failures:
                        batch_rows.append(
                            _failure_row(
                                failure,
                                batch_id,
                                engine=config.engine,
                                scoring=config.scoring,
                            )
                        )

                    if prepared:
                        docking_started = perf_counter()
                        results, invocations = dock_batch_resilient(
                            prepared,
                            config=dock_config,
                            work_dir=batch_dir,
                            label=f"batch_{batch_id:06d}",
                            retry_missing=True,
                        )
                        timings["docking_wall_seconds"] += (
                            perf_counter() - docking_started
                        )
                        timings["docking_invocation_seconds"] += sum(item.seconds for item in invocations)
                        result_processing_started = perf_counter()
                        _append_invocation_logs(
                            invocations,
                            stdout_handle=dock_stdout,
                            stderr_handle=dock_stderr,
                            batch_id=batch_id,
                        )
                        for result in results:
                            pose_ref = _persist_pose(
                                result,
                                output_dir=output_dir,
                                merged_handle=pose_handle,
                                pose_output=pose_output,
                            )
                            batch_rows.append(
                                _result_row(
                                    result,
                                    batch_id,
                                    pose_ref,
                                    engine=config.engine,
                                    scoring=config.scoring,
                                )
                            )
                            if result.status == "success":
                                counts["docked"] += 1
                            else:
                                counts["docking_failed"] += 1
                    else:
                        result_processing_started = perf_counter()
                    batch_rows.sort(key=lambda row: int(row["source_row"]))
                    writer.writerows(batch_rows)
                    timings["result_processing_seconds"] += perf_counter() - result_processing_started
                    counts["batches"] += 1
                    if not config.keep_work:
                        cleanup_started = perf_counter()
                        shutil.rmtree(batch_dir)
                        timings["cleanup_seconds"] += perf_counter() - cleanup_started
                    results_handle.flush()
                    if pose_handle is not None:
                        pose_handle.flush()
                    print(
                        f"[iGenVS] batch {batch_id}: prepared={len(prepared):,}, "
                        f"prep_failed={len(prep_failures):,}, total_docked={counts['docked']:,}",
                        flush=True,
                    )
                    current = following

        timings["screening_seconds"] = perf_counter() - screening_started
        manifest["status"] = "complete"
        manifest["completed_at"] = _utc_now()
        elapsed_seconds = perf_counter() - started
        manifest["elapsed_seconds"] = elapsed_seconds
        manifest["counts"] = counts
        manifest["performance"] = {
            "successful_ligands_per_second_end_to_end": counts["docked"] / max(elapsed_seconds, 1e-9),
            "successful_ligands_per_second_screening": counts["docked"]
            / max(timings["screening_seconds"], 1e-9),
            "successful_ligands_per_second_docking_wall": counts["docked"]
            / max(timings["docking_wall_seconds"], 1e-9),
            "docking_worker_seconds_sum": timings["docking_invocation_seconds"],
            "input_ligands_per_second_end_to_end": int(validation["input"])
            / max(elapsed_seconds, 1e-9),
        }
        manifest["outputs"] = {
            "results": str(output_dir / "results.csv"),
            "poses": None if pose_output == "none" else str(output_dir / ("poses" if pose_output == "individual" else "poses.pdbqt")),
            "manifest": str(manifest_path),
        }
        _write_manifest(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = _utc_now()
        manifest["elapsed_seconds"] = perf_counter() - started
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        if mps_manager is not None:
            mps_manager.close()

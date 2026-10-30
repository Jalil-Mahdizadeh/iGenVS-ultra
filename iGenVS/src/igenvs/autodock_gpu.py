"""AutoDock-GPU 1.6 batch invocation and strict result parsing."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .docking import DockInvocation, DockingConfig
from .records import DockingResult, PreparedLigand
from .pose_validation import validate_ring_closures


AUTODOCK_GPU_RUNS_BY_MODE = {
    "fast": 10,
    "balance": 20,
    "detail": 50,
}
AUTODOCK_GPU_LOCAL_SEARCH_METHODS = {"sw", "sd", "fire", "ad", "adam"}


def resolved_autodock_gpu_runs(config: DockingConfig) -> int:
    runs = (
        config.autodock_gpu_runs
        if config.autodock_gpu_runs is not None
        else AUTODOCK_GPU_RUNS_BY_MODE.get(config.search_mode)
    )
    if runs is None:
        raise ValueError(f"unsupported AutoDock-GPU search mode: {config.search_mode}")
    if not 1 <= runs <= 8192:
        raise ValueError("AutoDock-GPU runs must be between 1 and 8192")
    if not 1 <= config.num_modes <= 8192:
        raise ValueError("AutoDock-GPU pose count must be between 1 and 8192")
    if config.autodock_gpu_runs is not None and runs < config.num_modes:
        raise ValueError("AutoDock-GPU runs must be at least the requested pose count")
    return max(runs, config.num_modes)


def autodock_gpu_output_stem(ligand: PreparedLigand, output_dir: Path) -> Path:
    return output_dir / ligand.path.stem


def autodock_gpu_xml_path(ligand: PreparedLigand, output_dir: Path) -> Path:
    return autodock_gpu_output_stem(ligand, output_dir).with_suffix(".xml")


def autodock_gpu_pose_path(ligand: PreparedLigand, output_dir: Path) -> Path:
    stem = autodock_gpu_output_stem(ligand, output_dir)
    return stem.parent / f"{stem.name}-poses.pdbqt"


def _ranked_autodock_gpu_runs(path: Path, num_modes: int) -> list[tuple[int, float]]:
    """Rank run results by AD4 binding energy, retaining their coordinate IDs."""

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError("output is not well-formed AutoDock-GPU XML") from exc
    if root.tag != "autodock_gpu":
        raise ValueError("output root is not autodock_gpu")
    nodes = root.findall("./runs/run")
    if not nodes:
        raise ValueError("output contains no free_NRG_binding scores")
    runs = []
    seen = set()
    for node in nodes:
        try:
            score = float(node.findtext("free_NRG_binding", ""))
        except ValueError as exc:
            raise ValueError("output contains a non-numeric docking score") from exc
        if not math.isfinite(score):
            raise ValueError("output contains a non-finite docking score")
        try:
            run_id = int(node.attrib["id"])
        except (KeyError, ValueError) as exc:
            raise ValueError("output contains an invalid run ID") from exc
        if run_id < 1 or run_id in seen:
            raise ValueError("output contains an invalid or duplicate run ID")
        seen.add(run_id)
        runs.append((run_id, score))
    if num_modes < 1 or len(runs) < num_modes:
        raise ValueError(f"requested {num_modes} poses but AutoDock-GPU returned {len(runs)} runs")
    return sorted(runs, key=lambda run: (run[1], run[0]))[:num_modes]


def parse_autodock_gpu_xml(path: Path, num_modes: int = 1) -> tuple[float, ...]:
    return tuple(score for _, score in _ranked_autodock_gpu_runs(path, num_modes))


def _write_ranked_poses(
    ligand: PreparedLigand, output_dir: Path, runs: Sequence[tuple[int, float]],
) -> Path:
    """Join run coordinates in exactly the same order as CSV pose scores."""
    models = []
    for rank, (run_id, score) in enumerate(runs, 1):
        source = output_dir / f"{ligand.path.stem}_docked_run{run_id}_entity1.pdbqt"
        text = source.read_text(encoding="utf-8")
        if not any(line.startswith(("ATOM  ", "HETATM")) for line in text.splitlines()):
            raise ValueError(f"AutoDock-GPU run {run_id} has no atom coordinates")
        # Meeko recognizes native AutoDock energy remarks in multi-model PDBQT.
        models.append(
            f"MODEL {rank}\nREMARK AUTODOCK_GPU RUN_ID {run_id}\n"
            f"USER    Estimated Free Energy of Binding    = {score:.6f} kcal/mol\n"
            f"{text.rstrip()}\nENDMDL\n"
        )
    destination = autodock_gpu_pose_path(ligand, output_dir)
    destination.write_text("".join(models), encoding="utf-8")
    validate_ring_closures(ligand.path, destination)
    return destination


def build_autodock_gpu_command(
    *,
    config: DockingConfig,
    filelist_path: Path,
) -> list[str]:
    if config.autodock_gpu_fld is None:
        raise ValueError("AutoDock-GPU requires a .maps.fld grid descriptor")
    if config.scoring != "ad4":
        raise ValueError("AutoDock-GPU requires AD4 scoring")
    if config.device_id < 0:
        raise ValueError("device_id must be non-negative")
    if config.autodock_gpu_cpu_threads < 1:
        raise ValueError("AutoDock-GPU CPU thread count must be positive")
    if config.autodock_gpu_workers < 1:
        raise ValueError("AutoDock-GPU worker count must be positive")
    if config.autodock_gpu_workers > 64:
        raise ValueError("AutoDock-GPU worker count cannot exceed 64")
    if config.autodock_gpu_local_search not in AUTODOCK_GPU_LOCAL_SEARCH_METHODS:
        choices = ", ".join(sorted(AUTODOCK_GPU_LOCAL_SEARCH_METHODS))
        raise ValueError(f"AutoDock-GPU local search must be one of: {choices}")
    if (
        config.autodock_gpu_evaluations is not None
        and config.autodock_gpu_evaluations < 1
    ):
        raise ValueError("AutoDock-GPU evaluations must be positive")

    command = [
        config.autodock_gpu_executable,
        "--filelist",
        str(filelist_path),
        "--devnum",
        str(config.device_id + 1),
        "--nrun",
        str(resolved_autodock_gpu_runs(config)),
        "--heuristics",
        "1" if config.autodock_gpu_heuristics else "0",
        "--autostop",
        "1" if config.autodock_gpu_autostop else "0",
        "--lsmet",
        config.autodock_gpu_local_search,
        "--seed",
        str(config.seed),
        "--xmloutput",
        "1",
        "--dlgoutput",
        "0",
        "--clustering",
        "0",
        "--gbest",
        "0",
        "--npdb",
        "0" if config.scores_only_output else "1",
    ]
    if config.autodock_gpu_evaluations is not None:
        command.extend(["--nev", str(config.autodock_gpu_evaluations)])
    return command


def _write_filelist(
    ligands: Sequence[PreparedLigand],
    *,
    config: DockingConfig,
    filelist_path: Path,
    output_dir: Path,
) -> None:
    if config.autodock_gpu_fld is None:
        raise ValueError("AutoDock-GPU requires a .maps.fld grid descriptor")
    lines = [str(config.autodock_gpu_fld.resolve())]
    for ligand in ligands:
        source = ligand.path.resolve()
        if not config.scores_only_output:
            # --npdb writes beside the input ligand. Keep all run files inside
            # this invocation's scratch directory, including on a retry.
            source = output_dir / ligand.path.name
            shutil.copyfile(ligand.path, source)
        lines.extend(
            [
                str(source.resolve()),
                str(autodock_gpu_output_stem(ligand, output_dir).resolve()),
            ]
        )
    filelist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def invoke_autodock_gpu(
    ligands: Sequence[PreparedLigand],
    *,
    config: DockingConfig,
    work_dir: Path,
    label: str,
) -> DockInvocation:
    if not ligands:
        raise ValueError("cannot invoke AutoDock-GPU with an empty ligand batch")
    output_dir = work_dir / f"{label}_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    filelist_path = work_dir / f"{label}_filelist.txt"
    _write_filelist(
        ligands,
        config=config,
        filelist_path=filelist_path,
        output_dir=output_dir,
    )
    stdout_path = work_dir / f"{label}.stdout.log"
    stderr_path = work_dir / f"{label}.stderr.log"
    command = build_autodock_gpu_command(config=config, filelist_path=filelist_path)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(config.autodock_gpu_cpu_threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    started = perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            check=False,
        )
    return DockInvocation(
        process.returncode,
        perf_counter() - started,
        stdout_path,
        stderr_path,
        None,
        tuple(command),
    )


def _collect_outputs(
    ligands: Sequence[PreparedLigand],
    *,
    config: DockingConfig,
    output_dir: Path,
    seconds: float,
) -> tuple[list[DockingResult], list[PreparedLigand]]:
    results: list[DockingResult] = []
    missing: list[PreparedLigand] = []
    for ligand in ligands:
        xml_path = autodock_gpu_xml_path(ligand, output_dir)
        pose_path = (
            None
            if config.scores_only_output
            else autodock_gpu_pose_path(ligand, output_dir)
        )
        if not xml_path.is_file():
            missing.append(ligand)
            continue
        try:
            runs = _ranked_autodock_gpu_runs(xml_path, config.num_modes)
            scores = tuple(score for _, score in runs)
            if pose_path is not None:
                pose_path = _write_ranked_poses(ligand, output_dir, runs)
            results.append(
                DockingResult(ligand, "success", scores, pose_path, "", seconds)
            )
        except FileNotFoundError:
            missing.append(ligand)
        except Exception as exc:
            results.append(
                DockingResult(
                    ligand,
                    "invalid_output",
                    (),
                    pose_path,
                    f"{type(exc).__name__}: {exc}",
                    seconds,
                )
            )
    return results, missing


def partition_autodock_gpu_ligands(
    ligands: Sequence[PreparedLigand],
    worker_count: int,
) -> list[list[PreparedLigand]]:
    """Deterministically balance AD-GPU file lists by predicted search work.

    AutoDock-GPU time grows strongly with ligand torsions.  A longest-work-first
    assignment avoids leaving one MPS process with most of the flexible tail,
    while sorting each resulting file list back into source order keeps output
    and restart behavior stable.
    """

    if worker_count < 1:
        raise ValueError("AutoDock-GPU worker count must be positive")
    groups: list[list[PreparedLigand]] = [[] for _ in range(worker_count)]
    predicted_loads = [0] * worker_count
    source_order = {id(ligand): index for index, ligand in enumerate(ligands)}
    hardest_first = sorted(
        ligands,
        key=lambda ligand: (
            ligand.torsion_count,
            ligand.atom_count,
            -ligand.record.source_row,
        ),
        reverse=True,
    )
    for ligand in hardest_first:
        worker = min(range(worker_count), key=lambda index: (predicted_loads[index], index))
        groups[worker].append(ligand)
        # The additive model is deliberately simple and hardware-independent.
        # Retrospective GH200 logs show torsion count is the most robust single
        # predictor; +1 prevents rigid ligands from becoming zero-cost items.
        predicted_loads[worker] += ligand.torsion_count + 1
    for group in groups:
        group.sort(key=lambda ligand: source_order[id(ligand)])
    return groups


def dock_batch_resilient_autodock_gpu(
    ligands: Sequence[PreparedLigand],
    *,
    config: DockingConfig,
    work_dir: Path,
    label: str,
    retry_missing: bool = True,
) -> tuple[list[DockingResult], list[DockInvocation]]:
    """Dock file-list shards concurrently and retry missing ligands once."""

    if not ligands:
        return [], []
    if (
        config.autodock_gpu_workers > 1
        and not os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    ):
        raise ValueError(
            "multiple AutoDock-GPU workers require CUDA MPS and "
            "CUDA_MPS_PIPE_DIRECTORY"
        )

    ligand_order = {id(ligand): index for index, ligand in enumerate(ligands)}

    def restore_input_order(rows: list[DockingResult]) -> list[DockingResult]:
        return sorted(rows, key=lambda row: ligand_order[id(row.ligand)])

    def invoke_groups(
        groups: Sequence[Sequence[PreparedLigand]],
        *,
        phase: str,
    ) -> tuple[list[DockingResult], list[PreparedLigand], list[DockInvocation]]:
        jobs = [tuple(group) for group in groups if group]
        if not jobs:
            return [], [], []

        def invoke_one(
            indexed_group: tuple[int, tuple[PreparedLigand, ...]],
        ) -> tuple[list[DockingResult], list[PreparedLigand], DockInvocation]:
            group_index, group = indexed_group
            group_label = f"{label}_{phase}_worker{group_index:02d}"
            invocation = invoke_autodock_gpu(
                group,
                config=config,
                work_dir=work_dir,
                label=group_label,
            )
            group_results, group_missing = _collect_outputs(
                group,
                config=config,
                output_dir=work_dir / f"{group_label}_outputs",
                seconds=invocation.seconds,
            )
            return group_results, group_missing, invocation

        completed: list[
            tuple[list[DockingResult], list[PreparedLigand], DockInvocation]
        ]
        if len(jobs) == 1:
            completed = [invoke_one((0, jobs[0]))]
        else:
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                completed = list(executor.map(invoke_one, enumerate(jobs)))
        group_results: list[DockingResult] = []
        group_missing: list[PreparedLigand] = []
        group_invocations: list[DockInvocation] = []
        for result_rows, missing_rows, invocation in completed:
            group_results.extend(result_rows)
            group_missing.extend(missing_rows)
            group_invocations.append(invocation)
        return group_results, group_missing, group_invocations

    worker_count = min(config.autodock_gpu_workers, len(ligands))
    initial_groups = partition_autodock_gpu_ligands(ligands, worker_count)
    results, missing, invocations = invoke_groups(initial_groups, phase="initial")
    if not missing:
        return restore_input_order(results), invocations

    if not retry_missing:
        stderr_tail = " | ".join(
            invocation.stderr_path.read_text(
                encoding="utf-8", errors="replace"
            )[-500:].strip()
            for invocation in invocations
        )
        worker_seconds = max(
            (invocation.seconds for invocation in invocations),
            default=0.0,
        )
        results.extend(
            DockingResult(
                ligand,
                "docking_failed",
                (),
                None,
                f"missing output; AutoDock-GPU worker diagnostics: {stderr_tail}",
                worker_seconds,
            )
            for ligand in missing
        )
        return restore_input_order(results), invocations

    retry_worker_count = min(config.autodock_gpu_workers, len(missing))
    retry_groups = partition_autodock_gpu_ligands(missing, retry_worker_count)
    retry_results, still_missing, retry_invocations = invoke_groups(
        retry_groups,
        phase="retry",
    )
    results.extend(retry_results)
    invocations.extend(retry_invocations)
    retry_stderr = " | ".join(
        invocation.stderr_path.read_text(
            encoding="utf-8", errors="replace"
        )[-500:].strip()
        for invocation in retry_invocations
    )
    retry_seconds = max(
        (invocation.seconds for invocation in retry_invocations),
        default=0.0,
    )
    results.extend(
        DockingResult(
            ligand,
            "docking_failed",
            (),
            None,
            f"missing after one retry; AutoDock-GPU worker diagnostics: {retry_stderr}",
            retry_seconds,
        )
        for ligand in still_missing
    )
    return restore_input_order(results), invocations

"""Shared docking configuration plus the resilient Uni-Dock backend."""

from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .errors import ExternalToolError
from .records import DockingResult, PreparedLigand


VINA_RESULT = re.compile(
    r"^REMARK VINA RESULT:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class DockingConfig:
    receptor: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    engine: str = "unidock"
    search_mode: str = "balance"
    scoring: str = "vina"
    num_modes: int = 1
    energy_range: float = 3.0
    seed: int = 181129
    device_id: int = 0
    max_gpu_memory: int = 0
    refine_step: int = 3
    no_refine: bool = False
    verbosity: int = 0
    scores_only_output: bool = False
    unidock_executable: str = "unidock"
    autodock_gpu_fld: Path | None = None
    autodock_gpu_runs: int | None = None
    autodock_gpu_evaluations: int | None = None
    autodock_gpu_heuristics: bool = True
    autodock_gpu_autostop: bool = True
    autodock_gpu_local_search: str = "ad"
    autodock_gpu_cpu_threads: int = 4
    autodock_gpu_workers: int = 1
    autodock_gpu_executable: str = "autodock_gpu"


@dataclass(frozen=True)
class DockInvocation:
    returncode: int
    seconds: float
    stdout_path: Path
    stderr_path: Path
    score_path: Path | None
    command: tuple[str, ...]


def expected_output_path(ligand: PreparedLigand, output_dir: Path) -> Path:
    return output_dir / f"{ligand.path.stem}_out.pdbqt"


def parse_pdbqt_scores(path: Path) -> tuple[float, ...]:
    text = path.read_text(encoding="utf-8", errors="replace")
    scores = tuple(float(value) for value in VINA_RESULT.findall(text))
    if not scores:
        raise ValueError("output contains no REMARK VINA RESULT score")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("output contains a non-finite docking score")
    return scores


def build_unidock_command(
    ligands: Sequence[PreparedLigand],
    *,
    config: DockingConfig,
    score_path: Path | None = None,
    index_path: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        config.unidock_executable,
        "--receptor",
        str(config.receptor),
        "--ligand_index",
        str(index_path),
        "--dir",
        str(output_dir),
        "--center_x",
        str(config.center[0]),
        "--center_y",
        str(config.center[1]),
        "--center_z",
        str(config.center[2]),
        "--size_x",
        str(config.size[0]),
        "--size_y",
        str(config.size[1]),
        "--size_z",
        str(config.size[2]),
        "--search_mode",
        config.search_mode,
        "--scoring",
        config.scoring,
        "--num_modes",
        str(config.num_modes),
        "--energy_range",
        str(config.energy_range),
        "--refine_step",
        str(config.refine_step),
        "--seed",
        str(config.seed),
        "--device_id",
        str(config.device_id),
        "--verbosity",
        str(config.verbosity),
    ]
    if config.refine_step < 1:
        raise ValueError("refine_step must be positive")
    if config.verbosity not in {0, 1, 2}:
        raise ValueError("Uni-Dock verbosity must be 0, 1, or 2")
    if config.no_refine:
        command.append("--no_refine")
    if config.scores_only_output:
        if score_path is None:
            raise ValueError("score_path is required for scores-only output")
        command.extend(["--scores_only_output", "--score_file", str(score_path)])
    if config.max_gpu_memory > 0:
        command.extend(["--max_gpu_memory", str(config.max_gpu_memory)])
    return command


def invoke_unidock(
    ligands: Sequence[PreparedLigand],
    *,
    config: DockingConfig,
    work_dir: Path,
    label: str,
) -> DockInvocation:
    if not ligands:
        raise ValueError("cannot invoke Uni-Dock with an empty ligand batch")
    output_dir = work_dir / f"{label}_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = work_dir / f"{label}_ligands.txt"
    index_path.write_text("\n".join(str(ligand.path.resolve()) for ligand in ligands) + "\n", encoding="utf-8")
    stdout_path = work_dir / f"{label}.stdout.log"
    stderr_path = work_dir / f"{label}.stderr.log"
    score_path = work_dir / f"{label}.scores.tsv" if config.scores_only_output else None
    command = build_unidock_command(
        ligands,
        config=config,
        index_path=index_path,
        output_dir=output_dir,
        score_path=score_path,
    )
    started = perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    return DockInvocation(
        process.returncode,
        perf_counter() - started,
        stdout_path,
        stderr_path,
        score_path,
        tuple(command),
    )


def _collect_outputs(
    ligands: Sequence[PreparedLigand],
    *,
    output_dir: Path,
    seconds: float,
) -> tuple[list[DockingResult], list[PreparedLigand]]:
    results: list[DockingResult] = []
    missing: list[PreparedLigand] = []
    for ligand in ligands:
        output_path = expected_output_path(ligand, output_dir)
        if not output_path.is_file():
            missing.append(ligand)
            continue
        try:
            scores = parse_pdbqt_scores(output_path)
            results.append(DockingResult(ligand, "success", scores, output_path, "", seconds))
        except Exception as exc:
            results.append(
                DockingResult(
                    ligand,
                    "invalid_output",
                    (),
                    output_path,
                    f"{type(exc).__name__}: {exc}",
                    seconds,
                )
            )
    return results, missing


def parse_score_table(path: Path) -> dict[str, float]:
    """Parse the compact score table emitted by the patched Uni-Dock binary."""
    if not path.is_file():
        raise FileNotFoundError(path)
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) != 2:
                raise ValueError(f"malformed score row {line_number}")
            ligand_path, raw_score = fields
            if ligand_path in scores:
                raise ValueError(f"duplicate score row for {ligand_path}")
            scores[ligand_path] = float(raw_score)
    return scores


def _collect_invocation_outputs(
    ligands: Sequence[PreparedLigand],
    *,
    config: DockingConfig,
    invocation: DockInvocation,
    work_dir: Path,
    label: str,
) -> tuple[list[DockingResult], list[PreparedLigand]]:
    if not config.scores_only_output:
        return _collect_outputs(
            ligands,
            output_dir=work_dir / f"{label}_outputs",
            seconds=invocation.seconds,
        )
    if invocation.score_path is None:
        return [], list(ligands)
    try:
        score_by_path = parse_score_table(invocation.score_path)
    except (OSError, ValueError):
        return [], list(ligands)

    results: list[DockingResult] = []
    missing: list[PreparedLigand] = []
    for ligand in ligands:
        key = str(ligand.path.resolve())
        if key not in score_by_path:
            missing.append(ligand)
            continue
        score = score_by_path[key]
        if math.isfinite(score):
            results.append(DockingResult(ligand, "success", (score,), None, "", invocation.seconds))
        else:
            results.append(
                DockingResult(
                    ligand,
                    "invalid_output",
                    (),
                    None,
                    "score table contains a non-finite docking score",
                    invocation.seconds,
                )
            )
    return results, missing


def dock_batch_resilient(
    ligands: Sequence[PreparedLigand],
    *,
    config: DockingConfig,
    work_dir: Path,
    label: str,
    retry_missing: bool = True,
) -> tuple[list[DockingResult], list[DockInvocation]]:
    """Dock a batch and retry only missing outputs in split sub-batches once."""
    if config.engine == "autodock-gpu":
        from .autodock_gpu import dock_batch_resilient_autodock_gpu

        return dock_batch_resilient_autodock_gpu(
            ligands,
            config=config,
            work_dir=work_dir,
            label=label,
            retry_missing=retry_missing,
        )
    if config.engine != "unidock":
        raise ValueError(f"unsupported docking engine: {config.engine}")

    if not ligands:
        return [], []
    invocation = invoke_unidock(ligands, config=config, work_dir=work_dir, label=label)
    results, missing = _collect_invocation_outputs(
        ligands,
        config=config,
        invocation=invocation,
        work_dir=work_dir,
        label=label,
    )
    invocations = [invocation]
    if not missing:
        return results, invocations

    stderr_tail = invocation.stderr_path.read_text(encoding="utf-8", errors="replace")[-2_000:]
    if not retry_missing:
        results.extend(
            DockingResult(
                ligand,
                "docking_failed",
                (),
                None,
                f"missing output; Uni-Dock exit={invocation.returncode}; {stderr_tail.strip()}",
                invocation.seconds,
            )
            for ligand in missing
        )
        return results, invocations

    # Splitting prevents one poison ligand from discarding an otherwise healthy
    # batch. Each missing ligand is retried at most once.
    midpoint = max(1, len(missing) // 2)
    retry_groups = [missing[:midpoint], missing[midpoint:]] if len(missing) > 1 else [missing]
    for group_index, group in enumerate(retry_groups):
        if not group:
            continue
        retry_label = f"{label}_retry{group_index}"
        retry = invoke_unidock(group, config=config, work_dir=work_dir, label=retry_label)
        invocations.append(retry)
        retry_results, still_missing = _collect_invocation_outputs(
            group,
            config=config,
            invocation=retry,
            work_dir=work_dir,
            label=retry_label,
        )
        results.extend(retry_results)
        retry_stderr = retry.stderr_path.read_text(encoding="utf-8", errors="replace")[-2_000:]
        results.extend(
            DockingResult(
                ligand,
                "docking_failed",
                (),
                None,
                f"missing after one retry; Uni-Dock exit={retry.returncode}; {retry_stderr.strip()}",
                retry.seconds,
            )
            for ligand in still_missing
        )
    return results, invocations

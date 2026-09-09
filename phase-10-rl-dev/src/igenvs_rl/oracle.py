"""Docking oracle implemented as calls to the existing iGenVS CLI."""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DockingOutcome:
    molecule_id: str
    canonical_smiles: str
    status: str
    score: float | None
    error: str


def stable_molecule_id(smiles: str) -> str:
    import hashlib

    return "mol_" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:20]


def _write_input(path: Path, molecules: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["molecule_id", "smiles"])
        writer.writeheader()
        for molecule_id, smiles in molecules.items():
            writer.writerow({"molecule_id": molecule_id, "smiles": smiles})


def _visible_gpu_tokens() -> list[str]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if visible in {"", "-1", "NoDevFiles", "void"}:
            return []
        return [token.strip() for token in visible.split(",") if token.strip()]
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return []
    return [line.strip() for line in process.stdout.splitlines() if line.strip()]


class IGenVSOracle:
    def __init__(
        self,
        *,
        target: Path,
        engine: str,
        scoring: str,
        search_mode: str,
        shards: int,
        prep_workers: int,
        validation_workers: int,
        seed: int,
    ) -> None:
        self.target = target
        self.engine = engine
        self.scoring = scoring
        self.search_mode = search_mode
        self.shards = shards
        self.prep_workers = prep_workers
        self.validation_workers = validation_workers
        self.seed = seed

    def _command(
        self,
        input_path: Path,
        output_path: Path,
        shard_index: int,
        *,
        shard_count: int | None = None,
    ) -> list[str]:
        effective_shards = self.shards if shard_count is None else shard_count
        return [
            "igenvs",
            "screen",
            "--input",
            str(input_path),
            "--input-format",
            "csv",
            "--smiles-column",
            "smiles",
            "--id-column",
            "molecule_id",
            "--target",
            str(self.target),
            "--engine",
            self.engine,
            "--scoring",
            self.scoring,
            "--search-mode",
            self.search_mode,
            "--batch-size",
            "auto",
            "--prep-workers",
            str(self.prep_workers),
            "--validation-workers",
            str(self.validation_workers),
            "--prep-mode",
            "standard",
            "--pose-output",
            "none",
            "--seed",
            str(self.seed),
            "--device-id",
            "0",
            "--num-shards",
            str(effective_shards),
            "--shard-index",
            str(shard_index),
            "--output-dir",
            str(output_path),
        ]

    def dock(self, molecules: dict[str, str], output_dir: Path) -> dict[str, DockingOutcome]:
        if not molecules:
            return {}
        output_dir.mkdir(parents=True, exist_ok=False)
        input_path = output_dir / "input.csv"
        _write_input(input_path, molecules)
        # A highly concentrated policy can have fewer uncached molecules than
        # allocated GPUs. Do not launch empty iGenVS shards: the CLI correctly
        # rejects a shard with no molecules, which would otherwise turn normal
        # score-cache reuse into an infrastructure failure.
        effective_shards = min(self.shards, len(molecules))
        gpu_tokens = _visible_gpu_tokens()
        if len(gpu_tokens) < effective_shards:
            raise RuntimeError(
                f"requested {effective_shards} docking shards but only "
                f"{len(gpu_tokens)} GPU(s) are visible"
            )

        processes: list[tuple[int, subprocess.Popen[str], object, object]] = []
        for shard_index in range(effective_shards):
            shard_dir = output_dir / f"shard-{shard_index}"
            stdout_handle = (output_dir / f"shard-{shard_index}.stdout.log").open(
                "w", encoding="utf-8"
            )
            stderr_handle = (output_dir / f"shard-{shard_index}.stderr.log").open(
                "w", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_tokens[shard_index]
            command = self._command(
                input_path,
                shard_dir,
                shard_index,
                shard_count=effective_shards,
            )
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            processes.append((shard_index, process, stdout_handle, stderr_handle))

        failures: list[str] = []
        for shard_index, process, stdout_handle, stderr_handle in processes:
            return_code = process.wait()
            stdout_handle.close()
            stderr_handle.close()
            if return_code != 0:
                failures.append(
                    f"shard {shard_index} exited {return_code}; see "
                    f"{output_dir / f'shard-{shard_index}.stderr.log'}"
                )
        if failures:
            raise RuntimeError("iGenVS docking failed: " + "; ".join(failures))

        outcomes: dict[str, DockingOutcome] = {}
        for shard_index in range(effective_shards):
            result_path = output_dir / f"shard-{shard_index}" / "results.csv"
            if not result_path.is_file():
                raise RuntimeError(f"iGenVS did not create {result_path}")
            with result_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_score = (row.get("docking_score") or "").strip()
                    outcomes[row["molecule_id"]] = DockingOutcome(
                        molecule_id=row["molecule_id"],
                        canonical_smiles=row.get("canonical_smiles", ""),
                        status=row.get("status", "unknown"),
                        score=float(raw_score) if raw_score else None,
                        error=row.get("error", ""),
                    )
            rejected_path = (
                output_dir / f"shard-{shard_index}" / "validation" / "rejected.csv"
            )
            if rejected_path.is_file():
                with rejected_path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        outcomes[row["molecule_id"]] = DockingOutcome(
                            molecule_id=row["molecule_id"],
                            canonical_smiles="",
                            status=row.get("status", "validation_rejected"),
                            score=None,
                            error=row.get("error", ""),
                        )
        return outcomes


class FakeOracle:
    """Fast deterministic oracle for tests and trainer dry runs."""

    def dock(self, molecules: dict[str, str], output_dir: Path) -> dict[str, DockingOutcome]:
        from rdkit import Chem

        output_dir.mkdir(parents=True, exist_ok=False)
        outcomes: dict[str, DockingOutcome] = {}
        for molecule_id, smiles in molecules.items():
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                outcome = DockingOutcome(molecule_id, smiles, "invalid_smiles", None, "")
            else:
                heavy = molecule.GetNumHeavyAtoms()
                hetero = sum(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms())
                rings = molecule.GetRingInfo().NumRings()
                score = -(0.18 * heavy + 0.12 * hetero + 0.08 * rings)
                outcome = DockingOutcome(molecule_id, smiles, "success", score, "")
            outcomes[molecule_id] = outcome

        with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["molecule_id", "canonical_smiles", "status", "docking_score", "error"],
            )
            writer.writeheader()
            for outcome in outcomes.values():
                writer.writerow(
                    {
                        "molecule_id": outcome.molecule_id,
                        "canonical_smiles": outcome.canonical_smiles,
                        "status": outcome.status,
                        "docking_score": "" if outcome.score is None else outcome.score,
                        "error": outcome.error,
                    }
                )
        return outcomes

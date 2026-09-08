"""Job configuration and lightweight persistent state."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONFIG_NAME = "config.json"


@dataclass(frozen=True)
class JobConfig:
    schema_version: int
    target: str
    image: str
    igenvs_project: str
    model_root: str
    model_id: str = "base-isomeric"
    oracle: str = "igenvs"
    engine: str = "unidock"
    scoring: str = "auto"
    search_mode: str = "fast"
    shards: int = 1
    prep_workers: int = 16
    validation_workers: int = 8
    batch_size: int = 256
    reference_count: int = 512
    evaluation_count: int = 256
    evaluation_every: int = 5
    temperature: float = 1.0
    top_k: int = 64
    generator_seed: int = 13
    docking_seed: int = 181129
    learning_rate: float = 1e-5
    kl_beta: float = 0.02
    target_kl: float = 0.05
    max_grad_norm: float = 1.0
    reward_mode: str = "percentile"
    tail_fraction: float = 0.10
    tail_weight: float = 1.0
    reward_seen_molecules: bool = False
    elite_fraction: float = 0.01
    initial_model_root: str | None = None
    minimum_elite_unique: int = 0
    require_lipinski: bool = False
    minimum_qed: float = 0.0
    maximum_absolute_formal_charge: int | None = None
    minimum_fraction_csp3: float = 0.0
    maximum_aromatic_rings: int | None = None
    reward_occurrence_cap: int | None = None
    fresh_evaluation_docking: bool = False
    maximum_top_molecule_fraction: float = 1.0


def save_config(job_dir: Path, config: JobConfig) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / CONFIG_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_config(job_dir: Path) -> JobConfig:
    path = job_dir.expanduser().resolve() / CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"RL job configuration does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported RL job schema version")
    return JobConfig(**payload)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_seen_smiles(job_dir: Path) -> set[str]:
    path = job_dir / "seen.smi"
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_seen_smiles(job_dir: Path, smiles: list[str]) -> None:
    if not smiles:
        return
    with (job_dir / "seen.smi").open("a", encoding="utf-8") as handle:
        for value in smiles:
            handle.write(value + "\n")


def load_score_cache(job_dir: Path) -> dict[str, tuple[str, float]]:
    """Load successful scores cached under this frozen job protocol."""
    path = job_dir / "score-cache.csv"
    if not path.is_file():
        return {}
    cache: dict[str, tuple[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cache[row["molecule_id"]] = (
                row["canonical_smiles"],
                float(row["docking_score"]),
            )
    return cache


def append_score_cache(
    job_dir: Path,
    entries: list[tuple[str, str, float]],
) -> None:
    """Append newly successful unique dockings to the job-local cache."""
    if not entries:
        return
    path = job_dir / "score-cache.csv"
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["molecule_id", "canonical_smiles", "docking_score"],
        )
        if not exists:
            writer.writeheader()
        for molecule_id, canonical_smiles, docking_score in entries:
            writer.writerow(
                {
                    "molecule_id": molecule_id,
                    "canonical_smiles": canonical_smiles,
                    "docking_score": docking_score,
                }
            )

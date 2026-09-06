"""Small serializable records passed between screening stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRecord:
    molecule_id: str
    original_smiles: str
    source_row: int


@dataclass(frozen=True)
class ValidatedRecord:
    molecule_id: str
    original_smiles: str
    canonical_smiles: str
    source_row: int
    heavy_atoms: int
    rotatable_bonds: int


@dataclass(frozen=True)
class ValidationFailure:
    molecule_id: str
    original_smiles: str
    source_row: int
    status: str
    error: str


@dataclass(frozen=True)
class PreparedLigand:
    record: ValidatedRecord
    path: Path
    atom_count: int
    torsion_count: int
    seconds: float


@dataclass(frozen=True)
class PreparationFailure:
    record: ValidatedRecord
    status: str
    error: str
    seconds: float


@dataclass(frozen=True)
class DockingResult:
    ligand: PreparedLigand
    status: str
    scores: tuple[float, ...]
    pose_path: Path | None
    error: str
    batch_seconds: float

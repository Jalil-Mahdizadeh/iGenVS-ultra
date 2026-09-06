"""Randomized SMILES enumeration for derivative-generation seed preparation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class RandomizedSmilesResult:
    """Result metadata for a randomized SMILES enumeration run."""

    reference_smiles: str
    variants: list[str]
    attempts: int
    max_variants: int
    max_attempts: int
    stagnation_limit: int
    stop_reason: str
    isomeric_smiles: bool
    include_canonical: bool

    @property
    def variant_count(self) -> int:
        return len(self.variants)

    def metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["variant_count"] = self.variant_count
        data.pop("variants")
        return data


def _seed_rdkit(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        from rdkit import rdBase

        rdBase.SeedRandomNumberGenerator(int(seed))
    except Exception:
        pass


def _add_unique(seen: dict[str, None], smiles: str, max_variants: int) -> bool:
    if len(seen) >= max_variants:
        return False
    if smiles in seen:
        return False
    seen[smiles] = None
    return True


def enumerate_randomized_smiles(
    reference_smiles: str,
    *,
    max_variants: int = 10_000,
    max_attempts: int | None = None,
    stagnation_limit: int = 5_000,
    include_canonical: bool = True,
    isomeric_smiles: bool = True,
    seed: int | None = 13,
) -> RandomizedSmilesResult:
    """Enumerate canonical and randomized SMILES variants for one molecule.

    RDKit randomized SMILES enumeration is stochastic, so true mathematical
    exhaustion is not directly observable. This function stops when a high cap
    is reached, the maximum number of attempts is reached, or RDKit has not
    produced a new unique variant for ``stagnation_limit`` consecutive attempts.
    """

    if max_variants <= 0:
        raise ValueError("max_variants must be positive")
    if stagnation_limit <= 0:
        raise ValueError("stagnation_limit must be positive")
    if max_attempts is None:
        max_attempts = max(max_variants * 50, stagnation_limit)
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    mol = Chem.MolFromSmiles(reference_smiles)
    if mol is None:
        raise ValueError(f"Invalid reference SMILES: {reference_smiles}")

    _seed_rdkit(seed)
    seen: dict[str, None] = {}

    if include_canonical:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric_smiles)
        _add_unique(seen, canonical, max_variants)

    for atom_index in range(mol.GetNumAtoms()):
        rooted = Chem.MolToSmiles(
            mol,
            canonical=False,
            rootedAtAtom=atom_index,
            isomericSmiles=isomeric_smiles,
        )
        _add_unique(seen, rooted, max_variants)
        if len(seen) >= max_variants:
            return RandomizedSmilesResult(
                reference_smiles=reference_smiles,
                variants=list(seen),
                attempts=0,
                max_variants=max_variants,
                max_attempts=max_attempts,
                stagnation_limit=stagnation_limit,
                stop_reason="max_variants",
                isomeric_smiles=isomeric_smiles,
                include_canonical=include_canonical,
            )

    attempts = 0
    attempts_since_new = 0
    while len(seen) < max_variants and attempts < max_attempts and attempts_since_new < stagnation_limit:
        attempts += 1
        randomized = Chem.MolToSmiles(
            mol,
            canonical=False,
            doRandom=True,
            isomericSmiles=isomeric_smiles,
        )
        if _add_unique(seen, randomized, max_variants):
            attempts_since_new = 0
        else:
            attempts_since_new += 1

    if len(seen) >= max_variants:
        stop_reason = "max_variants"
    elif attempts >= max_attempts:
        stop_reason = "max_attempts"
    else:
        stop_reason = "stagnation_limit"

    return RandomizedSmilesResult(
        reference_smiles=reference_smiles,
        variants=list(seen),
        attempts=attempts,
        max_variants=max_variants,
        max_attempts=max_attempts,
        stagnation_limit=stagnation_limit,
        stop_reason=stop_reason,
        isomeric_smiles=isomeric_smiles,
        include_canonical=include_canonical,
    )


def write_randomized_smiles(result: RandomizedSmilesResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(result.variants) + "\n", encoding="utf-8")


def write_randomization_metadata(result: RandomizedSmilesResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.metadata(), indent=2), encoding="utf-8")

"""CPU-parallel 3D conformer and Meeko PDBQT preparation."""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .records import PreparedLigand, PreparationFailure, ValidatedRecord


PREPARATION_MODES = ("standard", "fast")
DEFAULT_EMBED_MAX_ATTEMPTS = 50
DEFAULT_EMBED_TIMEOUT_SECONDS = 3
_MEEKO_PREPARATOR = None


def _meeko_preparator():
    """Reuse Meeko's compiled atom-typing machinery within each worker process."""
    global _MEEKO_PREPARATOR
    if _MEEKO_PREPARATOR is None:
        from meeko import MoleculePreparation

        _MEEKO_PREPARATOR = MoleculePreparation()
    return _MEEKO_PREPARATOR


def ligand_basename(record: ValidatedRecord) -> str:
    digest = hashlib.sha256(record.molecule_id.encode("utf-8")).hexdigest()[:12]
    return f"lig_{record.source_row:012d}_{digest}"


def _molecule_seed(record: ValidatedRecord, base_seed: int) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{record.molecule_id}:{record.canonical_smiles}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return 1 + int.from_bytes(digest, "big") % 2_147_483_646


def _embedding_timed_out(params: object) -> bool:
    """Return whether RDKit stopped embedding at its native wall-time guard."""

    try:
        from rdkit.Chem.rdDistGeom import EmbedFailureCauses

        counts = params.GetFailureCounts()  # type: ignore[attr-defined]
        index = int(EmbedFailureCauses.EXCEEDED_TIMEOUT)
        return index < len(counts) and bool(counts[index])
    except (AttributeError, IndexError, TypeError, ValueError):
        # Pinned releases expose failure tracking. The fallback keeps source
        # compatibility with older RDKit builds while maxIterations still
        # bounds their deterministic amount of work.
        return False


def _preparation_priority(record: ValidatedRecord) -> tuple[int, int, int, int]:
    """Cheap deterministic proxy used only to drain likely stragglers first."""

    smiles = record.canonical_smiles
    stereocenters = smiles.count("@")
    ring_tokens = sum(character.isdigit() for character in smiles)
    return (
        stereocenters,
        ring_tokens,
        record.heavy_atoms + 2 * record.rotatable_bonds,
        -record.source_row,
    )


def _prepare_one(
    record: ValidatedRecord,
    output_dir: Path,
    base_seed: int,
    max_atoms: int,
    max_torsions: int,
    prep_mode: str = "standard",
    embed_max_attempts: int = DEFAULT_EMBED_MAX_ATTEMPTS,
    embed_timeout_seconds: int = DEFAULT_EMBED_TIMEOUT_SECONDS,
) -> PreparedLigand | PreparationFailure:
    started = perf_counter()
    try:
        from meeko import PDBQTWriterLegacy
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem

        if prep_mode not in PREPARATION_MODES:
            raise ValueError(f"prep_mode must be one of {', '.join(PREPARATION_MODES)}")
        if embed_max_attempts < 1:
            raise ValueError("embed_max_attempts must be positive")
        if embed_timeout_seconds < 1:
            raise ValueError("embed_timeout_seconds must be positive")

        # A molecule with more heavy atoms than the engine can consume cannot
        # become valid after adding hydrogens or Meeko typing. Reject it before
        # spending any time on 3D coordinates.
        if record.heavy_atoms > max_atoms:
            return PreparationFailure(
                record,
                "unsupported_size",
                f"molecule has {record.heavy_atoms} heavy atoms; configured engine limit is {max_atoms}",
                perf_counter() - started,
            )

        mol = Chem.MolFromSmiles(record.canonical_smiles)
        if mol is None:
            raise ValueError("RDKit could not reconstruct canonical SMILES")
        mol.SetProp("_Name", record.molecule_id)
        mol = Chem.AddHs(mol)

        # Valid hypervalent sulfur can make embedding/typing probes noisy even
        # when preparation succeeds. All return codes and exceptions remain
        # checked and become typed per-ligand failures.
        with rdBase.BlockLogs():
            params = AllChem.ETKDGv3()
            params.randomSeed = _molecule_seed(record, base_seed)
            params.numThreads = 1
            params.useSmallRingTorsions = True
            # The former RDKit defaults could spend tens of seconds, and in
            # observed cases nearly 100 seconds, trying to embed one molecule
            # that ultimately failed. Bound both deterministic attempts and
            # wall time. Most accepted molecules finish two orders of
            # magnitude below these guards.
            params.maxIterations = embed_max_attempts
            if hasattr(params, "timeout"):
                params.timeout = embed_timeout_seconds
            if hasattr(params, "trackFailures"):
                params.trackFailures = True
            embed_status = AllChem.EmbedMolecule(mol, params)
            if embed_status != 0:
                if _embedding_timed_out(params):
                    return PreparationFailure(
                        record,
                        "preparation_timeout",
                        (
                            "RDKit ETKDG exceeded the hard-molecule budget "
                            f"({embed_max_attempts} attempts/{embed_timeout_seconds}s); "
                            "random-coordinate retry skipped"
                        ),
                        perf_counter() - started,
                    )
                params.useRandomCoords = True
                # A quick deterministic failure can still be rescued, but the
                # fallback gets half the primary attempt/time budget. It is
                # never allowed to recreate the old long-tail barrier.
                params.maxIterations = max(5, embed_max_attempts // 2)
                if hasattr(params, "timeout"):
                    params.timeout = max(1, (embed_timeout_seconds + 1) // 2)
                embed_status = AllChem.EmbedMolecule(mol, params)
            if embed_status != 0:
                if _embedding_timed_out(params):
                    return PreparationFailure(
                        record,
                        "preparation_timeout",
                        (
                            "RDKit random-coordinate ETKDG retry exceeded the "
                            "bounded hard-molecule budget"
                        ),
                        perf_counter() - started,
                    )
                raise ValueError(
                    f"RDKit ETKDG embedding failed with status {embed_status}"
                )

            if prep_mode == "standard":
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                elif AllChem.UFFHasAllMoleculeParams(mol):
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            setups = _meeko_preparator().prepare(mol)
        if len(setups) != 1:
            raise ValueError(f"Meeko produced {len(setups)} setups; only standard nonreactive docking is supported")
        pdbqt_text, success, error = PDBQTWriterLegacy.write_string(setups[0])
        if not success:
            raise ValueError(f"Meeko PDBQT writer failed: {error}")

        atom_count = sum(line.startswith(("ATOM", "HETATM")) for line in pdbqt_text.splitlines())
        torsion_count = sum(line.startswith("BRANCH") for line in pdbqt_text.splitlines())
        if atom_count > max_atoms:
            return PreparationFailure(
                record,
                "unsupported_size",
                f"prepared ligand has {atom_count} atoms; configured engine limit is {max_atoms}",
                perf_counter() - started,
            )
        if torsion_count > max_torsions:
            return PreparationFailure(
                record,
                "unsupported_torsions",
                f"prepared ligand has {torsion_count} active torsions; configured engine limit is {max_torsions}",
                perf_counter() - started,
            )
        if atom_count == 0:
            raise ValueError("Meeko produced a PDBQT with no atoms")

        output_path = output_dir / f"{ligand_basename(record)}.pdbqt"
        output_path.write_text(pdbqt_text, encoding="utf-8")
        return PreparedLigand(record, output_path, atom_count, torsion_count, perf_counter() - started)
    except Exception as exc:
        return PreparationFailure(
            record,
            "preparation_failed",
            f"{type(exc).__name__}: {exc}",
            perf_counter() - started,
        )


def submit_preparation_batch(
    executor: ProcessPoolExecutor,
    records: Sequence[ValidatedRecord],
    *,
    output_dir: Path,
    seed: int,
    max_atoms: int = 300,
    max_torsions: int = 48,
    prep_mode: str = "standard",
    embed_max_attempts: int = DEFAULT_EMBED_MAX_ATTEMPTS,
    embed_timeout_seconds: int = DEFAULT_EMBED_TIMEOUT_SECONDS,
) -> list[Future[PreparedLigand | PreparationFailure]]:
    if prep_mode not in PREPARATION_MODES:
        raise ValueError(f"prep_mode must be one of {', '.join(PREPARATION_MODES)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Likely complex embeddings enter the process queue first. Return futures
    # in source order so output and docking identity remain unchanged.
    ranked = sorted(
        enumerate(records),
        key=lambda item: _preparation_priority(item[1]),
        reverse=True,
    )
    submitted = {
        index: executor.submit(
            _prepare_one,
            record,
            output_dir,
            seed,
            max_atoms,
            max_torsions,
            prep_mode,
            embed_max_attempts,
            embed_timeout_seconds,
        )
        for index, record in ranked
    }
    return [submitted[index] for index in range(len(records))]


def collect_preparation_batch(
    futures: Sequence[Future[PreparedLigand | PreparationFailure]],
) -> tuple[list[PreparedLigand], list[PreparationFailure]]:
    prepared: list[PreparedLigand] = []
    failed: list[PreparationFailure] = []
    for future in futures:
        result = future.result()
        if isinstance(result, PreparedLigand):
            prepared.append(result)
        else:
            failed.append(result)
    return prepared, failed


def prepare_records(
    records: Sequence[ValidatedRecord],
    *,
    output_dir: Path,
    workers: int,
    seed: int,
    max_atoms: int = 300,
    max_torsions: int = 48,
    prep_mode: str = "standard",
    embed_max_attempts: int = DEFAULT_EMBED_MAX_ATTEMPTS,
    embed_timeout_seconds: int = DEFAULT_EMBED_TIMEOUT_SECONDS,
) -> tuple[list[PreparedLigand], list[PreparationFailure]]:
    """Prepare a bounded pilot library; used by the batch tuner."""
    if not records:
        return [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = submit_preparation_batch(
            executor,
            records,
            output_dir=output_dir,
            seed=seed,
            max_atoms=max_atoms,
            max_torsions=max_torsions,
            prep_mode=prep_mode,
            embed_max_attempts=embed_max_attempts,
            embed_timeout_seconds=embed_timeout_seconds,
        )
        return collect_preparation_batch(futures)

"""Inference-only gMol policy validation without unused training split work."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from rdkit import Chem, RDLogger


def initialize_policy_worker() -> None:
    """Silence RDKit diagnostics in policy-validation workers."""
    RDLogger.DisableLog("rdApp.*")


def canonicalize_policy_chunk(
    payload: tuple[int, Sequence[tuple[str, str]], dict[str, Any]],
) -> list[tuple[int, str, str, str, int, str | None]]:
    """Match gMol inference acceptance/canonicalization in source order.

    The training canonicalizer also computes non-isomeric SMILES, Murcko
    scaffolds, split assignment, and hash buckets. None of those values enters
    inference. Omitting them here leaves acceptance, canonical SMILES,
    molecule hashes, atom counts, graph features, and embeddings unchanged.
    """
    start, rows, policy = payload
    allowed = set(policy["allowed_elements"])
    fragment_policy = str(policy["fragment_policy"])
    isomeric = bool(policy["isomeric_smiles"])
    minimum = int(policy["min_atoms"])
    maximum = int(policy["max_atoms"])
    output = []
    for offset, (molecule_id, raw_smiles) in enumerate(rows):
        index = start + offset
        reason = None
        canonical = ""
        molecule_hash = ""
        atom_count = 0
        if not raw_smiles or not raw_smiles.strip():
            reason = "empty_smiles"
        else:
            molecule = Chem.MolFromSmiles(raw_smiles)
            if molecule is None:
                reason = "parse_failure"
            else:
                fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
                if len(fragments) > 1:
                    if fragment_policy == "reject":
                        reason = "disconnected"
                    elif fragment_policy == "largest":
                        molecule = max(
                            fragments,
                            key=lambda item: (
                                item.GetNumHeavyAtoms(),
                                item.GetNumAtoms(),
                                Chem.MolToSmiles(item, canonical=True, isomericSmiles=True),
                            ),
                        )
                    else:
                        raise ValueError(f"unsupported fragment policy: {fragment_policy}")
                if reason is None and not {
                    atom.GetSymbol() for atom in molecule.GetAtoms()
                }.issubset(allowed):
                    reason = "unsupported_element"
                if reason is None:
                    atom_count = int(molecule.GetNumAtoms())
                    if atom_count < minimum:
                        reason = "too_few_atoms"
                    elif atom_count > maximum:
                        reason = "too_many_atoms"
                if reason is None:
                    canonical = Chem.MolToSmiles(
                        molecule,
                        canonical=True,
                        isomericSmiles=isomeric,
                    )
                    reparsed = Chem.MolFromSmiles(canonical)
                    if reparsed is None:
                        reason = "canonical_reparse_failure"
                        canonical = ""
                        atom_count = 0
                    else:
                        atom_count = int(reparsed.GetNumAtoms())
                        molecule_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        output.append(
            (index, molecule_id, canonical, molecule_hash, atom_count, reason)
        )
    return output

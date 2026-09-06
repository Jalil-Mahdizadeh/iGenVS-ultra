from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Scaffolds import MurckoScaffold

from .schema import (
    ATOM_TYPES,
    BOND_STEREO,
    BOND_TYPES,
    CHIRAL_TAGS,
    DEGREES,
    FORMAL_CHARGES,
    HYBRIDIZATIONS,
    TOTAL_HYDROGENS,
    feature_schema,
)
from .util import stable_fraction, stable_u64


_FEATURE_FACTORY = None


def _feature_factory():
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(f"{RDConfig.RDDataDir}/BaseFeatures.fdef")
    return _FEATURE_FACTORY


def _one_hot(value: Any, allowed: list[Any], include_other: bool = True) -> list[float]:
    width = len(allowed) + int(include_other)
    result = [0.0] * width
    try:
        index = allowed.index(value)
    except ValueError:
        if not include_other:
            raise ValueError(f"Unsupported categorical value {value!r}; allowed={allowed!r}") from None
        index = len(allowed)
    result[index] = 1.0
    return result


@dataclass(frozen=True)
class CanonicalMolecule:
    smiles: str
    nonisomeric_smiles: str
    molecule_hash: str
    scaffold: str
    split: str
    bucket: int
    atom_count: int
    bond_count: int


@dataclass(frozen=True)
class Rejection:
    reason: str


def canonicalize(
    raw_smiles: str,
    *,
    isomeric_smiles: bool,
    fragment_policy: str,
    allowed_elements: set[str],
    min_atoms: int,
    max_atoms: int,
    buckets: int,
    split_cfg: dict[str, Any],
) -> CanonicalMolecule | Rejection:
    if not raw_smiles or not raw_smiles.strip():
        return Rejection("empty_smiles")
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return Rejection("parse_failure")

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) > 1:
        if fragment_policy == "reject":
            return Rejection("disconnected")
        if fragment_policy != "largest":
            raise ValueError(f"Unsupported fragment policy: {fragment_policy}")
        mol = max(
            fragments,
            key=lambda item: (
                item.GetNumHeavyAtoms(),
                item.GetNumAtoms(),
                Chem.MolToSmiles(item, canonical=True, isomericSmiles=True),
            ),
        )

    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if not symbols.issubset(allowed_elements):
        return Rejection("unsupported_element")
    atom_count = mol.GetNumAtoms()
    if atom_count < min_atoms:
        return Rejection("too_few_atoms")
    if atom_count > max_atoms:
        return Rejection("too_many_atoms")

    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric_smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Rejection("canonical_reparse_failure")
    nonisomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    molecule_hash = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=False)
    split_key = f"SCAFFOLD:{scaffold}" if scaffold else f"ACYCLIC:{nonisomeric}"
    split_value = stable_fraction(split_key, int(split_cfg["seed"]))
    train_end = float(split_cfg["train_fraction"])
    validation_end = train_end + float(split_cfg["validation_fraction"])
    if split_value < train_end:
        split = "train"
    elif split_value < validation_end:
        split = "validation"
    else:
        split = "test"
    return CanonicalMolecule(
        smiles=smiles,
        nonisomeric_smiles=nonisomeric,
        molecule_hash=molecule_hash,
        scaffold=scaffold,
        split=split,
        bucket=stable_u64(smiles) % buckets,
        atom_count=mol.GetNumAtoms(),
        bond_count=mol.GetNumBonds(),
    )


def _hydrogen_bond_flags(mol: Chem.Mol) -> dict[int, tuple[float, float]]:
    flags = {index: [0.0, 0.0] for index in range(mol.GetNumAtoms())}
    for feature in _feature_factory().GetFeaturesForMol(mol):
        family = feature.GetFamily()
        if family not in {"Donor", "Acceptor"}:
            continue
        position = 0 if family == "Donor" else 1
        for atom_index in feature.GetAtomIds():
            flags[int(atom_index)][position] = 1.0
    return {key: (value[0], value[1]) for key, value in flags.items()}


def _position_encoding(mol: Chem.Mol, width: int, base: int = 100) -> np.ndarray:
    if width == 0:
        return np.empty((mol.GetNumAtoms(), 0), dtype=np.float32)
    if width % 2:
        raise ValueError("canonical_position_encoding_dim must be even")
    count = mol.GetNumAtoms()
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=True))
    order = np.argsort(ranks)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(count)
    positions = np.zeros((count, width), dtype=np.float32)
    for position in range(count):
        for pair in range(width // 2):
            denominator = base ** (2 * pair / width)
            positions[position, 2 * pair] = np.sin(position / denominator)
            positions[position, 2 * pair + 1] = np.cos(position / denominator)
    return positions[inverse]


def featurize_molecule(
    mol: Chem.Mol, *, include_chirality: bool = True, position_dim: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    schema = feature_schema(include_chirality, position_dim)
    hbond = _hydrogen_bond_flags(mol)
    node_rows: list[list[float]] = []
    for atom in mol.GetAtoms():
        row = []
        row.extend(_one_hot(atom.GetSymbol(), ATOM_TYPES, include_other=False))
        row.extend(_one_hot(atom.GetFormalCharge(), FORMAL_CHARGES, include_other=True))
        row.extend(_one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS, include_other=True))
        row.extend(hbond[atom.GetIdx()])
        row.append(float(atom.GetIsAromatic()))
        row.extend(_one_hot(atom.GetTotalDegree(), DEGREES, include_other=True))
        row.extend(_one_hot(atom.GetTotalNumHs(), TOTAL_HYDROGENS, include_other=True))
        if include_chirality:
            row.extend(_one_hot(str(atom.GetChiralTag()), CHIRAL_TAGS, include_other=True))
        node_rows.append(row)
    x = np.asarray(node_rows, dtype=np.float32)
    if position_dim:
        x = np.concatenate([x, _position_encoding(mol, position_dim)], axis=1)

    edge_pairs: list[tuple[int, int]] = []
    edge_rows: list[list[float]] = []
    for bond in mol.GetBonds():
        row = []
        row.extend(_one_hot(str(bond.GetBondType()), BOND_TYPES, include_other=True))
        row.append(float(bond.IsInRing()))
        row.append(float(bond.GetIsConjugated()))
        row.extend(_one_hot(str(bond.GetStereo()), BOND_STEREO, include_other=True))
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_pairs.extend(((begin, end), (end, begin)))
        edge_rows.extend((row, row.copy()))
    if edge_pairs:
        edge_index = np.asarray(edge_pairs, dtype=np.int64).T
        edge_attr = np.asarray(edge_rows, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, int(schema["edge_dim"])), dtype=np.float32)

    if x.shape[1] != schema["node_input_dim"] or edge_attr.shape[1] != schema["edge_dim"]:
        raise RuntimeError("Feature implementation and feature schema disagree")
    return x, edge_index, edge_attr

from __future__ import annotations

"""Exact, low-overhead graph construction for promoted gMolAI inference.

This module deliberately has no PyTorch imports so spawned preprocessing
workers do not initialize a CUDA/PyTorch runtime.  The reference featurizer in
``chem.py`` remains the scientific oracle and is used by qualification tests.
"""

from dataclasses import dataclass
import os
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures

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


FAST_GRAPH_VERSION = "1"
_PROMOTED_SCHEMA = feature_schema(include_chirality=True, position_dim=0)
NODE_DIM = int(_PROMOTED_SCHEMA["node_input_dim"])
EDGE_DIM = int(_PROMOTED_SCHEMA["edge_dim"])


def _group_offsets(groups: list[dict]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for group in groups:
        offsets[str(group["name"])] = offset
        if group["kind"] == "binary":
            offset += 1
        else:
            offset += len(group["values"]) + int(bool(group["other"]))
    return offsets


_NODE_OFFSETS = _group_offsets(_PROMOTED_SCHEMA["node_groups"])
_EDGE_OFFSETS = _group_offsets(_PROMOTED_SCHEMA["edge_groups"])
_ATOM_INDEX = {value: index for index, value in enumerate(ATOM_TYPES)}
_CHARGE_INDEX = {value: index for index, value in enumerate(FORMAL_CHARGES)}
_HYBRID_INDEX = {value: index for index, value in enumerate(HYBRIDIZATIONS)}
_DEGREE_INDEX = {value: index for index, value in enumerate(DEGREES)}
_HYDROGEN_INDEX = {value: index for index, value in enumerate(TOTAL_HYDROGENS)}
_CHIRAL_INDEX = {value: index for index, value in enumerate(CHIRAL_TAGS)}
_BOND_INDEX = {value: index for index, value in enumerate(BOND_TYPES)}
_STEREO_INDEX = {value: index for index, value in enumerate(BOND_STEREO)}


# These are the Donor/Acceptor definitions used by RDKit 2025.09.3's
# BaseFeatures.fdef.  gMolAI consumes only these two families; avoiding every
# unused feature family removes the dominant preprocessing cost.
_HBOND_FDEF = r"""
AtomType NDonor [N&!H0&v3,N&!H0&+1&v4,n&H1&+0]
AtomType AmideN [$(N-C(=O))]
AtomType SulfonamideN [$([N;H0]S(=O)(=O))]
AtomType NDonor [$([Nv3](-C)(-C)-C)]
AtomType NDonor [$(n[n;H1]),$(nc[n;H1])]
AtomType ChalcDonor [O,S;H1;+0]
DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor}]
  Family Donor
  Weights 1
EndFeature
AtomType NAcceptor [n;+0;!X3;!$([n;H1](cc)cc)]
AtomType NAcceptor [$([N;H0]#[C&v4])]
AtomType NAcceptor [N&v3;H0;$(Nc)]
AtomType ChalcAcceptor [O;H0;v2;!$(O=N-*)]
Atomtype ChalcAcceptor [O;-;!$(*-N=O)]
Atomtype ChalcAcceptor [o;+0]
AtomType Hydroxyl [O;H1;v2]
AtomType HalogenAcceptor [F;$(F-[#6]);!$(FC[F,Cl,Br,I])]
DefineFeature SingleAtomAcceptor [{Hydroxyl},{ChalcAcceptor},{NAcceptor},{HalogenAcceptor}]
  Family Acceptor
  Weights 1
EndFeature
"""

_HBOND_FACTORY = None


def _hbond_factory():
    global _HBOND_FACTORY
    if _HBOND_FACTORY is None:
        _HBOND_FACTORY = ChemicalFeatures.BuildFeatureFactoryFromString(_HBOND_FDEF)
    return _HBOND_FACTORY


def _category(index: dict, value, other: int | None, label: str) -> int:
    try:
        return int(index[value])
    except KeyError:
        if other is None:
            raise ValueError(f"Unsupported {label} value {value!r}") from None
        return int(other)


def fast_featurize_molecule(
    mol: Chem.Mol,
    *,
    include_chirality: bool = True,
    position_dim: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays equivalent to the promoted reference feature contract."""

    if not include_chirality or int(position_dim) != 0:
        raise ValueError(
            "Optimized inference supports the promoted feature contract only "
            "(include_chirality=True, position_dim=0)"
        )
    atom_count = int(mol.GetNumAtoms())
    x = np.zeros((atom_count, NODE_DIM), dtype=np.float32)

    donor_column = _NODE_OFFSETS["hydrogen_bond_donor"]
    acceptor_column = _NODE_OFFSETS["hydrogen_bond_acceptor"]
    for feature in _hbond_factory().GetFeaturesForMol(mol):
        family = feature.GetFamily()
        if family not in {"Donor", "Acceptor"}:
            continue
        column = donor_column if family == "Donor" else acceptor_column
        for atom_index in feature.GetAtomIds():
            x[int(atom_index), column] = 1.0

    for atom in mol.GetAtoms():
        row = int(atom.GetIdx())
        x[
            row,
            _NODE_OFFSETS["atom_type"]
            + _category(_ATOM_INDEX, atom.GetSymbol(), None, "atom type"),
        ] = 1.0
        x[
            row,
            _NODE_OFFSETS["formal_charge"]
            + _category(
                _CHARGE_INDEX,
                atom.GetFormalCharge(),
                len(FORMAL_CHARGES),
                "formal charge",
            ),
        ] = 1.0
        x[
            row,
            _NODE_OFFSETS["hybridization"]
            + _category(
                _HYBRID_INDEX,
                str(atom.GetHybridization()),
                len(HYBRIDIZATIONS),
                "hybridization",
            ),
        ] = 1.0
        x[row, _NODE_OFFSETS["aromatic"]] = float(atom.GetIsAromatic())
        x[
            row,
            _NODE_OFFSETS["total_degree"]
            + _category(
                _DEGREE_INDEX,
                atom.GetTotalDegree(),
                len(DEGREES),
                "total degree",
            ),
        ] = 1.0
        x[
            row,
            _NODE_OFFSETS["total_hydrogens"]
            + _category(
                _HYDROGEN_INDEX,
                atom.GetTotalNumHs(),
                len(TOTAL_HYDROGENS),
                "total hydrogens",
            ),
        ] = 1.0
        x[
            row,
            _NODE_OFFSETS["chirality"]
            + _category(
                _CHIRAL_INDEX,
                str(atom.GetChiralTag()),
                len(CHIRAL_TAGS),
                "chirality",
            ),
        ] = 1.0

    edge_count = 2 * int(mol.GetNumBonds())
    edge_index = np.empty((2, edge_count), dtype=np.int64)
    edge_attr = np.zeros((edge_count, EDGE_DIM), dtype=np.float32)
    for bond_number, bond in enumerate(mol.GetBonds()):
        first = 2 * bond_number
        second = first + 1
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        edge_index[:, first] = (begin, end)
        edge_index[:, second] = (end, begin)
        edge_attr[
            first,
            _EDGE_OFFSETS["bond_type"]
            + _category(
                _BOND_INDEX,
                str(bond.GetBondType()),
                len(BOND_TYPES),
                "bond type",
            ),
        ] = 1.0
        edge_attr[first, _EDGE_OFFSETS["same_ring"]] = float(bond.IsInRing())
        edge_attr[first, _EDGE_OFFSETS["conjugated"]] = float(
            bond.GetIsConjugated()
        )
        edge_attr[
            first,
            _EDGE_OFFSETS["stereo"]
            + _category(
                _STEREO_INDEX,
                str(bond.GetStereo()),
                len(BOND_STEREO),
                "bond stereo",
            ),
        ] = 1.0
        edge_attr[second] = edge_attr[first]

    if x.shape != (atom_count, NODE_DIM) or edge_attr.shape != (
        edge_count,
        EDGE_DIM,
    ):
        raise RuntimeError("Optimized features disagree with the promoted schema")
    return x, edge_index, edge_attr


@dataclass(slots=True)
class PackedBatch:
    start: int
    graph_count: int
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    batch: np.ndarray
    node_counts: np.ndarray


def pack_feature_arrays(
    features: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    start: int = 0,
) -> PackedBatch:
    if not features:
        raise ValueError("Cannot pack an empty graph batch")
    node_counts = np.asarray([len(item[0]) for item in features], dtype=np.int64)
    edge_counts = np.asarray(
        [item[1].shape[1] for item in features], dtype=np.int64
    )
    total_nodes = int(node_counts.sum())
    total_edges = int(edge_counts.sum())
    x = np.empty((total_nodes, NODE_DIM), dtype=np.float32)
    edge_index = np.empty((2, total_edges), dtype=np.int64)
    edge_attr = np.empty((total_edges, EDGE_DIM), dtype=np.float32)
    batch = np.repeat(np.arange(len(features), dtype=np.int64), node_counts)

    node_offset = 0
    edge_offset = 0
    for graph_x, graph_edge_index, graph_edge_attr in features:
        next_node = node_offset + len(graph_x)
        next_edge = edge_offset + graph_edge_index.shape[1]
        x[node_offset:next_node] = graph_x
        edge_index[:, edge_offset:next_edge] = graph_edge_index + node_offset
        edge_attr[edge_offset:next_edge] = graph_edge_attr
        node_offset = next_node
        edge_offset = next_edge
    return PackedBatch(
        start=int(start),
        graph_count=len(features),
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        node_counts=node_counts,
    )


def pack_molecules(
    molecules: Sequence[Chem.Mol], *, start: int = 0
) -> PackedBatch:
    return pack_feature_arrays(
        [fast_featurize_molecule(molecule) for molecule in molecules],
        start=start,
    )


def pack_smiles_task(task: tuple[int, Sequence[str]]) -> PackedBatch:
    start, smiles = task
    molecules: list[Chem.Mol] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"gMolAI could not parse canonical SMILES {value!r}")
        molecules.append(molecule)
    return pack_molecules(molecules, start=start)


def smiles_tasks(
    values: Sequence[str],
    *,
    batch_size: int,
    node_budget: int | None = None,
    atom_counts: Sequence[int] | None = None,
) -> Iterable[tuple[int, Sequence[str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if node_budget is not None and node_budget <= 0:
        raise ValueError("node_budget must be positive")
    if atom_counts is not None and len(atom_counts) != len(values):
        raise ValueError("atom_counts must align with values")

    start = 0
    while start < len(values):
        stop = min(start + batch_size, len(values))
        if node_budget is not None and atom_counts is not None:
            nodes = 0
            stop = start
            while stop < len(values) and stop - start < batch_size:
                count = int(atom_counts[stop])
                if count <= 0:
                    raise ValueError("atom counts must be positive")
                if stop > start and nodes + count > node_budget:
                    break
                nodes += count
                stop += 1
            if stop == start:
                stop = start + 1
        yield start, values[start:stop]
        start = stop


def initialize_worker() -> None:
    """Warm immutable RDKit state and prevent nested CPU oversubscription."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _hbond_factory()


def resolve_worker_count(requested: int | str | None, *, cap: int = 48) -> int:
    if cap <= 0:
        raise ValueError("worker cap must be positive")
    affinity = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    limits = [max(1, int(affinity))]
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name)
        if value and value.isdigit() and int(value) > 0:
            limits.append(int(value))
    available = min(limits)
    if requested is None or str(requested).strip().lower() == "auto":
        return max(1, min(int(cap), available))
    workers = int(requested)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers > available:
        raise ValueError(
            f"workers={workers} exceeds the available CPU allocation ({available})"
        )
    return workers

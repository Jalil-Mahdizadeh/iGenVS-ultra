from __future__ import annotations

from dataclasses import dataclass
import copy

from .config import object_hash


ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", "B", "Si"]
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"]
DEGREES = [0, 1, 2, 3, 4, 5, 6]
TOTAL_HYDROGENS = [0, 1, 2, 3, 4]
CHIRAL_TAGS = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE"]
BOND_STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]


def categorical_width(values: list[object], include_other: bool = True) -> int:
    return len(values) + int(include_other)


def feature_schema(include_chirality: bool = True, position_dim: int = 0) -> dict:
    node_groups = [
        {"name": "atom_type", "kind": "one_hot", "values": ATOM_TYPES, "other": False},
        {"name": "formal_charge", "kind": "one_hot", "values": FORMAL_CHARGES, "other": True},
        {"name": "hybridization", "kind": "one_hot", "values": HYBRIDIZATIONS, "other": True},
        {"name": "hydrogen_bond_donor", "kind": "binary"},
        {"name": "hydrogen_bond_acceptor", "kind": "binary"},
        {"name": "aromatic", "kind": "binary"},
        {"name": "total_degree", "kind": "one_hot", "values": DEGREES, "other": True},
        {"name": "total_hydrogens", "kind": "one_hot", "values": TOTAL_HYDROGENS, "other": True},
    ]
    if include_chirality:
        node_groups.append({"name": "chirality", "kind": "one_hot", "values": CHIRAL_TAGS, "other": True})
    target_dim = sum(
        1 if group["kind"] == "binary" else categorical_width(group["values"], group["other"])
        for group in node_groups
    )
    edge_groups = [
        {"name": "bond_type", "kind": "one_hot", "values": BOND_TYPES, "other": True},
        {"name": "same_ring", "kind": "binary"},
        {"name": "conjugated", "kind": "binary"},
        {"name": "stereo", "kind": "one_hot", "values": BOND_STEREO, "other": True},
    ]
    edge_dim = sum(
        1 if group["kind"] == "binary" else categorical_width(group["values"], group["other"])
        for group in edge_groups
    )
    result = {
        "schema_version": 1,
        "node_groups": node_groups,
        "edge_groups": edge_groups,
        "node_target_dim": target_dim,
        "position_encoding_dim": int(position_dim),
        "node_input_dim": target_dim + int(position_dim),
        "edge_dim": edge_dim,
        "directed_edges": True,
        "directed_pair_order": "each RDKit bond emits begin->end then end->begin",
    }
    result["hash"] = object_hash(result)
    return result


@dataclass(frozen=True)
class FeatureDimensions:
    node_input: int
    node_target: int
    edge: int


def dimensions(schema: dict) -> FeatureDimensions:
    return FeatureDimensions(
        node_input=int(schema["node_input_dim"]),
        node_target=int(schema["node_target_dim"]),
        edge=int(schema["edge_dim"]),
    )


def validate_feature_schema(schema: dict) -> None:
    value = copy.deepcopy(schema)
    claimed = value.pop("hash", None)
    actual = object_hash(value)
    if claimed != actual:
        raise ValueError(f"Feature schema hash mismatch: claimed {claimed}, computed {actual}")

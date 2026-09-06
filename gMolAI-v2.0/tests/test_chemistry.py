import numpy as np
from rdkit import Chem

from gmolai_retrain.chem import CanonicalMolecule, Rejection, canonicalize, featurize_molecule
from gmolai_retrain.schema import feature_schema


SPLIT = {
    "seed": 9,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
}
ALLOWED = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", "B", "Si"}


def canonical(smiles):
    return canonicalize(
        smiles,
        isomeric_smiles=True,
        fragment_policy="reject",
        allowed_elements=ALLOWED,
        min_atoms=2,
        max_atoms=256,
        buckets=16,
        split_cfg=SPLIT,
    )


def test_stereochemistry_is_retained_and_fragments_rejected():
    clockwise = canonical("C[C@H](O)F")
    counterclockwise = canonical("C[C@@H](O)F")
    assert isinstance(clockwise, CanonicalMolecule)
    assert isinstance(counterclockwise, CanonicalMolecule)
    assert clockwise.smiles != counterclockwise.smiles
    assert isinstance(canonical("CC.O"), Rejection)


def test_explicit_feature_dimensions_and_bidirectional_edges():
    mol = Chem.MolFromSmiles("C[C@H](O)F")
    schema = feature_schema(include_chirality=True, position_dim=0)
    x, edge_index, edge_attr = featurize_molecule(mol, include_chirality=True, position_dim=0)
    assert x.shape == (mol.GetNumAtoms(), schema["node_input_dim"])
    assert edge_index.shape == (2, 2 * mol.GetNumBonds())
    assert edge_attr.shape == (2 * mol.GetNumBonds(), schema["edge_dim"])
    assert np.array_equal(edge_index[:, 0], edge_index[::-1, 1])

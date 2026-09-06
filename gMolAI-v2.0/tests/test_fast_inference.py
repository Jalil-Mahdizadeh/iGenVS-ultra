from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from gmolai_retrain.chem import featurize_molecule
from gmolai_retrain.downstream import _encode_molecules
from gmolai_retrain.fast_graph import (
    fast_featurize_molecule,
    pack_molecules,
    smiles_tasks,
)
from gmolai_retrain.fast_inference import (
    OptimizedRawCore,
    OptimizedSmilesEncoder,
    ReferenceSmilesEncoder,
    VerifyingSmilesEncoder,
    compare_embedding_matrices,
    packed_to_device,
)
from gmolai_retrain.model import MolecularRepresentationModel
from gmolai_retrain.schema import feature_schema


CURATED_SMILES = (
    "CCO",
    "c1cc[nH]c1",
    "CC(=O)N",
    "C[N+](C)(C)C",
    "N#CC(F)(F)F",
    "C[C@H](O)Cl",
    "B(O)O",
    "C[Si](C)(C)C",
    "CS(=O)(=O)N",
    "OP(=O)(O)O",
)


def small_representation_model() -> MolecularRepresentationModel:
    torch.manual_seed(17)
    model = MolecularRepresentationModel(
        feature_schema(include_chirality=True, position_dim=0),
        descriptor_count=3,
        model_cfg={
            "hidden_dim": 32,
            "node_latent_dim": 12,
            "graph_latent_dim": 16,
            "gine_layers": 2,
            "dropout": 0.1,
            "vicreg_projector": False,
        },
    )
    return model.eval().requires_grad_(False)


def molecules(values=CURATED_SMILES) -> list[Chem.Mol]:
    result = [Chem.MolFromSmiles(value) for value in values]
    assert all(item is not None for item in result)
    return result


def test_fast_features_are_exact_on_curated_chemistry():
    for molecule in molecules():
        expected = featurize_molecule(
            molecule, include_chirality=True, position_dim=0
        )
        observed = fast_featurize_molecule(molecule)
        for expected_array, observed_array in zip(expected, observed, strict=True):
            assert np.array_equal(expected_array, observed_array)


def test_fast_features_fail_closed_outside_promoted_schema():
    molecule = Chem.MolFromSmiles("CCO")
    with pytest.raises(ValueError, match="promoted feature contract"):
        fast_featurize_molecule(molecule, include_chirality=False)
    with pytest.raises(ValueError, match="promoted feature contract"):
        fast_featurize_molecule(molecule, position_dim=2)


def test_direct_packing_matches_pyg_batch_exactly():
    selected = molecules(CURATED_SMILES[:4])
    packed = pack_molecules(selected, start=11)
    graphs = []
    for molecule in selected:
        x, edge_index, edge_attr = featurize_molecule(
            molecule, include_chirality=True, position_dim=0
        )
        graphs.append(
            Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr),
            )
        )
    reference = Batch.from_data_list(graphs)
    assert packed.start == 11
    assert packed.graph_count == len(selected)
    assert np.array_equal(packed.x, reference.x.numpy())
    assert np.array_equal(packed.edge_index, reference.edge_index.numpy())
    assert np.array_equal(packed.edge_attr, reference.edge_attr.numpy())
    assert np.array_equal(packed.batch, reference.batch.numpy())


def test_smiles_tasks_respect_graph_and_node_budgets():
    values = ["CC", "CCC", "CCCC", "CCCCC"]
    tasks = list(
        smiles_tasks(
            values,
            batch_size=3,
            node_budget=6,
            atom_counts=[2, 3, 4, 5],
        )
    )
    assert [(start, list(batch)) for start, batch in tasks] == [
        (0, ["CC", "CCC"]),
        (2, ["CCCC"]),
        (3, ["CCCCC"]),
    ]


def test_optimized_raw_core_matches_authoritative_model_encode():
    model = small_representation_model()
    selected = molecules(CURATED_SMILES[:5])
    packed = pack_molecules(selected)
    x, edge_index, edge_attr, batch = packed_to_device(
        packed, torch.device("cpu")
    )
    with torch.inference_mode():
        node_z, graph_z = model.encode(x, edge_index, edge_attr, batch)
        expected = torch.cat((graph_z, global_mean_pool(node_z, batch)), dim=1)
        observed = OptimizedRawCore(model).eval()(x, edge_index, edge_attr, batch)
    assert torch.allclose(expected, observed, rtol=1.0e-6, atol=1.0e-6)


def test_optimized_smiles_encoder_preserves_reference_values_and_order():
    model = small_representation_model()
    dimensions = int(model.graph_latent_dim + model.node_latent_dim)
    mean = np.linspace(-0.2, 0.2, dimensions, dtype=np.float32)
    scale = np.linspace(0.8, 1.2, dimensions, dtype=np.float32)
    values = list(CURATED_SMILES[:7])
    counts = [molecule.GetNumAtoms() for molecule in molecules(values)]
    reference = ReferenceSmilesEncoder(
        model,
        mean,
        scale,
        device=torch.device("cpu"),
        batch_size=3,
        node_budget=12,
    )
    optimized = OptimizedSmilesEncoder(
        model,
        mean,
        scale,
        device=torch.device("cpu"),
        batch_size=3,
        node_budget=12,
        workers=1,
    )
    expected = reference.encode(values, atom_counts=counts)
    observed = optimized.encode(values, atom_counts=counts)
    comparison = compare_embedding_matrices(expected, observed)
    assert comparison["minimum_cosine"] > 0.999999
    assert comparison["maximum_relative_l2"] < 1.0e-5

    reversed_values = list(reversed(values))
    reversed_counts = list(reversed(counts))
    reversed_observed = optimized.encode(
        reversed_values, atom_counts=reversed_counts
    )
    assert np.allclose(reversed_observed, observed[::-1], rtol=1.0e-5, atol=1.0e-5)
    optimized.close()


def test_verify_backend_returns_optimized_matrix_after_reference_gate():
    model = small_representation_model()
    dimensions = int(model.graph_latent_dim + model.node_latent_dim)
    mean = np.zeros(dimensions, dtype=np.float32)
    scale = np.ones(dimensions, dtype=np.float32)
    values = list(CURATED_SMILES[:4])
    counts = [molecule.GetNumAtoms() for molecule in molecules(values)]
    optimized = OptimizedSmilesEncoder(
        model,
        mean,
        scale,
        device=torch.device("cpu"),
        batch_size=4,
        workers=1,
    )
    reference = ReferenceSmilesEncoder(
        model,
        mean,
        scale,
        device=torch.device("cpu"),
        batch_size=4,
    )
    verifying = VerifyingSmilesEncoder(
        optimized,
        reference,
        verify_rows=4,
        minimum_cosine=0.999,
        maximum_relative_l2=0.01,
    )
    observed = verifying.encode(values, atom_counts=counts)
    assert observed.shape == (4, dimensions)
    assert verifying.remaining == 0
    assert verifying.last_comparison is not None
    verifying.close()


def test_downstream_consumer_uses_equivalent_optimized_blocks():
    model = small_representation_model()
    selected = molecules(CURATED_SMILES[:5])
    observed, blocks = _encode_molecules(
        model,
        selected,
        {
            "features": {
                "include_atom_chirality": True,
                "canonical_position_encoding_dim": 0,
            }
        },
        torch.device("cpu"),
    )
    packed = pack_molecules(selected)
    x, edge_index, edge_attr, batch = packed_to_device(
        packed, torch.device("cpu")
    )
    with torch.inference_mode():
        node_z, graph_z = model.encode(x, edge_index, edge_attr, batch)
        mean_node_z = global_mean_pool(node_z, batch)
        expected = model.combine_molecule_embedding(
            node_z, graph_z, batch, mean_node_weight=3.0
        ).numpy()
    assert np.allclose(observed, expected, rtol=1.0e-6, atol=1.0e-6)
    assert np.allclose(
        blocks["raw_graph_z"], graph_z.numpy(), rtol=1.0e-6, atol=1.0e-6
    )
    assert np.allclose(
        blocks["raw_mean_node_z"],
        mean_node_z.numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_optimized_core_rejects_nonpromoted_feature_schema():
    model = MolecularRepresentationModel(
        feature_schema(include_chirality=False, position_dim=0),
        descriptor_count=2,
        model_cfg={
            "hidden_dim": 16,
            "node_latent_dim": 8,
            "graph_latent_dim": 8,
            "gine_layers": 1,
            "dropout": 0.0,
            "vicreg_projector": False,
        },
    ).eval()
    with pytest.raises(ValueError, match="promoted optimized contract"):
        OptimizedRawCore(model)

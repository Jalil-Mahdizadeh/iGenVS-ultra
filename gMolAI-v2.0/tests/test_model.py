import pytest
import torch
from torch_geometric.data import Batch, Data

from gmolai_retrain.model import (
    MolecularRepresentationModel,
    MolecularVGAE,
    corrupt_graph_inputs,
    nt_xent_loss,
    vicreg_terms,
)
from gmolai_retrain.schema import feature_schema
from gmolai_retrain.representations import _resolve_embedding_definition
from gmolai_retrain.train import _initialize_model_from_checkpoint, _losses_for_batch
from gmolai_retrain.util import sha256_file


def test_model_forward_and_symmetric_edge_decoder():
    schema = feature_schema(True, 0)
    model = MolecularVGAE(
        schema,
        descriptor_count=13,
        model_cfg={
            "hidden_dim": 32,
            "latent_dim": 16,
            "gine_layers": 2,
            "dropout": 0.0,
            "logvar_min": -10.0,
            "logvar_max": 6.0,
        },
    )
    x = torch.zeros((3, schema["node_input_dim"]))
    x[:, 0] = 1
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.zeros((4, schema["edge_dim"]))
    edge_attr[:, 0] = 1
    batch = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.zeros((1, 13)))])
    existence_index = torch.tensor([[0, 1], [1, 2]])
    z, mu, logvar, node_logits, descriptors, existence_logits, edge_logits = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        existence_index,
        edge_index[:, edge_index[0] < edge_index[1]],
        sample=False,
    )
    assert z.shape == mu.shape == logvar.shape == (3, 16)
    assert node_logits.shape == (3, schema["node_target_dim"])
    assert descriptors.shape == (1, 13)
    assert existence_logits is not None and existence_logits.shape == (2,)
    assert edge_logits is not None and edge_logits.shape == (2, schema["edge_dim"])
    forward = model.edge_existence_decoder(z, torch.tensor([[0], [1]]))
    reverse = model.edge_existence_decoder(z, torch.tensor([[1], [0]]))
    assert torch.allclose(forward, reverse)


def test_descriptor_prediction_is_independent_of_posterior_sampling():
    """The descriptor head must see the same representation in train/eval paths."""
    torch.manual_seed(7)
    schema = feature_schema(True, 0)
    model = MolecularVGAE(
        schema,
        descriptor_count=13,
        model_cfg={
            "hidden_dim": 32,
            "latent_dim": 16,
            "gine_layers": 2,
            "dropout": 0.0,
            "logvar_min": -10.0,
            "logvar_max": 6.0,
        },
    )
    model.eval()
    x = torch.zeros((3, schema["node_input_dim"]))
    x[:, 0] = 1
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.zeros((4, schema["edge_dim"]))
    edge_attr[:, 0] = 1
    batch = torch.zeros(3, dtype=torch.long)

    sampled = model(x, edge_index, edge_attr, batch, sample=True)
    deterministic = model(x, edge_index, edge_attr, batch, sample=False)

    assert not torch.equal(sampled[0], deterministic[0])
    assert torch.equal(sampled[1], deterministic[1])
    assert torch.equal(sampled[4], deterministic[4])


def test_corruption_masks_both_directions_together():
    schema = feature_schema(True, 0)
    x = torch.ones((2, schema["node_input_dim"]))
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.ones((2, schema["edge_dim"]))
    generator = torch.Generator().manual_seed(1)
    result = corrupt_graph_inputs(
        x,
        edge_index,
        edge_attr,
        node_probability=0.0,
        edge_feature_probability=0.0,
        edge_dropout_probability=1.0,
        generator=generator,
    )
    assert result.edge_index.shape[1] == 0
    assert result.edge_target_mask.tolist() == [True]
    assert result.edge_drop_mask.tolist() == [True]


def test_complete_corrected_objective_backpropagates():
    schema = feature_schema(True, 0)
    model_cfg = {
        "hidden_dim": 32,
        "latent_dim": 16,
        "gine_layers": 2,
        "dropout": 0.0,
        "logvar_min": -10.0,
        "logvar_max": 6.0,
    }
    model = MolecularVGAE(schema, descriptor_count=13, model_cfg=model_cfg)
    x = torch.zeros((3, schema["node_input_dim"]))
    x[:, 0] = 1
    # Fill one valid class in every categorical group and zeros for binaries.
    offset = 0
    for group in schema["node_groups"]:
        if group["kind"] == "binary":
            offset += 1
        else:
            x[:, offset] = 1
            offset += len(group["values"]) + int(group["other"])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.zeros((4, schema["edge_dim"]))
    offset = 0
    for group in schema["edge_groups"]:
        if group["kind"] == "binary":
            offset += 1
        else:
            edge_attr[:, offset] = 1
            offset += len(group["values"]) + int(group["other"])
    batch = Batch.from_data_list(
        [
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.zeros((1, 13)),
                graph_id=torch.tensor([123]),
            )
        ]
    )
    cfg = {
        "seed": 42,
        "objective": {
            "node_mask_probability": 1.0,
            "bond_feature_mask_probability": 1.0,
            "bond_dropout_probability": 0.5,
            "easy_negative_ratio": 0.0,
            "hard_negative_ratio": 1.0,
            "hard_pool_ratio": 5.0,
            "node_weight": 1.0,
            "edge_existence_weight": 1.0,
            "edge_feature_weight": 1.0,
            "descriptor_weight": 1.0,
            "kl_beta_max": 1 / 16,
            "kl_warmup_steps": 10,
        },
    }
    losses = _losses_for_batch(model, batch, torch.device("cpu"), cfg, step=1, rank=0, training=True)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def _valid_features(groups, rows: int, class_offset: int = 0) -> torch.Tensor:
    width = sum(
        1 if group["kind"] == "binary" else len(group["values"]) + int(group["other"])
        for group in groups
    )
    value = torch.zeros((rows, width))
    offset = 0
    for group in groups:
        if group["kind"] == "binary":
            offset += 1
        else:
            group_width = len(group["values"]) + int(group["other"])
            value[:, offset + class_offset % group_width] = 1
            offset += group_width
    return value


def _representation_model_and_batch():
    schema = feature_schema(True, 0)
    model_cfg = {
        "architecture": "masked_graph_vicreg",
        "hidden_dim": 32,
        "node_latent_dim": 16,
        "graph_latent_dim": 12,
        "gine_layers": 2,
        "dropout": 0.0,
    }
    model = MolecularRepresentationModel(schema, descriptor_count=13, model_cfg=model_cfg)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    first = Data(
        x=_valid_features(schema["node_groups"], 3, 0),
        edge_index=edge_index,
        edge_attr=_valid_features(schema["edge_groups"], 4, 0),
        y=torch.zeros((1, 13)),
        graph_id=torch.tensor([101]),
    )
    second = Data(
        x=_valid_features(schema["node_groups"], 3, 1),
        edge_index=edge_index,
        edge_attr=_valid_features(schema["edge_groups"], 4, 1),
        y=torch.ones((1, 13)),
        graph_id=torch.tensor([202]),
    )
    return schema, model_cfg, model, Batch.from_data_list([first, second])


def test_representation_model_exports_deterministic_graph_vectors_and_symmetric_edges():
    schema, _, model, batch = _representation_model_and_batch()
    model.eval()
    unique_edges = batch.edge_index[:, batch.edge_index[0] < batch.edge_index[1]]
    outputs = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        unique_edges,
        unique_edges,
        view2_x=batch.x,
        view2_edge_index=batch.edge_index,
        view2_edge_attr=batch.edge_attr,
    )
    repeated_node, repeated_graph = model.encode(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch
    )
    molecule_embedding = model.molecule_embedding(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, mean_node_weight=3.0
    )
    assert outputs["node_z"].shape == (6, 16)
    assert outputs["mean_node_z"].shape == (2, 16)
    assert outputs["mean_node_z2"].shape == (2, 16)
    assert outputs["graph_z"].shape == (2, 12)
    assert outputs["node_logits"].shape == (6, schema["node_target_dim"])
    assert outputs["edge_logits"].shape == (4, schema["edge_dim"])
    assert torch.equal(outputs["node_z"], repeated_node)
    assert torch.equal(outputs["graph_z"], repeated_graph)
    assert molecule_embedding.shape == (2, 28)
    assert torch.allclose(molecule_embedding[:, :12].norm(dim=-1), torch.ones(2))
    assert torch.allclose(
        molecule_embedding[:, 12:].norm(dim=-1), torch.full((2,), 3.0)
    )
    raw_embedding = model.combine_raw_molecule_embedding(
        repeated_node, repeated_graph, batch.batch
    )
    calibrated = model.apply_molecule_calibration(
        raw_embedding, torch.zeros(28), 2.0 * torch.ones(28)
    )
    assert raw_embedding.shape == calibrated.shape == (2, 28)
    assert torch.allclose(calibrated, raw_embedding / 2.0)
    with pytest.raises(ValueError, match="calibration tensors"):
        model.apply_molecule_calibration(
            raw_embedding, torch.zeros(28), torch.zeros(28)
        )
    forward = model.edge_existence_decoder(
        repeated_node, repeated_graph, batch.batch, torch.tensor([[0], [1]])
    )
    reverse = model.edge_existence_decoder(
        repeated_node, repeated_graph, batch.batch, torch.tensor([[1], [0]])
    )
    assert torch.allclose(forward, reverse)


def test_molecule_embedding_is_invariant_to_atom_index_permutation():
    _, _, model, batch = _representation_model_and_batch()
    model.eval()
    x = batch.x[:3]
    edge_index = batch.edge_index[:, :4]
    edge_attr = batch.edge_attr[:4]
    graph_batch = torch.zeros(3, dtype=torch.long)
    expected = model.molecule_embedding(x, edge_index, edge_attr, graph_batch)

    permutation = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    observed = model.molecule_embedding(
        x[permutation], inverse[edge_index], edge_attr, graph_batch
    )
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_molecule_embedding_does_not_depend_on_batch_companions():
    _, _, model, batch = _representation_model_and_batch()
    model.eval()
    batched = model.molecule_embedding(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch
    )[0]
    isolated = model.molecule_embedding(
        batch.x[:3],
        batch.edge_index[:, :4],
        batch.edge_attr[:4],
        torch.zeros(3, dtype=torch.long),
    )[0]
    assert torch.allclose(isolated, batched, atol=1.0e-6, rtol=1.0e-6)


def test_vicreg_and_complete_representation_objective_prevent_silent_unused_parameters():
    _, model_cfg, model, batch = _representation_model_and_batch()
    collapsed = torch.zeros((8, 12))
    invariance, variance, covariance = vicreg_terms(collapsed, collapsed)
    assert invariance == 0
    assert variance > 0.9
    assert covariance == 0
    cfg = {
        "seed": 42,
        "model": model_cfg,
        "objective": {
            "node_mask_probability": 1.0,
            "bond_feature_mask_probability": 1.0,
            "bond_dropout_probability": 0.5,
            "easy_negative_ratio": 1.0,
            "hard_negative_ratio": 1.0,
            "hard_pool_ratio": 5.0,
            "node_weight": 1.0,
            "edge_existence_weight": 0.5,
            "edge_feature_weight": 1.0,
            "descriptor_weight": 0.25,
            "invariance_weight": 1.0,
            "variance_weight": 1.0,
            "covariance_weight": 0.04,
            "variance_target": 1.0,
        },
    }
    losses = _losses_for_batch(
        model, batch, torch.device("cpu"), cfg, step=1, rank=0, training=True
    )
    assert torch.isfinite(losses["total"])
    assert losses["_graph_z_clean"].shape == (2, 12)
    losses["total"].backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not missing


def test_nt_xent_prefers_correct_cross_view_pairs():
    first = torch.eye(8)
    matching = first.clone()
    mismatched = matching.roll(1, dims=0)
    assert nt_xent_loss(first, matching, temperature=0.1) < nt_xent_loss(
        first, mismatched, temperature=0.1
    )


def test_contrastive_objective_can_target_canonical_mean_node_block():
    _, model_cfg, model, batch = _representation_model_and_batch()
    cfg = {
        "seed": 42,
        "model": model_cfg,
        "objective": {
            "node_mask_probability": 1.0,
            "bond_feature_mask_probability": 1.0,
            "bond_dropout_probability": 0.5,
            "easy_negative_ratio": 1.0,
            "hard_negative_ratio": 1.0,
            "hard_pool_ratio": 5.0,
            "node_weight": 1.0,
            "edge_existence_weight": 0.5,
            "edge_feature_weight": 1.0,
            "descriptor_weight": 0.5,
            "invariance_weight": 0.0,
            "variance_weight": 0.0,
            "covariance_weight": 0.0,
            "contrastive_space": "mean_node_z",
            "contrastive_weight": 0.01,
            "contrastive_temperature": 0.1,
            "variance_target": 1.0,
        },
    }
    losses = _losses_for_batch(
        model, batch, torch.device("cpu"), cfg, step=1, rank=0, training=True
    )
    assert losses["_contrastive_z"].shape == (2, 16)
    assert losses["_contrastive_z_clean"].shape == (2, 16)
    assert torch.isfinite(losses["contrastive"])
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_embedding_auto_resolves_to_canonical_hybrid_without_changing_legacy():
    assert _resolve_embedding_definition("auto", representation_model=True) == "hybrid"
    assert _resolve_embedding_definition("graph_z", representation_model=True) == "graph_z"
    assert (
        _resolve_embedding_definition("projector_z", representation_model=True)
        == "projector_z"
    )
    assert (
        _resolve_embedding_definition("raw_hybrid", representation_model=True)
        == "raw_hybrid"
    )
    assert (
        _resolve_embedding_definition(
            "standardized_raw_hybrid", representation_model=True
        )
        == "standardized_raw_hybrid"
    )
    assert _resolve_embedding_definition("auto", representation_model=False) == "graph_z"
    with pytest.raises(ValueError, match="representation model"):
        _resolve_embedding_definition("hybrid", representation_model=False)
    with pytest.raises(ValueError, match="representation model"):
        _resolve_embedding_definition("projector_z", representation_model=False)
    with pytest.raises(ValueError, match="representation model"):
        _resolve_embedding_definition("raw_hybrid", representation_model=False)
    with pytest.raises(ValueError, match="representation model"):
        _resolve_embedding_definition(
            "standardized_raw_hybrid", representation_model=False
        )


def test_projector_regularization_keeps_exported_graph_vector_separate_and_backpropagates():
    schema, model_cfg, _, batch = _representation_model_and_batch()
    model_cfg = {
        **model_cfg,
        "vicreg_projector": True,
        "vicreg_projector_dim": 10,
    }
    model = MolecularRepresentationModel(schema, descriptor_count=13, model_cfg=model_cfg)
    cfg = {
        "seed": 42,
        "model": model_cfg,
        "objective": {
            "node_mask_probability": 1.0,
            "bond_feature_mask_probability": 1.0,
            "bond_dropout_probability": 0.5,
            "easy_negative_ratio": 1.0,
            "hard_negative_ratio": 1.0,
            "hard_pool_ratio": 5.0,
            "node_weight": 1.0,
            "edge_existence_weight": 0.5,
            "edge_feature_weight": 1.0,
            "descriptor_weight": 0.25,
            "invariance_weight": 1.0,
            "variance_weight": 1.0,
            "covariance_weight": 0.04,
            "variance_target": 1.0,
            "invariance_space": "projector",
        },
    }
    losses = _losses_for_batch(
        model, batch, torch.device("cpu"), cfg, step=1, rank=0, training=True
    )
    assert losses["_graph_z_clean"].shape == (2, 12)
    assert losses["_regularization_z_clean"].shape == (2, 10)
    assert losses["_metric_counts"]["invariance"] == 20
    expected_invariance = torch.nn.functional.mse_loss(
        losses["_regularization_z"], losses["_regularization_z_clean"]
    )
    assert torch.allclose(losses["invariance"], expected_invariance)
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.vicreg_projector.parameters())


def test_warm_start_loads_shared_weights_and_only_initializes_new_projector(tmp_path):
    schema, model_cfg, source_model, _ = _representation_model_and_batch()
    target_model = MolecularRepresentationModel(
        schema,
        descriptor_count=13,
        model_cfg={
            **model_cfg,
            "vicreg_projector": True,
            "vicreg_projector_dim": 10,
        },
    )
    projector_before = {
        name: value.detach().clone()
        for name, value in target_model.state_dict().items()
        if name.startswith("vicreg_projector.")
    }
    identity = {
        "config_hash": "config",
        "graph_manifest_hash": "graphs",
        "descriptor_schema_hash": "descriptors",
        "feature_schema_hash": "features",
        "scaler_hash": "scaler",
        "training_implementation_version": "5",
        "training_plan_hash": "new-plan",
    }
    checkpoint = {
        "checkpoint_version": 1,
        **{key: value for key, value in identity.items() if key != "training_plan_hash"},
        "training_plan_hash": "source-plan",
        "global_step": 15000,
        "model": source_model.state_dict(),
    }
    source_path = tmp_path / "source.pt"
    torch.save(checkpoint, source_path)
    report = _initialize_model_from_checkpoint(
        target_model, source_path, sha256_file(source_path), identity
    )
    for name, value in source_model.state_dict().items():
        assert torch.equal(target_model.state_dict()[name], value)
    for name, value in projector_before.items():
        assert torch.equal(target_model.state_dict()[name], value)
    assert report["source_global_step"] == 15000
    assert report["source_training_plan_hash"] == "source-plan"

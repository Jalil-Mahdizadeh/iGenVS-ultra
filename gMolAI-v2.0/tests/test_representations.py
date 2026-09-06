import json

import pytest
import torch

from gmolai_retrain import representations
from gmolai_retrain.cli import build_parser
from gmolai_retrain.config import object_hash


def test_embed_cli_accepts_independent_sampling_seed():
    args = build_parser().parse_args(
        ["embed", "--sampling-seed", "20260810", "--calibrator", "stats.pt"]
    )
    assert args.sampling_seed == 20260810
    assert args.calibrator == "stats.pt"


def test_downstream_cli_accepts_calibrated_embedding_definition():
    args = build_parser().parse_args(
        [
            "benchmark-downstream",
            "--run-dir",
            "run",
            "--datasets-dir",
            "datasets",
            "--output",
            "report.json",
            "--embedding-definition",
            "standardized_raw_hybrid",
            "--calibrator",
            "stats.pt",
        ]
    )
    assert args.embedding_definition == "standardized_raw_hybrid"
    assert args.calibrator == "stats.pt"


def test_train_only_embedding_calibrator_is_atomic_and_provenance_checked(tmp_path):
    source = tmp_path / "raw.pt"
    destination = tmp_path / "calibrator.pt"
    identity = {
        "checkpoint": "checkpoints/step-000005000.pt",
        "checkpoint_sha256": "checkpoint-sha",
        "global_step": 5000,
        "config_hash": "config-hash",
        "training_plan_hash": "plan-hash",
        "graph_manifest_hash": "manifest-hash",
        "descriptor_schema_hash": "descriptor-hash",
    }
    embeddings = torch.arange(40_000, dtype=torch.float32).reshape(10_000, 4)
    torch.save(
        {
            "metadata": {
                **identity,
                "embedding_definition": (
                    "clean_graph_z_plus_mean_node_z_raw_blocks"
                ),
                "split": "train",
                "sampling": (
                    "deterministic_hash_bucket_stratified_without_replacement"
                ),
                "sampling_seed": 42,
                "sampled_source_buckets": 256,
            },
            "embeddings": embeddings,
        },
        source,
    )
    result = representations.fit_embedding_calibrator(
        source, destination
    )
    saved = torch.load(destination, map_location="cpu", weights_only=False)
    assert torch.allclose(saved["coordinate_mean"], embeddings.mean(dim=0))
    assert torch.allclose(
        saved["coordinate_scale"], embeddings.std(dim=0, unbiased=False)
    )
    assert result["checkpoint_sha256"] == identity["checkpoint_sha256"]
    mean, scale, metadata, digest = representations._load_embedding_calibrator(
        destination,
        expected={key: identity[key] for key in identity if key != "checkpoint"},
        dimensions=4,
    )
    assert mean.shape == scale.shape == (4,)
    assert metadata["graphs"] == 10_000
    assert digest == result["sha256"]
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        representations._load_embedding_calibrator(
            destination,
            expected={"checkpoint_sha256": "different"},
            dimensions=4,
        )


def test_embedding_calibrator_rejects_validation_split(tmp_path):
    source = tmp_path / "raw.pt"
    torch.save(
        {
            "metadata": {
                "embedding_definition": (
                    "clean_graph_z_plus_mean_node_z_raw_blocks"
                ),
                "split": "validation",
                "sampling": (
                    "deterministic_hash_bucket_stratified_without_replacement"
                ),
                "sampled_source_buckets": 256,
            },
            "embeddings": torch.randn(12, 4),
        },
        source,
    )
    with pytest.raises(ValueError, match="pretraining train split"):
        representations.fit_embedding_calibrator(
            source, tmp_path / "calibrator.pt", minimum_graphs=10
        )


def test_hybrid_reweighting_is_atomic_and_preserves_identity(tmp_path):
    source = tmp_path / "source.pt"
    destination = tmp_path / "destination.pt"
    embeddings = torch.cat((torch.ones((2, 3)), 2.0 * torch.ones((2, 2))), dim=1)
    payload = {
        "metadata": {
            "embedding_definition": "clean_graph_z_plus_mean_node_z_unit_blocks",
            "embedding_parameters": {
                "graph_dimensions": 3,
                "mean_node_dimensions": 2,
                "mean_node_weight": 2.0,
            },
            "checkpoint_sha256": "immutable-checkpoint",
        },
        "embeddings": embeddings,
        "molecule_hashes": ["a", "b"],
    }
    torch.save(payload, source)
    result = representations.reweight_hybrid_embeddings(
        source, destination, mean_node_weight=3.0
    )
    saved = torch.load(destination, map_location="cpu", weights_only=False)
    assert torch.equal(saved["embeddings"][:, :3], embeddings[:, :3])
    assert torch.equal(saved["embeddings"][:, 3:], 1.5 * embeddings[:, 3:])
    assert saved["metadata"]["checkpoint_sha256"] == "immutable-checkpoint"
    assert result["embedding_parameters"]["mean_node_weight"] == 3.0
    assert torch.equal(torch.load(source, weights_only=False)["embeddings"], embeddings)


def test_reweight_train_standardized_hybrid_embeddings(tmp_path):
    source = tmp_path / "source.pt"
    destination = tmp_path / "destination.pt"
    embeddings = torch.cat((torch.ones((2, 3)), 2.0 * torch.ones((2, 2))), dim=1)
    payload = {
        "metadata": {
            "embedding_definition": (
                "clean_graph_z_plus_mean_node_z_train_standardized_raw_blocks"
            ),
            "embedding_parameters": {
                "graph_dimensions": 3,
                "mean_node_dimensions": 2,
                "mean_node_weight": 1.0,
                "coordinate_transform": "train_mean_and_population_std",
            },
        },
        "embeddings": embeddings,
    }
    torch.save(payload, source)

    representations.reweight_hybrid_embeddings(
        source, destination, mean_node_weight=4.0
    )

    saved = torch.load(destination, map_location="cpu", weights_only=False)
    assert torch.equal(saved["embeddings"][:, :3], embeddings[:, :3])
    assert torch.equal(saved["embeddings"][:, 3:], 4.0 * embeddings[:, 3:])
    assert saved["metadata"]["embedding_parameters"]["mean_node_weight"] == 4.0


def test_promotion_requires_identical_checkpoint_and_calibrator(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    checkpoint_name = "checkpoints/step-000005000.pt"
    checkpoint_path = run_dir / checkpoint_name
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"retained checkpoint bytes")

    cfg = {
        "paths": {"run_dir": str(run_dir)},
        "_config_hash": "config-hash",
        "_descriptor_schema_hash": "descriptor-hash",
        "model": {"graph_latent_dim": 256, "node_latent_dim": 128},
        "objective": {"contrastive_weight": 0.02},
        "training": {"max_steps": 20_000},
    }
    plan_hash = object_hash(
        {
            "model": cfg["model"],
            "objective": cfg["objective"],
            "training": cfg["training"],
        }
    )
    checkpoint_hash = representations._file_sha256(checkpoint_path)
    calibrator_path = run_dir / "calibrator.pt"
    calibration_metadata = {
        "schema_version": 1,
        "calibration_definition": "coordinate_mean_and_population_std",
        "source_embedding_definition": (
            "clean_graph_z_plus_mean_node_z_raw_blocks"
        ),
        "split": "train",
        "graphs": 100_000,
        "dimensions": 384,
        "sampling": "deterministic_hash_bucket_stratified_without_replacement",
        "sampling_seed": 42,
        "sampled_source_buckets": 256,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checkpoint_hash,
        "global_step": 5000,
        "config_hash": cfg["_config_hash"],
        "training_plan_hash": plan_hash,
        "graph_manifest_hash": "manifest-hash",
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
    }
    torch.save(
        {
            "metadata": calibration_metadata,
            "coordinate_mean": torch.zeros(384),
            "coordinate_scale": torch.ones(384),
        },
        calibrator_path,
    )
    calibrator_hash = representations._file_sha256(calibrator_path)
    shared = {
        "global_step": 5000,
        "training_plan_hash": plan_hash,
        "config_hash": cfg["_config_hash"],
        "graph_manifest_hash": "manifest-hash",
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "checkpoint_sha256": checkpoint_hash,
        "embedding_definition": (
            "clean_graph_z_plus_mean_node_z_train_standardized_raw_blocks"
        ),
        "embedding_parameters": {
            "graph_dimensions": 256,
            "mean_node_dimensions": 128,
            "mean_node_weight": 3.0,
            "coordinate_transform": "train_mean_and_population_std",
            "calibrator_sha256": calibrator_hash,
            "calibration_graphs": 100_000,
            "calibration_sampling_seed": 42,
        },
    }
    probe = {
        "train_embedding_metadata": {
            **shared,
            "checkpoint": checkpoint_name,
            "dimensions": 384,
            "graphs": 10000,
            "sampling": "deterministic_hash_bucket_stratified_without_replacement",
            "sampling_seed": 42,
            "sampled_source_buckets": 256,
        },
        "checkpoint_metadata": {
            **shared,
            "checkpoint": checkpoint_name,
            "dimensions": 384,
            "graphs": 50000,
            "sampling": "deterministic_hash_bucket_stratified_without_replacement",
            "sampling_seed": 42,
            "sampled_source_buckets": 256,
        },
        "embedding_diagnostics": {"effective_rank": 31.0},
        "held_out_linear_probe": {
            "mean_r2": 0.96,
            "median_r2": 0.97,
            "mean_standardized_mae": 0.10,
            "train_graphs": 10000,
            "validation_graphs": 50000,
        },
        "scaffold_disjoint_linear_probe": {"mean_r2": 0.96},
        "similarity": {
            "graphs": 5000,
            "available_graphs": 50000,
            "sampling": "seeded_without_replacement_across_export",
            "latent_to_morgan_recall_at_10": 0.19,
            "latent_cosine_vs_morgan_spearman": 0.62,
            "latent_neighbor_mean_tanimoto": 0.21,
            "neighbor_tanimoto_enrichment": 1.8,
            "scaffold_neighbor_purity_at_10": 0.21,
            "scaffold_purity_enrichment": 30.0,
        },
        "clustering": {
            "available": True,
            "graphs": 10000,
            "sampled_graphs": 50000,
            "kmeans_repetitions": 5,
            "sampling": "seeded_without_replacement_across_export",
            "latent_spherical_kmeans": {
                "adjusted_rand_index": 0.40,
                "normalized_mutual_information": 0.78,
            },
            "morgan_spherical_kmeans": {
                "adjusted_rand_index": 0.35,
                "normalized_mutual_information": 0.77,
            },
        },
    }
    downstream_scores = {
        "bace": ("roc_auc", 0.84),
        "bbbp": ("roc_auc", 0.88),
        "esol": ("rmse", 0.68),
        "freesolv": ("rmse", 1.17),
        "lipophilicity": ("rmse", 0.80),
    }
    diagnostic_features = {
        "molecule_embedding",
        "morgan_radius2_2048",
        "unit_graph_z",
        "unit_mean_node_z",
        "graph_z",
        "mean_node_z",
        "raw_graph_z_plus_mean_node_z",
    }
    downstream = {
        "checkpoint": {
            **shared,
            "name": checkpoint_name,
            "embedding_dimensions": 384,
        },
        "datasets": {
            name: {
                "scaffold_splits": 10,
                "preparation": {"molecules": 5000},
                "feature_results": {
                    feature_name: {
                        "summary": {metric: {"mean": value}}
                    }
                    for feature_name in diagnostic_features
                }
            }
            for name, (metric, value) in downstream_scores.items()
        },
    }
    probe_path = run_dir / "probe.json"
    downstream_path = run_dir / "downstream.json"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")

    def fake_load_saved_model(requested_cfg, requested_name, device):
        assert requested_cfg is cfg
        assert requested_name == checkpoint_name
        return cfg, {"graph_manifest_hash": "manifest-hash"}, None, None, {
            "global_step": 5000
        }

    monkeypatch.setattr(representations, "load_saved_model", fake_load_saved_model)

    downstream["checkpoint"]["embedding_parameters"]["calibrator_sha256"] = "different"
    downstream_path.write_text(json.dumps(downstream), encoding="utf-8")
    with pytest.raises(ValueError, match="different calibrator"):
        representations.promote_representation_checkpoint(
            cfg,
            checkpoint_name=checkpoint_name,
            calibrator=calibrator_path,
            representation_probe=probe_path,
            downstream_benchmark=downstream_path,
        )

    downstream["checkpoint"]["embedding_parameters"][
        "calibrator_sha256"
    ] = calibrator_hash
    downstream["selected_only"] = True
    downstream_path.write_text(json.dumps(downstream), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic-baseline panel"):
        representations.promote_representation_checkpoint(
            cfg,
            checkpoint_name=checkpoint_name,
            calibrator=calibrator_path,
            representation_probe=probe_path,
            downstream_benchmark=downstream_path,
        )

    downstream["selected_only"] = False
    downstream["datasets"]["bace"]["feature_results"].pop("morgan_radius2_2048")
    downstream_path.write_text(json.dumps(downstream), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks diagnostic feature results"):
        representations.promote_representation_checkpoint(
            cfg,
            checkpoint_name=checkpoint_name,
            calibrator=calibrator_path,
            representation_probe=probe_path,
            downstream_benchmark=downstream_path,
        )

    downstream["datasets"]["bace"]["feature_results"]["morgan_radius2_2048"] = {
        "summary": {"roc_auc": {"mean": 0.84}}
    }
    downstream_path.write_text(json.dumps(downstream), encoding="utf-8")
    selection = representations.promote_representation_checkpoint(
        cfg,
        checkpoint_name=checkpoint_name,
        calibrator=calibrator_path,
        representation_probe=probe_path,
        downstream_benchmark=downstream_path,
    )
    promoted = run_dir / "representation-best.pt"
    assert promoted.read_bytes() == checkpoint_path.read_bytes()
    assert (run_dir / "representation-calibrator.pt").read_bytes() == calibrator_path.read_bytes()
    assert selection["checkpoint_sha256"] == checkpoint_hash
    assert selection["embedding_parameters"]["calibrator_sha256"] == calibrator_hash
    assert selection["promotion_gates"]["downstream"]["freesolv"]["observed"] == 1.17


def test_automatic_representation_export_requires_promotion(tmp_path):
    cfg = {
        "paths": {"run_dir": str(tmp_path)},
        "model": {"architecture": "masked_graph_vicreg"},
    }
    with pytest.raises(FileNotFoundError, match="validate and promote"):
        representations._automatic_checkpoint_name(cfg)
    (tmp_path / "representation-best.pt").write_bytes(b"promoted")
    assert representations._automatic_checkpoint_name(cfg) == "representation-best.pt"
    calibrator = tmp_path / "representation-calibrator.pt"
    calibrator.write_bytes(b"calibrator")
    (tmp_path / "representation_selection.json").write_text(
        json.dumps(
            {
                "embedding_definition": (
                    "clean_graph_z_plus_mean_node_z_train_standardized_raw_blocks"
                ),
                "calibrator": {
                    "promoted": calibrator.name,
                    "sha256": representations._file_sha256(calibrator),
                },
            }
        ),
        encoding="utf-8",
    )
    assert representations._automatic_representation_calibrator(cfg) == calibrator
    calibrator.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        representations._automatic_representation_calibrator(cfg)

    cfg["model"]["architecture"] = "vgae"
    assert representations._automatic_checkpoint_name(cfg) == "best.pt"


def test_promotion_rejects_mutable_checkpoint_aliases(tmp_path):
    cfg = {"paths": {"run_dir": str(tmp_path)}}
    with pytest.raises(ValueError, match="immutable"):
        representations.promote_representation_checkpoint(
            cfg,
            checkpoint_name="last.pt",
            calibrator=tmp_path / "calibrator.pt",
            representation_probe=tmp_path / "probe.json",
            downstream_benchmark=tmp_path / "downstream.json",
        )

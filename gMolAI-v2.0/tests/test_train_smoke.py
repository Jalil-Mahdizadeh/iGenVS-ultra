import copy
import json
from pathlib import Path

import torch
from rdkit import Chem

from gmolai_retrain.chem import featurize_molecule
from gmolai_retrain.graph_shards import _ShardBuffer
from gmolai_retrain.schema import feature_schema
import gmolai_retrain.train as training_module


def _write_shard(path: Path, schema: dict, split: str, start_id: int):
    buffer = _ShardBuffer()
    for offset, smiles in enumerate(("CCO", "CCN", "CCF")):
        x, edge_index, edge_attr = featurize_molecule(Chem.MolFromSmiles(smiles))
        molecule_hash = f"{start_id + offset:064x}"
        buffer.append(
            torch.from_numpy(x).to(torch.uint8),
            torch.from_numpy(edge_index).to(torch.int32),
            torch.from_numpy(edge_attr).to(torch.uint8),
            torch.full((13,), float(offset), dtype=torch.float32),
            molecule_hash,
        )
    payload = buffer.pack(
        {
            "schema_version": 1,
            "config_hash": "config",
            "descriptor_schema_hash": "descriptors",
            "feature_schema_hash": schema["hash"],
            "split": split,
        }
    )
    torch.save(payload, path)
    return {
        "path": str(path),
        "split": split,
        "bucket": 0,
        "sequence": 0,
        "graphs": 3,
        "nodes": int(payload["node_ptr"][-1]),
        "directed_edges": int(payload["edge_ptr"][-1]),
        "size_bytes": path.stat().st_size,
    }


def test_cpu_training_writes_full_resumable_checkpoint(tmp_path, monkeypatch):
    # This regression is intentionally bitwise and therefore must really use
    # CPU even when pytest runs inside a GPU-enabled validation allocation.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    schema = feature_schema(True, 0)
    work = tmp_path / "work"
    run = tmp_path / "run"
    work.mkdir()
    shards = [
        _write_shard(work / "train.pt", schema, "train", 100),
        _write_shard(work / "validation.pt", schema, "validation", 200),
        _write_shard(work / "test.pt", schema, "test", 300),
    ]
    graph_manifest = {
        "schema_version": 1,
        "config_hash": "config",
        "graph_manifest_hash": "graphs",
        "descriptor_schema_hash": "descriptors",
        "scaler_hash": "scaler",
        "feature_schema": schema,
        "shards": shards,
    }
    (work / "graph_manifest.json").write_text(json.dumps(graph_manifest), encoding="utf-8")
    scaler = {
        "schema_version": 1,
        "descriptor_schema_hash": "descriptors",
        "scaler_hash": "scaler",
        "mean": [0.0] * 13,
        "scale": [1.0] * 13,
    }
    (work / "descriptor_scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    names = [f"d{index}" for index in range(13)]
    cfg = {
        "schema_version": 1,
        "seed": 42,
        "_config_hash": "config",
        "_descriptor_schema_hash": "descriptors",
        "_config_path": str(tmp_path / "config.yaml"),
        "_descriptors": {"features": [{"name": name} for name in names]},
        "paths": {"work_dir": str(work), "run_dir": str(run)},
        "data": {"descriptor_columns": [str(index) for index in range(13)]},
        "model": {
            "hidden_dim": 16,
            "latent_dim": 8,
            "gine_layers": 2,
            "dropout": 0.0,
            "logvar_min": -10.0,
            "logvar_max": 6.0,
        },
        "objective": {
            "node_mask_probability": 1.0,
            "bond_feature_mask_probability": 1.0,
            "bond_dropout_probability": 0.25,
            "easy_negative_ratio": 1.0,
            "hard_negative_ratio": 1.0,
            "hard_pool_ratio": 5.0,
            "node_weight": 1.0,
            "edge_existence_weight": 1.0,
            "edge_feature_weight": 1.0,
            "descriptor_weight": 1.0,
            "kl_beta_max": 0.125,
            "kl_warmup_steps": 2,
        },
        "training": {
            "max_steps": 2,
            "node_budget_per_gpu": 100,
            "max_graphs_per_gpu": 3,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-5,
            "warmup_steps": 1,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "precision": "fp32",
            "checkpoint_every_steps": 1,
            "retain_every_steps": 1,
            "validate_every_steps": 1,
            "log_every_steps": 1,
            "validation_max_graphs": 3,
            "test_max_graphs": 3,
            "resume": "auto",
        },
    }
    training_module._SIGNAL_REQUESTED = True
    assert training_module.train(cfg, allow_cpu=True) == 99
    interrupted = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert interrupted["global_step"] == 1
    assert not (run / "COMPLETE").exists()
    training_module._SIGNAL_REQUESTED = False
    assert training_module.train(cfg, allow_cpu=True) == 0
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 2
    assert checkpoint["config_hash"] == "config"
    assert checkpoint["world_size"] == 1
    assert (run / "checkpoints" / "step-000000001.pt").is_file()
    assert (run / "checkpoints" / "step-000000002.pt").is_file()
    assert len(checkpoint["data_states"]) == 1
    assert "optimizer" in checkpoint and "scheduler" in checkpoint and "rng_states" in checkpoint
    assert (run / "best.pt").is_file()
    assert (run / "COMPLETE").is_file()

    reference_cfg = copy.deepcopy(cfg)
    reference_run = tmp_path / "reference-run"
    reference_cfg["paths"]["run_dir"] = str(reference_run)
    training_module._SIGNAL_REQUESTED = False
    assert training_module.train(reference_cfg, allow_cpu=True) == 0
    reference = torch.load(reference_run / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == reference["global_step"]
    assert checkpoint["data_states"] == reference["data_states"]
    assert checkpoint["scheduler"] == reference["scheduler"]
    assert checkpoint["model"].keys() == reference["model"].keys()
    for name in checkpoint["model"]:
        assert torch.equal(checkpoint["model"][name], reference["model"][name]), name

    # Evaluation must recover runtime batch-budget overrides from the saved
    # resolved plan rather than rejecting an otherwise matching checkpoint.
    evaluation_cfg = copy.deepcopy(cfg)
    evaluation_cfg["training"]["node_budget_per_gpu"] = 1
    evaluation_cfg["training"]["max_graphs_per_gpu"] = 1
    result = training_module.evaluate_saved(
        evaluation_cfg,
        checkpoint_name="best.pt",
        split="test",
        allow_cpu=True,
    )
    assert result["graphs"] == 3
    assert result["metric_denominators"]["node"] == 9
    assert result["metric_denominators"]["edge_feature"] == 6
    assert result["metric_denominators"]["descriptor"] == 39
    assert result["metric_denominators"]["kl"] == 9
    assert "model_hard" in result["edge_existence_metrics"]
    assert result["checkpoint"]["name"] == "best.pt"
    assert result["checkpoint"]["sha256"] == training_module.sha256_file(
        Path(cfg["paths"]["run_dir"]) / "best.pt"
    )
    assert result["checkpoint"]["global_step"] == result["step"]
    assert result["evaluation"] == {
        "split": "test",
        "maximum_graphs": cfg["training"]["test_max_graphs"],
        "world_size": 1,
    }

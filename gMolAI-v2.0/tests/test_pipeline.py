import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import torch
import yaml

from gmolai_retrain.config import load_config
from gmolai_retrain.deduplicate import deduplicate_bucket, finalize_dataset, fit_train_scaler
from gmolai_retrain.graph_shards import featurize_bucket, finalize_graphs
from gmolai_retrain.preprocess import canonicalize_task, prepare_tasks, verify_sources


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_source(path, smiles, offset):
    values = {"0": smiles}
    for column in range(1, 14):
        values[str(column)] = [float(offset + row + column / 100) for row in range(len(smiles))]
    feather.write_feather(pa.table(values), path, compression="uncompressed", chunksize=2)


def test_end_to_end_data_contract_dedup_scaler_and_graph_shards(tmp_path):
    inputs = tmp_path / "inputs"
    configs = tmp_path / "configs"
    inputs.mkdir()
    configs.mkdir()
    zinc = inputs / "ZINC.feather"
    pubchem = inputs / "PubChem.feather"
    _write_source(zinc, ["CCO", "CCN", "CC.O", "C[C@H](O)F"], 0)
    _write_source(pubchem, ["OCC", "CCCl", "C[C@@H](O)F"], 0)
    # Make OCC agree with CCO while retaining a deliberate second CCO conflict.
    table = feather.read_table(pubchem).to_pydict()
    for column in range(1, 14):
        table[str(column)][0] = float(column / 100)
    table["0"].append("CCO")
    for column in range(1, 14):
        table[str(column)].append(float(999 + column))
    feather.write_feather(pa.table(table), pubchem, compression="uncompressed", chunksize=2)

    descriptors = {
        "schema_version": 1,
        "confirmed_identical_across_sources": True,
        "confirmation_note": "synthetic unit test",
        "features": [
            {"column": str(index), "name": f"descriptor_{index}", "unit": "test", "generator": "test"}
            for index in range(1, 14)
        ],
    }
    (configs / "descriptors.yaml").write_text(yaml.safe_dump(descriptors), encoding="utf-8")
    config = {
        "schema_version": 1,
        "experiment_name": "test",
        "seed": 42,
        "paths": {
            "project_root": "..",
            "work_dir": "./work",
            "run_dir": "./runs/test",
            "descriptor_manifest": "./configs/descriptors.yaml",
            "sources": [
                {"name": "zinc", "path": "./inputs/ZINC.feather", "priority": 0, "sha256": _sha256(zinc), "rows": 4},
                {"name": "pubchem", "path": "./inputs/PubChem.feather", "priority": 1, "sha256": _sha256(pubchem), "rows": 4},
            ],
        },
        "data": {
            "smiles_column": "0",
            "descriptor_columns": [str(index) for index in range(1, 14)],
            "record_batches_per_task": 1,
            "hash_buckets": 4,
            "canonicalization": {
                "isomeric_smiles": True,
                "fragment_policy": "reject",
                "allowed_elements": ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", "B", "Si"],
                "min_atoms": 2,
                "max_atoms": 256,
            },
            "deduplication": {
                "key": "canonical_isomeric_smiles",
                "descriptor_conflict_policy": "exclude",
                "descriptor_atol": 1e-8,
                "descriptor_rtol": 1e-5,
                "max_conflict_fraction": 1.0,
            },
            "split": {
                "method": "bemis_murcko_scaffold_hash",
                "train_fraction": 1.0,
                "validation_fraction": 0.0,
                "test_fraction": 0.0,
                "seed": 7,
            },
            "graph_shards": {"graphs_per_shard": 2},
        },
        "features": {"include_atom_chirality": True, "canonical_position_encoding_dim": 0},
        "model": {"hidden_dim": 32, "latent_dim": 16, "gine_layers": 2, "dropout": 0.0, "logvar_min": -10.0, "logvar_max": 6.0},
        "objective": {},
        "training": {},
    }
    config_path = configs / "retrain.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cfg = load_config(config_path)
    prepared = prepare_tasks(cfg)
    verify_sources(cfg)
    for task in range(prepared["canonicalize_task_count"]):
        canonicalize_task(cfg, task)
    for bucket in range(4):
        deduplicate_bucket(cfg, bucket, threads=1)
    manifest = finalize_dataset(cfg)
    scaler = fit_train_scaler(cfg, batch_size=2)
    assert manifest["canonicalization_counts"]["reject_disconnected"] == 1
    assert manifest["deduplication"]["descriptor_conflict_groups_excluded"] == 1
    assert manifest["split_counts"]["train"] == 4
    assert scaler["count"] == 4
    for bucket in range(4):
        featurize_bucket(cfg, bucket)
    graph_manifest = finalize_graphs(cfg)
    assert graph_manifest["counts"]["graphs_train"] == 4
    shard = torch.load(graph_manifest["shards"][0]["path"], map_location="cpu", weights_only=True)
    assert shard["metadata"]["feature_schema_hash"] == graph_manifest["feature_schema"]["hash"]

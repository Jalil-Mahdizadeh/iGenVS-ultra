import numpy as np

import pytest

from gmolai_retrain import downstream
from gmolai_retrain.downstream import (
    _resolve_dataset_names,
    _scaffold_splits,
    _select_representation_embedding,
)
from gmolai_retrain.cli import build_parser


def test_repeated_scaffold_splits_are_disjoint_deterministic_and_stratified():
    groups = np.asarray([f"group-{index // 2}" for index in range(80)], dtype=object)
    targets = np.asarray([(index // 2) % 2 for index in range(80)], dtype=np.float64)
    first = _scaffold_splits(
        groups, targets, task="classification", count=3, seed=42
    )
    repeated = _scaffold_splits(
        groups, targets, task="classification", count=3, seed=42
    )
    for split, same_split in zip(first, repeated):
        for indices, same_indices in zip(split[:2], same_split[:2]):
            assert np.array_equal(indices, same_indices)
            assert set(np.unique(targets[indices])) == {0.0, 1.0}
        assert split[2] == same_split[2]
        train, test, _ = split
        assert not set(groups[train]) & set(groups[test])


def test_downstream_dataset_selection_is_ordered_deduplicated_and_validated():
    assert _resolve_dataset_names(["BBBP", "esol", "bbbp"]) == ["bbbp", "esol"]
    assert _resolve_dataset_names(["HIV"]) == ["hiv"]
    with pytest.raises(ValueError, match="Unknown MoleculeNet datasets"):
        _resolve_dataset_names(["not-a-dataset"])


def test_downstream_cli_can_request_only_the_selected_representation():
    args = build_parser().parse_args(
        [
            "benchmark-downstream",
            "--run-dir",
            "run",
            "--datasets-dir",
            "datasets",
            "--output",
            "result.json",
            "--selected-only",
        ]
    )
    assert args.selected_only is True


def test_selected_representation_blocks_do_not_depend_on_diagnostic_exports():
    graph = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean_node = np.asarray([[5.0], [6.0]], dtype=np.float32)
    unit_hybrid = np.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    blocks = {
        "graph_dimensions": 2,
        "mean_node_dimensions": 1,
        "raw_graph_z": graph,
        "raw_mean_node_z": mean_node,
    }

    assert np.array_equal(
        _select_representation_embedding("graph_z", unit_hybrid, blocks), graph
    )
    assert np.array_equal(
        _select_representation_embedding("mean_node_z", unit_hybrid, blocks),
        mean_node,
    )


def test_downstream_artifact_records_complete_checkpoint_provenance(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_name = "checkpoint.pt"
    (run_dir / checkpoint_name).write_bytes(b"checkpoint")
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "HIV.csv").write_text("pinned fixture", encoding="utf-8")
    cfg = {
        "paths": {"run_dir": str(run_dir)},
        "seed": 42,
        "model": {"architecture": "vgae"},
        "_config_hash": "config-hash",
        "_descriptor_schema_hash": "descriptor-hash",
    }
    manifest = {
        "graph_manifest_hash": "manifest-hash",
        "feature_schema": {"hash": "feature-hash"},
    }
    standardizer = type("StandardizerStub", (), {"scaler_hash": "scaler-hash"})()
    model = object()
    monkeypatch.setattr(
        downstream,
        "_distributed_context",
        lambda allow_cpu: (0, 1, 0, object()),
    )
    monkeypatch.setattr(
        downstream,
        "load_saved_model",
        lambda requested_cfg, requested_checkpoint, device: (
            cfg,
            manifest,
            standardizer,
            model,
            {"global_step": 10_000},
        ),
    )
    monkeypatch.setattr(downstream, "_training_plan_hash", lambda value: "plan-hash")
    molecules = [object(), object(), object(), object()]
    targets = np.asarray([0.0, 1.0, 0.0, 1.0])
    groups = np.asarray(["a", "b", "c", "d"], dtype=object)
    monkeypatch.setattr(
        downstream,
        "_prepare_dataset",
        lambda path, spec, value: (
            molecules,
            targets,
            groups,
            {"molecules": 4},
        ),
    )
    monkeypatch.setattr(
        downstream,
        "_encode_molecules",
        lambda requested_model, requested_molecules, value, device: (
            np.ones((4, 2), dtype=np.float32),
            {"graph_dimensions": 2, "mean_node_dimensions": 0},
        ),
    )
    monkeypatch.setattr(
        downstream,
        "_scaffold_splits",
        lambda groups, targets, task, count, seed: [
            (np.asarray([0, 1]), np.asarray([2, 3]), seed)
        ],
    )
    monkeypatch.setattr(
        downstream,
        "_classification_probe",
        lambda features, targets, groups, splits: [
            {
                "roc_auc": 0.75,
                "average_precision": 0.50,
                "balanced_accuracy": 0.60,
            }
        ],
    )

    result = downstream.benchmark_moleculenet(
        cfg,
        checkpoint_name=checkpoint_name,
        datasets_dir=datasets_dir,
        output=tmp_path / "result.json",
        scaffold_splits=1,
        dataset_names=["hiv"],
        selected_only=True,
        allow_cpu=True,
    )

    assert result["checkpoint"]["descriptor_schema_hash"] == "descriptor-hash"
    assert result["checkpoint"]["feature_schema_hash"] == "feature-hash"
    assert result["checkpoint"]["scaler_hash"] == "scaler-hash"

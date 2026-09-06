import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import Chem

from gmolai_retrain.cli import build_parser
from gmolai_retrain.downstream import PreparedMoleculeNetDataset
from gmolai_retrain.downstream_audit import (
    _descriptor_matrix,
    _join_pretraining_rows,
)
from gmolai_retrain.downstream_exposure import (
    _load_graph_shard_identities,
    _scan_target_locations,
    _seen_at_cycle_zero_cursor,
)
from gmolai_retrain.exposure import _rank_exposure
from gmolai_retrain.util import atomic_write_csv


DESCRIPTOR_NAMES = [
    "qed",
    "MolWt",
    "NumValenceElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "BalabanJ",
    "LabuteASA",
    "TPSA",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "MolLogP",
    "MolMR",
]


def _hash(smiles):
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


def _cfg():
    return {
        "data": {
            "hash_buckets": 1,
            "descriptor_columns": [str(index) for index in range(13)],
        },
        "_descriptors": {
            "features": [{"name": name} for name in DESCRIPTOR_NAMES]
        },
    }


def test_bucket_join_returns_exact_corpus_split_and_frozen_descriptors(tmp_path):
    first_smiles, second_smiles = "CC", "CCC"
    first_hash, second_hash = _hash(first_smiles), _hash(second_smiles)
    columns = {
        "molecule_hash": [first_hash],
        "canonical_smiles": [first_smiles],
        "split": ["train"],
    }
    for index in range(13):
        columns[f"d{index:02d}"] = [float(index + 1)]
    parquet = tmp_path / "bucket-0000.parquet"
    pq.write_table(pa.table(columns), parquet)
    prepared = PreparedMoleculeNetDataset(
        molecules=[Chem.MolFromSmiles(first_smiles), Chem.MolFromSmiles(second_smiles)],
        targets=np.asarray([0.0, 1.0]),
        scaffold_groups=np.asarray(["a", "b"], dtype=object),
        canonical_smiles=(first_smiles, second_smiles),
        molecule_hashes=(first_hash, second_hash),
        source_buckets=np.asarray([0, 0], dtype=np.int16),
        preparation={"molecules": 2, "scaffold_groups": 2},
    )
    result = _join_pretraining_rows(
        _cfg(),
        {"parquet_files": [str(parquet)]},
        {"fixture": prepared},
        include_descriptors=True,
    )["fixture"]

    assert result[0]["split"] == "train"
    assert result[0]["descriptors"] == [float(index + 1) for index in range(13)]
    assert result[1] is None


def test_descriptor_matrix_uses_the_frozen_13_feature_order():
    names, values = _descriptor_matrix(
        _cfg(), [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    )
    assert names == DESCRIPTOR_NAMES
    assert values.shape == (2, 13)
    assert np.isfinite(values).all()


def test_rank_exposure_counts_cycles_presentations_and_unique_graphs():
    shards = [{"graphs": 10, "path": "shard.pt", "split": "train"}]
    first_cycle = _rank_exposure(
        shards,
        seed=42,
        rank=0,
        world_size=1,
        cursor={"cycle": 0, "shard_position": 0, "graph_position": 7},
    )
    second_cycle = _rank_exposure(
        shards,
        seed=42,
        rank=0,
        world_size=1,
        cursor={"cycle": 1, "shard_position": 0, "graph_position": 3},
    )
    assert first_cycle["total_presentations"] == 7
    assert first_cycle["unique_graphs_presented"] == 7
    assert second_cycle["total_presentations"] == 13
    assert second_cycle["unique_graphs_presented"] == 10


def _write_identity_shard(path, *, hashes, bucket, sequence):
    payload = {
        "metadata": {
            "split": "train",
            "bucket": bucket,
            "sequence": sequence,
            "graphs": len(hashes),
        },
        "x": torch.zeros((1, 1), dtype=torch.uint8),
        "edge_index": torch.zeros((2, 0), dtype=torch.int32),
        "edge_attr": torch.zeros((0, 1), dtype=torch.uint8),
        "y": torch.zeros((len(hashes), 1), dtype=torch.float32),
        "node_ptr": torch.zeros(len(hashes) + 1, dtype=torch.int64),
        "edge_ptr": torch.zeros(len(hashes) + 1, dtype=torch.int64),
        "graph_ids": torch.zeros(len(hashes), dtype=torch.int64),
        "molecule_hashes": list(hashes),
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "split": "train",
        "bucket": bucket,
        "sequence": sequence,
        "graphs": len(hashes),
    }


def test_identity_metadata_reader_and_exact_target_locations(tmp_path):
    shards = []
    all_hashes = []
    for sequence in range(4):
        hashes = [_hash(f"C{sequence}{index}") for index in range(3)]
        all_hashes.extend(hashes)
        shards.append(
            _write_identity_shard(
                tmp_path / f"shard-{sequence}.pt",
                hashes=hashes,
                bucket=0,
                sequence=sequence,
            )
        )

    identities, metadata_bytes = _load_graph_shard_identities(shards[0])
    assert identities == all_hashes[:3]
    assert metadata_bytes > 0

    targets = {all_hashes[1], all_hashes[7], all_hashes[11]}
    locations, audit = _scan_target_locations(
        shards,
        target_hashes=targets,
        seed=42,
        world_size=2,
        workers=2,
    )
    assert set(locations) == targets
    assert audit["training_shards_scanned"] == 4
    assert audit["training_graph_hashes_scanned"] == 12
    assert audit["tensor_storage_members_loaded"] is False
    assert audit["rank_shards_exclusive"] is True
    for molecule_hash, location in locations.items():
        assert location["rank"] == location["manifest_train_shard_index"] % 2
        assert 0 <= location["stream_graph_position_cycle0"] < 3
        shard_hashes, _ = _load_graph_shard_identities(
            shards[location["manifest_train_shard_index"]]
        )
        assert shard_hashes[location["graph_index_in_shard"]] == molecule_hash


def test_cycle_zero_seen_boundary_is_strict():
    location = {
        "stream_shard_position_cycle0": 4,
        "stream_graph_position_cycle0": 7,
    }
    assert not _seen_at_cycle_zero_cursor(
        location, {"cycle": 0, "shard_position": 4, "graph_position": 7}
    )
    assert _seen_at_cycle_zero_cursor(
        location, {"cycle": 0, "shard_position": 4, "graph_position": 8}
    )
    assert _seen_at_cycle_zero_cursor(
        location, {"cycle": 0, "shard_position": 5, "graph_position": 0}
    )
    assert not _seen_at_cycle_zero_cursor(
        location, {"cycle": 0, "shard_position": 3, "graph_position": 100}
    )


def test_atomic_csv_artifacts_use_lf_line_endings(tmp_path):
    output = tmp_path / "audit.csv"
    atomic_write_csv(output, [{"name": "example", "value": 1}])

    assert output.read_bytes() == b"name,value\nexample,1\n"


def test_no_training_audit_commands_are_registered():
    overlap = build_parser().parse_args(
        [
            "audit-downstream-overlap",
            "--datasets-dir",
            "datasets",
            "--output",
            "overlap.json",
            "--summary-csv",
            "overlap.csv",
        ]
    )
    descriptor = build_parser().parse_args(
        [
            "benchmark-descriptor-control",
            "--datasets-dir",
            "datasets",
            "--reference-benchmark",
            "reference.json",
            "--output",
            "descriptor.json",
            "--summary-csv",
            "descriptor.csv",
        ]
    )
    exposure = build_parser().parse_args(
        [
            "audit-training-exposure",
            "--run-dir",
            "run",
            "--checkpoint",
            "checkpoints/step-000010000.pt",
            "--output",
            "exposure.json",
            "--summary-csv",
            "exposure.csv",
        ]
    )
    downstream_exposure = build_parser().parse_args(
        [
            "audit-downstream-exposure",
            "--run-dir",
            "run",
            "--checkpoint",
            "checkpoints/step-000005000.pt",
            "--datasets-dir",
            "datasets",
            "--output",
            "downstream-exposure.json",
            "--summary-csv",
            "downstream-exposure.csv",
            "--identity-ledger-csv",
            "downstream-exposure-identities.csv",
        ]
    )

    assert overlap.command == "audit-downstream-overlap"
    assert descriptor.command == "benchmark-descriptor-control"
    assert exposure.command == "audit-training-exposure"
    assert exposure.checkpoints == ["checkpoints/step-000010000.pt"]
    assert downstream_exposure.command == "audit-downstream-exposure"
    assert downstream_exposure.checkpoints == ["checkpoints/step-000005000.pt"]

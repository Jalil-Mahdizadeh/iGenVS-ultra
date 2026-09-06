from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .representations import _file_sha256
from .util import atomic_write_csv, atomic_write_json, runtime_versions


def _load_checkpoint_safely(path: Path) -> dict[str, Any]:
    """Load a local checkpoint with the weights-only unpickler and a minimal NumPy allowlist."""

    numpy_dtype_type = type(np.dtype(np.uint32))
    safe_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        numpy_dtype_type,
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    for key in ("global_step", "world_size", "data_states", "graph_manifest_hash"):
        if key not in checkpoint:
            raise RuntimeError(f"Checkpoint lacks {key}: {path}")
    return checkpoint


def _rank_exposure(
    train_shards: list[dict[str, Any]],
    *,
    seed: int,
    rank: int,
    world_size: int,
    cursor: dict[str, int],
) -> dict[str, Any]:
    local_shards = [
        entry for index, entry in enumerate(train_shards) if index % world_size == rank
    ]
    if not local_shards:
        raise RuntimeError(f"Rank {rank} has no training shards")
    local_population = sum(int(entry["graphs"]) for entry in local_shards)
    cycle = int(cursor["cycle"])
    shard_position = int(cursor["shard_position"])
    graph_position = int(cursor["graph_position"])
    if cycle < 0 or shard_position < 0 or graph_position < 0:
        raise RuntimeError(f"Rank {rank} has a negative cursor component")
    shard_order = list(range(len(local_shards)))
    random.Random(seed + 1_000_003 * cycle + rank).shuffle(shard_order)
    if shard_position >= len(shard_order):
        raise RuntimeError(f"Rank {rank} shard cursor is outside the local shard order")
    current_shard = local_shards[shard_order[shard_position]]
    current_shard_graphs = int(current_shard["graphs"])
    if graph_position > current_shard_graphs:
        raise RuntimeError(f"Rank {rank} graph cursor exceeds its current shard")
    completed_current_cycle = sum(
        int(local_shards[shard_order[position]]["graphs"])
        for position in range(shard_position)
    )
    presentations = (
        cycle * local_population + completed_current_cycle + graph_position
    )
    unique_graphs = (
        local_population
        if cycle > 0
        else completed_current_cycle + graph_position
    )
    return {
        "rank": rank,
        "cursor": {
            "cycle": cycle,
            "shard_position": shard_position,
            "graph_position": graph_position,
        },
        "local_shards": len(local_shards),
        "local_population": local_population,
        "completed_shards_in_current_cycle": shard_position,
        "current_shard": str(current_shard["path"]),
        "current_shard_graphs": current_shard_graphs,
        "total_presentations": presentations,
        "unique_graphs_presented": unique_graphs,
    }


def audit_training_exposure(
    cfg: dict[str, Any],
    *,
    checkpoint_names: list[str] | tuple[str, ...],
    output: str | Path,
    summary_csv: str | Path,
) -> dict[str, Any]:
    if not checkpoint_names:
        raise ValueError("At least one checkpoint must be requested")
    work_dir = Path(cfg["paths"]["work_dir"])
    run_dir = Path(cfg["paths"]["run_dir"])
    graph_manifest_path = work_dir / "graph_manifest.json"
    dataset_manifest_path = work_dir / "dataset_manifest.json"
    if not graph_manifest_path.is_file():
        raise FileNotFoundError(graph_manifest_path)
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(dataset_manifest_path)
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    graph_counts = graph_manifest["counts"]
    corpus_graphs = int(graph_counts["graphs_total"])
    train_graphs = int(graph_counts["graphs_train"])
    if corpus_graphs != int(dataset_manifest["deduplication"]["rows_after_deduplication"]):
        raise RuntimeError("Graph and dataset manifests disagree on total graph count")
    if train_graphs != int(dataset_manifest["split_counts"]["train"]):
        raise RuntimeError("Graph and dataset manifests disagree on training graph count")
    train_shards = [
        entry for entry in graph_manifest["shards"] if entry["split"] == "train"
    ]
    if sum(int(entry["graphs"]) for entry in train_shards) != train_graphs:
        raise RuntimeError("Training shard graph counts do not match the manifest")

    training_seed = int(
        cfg.get("training", {}).get("seed", cfg["seed"])
    )
    checkpoints: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for checkpoint_name in checkpoint_names:
        path = Path(checkpoint_name)
        if not path.is_absolute():
            path = run_dir / path
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = _load_checkpoint_safely(path)
        if checkpoint["graph_manifest_hash"] != graph_manifest["graph_manifest_hash"]:
            raise RuntimeError(f"Checkpoint graph manifest mismatch: {path}")
        step = int(checkpoint["global_step"])
        if step in seen_steps:
            raise RuntimeError(f"Duplicate checkpoint step requested: {step}")
        seen_steps.add(step)
        world_size = int(checkpoint["world_size"])
        data_states = checkpoint["data_states"]
        if len(data_states) != world_size:
            raise RuntimeError(f"Checkpoint {path} has incomplete DDP data cursors")
        ranks = [
            _rank_exposure(
                train_shards,
                seed=training_seed,
                rank=rank,
                world_size=world_size,
                cursor=data_states[rank],
            )
            for rank in range(world_size)
        ]
        total_presentations = sum(int(item["total_presentations"]) for item in ranks)
        unique_graphs = sum(int(item["unique_graphs_presented"]) for item in ranks)
        if unique_graphs > train_graphs:
            raise RuntimeError("Unique exposure count exceeds the training population")
        record = {
            "checkpoint": str(path),
            "checkpoint_sha256": _file_sha256(path),
            "global_step": step,
            "world_size": world_size,
            "rank_accounting": ranks,
            "training_graph_presentations": total_presentations,
            "unique_training_graphs_presented": unique_graphs,
            "repeated_training_graph_presentations": total_presentations - unique_graphs,
            "training_partition_fraction_presented": unique_graphs / train_graphs,
            "full_corpus_fraction_presented": unique_graphs / corpus_graphs,
            "completed_one_training_partition_pass": unique_graphs == train_graphs,
            "completed_one_full_corpus_pass": unique_graphs == corpus_graphs,
        }
        checkpoints.append(record)
        csv_rows.append(
            {
                "global_step": step,
                "world_size": world_size,
                "training_graph_presentations": total_presentations,
                "unique_training_graphs_presented": unique_graphs,
                "repeated_training_graph_presentations": total_presentations - unique_graphs,
                "training_partition_fraction_percent": 100.0 * unique_graphs / train_graphs,
                "full_corpus_fraction_percent": 100.0 * unique_graphs / corpus_graphs,
                "completed_one_training_partition_pass": unique_graphs == train_graphs,
                "completed_one_full_corpus_pass": unique_graphs == corpus_graphs,
            }
        )
    checkpoints.sort(key=lambda item: int(item["global_step"]))
    csv_rows.sort(key=lambda item: int(item["global_step"]))
    result = {
        "schema_version": 1,
        "audit": f"Exact seed-{training_seed} training exposure from serialized DDP cursors",
        "training_stream_seed": training_seed,
        "pretrained_model_executed": False,
        "training_permitted": False,
        "counting_rule": (
            "For each rank: completed cycles times the exclusive local population, "
            "plus graph counts in completed shards of the current deterministic shuffle, "
            "plus graph_position in the current shard."
        ),
        "presentation_definition": (
            "One source molecular graph consumed into one rank's optimizer batch; "
            "masked or corrupted internal views are not counted separately."
        ),
        "rank_partition_rule": (
            "Training shards at manifest index i are assigned to rank i modulo world_size."
        ),
        "corpus": {
            "graphs": corpus_graphs,
            "training": train_graphs,
            "validation": int(graph_counts["graphs_validation"]),
            "test": int(graph_counts["graphs_test"]),
            "graph_manifest": str(graph_manifest_path),
            "graph_manifest_sha256": _file_sha256(graph_manifest_path),
            "graph_manifest_hash": graph_manifest["graph_manifest_hash"],
            "dataset_manifest": str(dataset_manifest_path),
            "dataset_manifest_sha256": _file_sha256(dataset_manifest_path),
            "dataset_manifest_hash": dataset_manifest["manifest_hash"],
        },
        "runtime": runtime_versions(),
        "checkpoints": checkpoints,
    }
    atomic_write_json(output, result)
    atomic_write_csv(summary_csv, csv_rows)
    return result

from __future__ import annotations

import io
import json
import os
import pickle
import random
import zipfile
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .downstream import MOLECULENET_DATASETS
from .downstream_audit import (
    OVERLAP_AUDIT_DATASETS,
    _dataset_source,
    _identity_digest,
    _join_pretraining_rows,
    _load_dataset_manifest,
    _prepare_selected,
    _resolve_dataset_names,
)
from .exposure import _load_checkpoint_safely, _rank_exposure
from .representations import _file_sha256
from .util import atomic_write_csv, atomic_write_json, runtime_versions, stable_u64


def _discard_tensor(*_args: Any) -> None:
    """Replace tensor rebuilds while reading the identity-only pickle member."""

    return None


class _StoragePlaceholder:
    pass


class _IdentityMetadataUnpickler(pickle.Unpickler):
    """Allow only the globals used by the immutable graph-shard metadata."""

    _STORAGE_TYPES = {
        "BFloat16Storage",
        "BoolStorage",
        "ByteStorage",
        "CharStorage",
        "ComplexDoubleStorage",
        "ComplexFloatStorage",
        "DoubleStorage",
        "FloatStorage",
        "HalfStorage",
        "IntStorage",
        "LongStorage",
        "ShortStorage",
    }

    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _discard_tensor
        if module == "torch" and name in self._STORAGE_TYPES:
            return _StoragePlaceholder
        raise pickle.UnpicklingError(
            f"Blocked graph-shard pickle global {module}.{name}"
        )

    def persistent_load(self, _persistent_id: Any) -> _StoragePlaceholder:
        return _StoragePlaceholder()


def _load_graph_shard_identities(
    entry: dict[str, Any],
) -> tuple[list[str], int]:
    """Read graph identities without materializing any tensor-storage member."""

    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        members = [
            item for item in archive.infolist() if item.filename.endswith("/data.pkl")
        ]
        if len(members) != 1:
            raise RuntimeError(f"Expected one data.pkl member in graph shard {path}")
        member = members[0]
        with archive.open(member) as handle:
            payload = _IdentityMetadataUnpickler(io.BytesIO(handle.read())).load()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Graph shard metadata is not a mapping: {path}")
    metadata = payload.get("metadata")
    hashes = payload.get("molecule_hashes")
    if not isinstance(metadata, dict) or not isinstance(hashes, list):
        raise RuntimeError(f"Graph shard lacks identity metadata: {path}")
    expected = {
        "split": str(entry["split"]),
        "bucket": int(entry["bucket"]),
        "sequence": int(entry["sequence"]),
        "graphs": int(entry["graphs"]),
    }
    observed = {
        "split": str(metadata.get("split")),
        "bucket": int(metadata.get("bucket", -1)),
        "sequence": int(metadata.get("sequence", -1)),
        "graphs": int(metadata.get("graphs", -1)),
    }
    if observed != expected:
        raise RuntimeError(
            f"Graph shard metadata disagrees with manifest for {path}: "
            f"expected {expected}, observed {observed}"
        )
    if len(hashes) != expected["graphs"] or any(
        not isinstance(value, str) for value in hashes
    ):
        raise RuntimeError(f"Graph identity count/type mismatch for {path}")
    return hashes, int(member.file_size)


def _cycle_zero_shard_positions(
    train_shards: list[dict[str, Any]], *, seed: int, world_size: int
) -> tuple[dict[int, dict[str, int]], list[set[int]]]:
    locations: dict[int, dict[str, int]] = {}
    rank_indices: list[set[int]] = []
    for rank in range(world_size):
        global_indices = list(range(rank, len(train_shards), world_size))
        local_order = list(range(len(global_indices)))
        random.Random(seed + rank).shuffle(local_order)
        assigned = set(global_indices)
        rank_indices.append(assigned)
        for stream_position, local_index in enumerate(local_order):
            global_index = global_indices[local_index]
            locations[global_index] = {
                "rank": rank,
                "local_shard_index": local_index,
                "stream_shard_position_cycle0": stream_position,
            }
    if set(locations) != set(range(len(train_shards))):
        raise RuntimeError(
            "DDP shard assignment did not cover the training manifest exactly"
        )
    for left in range(world_size):
        for right in range(left + 1, world_size):
            if rank_indices[left] & rank_indices[right]:
                raise RuntimeError(
                    f"DDP ranks {left} and {right} share training shards"
                )
    return locations, rank_indices


def _scan_target_locations(
    train_shards: list[dict[str, Any]],
    *,
    target_hashes: set[str],
    seed: int,
    world_size: int,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    shard_locations, rank_indices = _cycle_zero_shard_positions(
        train_shards, seed=seed, world_size=world_size
    )

    def inspect(
        item: tuple[int, dict[str, Any]],
    ) -> tuple[int, int, list[tuple[str, int]]]:
        global_index, entry = item
        hashes, metadata_bytes = _load_graph_shard_identities(entry)
        matches = [
            (molecule_hash, graph_index)
            for graph_index, molecule_hash in enumerate(hashes)
            if molecule_hash in target_hashes
        ]
        return global_index, metadata_bytes, matches

    locations: dict[str, dict[str, Any]] = {}
    metadata_bytes = 0
    items = list(enumerate(train_shards))
    executor: ThreadPoolExecutor | None = None
    inspected = map(inspect, items)
    if workers != 1:
        executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="identity-shard"
        )
        inspected = executor.map(inspect, items)
    try:
        for global_index, member_bytes, matches in inspected:
            metadata_bytes += member_bytes
            entry = train_shards[global_index]
            base = shard_locations[global_index]
            for molecule_hash, graph_index in matches:
                if molecule_hash in locations:
                    raise RuntimeError(
                        f"Training graph shards returned duplicate identity {molecule_hash}"
                    )
                locations[molecule_hash] = {
                    "manifest_train_shard_index": global_index,
                    "rank": int(base["rank"]),
                    "local_shard_index": int(base["local_shard_index"]),
                    "stream_shard_position_cycle0": int(
                        base["stream_shard_position_cycle0"]
                    ),
                    "graph_index_in_shard": int(graph_index),
                    "shard_path": str(entry["path"]),
                    "shard_graphs": int(entry["graphs"]),
                }
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    missing = sorted(target_hashes - set(locations))
    extra = sorted(set(locations) - target_hashes)
    if missing or extra:
        raise RuntimeError(
            "Training-overlap identities did not map one-to-one onto graph shards: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    by_shard: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for molecule_hash, location in locations.items():
        by_shard[int(location["manifest_train_shard_index"])].append(
            (molecule_hash, int(location["graph_index_in_shard"]))
        )
    for global_index, values in by_shard.items():
        entry = train_shards[global_index]
        order = list(range(int(entry["graphs"])))
        random.Random(seed + stable_u64(str(entry["path"]))).shuffle(order)
        wanted_indices = {graph_index for _, graph_index in values}
        inverse = {
            graph_index: stream_position
            for stream_position, graph_index in enumerate(order)
            if graph_index in wanted_indices
        }
        if set(inverse) != wanted_indices:
            raise RuntimeError(f"Could not invert graph order for {entry['path']}")
        for molecule_hash, graph_index in values:
            locations[molecule_hash]["stream_graph_position_cycle0"] = int(
                inverse[graph_index]
            )

    return locations, {
        "training_shards_scanned": len(train_shards),
        "training_graph_hashes_scanned": sum(
            int(entry["graphs"]) for entry in train_shards
        ),
        "identity_pickle_bytes_read": metadata_bytes,
        "tensor_storage_members_loaded": False,
        "target_training_identity_union": len(target_hashes),
        "target_locations_resolved": len(locations),
        "rank_shard_counts": [len(indices) for indices in rank_indices],
        "rank_shards_exclusive": True,
    }


def _seen_at_cycle_zero_cursor(
    location: dict[str, Any], cursor: dict[str, int]
) -> bool:
    if int(cursor["cycle"]) != 0:
        raise ValueError(
            "cycle-zero exposure predicate received a nonzero-cycle cursor"
        )
    shard_position = int(location["stream_shard_position_cycle0"])
    cursor_shard = int(cursor["shard_position"])
    if shard_position != cursor_shard:
        return shard_position < cursor_shard
    return int(location["stream_graph_position_cycle0"]) < int(cursor["graph_position"])


def _checkpoint_records(
    cfg: dict[str, Any],
    *,
    graph_manifest: dict[str, Any],
    train_shards: list[dict[str, Any]],
    checkpoint_names: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    if not checkpoint_names:
        raise ValueError("At least one checkpoint must be requested")
    run_dir = Path(cfg["paths"]["run_dir"])
    training_seed = int(cfg.get("training", {}).get("seed", cfg["seed"]))
    records: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    expected_world_size: int | None = None
    for checkpoint_name in checkpoint_names:
        path = Path(checkpoint_name)
        if not path.is_absolute():
            path = run_dir / path
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = _load_checkpoint_safely(path)
        if checkpoint["graph_manifest_hash"] != graph_manifest["graph_manifest_hash"]:
            raise RuntimeError(f"Checkpoint graph manifest mismatch: {path}")
        if checkpoint.get("config_hash") != cfg["_config_hash"]:
            raise RuntimeError(f"Checkpoint configuration mismatch: {path}")
        step = int(checkpoint["global_step"])
        if step in seen_steps:
            raise RuntimeError(f"Duplicate checkpoint step requested: {step}")
        seen_steps.add(step)
        world_size = int(checkpoint["world_size"])
        if expected_world_size is None:
            expected_world_size = world_size
        elif world_size != expected_world_size:
            raise RuntimeError("Retained checkpoints used different DDP world sizes")
        data_states = checkpoint["data_states"]
        if len(data_states) != world_size:
            raise RuntimeError(f"Checkpoint {path} has incomplete DDP data cursors")
        if any(int(cursor["cycle"]) != 0 for cursor in data_states):
            raise RuntimeError(
                "This exact retained-checkpoint identity audit requires all cursors in cycle 0"
            )
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
        records.append(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": _file_sha256(path),
                "global_step": step,
                "world_size": world_size,
                "data_states": [
                    {
                        "cycle": int(cursor["cycle"]),
                        "shard_position": int(cursor["shard_position"]),
                        "graph_position": int(cursor["graph_position"]),
                    }
                    for cursor in data_states
                ],
                "rank_accounting": ranks,
                "training_graph_presentations": sum(
                    int(item["total_presentations"]) for item in ranks
                ),
                "unique_training_graphs_presented": sum(
                    int(item["unique_graphs_presented"]) for item in ranks
                ),
            }
        )
    records.sort(key=lambda item: int(item["global_step"]))
    for previous, current in zip(records, records[1:]):
        if int(current["global_step"]) <= int(previous["global_step"]):
            raise RuntimeError("Checkpoint steps are not strictly increasing")
        for rank, (left, right) in enumerate(
            zip(previous["data_states"], current["data_states"])
        ):
            left_position = (int(left["shard_position"]), int(left["graph_position"]))
            right_position = (
                int(right["shard_position"]),
                int(right["graph_position"]),
            )
            if right_position < left_position:
                raise RuntimeError(
                    f"Rank {rank} cursor regressed between retained checkpoints"
                )
    return records


def audit_downstream_checkpoint_exposure(
    cfg: dict[str, Any],
    *,
    checkpoint_names: list[str] | tuple[str, ...],
    datasets_dir: str | Path,
    output: str | Path,
    summary_csv: str | Path,
    identity_ledger_csv: str | Path,
    dataset_names: list[str] | tuple[str, ...] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Audit exact downstream identities consumed before retained checkpoints."""

    names = (
        list(OVERLAP_AUDIT_DATASETS)
        if dataset_names is None
        else _resolve_dataset_names(dataset_names)
    )
    workers = int(workers or min(8, os.cpu_count() or 1))
    dataset_manifest_path, dataset_manifest = _load_dataset_manifest(cfg)
    graph_manifest_path = Path(cfg["paths"]["work_dir"]) / "graph_manifest.json"
    if not graph_manifest_path.is_file():
        raise FileNotFoundError(graph_manifest_path)
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    if graph_manifest.get("config_hash") != cfg["_config_hash"]:
        raise RuntimeError("Graph manifest belongs to a different configuration")
    if graph_manifest.get("dataset_manifest_hash") != dataset_manifest["manifest_hash"]:
        raise RuntimeError("Graph and dataset manifest identities disagree")
    train_shards = [
        entry for entry in graph_manifest["shards"] if entry["split"] == "train"
    ]
    train_graphs = int(graph_manifest["counts"]["graphs_train"])
    if sum(int(entry["graphs"]) for entry in train_shards) != train_graphs:
        raise RuntimeError("Training shard counts disagree with the graph manifest")

    checkpoints = _checkpoint_records(
        cfg,
        graph_manifest=graph_manifest,
        train_shards=train_shards,
        checkpoint_names=checkpoint_names,
    )
    world_size = int(checkpoints[0]["world_size"])
    training_seed = int(cfg.get("training", {}).get("seed", cfg["seed"]))
    prepared = _prepare_selected(cfg, datasets_dir, names)
    matches = _join_pretraining_rows(
        cfg, dataset_manifest, prepared, include_descriptors=False
    )

    canonical_by_hash: dict[str, str] = {}
    train_hashes_by_dataset: dict[str, set[str]] = {}
    corpus_hashes_by_dataset: dict[str, set[str]] = {}
    for name in names:
        dataset = prepared[name]
        corpus_hashes: set[str] = set()
        train_hashes: set[str] = set()
        if len(set(dataset.molecule_hashes)) != len(dataset.molecule_hashes):
            raise RuntimeError(f"Downstream dataset {name} contains duplicate hashes")
        for molecule_hash, canonical_smiles, match in zip(
            dataset.molecule_hashes, dataset.canonical_smiles, matches[name]
        ):
            previous = canonical_by_hash.setdefault(molecule_hash, canonical_smiles)
            if previous != canonical_smiles:
                raise RuntimeError(
                    f"SHA-256 collision across downstream canonical identities: {molecule_hash}"
                )
            if match is None:
                continue
            if (
                match["molecule_hash"] != molecule_hash
                or match["canonical_smiles"] != canonical_smiles
            ):
                raise RuntimeError(
                    f"Corpus identity mismatch for {name} {molecule_hash}"
                )
            corpus_hashes.add(molecule_hash)
            if match["split"] == "train":
                train_hashes.add(molecule_hash)
        corpus_hashes_by_dataset[name] = corpus_hashes
        train_hashes_by_dataset[name] = train_hashes

    target_hashes = set().union(*train_hashes_by_dataset.values())
    locations, scan = _scan_target_locations(
        train_shards,
        target_hashes=target_hashes,
        seed=training_seed,
        world_size=world_size,
        workers=workers,
    )

    seen_by_step: dict[int, set[str]] = {}
    for checkpoint in checkpoints:
        step = int(checkpoint["global_step"])
        seen: set[str] = set()
        for molecule_hash, location in locations.items():
            rank = int(location["rank"])
            if _seen_at_cycle_zero_cursor(location, checkpoint["data_states"][rank]):
                seen.add(molecule_hash)
        seen_by_step[step] = seen
    for previous, current in zip(checkpoints, checkpoints[1:]):
        left = int(previous["global_step"])
        right = int(current["global_step"])
        if not seen_by_step[left].issubset(seen_by_step[right]):
            raise RuntimeError(
                f"Exact downstream exposure is not monotonic from {left} to {right}"
            )

    datasets: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for name in names:
        dataset = prepared[name]
        accepted = len(dataset.molecule_hashes)
        corpus_hashes = corpus_hashes_by_dataset[name]
        train_hashes = train_hashes_by_dataset[name]
        trajectory: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "dataset": name,
            "post_filter_downstream_size": accepted,
            "corpus_overlap_n": len(corpus_hashes),
            "corpus_overlap_percent": 100.0 * len(corpus_hashes) / accepted,
            "train_overlap_n": len(train_hashes),
            "train_overlap_percent": 100.0 * len(train_hashes) / accepted,
        }
        for checkpoint in checkpoints:
            step = int(checkpoint["global_step"])
            dataset_seen = train_hashes & seen_by_step[step]
            record = {
                "global_step": step,
                "seen_training_overlap_molecules": len(dataset_seen),
                "full_downstream_fraction_seen": len(dataset_seen) / accepted,
                "training_overlap_fraction_seen": (
                    len(dataset_seen) / len(train_hashes) if train_hashes else None
                ),
                "seen_identity_set_sha256": _identity_digest(sorted(dataset_seen)),
            }
            trajectory.append(record)
            prefix = f"seen_{step}"
            summary[f"{prefix}_n"] = len(dataset_seen)
            summary[f"{prefix}_percent_downstream"] = (
                100.0 * len(dataset_seen) / accepted
            )
            summary[f"{prefix}_percent_train_overlap"] = (
                100.0 * len(dataset_seen) / len(train_hashes) if train_hashes else None
            )
        summary_rows.append(summary)
        datasets[name] = {
            "source": _dataset_source(MOLECULENET_DATASETS[name]),
            "task": MOLECULENET_DATASETS[name]["task"],
            "preparation": dataset.preparation,
            "post_filter_downstream_size": accepted,
            "accepted_identity_set_sha256": _identity_digest(dataset.molecule_hashes),
            "corpus_overlap": {
                "count": len(corpus_hashes),
                "fraction_of_downstream": len(corpus_hashes) / accepted,
                "identity_set_sha256": _identity_digest(sorted(corpus_hashes)),
            },
            "training_partition_overlap": {
                "count": len(train_hashes),
                "fraction_of_downstream": len(train_hashes) / accepted,
                "identity_set_sha256": _identity_digest(sorted(train_hashes)),
            },
            "trajectory": trajectory,
            "monotonic": all(
                int(left["seen_training_overlap_molecules"])
                <= int(right["seen_training_overlap_molecules"])
                for left, right in zip(trajectory, trajectory[1:])
            ),
        }

        for dataset_index, (molecule_hash, canonical_smiles, match) in enumerate(
            zip(dataset.molecule_hashes, dataset.canonical_smiles, matches[name])
        ):
            location = locations.get(molecule_hash)
            row: dict[str, Any] = {
                "dataset": name,
                "downstream_index": dataset_index,
                "molecule_hash": molecule_hash,
                "canonical_smiles": canonical_smiles,
                "pretraining_corpus_overlap": match is not None,
                "pretraining_split": "" if match is None else match["split"],
                "manifest_train_shard_index": (
                    "" if location is None else location["manifest_train_shard_index"]
                ),
                "rank": "" if location is None else location["rank"],
                "shard_path": "" if location is None else location["shard_path"],
                "graph_index_in_shard": (
                    "" if location is None else location["graph_index_in_shard"]
                ),
                "stream_shard_position_cycle0": (
                    "" if location is None else location["stream_shard_position_cycle0"]
                ),
                "stream_graph_position_cycle0": (
                    "" if location is None else location["stream_graph_position_cycle0"]
                ),
            }
            for checkpoint in checkpoints:
                step = int(checkpoint["global_step"])
                row[f"seen_{step}"] = (
                    molecule_hash in train_hashes
                    and molecule_hash in seen_by_step[step]
                )
            ledger_rows.append(row)

    if not all(item["monotonic"] for item in datasets.values()):
        raise RuntimeError(
            "At least one dataset failed the monotonic exposure invariant"
        )
    atomic_write_csv(summary_csv, summary_rows)
    atomic_write_csv(identity_ledger_csv, ledger_rows)
    result = {
        "schema_version": 1,
        "audit": (
            f"Exact seed-{training_seed} downstream-molecule exposure from canonical identities, "
            "graph-shard order and serialized DDP cursors"
        ),
        "training_stream_seed": training_seed,
        "pretrained_model_executed": False,
        "training_permitted": False,
        "checkpoints_modified": False,
        "embeddings_regenerated": False,
        "promotion_criteria_modified": False,
        "promoted_seed42_step10000_artifact_modified": False,
        "presentation_definition": (
            "A source molecular graph included in a completed optimizer batch before the "
            "checkpoint cursor; masked or corrupted internal views are not counted separately."
        ),
        "current_shard_boundary_rule": (
            "A graph is consumed only when its deterministic shuffled position is strictly "
            "less than graph_position; graph_position names the next unread graph."
        ),
        "identity_rule": (
            "SHA-256 of canonical isomeric SMILES, with canonical-SMILES equality required "
            "after every corpus hash join."
        ),
        "rank_partition_rule": (
            "Training shard at manifest index i is assigned exclusively to rank i modulo "
            "world_size; shard and graph permutations reproduce InfiniteGraphBatchIterator."
        ),
        "corpus": {
            "graphs": int(graph_manifest["counts"]["graphs_total"]),
            "training": train_graphs,
            "validation": int(graph_manifest["counts"]["graphs_validation"]),
            "test": int(graph_manifest["counts"]["graphs_test"]),
            "dataset_manifest": str(dataset_manifest_path),
            "dataset_manifest_sha256": _file_sha256(dataset_manifest_path),
            "dataset_manifest_hash": dataset_manifest["manifest_hash"],
            "graph_manifest": str(graph_manifest_path),
            "graph_manifest_sha256": _file_sha256(graph_manifest_path),
            "graph_manifest_hash": graph_manifest["graph_manifest_hash"],
        },
        "checkpoints": checkpoints,
        "datasets": datasets,
        "validation": {
            **scan,
            "all_checkpoint_cursors_cycle_zero": True,
            "all_ranks_accounted_exactly": all(
                len(item["data_states"]) == int(item["world_size"])
                for item in checkpoints
            ),
            "checkpoint_cursors_monotonic": True,
            "dataset_seen_sets_monotonic": True,
            "no_double_counting": True,
            "canonical_hash_matching_collision_safe": True,
            "all_training_overlap_locations_resolved_once": len(locations)
            == len(target_hashes),
            "identity_ledger": str(Path(identity_ledger_csv)),
            "identity_ledger_sha256": _file_sha256(Path(identity_ledger_csv)),
            "summary_csv": str(Path(summary_csv)),
            "summary_csv_sha256": _file_sha256(Path(summary_csv)),
        },
        "runtime": runtime_versions(),
    }
    atomic_write_json(output, result)
    return result

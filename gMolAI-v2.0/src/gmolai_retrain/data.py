from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch_geometric.data import Batch, Data

from .util import stable_u64


@dataclass
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor
    schema_hash: str
    scaler_hash: str

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=torch.tensor(value["mean"], dtype=torch.float32),
            scale=torch.tensor(value["scale"], dtype=torch.float32),
            schema_hash=value["descriptor_schema_hash"],
            scaler_hash=value["scaler_hash"],
        )

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.scale


def load_graph_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_shard(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def graph_from_shard(payload: dict[str, Any], index: int, standardizer: Standardizer) -> Data:
    node_start, node_end = (int(payload["node_ptr"][index]), int(payload["node_ptr"][index + 1]))
    edge_start, edge_end = (int(payload["edge_ptr"][index]), int(payload["edge_ptr"][index + 1]))
    x = payload["x"][node_start:node_end].to(torch.float32)
    edge_index = payload["edge_index"][:, edge_start:edge_end].to(torch.int64) - node_start
    edge_attr = payload["edge_attr"][edge_start:edge_end].to(torch.float32)
    y = standardizer.transform(payload["y"][index].to(torch.float32))
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y.unsqueeze(0),
        graph_id=payload["graph_ids"][index].view(1),
        source_bucket=torch.tensor([int(payload["metadata"].get("bucket", -1))], dtype=torch.int16),
        molecule_hash=payload["molecule_hashes"][index],
        num_nodes=node_end - node_start,
    )


class InfiniteGraphBatchIterator:
    """Deterministic shard-exclusive stream with an exactly serializable cursor."""

    def __init__(
        self,
        manifest: dict[str, Any],
        standardizer: Standardizer,
        *,
        split: str,
        rank: int,
        world_size: int,
        seed: int,
        node_budget: int,
        graph_budget: int,
        state: dict[str, int] | None = None,
    ) -> None:
        all_shards = [entry for entry in manifest["shards"] if entry["split"] == split]
        self.shards = [entry for index, entry in enumerate(all_shards) if index % world_size == rank]
        if not self.shards:
            raise RuntimeError(f"Rank {rank} received no {split} shards")
        self.standardizer = standardizer
        self.seed = int(seed)
        self.node_budget = int(node_budget)
        self.graph_budget = int(graph_budget)
        self.rank = int(rank)
        self.cycle = int((state or {}).get("cycle", 0))
        self.shard_position = int((state or {}).get("shard_position", 0))
        self.graph_position = int((state or {}).get("graph_position", 0))
        self._payload = None
        self._graph_order: list[int] | None = None
        self._shard_order: list[int] = []
        self._refresh_cycle_order()

    def _refresh_cycle_order(self) -> None:
        self._shard_order = list(range(len(self.shards)))
        random.Random(self.seed + 1_000_003 * self.cycle + self.rank).shuffle(self._shard_order)

    def _ensure_payload(self) -> None:
        if self._payload is not None:
            return
        shard_index = self._shard_order[self.shard_position]
        entry = self.shards[shard_index]
        self._payload = _load_shard(entry["path"])
        count = int(self._payload["y"].shape[0])
        self._graph_order = list(range(count))
        random.Random(
            self.seed + 10_000_019 * self.cycle + stable_u64(str(entry["path"]))
        ).shuffle(self._graph_order)
        if self.graph_position > count:
            raise RuntimeError("Resume cursor points beyond its shard")

    def _advance_shard_if_needed(self) -> None:
        self._ensure_payload()
        assert self._graph_order is not None
        if self.graph_position < len(self._graph_order):
            return
        self._payload = None
        self._graph_order = None
        self.graph_position = 0
        self.shard_position += 1
        if self.shard_position >= len(self._shard_order):
            self.cycle += 1
            self.shard_position = 0
            self._refresh_cycle_order()
        self._ensure_payload()

    def state_dict(self) -> dict[str, int]:
        return {
            "cycle": self.cycle,
            "shard_position": self.shard_position,
            "graph_position": self.graph_position,
        }

    def next_batch(self) -> Batch:
        graphs: list[Data] = []
        nodes = 0
        while len(graphs) < self.graph_budget:
            self._advance_shard_if_needed()
            assert self._payload is not None and self._graph_order is not None
            graph_index = self._graph_order[self.graph_position]
            next_nodes = int(self._payload["node_ptr"][graph_index + 1] - self._payload["node_ptr"][graph_index])
            if graphs and nodes + next_nodes > self.node_budget:
                break
            graph = graph_from_shard(self._payload, graph_index, self.standardizer)
            graphs.append(graph)
            nodes += next_nodes
            self.graph_position += 1
        return Batch.from_data_list(graphs)


def finite_batches(
    manifest: dict[str, Any],
    standardizer: Standardizer,
    *,
    split: str,
    rank: int,
    world_size: int,
    max_graphs: int,
    node_budget: int,
    graph_budget: int,
    seed: int,
    skip_graphs: int = 0,
) -> Iterator[Batch]:
    plan = _finite_shard_window_plan(
        manifest,
        split=split,
        max_graphs=max_graphs,
        seed=seed,
        skip_graphs=skip_graphs,
    )
    shards = [item for index, item in enumerate(plan) if index % world_size == rank]
    pending: list[Data] = []
    pending_nodes = 0
    for entry, graph_indices in shards:
        payload = _load_shard(entry["path"])
        count = int(payload["y"].shape[0])
        if count != int(entry["graphs"]):
            raise RuntimeError(f"Graph manifest count mismatch for {entry['path']}")
        for index in graph_indices:
            graph = graph_from_shard(payload, index, standardizer)
            graph_nodes = int(graph.num_nodes)
            if pending and (pending_nodes + graph_nodes > node_budget or len(pending) >= graph_budget):
                yield Batch.from_data_list(pending)
                pending, pending_nodes = [], 0
            pending.append(graph)
            pending_nodes += graph_nodes
    if pending:
        yield Batch.from_data_list(pending)


def _balanced_allocation(
    capacities: list[int], target: int, order: list[int]
) -> list[int]:
    """Allocate a target evenly, using ``order`` only to break ties."""
    quotas = [0] * len(capacities)
    remaining = min(max(0, int(target)), sum(capacities))
    active = [index for index in order if capacities[index] > 0]
    while remaining and active:
        share = max(1, remaining // len(active))
        next_active: list[int] = []
        for index in active:
            if remaining == 0:
                break
            room = capacities[index] - quotas[index]
            take = min(room, share, remaining)
            quotas[index] += take
            remaining -= take
            if quotas[index] < capacities[index]:
                next_active.append(index)
        active = next_active
    if remaining:
        raise RuntimeError("Finite sampling could not satisfy the requested graph count")
    return quotas


def _finite_shard_plan(
    manifest: dict[str, Any],
    *,
    split: str,
    max_graphs: int,
    seed: int,
) -> list[tuple[dict[str, Any], int]]:
    """Plan a deterministic sample spread across every molecule-hash bucket.

    Manifests are ordered by hash bucket, so taking a finite prefix silently
    evaluates only the first buckets.  We allocate evenly across buckets,
    randomize tie-breaking, and randomly select shard segments within each
    bucket.  This retains bounded I/O while making finite validation/export
    representative of the full split.
    """
    if int(max_graphs) <= 0:
        raise ValueError("max_graphs must be positive")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in manifest["shards"]:
        if entry["split"] != split:
            continue
        bucket = int(entry.get("bucket", -1))
        grouped.setdefault(bucket, []).append(entry)
    if not grouped:
        raise RuntimeError(f"No graph shards exist for split {split}")

    bucket_keys = sorted(grouped)
    bucket_order = list(range(len(bucket_keys)))
    random.Random(int(seed) ^ stable_u64(f"finite:{split}:buckets")).shuffle(
        bucket_order
    )
    capacities = [
        sum(int(entry["graphs"]) for entry in grouped[key]) for key in bucket_keys
    ]
    bucket_quotas = _balanced_allocation(capacities, int(max_graphs), bucket_order)

    plan: list[tuple[dict[str, Any], int]] = []
    for bucket_index in bucket_order:
        quota = bucket_quotas[bucket_index]
        if quota == 0:
            continue
        bucket = bucket_keys[bucket_index]
        entries = list(grouped[bucket])
        random.Random(
            int(seed) ^ stable_u64(f"finite:{split}:bucket:{bucket}")
        ).shuffle(entries)
        remaining = quota
        for entry in entries:
            take = min(remaining, int(entry["graphs"]))
            if take:
                plan.append((entry, take))
                remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(f"Finite sampling exhausted bucket {bucket}")
    if sum(quota for _, quota in plan) != min(int(max_graphs), sum(capacities)):
        raise RuntimeError("Finite sampling plan has an incorrect graph count")
    return plan


def _finite_shard_window_plan(
    manifest: dict[str, Any],
    *,
    split: str,
    max_graphs: int,
    seed: int,
    skip_graphs: int = 0,
) -> list[tuple[dict[str, Any], list[int]]]:
    """Select one efficient, non-overlapping window of the stratified sequence.

    Balanced allocations are nested as their target grows.  The difference
    between allocations at ``skip`` and ``skip + maximum`` therefore identifies
    an exact per-bucket window.  Within each bucket, shuffled shard and row order
    is stable, so later windows neither re-encode nor overlap earlier graphs.
    """
    if int(max_graphs) <= 0:
        raise ValueError("max_graphs must be positive")
    if int(skip_graphs) < 0:
        raise ValueError("skip_graphs must be non-negative")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in manifest["shards"]:
        if entry["split"] == split:
            grouped.setdefault(int(entry.get("bucket", -1)), []).append(entry)
    if not grouped:
        raise RuntimeError(f"No graph shards exist for split {split}")

    bucket_keys = sorted(grouped)
    bucket_order = list(range(len(bucket_keys)))
    random.Random(int(seed) ^ stable_u64(f"finite:{split}:buckets")).shuffle(
        bucket_order
    )
    capacities = [
        sum(int(entry["graphs"]) for entry in grouped[key]) for key in bucket_keys
    ]
    population = sum(capacities)
    offset = int(skip_graphs)
    if offset >= population:
        raise ValueError(
            f"skip_graphs={offset} reaches or exceeds the {population}-graph {split} split"
        )
    stop = min(population, offset + int(max_graphs))
    start_quotas = _balanced_allocation(capacities, offset, bucket_order)
    stop_quotas = _balanced_allocation(capacities, stop, bucket_order)

    plan: list[tuple[dict[str, Any], list[int]]] = []
    for bucket_index in bucket_order:
        local_start = start_quotas[bucket_index]
        local_stop = stop_quotas[bucket_index]
        if local_start == local_stop:
            continue
        bucket = bucket_keys[bucket_index]
        entries = list(grouped[bucket])
        random.Random(
            int(seed) ^ stable_u64(f"finite:{split}:bucket:{bucket}")
        ).shuffle(entries)
        cursor = 0
        for entry in entries:
            count = int(entry["graphs"])
            entry_stop = cursor + count
            overlap_start = max(local_start, cursor)
            overlap_stop = min(local_stop, entry_stop)
            if overlap_start < overlap_stop:
                graph_order = list(range(count))
                random.Random(
                    int(seed) ^ stable_u64(f"finite:{split}:{entry['path']}")
                ).shuffle(graph_order)
                relative_start = overlap_start - cursor
                relative_stop = overlap_stop - cursor
                plan.append(
                    (entry, graph_order[relative_start:relative_stop])
                )
            cursor = entry_stop
            if cursor >= local_stop:
                break
        if cursor < local_stop:
            raise RuntimeError(f"Finite sampling exhausted bucket {bucket}")
    if sum(len(indices) for _, indices in plan) != stop - offset:
        raise RuntimeError("Finite sampling window has an incorrect graph count")
    return plan

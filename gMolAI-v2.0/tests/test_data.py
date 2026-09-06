from collections import defaultdict
import random

import pytest

from gmolai_retrain.data import (
    _balanced_allocation,
    _finite_shard_plan,
    _finite_shard_window_plan,
)
from gmolai_retrain.util import stable_u64


def _manifest(buckets: int = 8, shards_per_bucket: int = 2, graphs: int = 10):
    return {
        "shards": [
            {
                "split": "validation",
                "bucket": bucket,
                "sequence": sequence,
                "graphs": graphs,
                "path": f"bucket-{bucket:04d}/shard-{sequence:05d}.pt",
            }
            for bucket in range(buckets)
            for sequence in range(shards_per_bucket)
        ]
    }


def test_finite_plan_is_deterministic_and_spans_all_hash_buckets():
    manifest = _manifest()
    first = _finite_shard_plan(
        manifest, split="validation", max_graphs=8, seed=42
    )
    repeated = _finite_shard_plan(
        manifest, split="validation", max_graphs=8, seed=42
    )
    assert first == repeated
    assert sum(quota for _, quota in first) == 8
    assert {entry["bucket"] for entry, _ in first} == set(range(8))


def test_finite_plan_balances_larger_samples_and_randomizes_small_ones():
    manifest = _manifest()
    plan = _finite_shard_plan(
        manifest, split="validation", max_graphs=100, seed=42
    )
    by_bucket = defaultdict(int)
    for entry, quota in plan:
        by_bucket[entry["bucket"]] += quota
    assert sum(by_bucket.values()) == 100
    assert max(by_bucket.values()) - min(by_bucket.values()) <= 1

    small = _finite_shard_plan(
        manifest, split="validation", max_graphs=4, seed=42
    )
    assert len({entry["bucket"] for entry, _ in small}) == 4
    assert [entry["bucket"] for entry, _ in small] != [0, 1, 2, 3]


def test_balanced_allocation_respects_capacity_and_exact_target():
    quotas = _balanced_allocation([1, 5, 20], 17, [2, 0, 1])
    assert sum(quotas) == 17
    assert all(quota <= capacity for quota, capacity in zip(quotas, [1, 5, 20]))


def test_finite_windows_are_nonoverlapping_complete_and_read_only_the_window():
    manifest = _manifest(buckets=8, shards_per_bucket=3, graphs=7)

    first = _finite_shard_window_plan(
        manifest, split="validation", max_graphs=53, skip_graphs=0, seed=42
    )
    second = _finite_shard_window_plan(
        manifest, split="validation", max_graphs=61, skip_graphs=53, seed=42
    )
    combined = _finite_shard_window_plan(
        manifest, split="validation", max_graphs=114, skip_graphs=0, seed=42
    )

    def identities(plan):
        return {
            (entry["path"], graph_index)
            for entry, graph_indices in plan
            for graph_index in graph_indices
        }

    first_ids = identities(first)
    second_ids = identities(second)
    assert len(first_ids) == 53
    assert len(second_ids) == 61
    assert first_ids.isdisjoint(second_ids)
    assert first_ids | second_ids == identities(combined)
    assert sum(len(indices) for _, indices in second) == 61


def test_zero_offset_window_preserves_the_existing_stratified_sample():
    manifest = _manifest(buckets=8, shards_per_bucket=3, graphs=7)
    legacy = _finite_shard_plan(
        manifest, split="validation", max_graphs=53, seed=42
    )
    expected = []
    for entry, quota in legacy:
        graph_order = list(range(entry["graphs"]))
        random.Random(
            42 ^ stable_u64(f"finite:validation:{entry['path']}")
        ).shuffle(graph_order)
        expected.append((entry["path"], graph_order[:quota]))
    window = _finite_shard_window_plan(
        manifest, split="validation", max_graphs=53, skip_graphs=0, seed=42
    )
    assert [(entry["path"], indices) for entry, indices in window] == expected


def test_finite_window_rejects_offsets_beyond_the_split():
    with pytest.raises(ValueError, match="reaches or exceeds"):
        _finite_shard_window_plan(
            _manifest(buckets=2, shards_per_bucket=1, graphs=3),
            split="validation",
            max_graphs=1,
            skip_graphs=6,
            seed=42,
        )

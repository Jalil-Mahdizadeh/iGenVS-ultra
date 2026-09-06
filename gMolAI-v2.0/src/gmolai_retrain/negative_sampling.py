from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


@dataclass
class NegativeCandidates:
    positive: torch.Tensor
    easy: torch.Tensor
    hard_pool: torch.Tensor
    hard_counts: torch.Tensor
    pool_graph_offsets: torch.Tensor

    def __post_init__(self) -> None:
        self.hard_counts = torch.as_tensor(self.hard_counts, dtype=torch.long)
        self.pool_graph_offsets = torch.as_tensor(self.pool_graph_offsets, dtype=torch.long)

    def pin_memory(self) -> "NegativeCandidates":
        return NegativeCandidates(
            positive=self.positive.pin_memory(),
            easy=self.easy.pin_memory(),
            hard_pool=self.hard_pool.pin_memory(),
            hard_counts=self.hard_counts.pin_memory(),
            pool_graph_offsets=self.pool_graph_offsets.pin_memory(),
        )


def _edge_tensor(chunks: list[np.ndarray]) -> torch.Tensor:
    if not chunks:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(np.concatenate(chunks, axis=1)).to(torch.long)


@lru_cache(maxsize=256)
def _pair_template(node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return lexicographic upper-triangle pairs for a graph size."""
    source, destination = np.triu_indices(node_count, k=1)
    return source, destination


def sample_per_graph_negatives(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    graph_ids: torch.Tensor,
    *,
    easy_ratio: float,
    hard_ratio: float,
    hard_pool_ratio: float,
    seed: int,
) -> NegativeCandidates:
    """Sample unique undirected negatives independently inside each graph.

    Easy candidates and hard-pool candidates are disjoint, and neither can
    overlap a positive edge. Counts are ratios of each graph's unique positive
    bonds, never a batch-level integer reused for every graph.
    """
    if edge_index.device.type != "cpu" or ptr.device.type != "cpu" or graph_ids.device.type != "cpu":
        raise ValueError("Negative candidate construction must run on CPU before device transfer")
    if min(easy_ratio, hard_ratio, hard_pool_ratio) < 0:
        raise ValueError("Negative ratios must be non-negative")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge_count]")
    if ptr.ndim != 1 or ptr.numel() < 2:
        raise ValueError("ptr must be a one-dimensional graph-boundary tensor")
    edges_np = edge_index.numpy()
    ptr_np = ptr.numpy()
    graph_id_list = graph_ids.view(-1).tolist()
    graph_count = ptr_np.size - 1
    if len(graph_id_list) != graph_count:
        raise ValueError("graph_ids and ptr describe different graph counts")
    if ptr_np[0] != 0 or bool(np.any(ptr_np[1:] < ptr_np[:-1])):
        raise ValueError("ptr must start at zero and be non-decreasing")
    total_nodes = int(ptr_np[-1])
    if edges_np.size and (int(edges_np.min()) < 0 or int(edges_np.max()) >= total_nodes):
        raise ValueError("edge_index contains a node outside ptr's range")
    if edges_np.shape[1]:
        source_graph = np.searchsorted(ptr_np[1:], edges_np[0], side="right")
        destination_graph = np.searchsorted(ptr_np[1:], edges_np[1], side="right")
        if bool(np.any(source_graph != destination_graph)):
            raise ValueError("edge_index contains a cross-graph positive edge")

    canonical_mask = edges_np[0] < edges_np[1]
    positive_source = edges_np[0, canonical_mask]
    positive_destination = edges_np[1, canonical_mask]
    positive_graph = np.searchsorted(ptr_np[1:], positive_source, side="right")
    if positive_graph.size:
        order = np.argsort(positive_graph, kind="stable")
        positive_source = positive_source[order]
        positive_destination = positive_destination[order]
        positive_graph = positive_graph[order]
    positive_offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(np.bincount(positive_graph, minlength=graph_count), dtype=np.int64),
        )
    )

    positives: list[np.ndarray] = []
    easy: list[np.ndarray] = []
    pool: list[np.ndarray] = []
    hard_counts: list[int] = []
    pool_offsets = [0]

    for graph_index, (start_value, end_value) in enumerate(zip(ptr_np[:-1], ptr_np[1:])):
        start, end = int(start_value), int(end_value)
        node_count = end - start
        positive_start, positive_end = positive_offsets[graph_index : graph_index + 2]
        local_source = positive_source[positive_start:positive_end] - start
        local_destination = positive_destination[positive_start:positive_end] - start
        positive_keys = np.unique(local_source * node_count + local_destination)
        local_source = positive_keys // node_count
        local_destination = positive_keys % node_count
        positive_count = positive_keys.size
        if positive_count:
            positives.append(
                np.stack((local_source + start, local_destination + start)).astype(np.int64, copy=False)
            )

        pair_source, pair_destination = _pair_template(node_count)
        available_mask = np.ones(pair_source.size, dtype=np.bool_)
        if positive_count:
            positive_positions = (
                local_source * (2 * node_count - local_source - 1) // 2
                + local_destination
                - local_source
                - 1
            )
            available_mask[positive_positions] = False
        available = np.flatnonzero(available_mask)

        easy_count = min(available.size, int(math.ceil(positive_count * easy_ratio)))
        desired_hard = int(math.ceil(positive_count * hard_ratio))
        desired_pool = max(desired_hard, int(math.ceil(positive_count * hard_pool_ratio)))
        pool_count = min(available.size - easy_count, desired_pool)
        selected_count = easy_count + pool_count
        unsigned_graph_id = int(graph_id_list[graph_index]) & ((1 << 64) - 1)
        graph_seed = (int(seed) ^ unsigned_graph_id) & ((1 << 64) - 1)
        if selected_count:
            selected = available[
                np.random.default_rng(graph_seed).choice(
                    available.size,
                    size=selected_count,
                    replace=False,
                )
            ]
            selected_source = pair_source[selected] + start
            selected_destination = pair_destination[selected] + start
            if easy_count:
                easy.append(
                    np.stack(
                        (selected_source[:easy_count], selected_destination[:easy_count])
                    ).astype(np.int64, copy=False)
                )
            if pool_count:
                pool.append(
                    np.stack(
                        (selected_source[easy_count:], selected_destination[easy_count:])
                    ).astype(np.int64, copy=False)
                )
        hard_counts.append(min(desired_hard, pool_count))
        pool_offsets.append(pool_offsets[-1] + pool_count)

    return NegativeCandidates(
        positive=_edge_tensor(positives),
        easy=_edge_tensor(easy),
        hard_pool=_edge_tensor(pool),
        hard_counts=torch.tensor(hard_counts, dtype=torch.long),
        pool_graph_offsets=torch.tensor(pool_offsets, dtype=torch.long),
    )


def select_hard_negative_logits(
    pool_logits: torch.Tensor,
    candidates: NegativeCandidates,
) -> torch.Tensor:
    """Select hard-pool logits while retaining gradients for selected values."""
    if pool_logits.ndim != 1:
        raise ValueError("Hard-negative pool logits must be one-dimensional")
    if pool_logits.shape[0] != candidates.hard_pool.shape[1]:
        raise ValueError("Hard-negative pool logits and candidate counts differ")
    if pool_logits.numel() == 0:
        return pool_logits
    pool_lengths_cpu = candidates.pool_graph_offsets[1:] - candidates.pool_graph_offsets[:-1]
    if int(pool_lengths_cpu.sum()) != pool_logits.shape[0]:
        raise ValueError("Hard-negative pool offsets do not cover all logits")
    max_pool = int(pool_lengths_cpu.max()) if pool_lengths_cpu.numel() else 0
    max_hard = int(candidates.hard_counts.max()) if candidates.hard_counts.numel() else 0
    if max_pool == 0 or max_hard == 0:
        return pool_logits[:0]

    device = pool_logits.device
    lengths = pool_lengths_cpu.to(device, non_blocking=True)
    rows = torch.repeat_interleave(
        torch.arange(pool_lengths_cpu.numel(), device=device),
        lengths,
        output_size=pool_logits.shape[0],
    )
    starts = candidates.pool_graph_offsets[:-1].to(device, non_blocking=True)
    columns = torch.arange(pool_logits.shape[0], device=device) - torch.repeat_interleave(
        starts,
        lengths,
        output_size=pool_logits.shape[0],
    )
    padded = pool_logits.new_full((pool_lengths_cpu.numel(), max_pool), -torch.inf)
    padded[rows, columns] = pool_logits
    top_values = torch.topk(padded, k=max_hard, dim=1, largest=True).values
    counts = candidates.hard_counts.to(device, non_blocking=True)
    selected = torch.arange(max_hard, device=device).unsqueeze(0) < counts.unsqueeze(1)
    return top_values[selected]


def assert_valid_candidates(candidates: NegativeCandidates, ptr: torch.Tensor) -> None:
    positive = {tuple(edge) for edge in candidates.positive.t().tolist()}
    easy = [tuple(edge) for edge in candidates.easy.t().tolist()]
    pool = [tuple(edge) for edge in candidates.hard_pool.t().tolist()]
    if len(easy) != len(set(easy)) or len(pool) != len(set(pool)):
        raise AssertionError("Negative candidates contain duplicates")
    if positive.intersection(easy) or positive.intersection(pool) or set(easy).intersection(pool):
        raise AssertionError("Positive, easy, and hard-pool edge sets are not disjoint")
    boundaries = ptr.tolist()
    for source, destination in [*easy, *pool]:
        if source >= destination:
            raise AssertionError("Candidate is not canonical undirected i<j")
        if not any(start <= source < end and start <= destination < end for start, end in zip(boundaries[:-1], boundaries[1:])):
            raise AssertionError("Cross-graph negative candidate detected")

"""Microbenchmark the deterministic molecular negative sampler."""

import argparse
import json
import time

import torch

from gmolai_retrain.negative_sampling import sample_per_graph_negatives


def synthetic_path_batch(graphs: int, nodes: int):
    ptr = torch.arange(0, (graphs + 1) * nodes, nodes, dtype=torch.long)
    local_source = torch.arange(nodes - 1, dtype=torch.long)
    local_destination = local_source + 1
    undirected = torch.stack((local_source, local_destination))
    directed = torch.cat((undirected, undirected.flip(0)), dim=1)
    edge_index = torch.cat([directed + graph * nodes for graph in range(graphs)], dim=1)
    graph_ids = torch.arange(10_000, 10_000 + graphs, dtype=torch.long)
    return edge_index, ptr, graph_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=1400)
    parser.add_argument("--nodes", type=int, default=23)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    edge_index, ptr, graph_ids = synthetic_path_batch(args.graphs, args.nodes)

    durations = []
    candidate_counts = None
    for repeat in range(args.repeats + 1):
        started = time.perf_counter()
        candidates = sample_per_graph_negatives(
            edge_index,
            ptr,
            graph_ids,
            easy_ratio=1.0,
            hard_ratio=1.0,
            hard_pool_ratio=5.0,
            seed=42 + repeat,
        )
        elapsed = time.perf_counter() - started
        if repeat:
            durations.append(elapsed)
        candidate_counts = {
            "positive": candidates.positive.shape[1],
            "easy": candidates.easy.shape[1],
            "hard_pool": candidates.hard_pool.shape[1],
        }
    print(
        json.dumps(
            {
                "graphs": args.graphs,
                "nodes_per_graph": args.nodes,
                "seconds_mean": sum(durations) / len(durations),
                "graphs_per_second": args.graphs * len(durations) / sum(durations),
                "candidate_counts": candidate_counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

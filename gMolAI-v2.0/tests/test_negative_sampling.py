import pytest
import torch

from gmolai_retrain.negative_sampling import (
    NegativeCandidates,
    assert_valid_candidates,
    sample_per_graph_negatives,
    select_hard_negative_logits,
)


def test_per_graph_negatives_are_unique_disjoint_and_deterministic():
    # Graph 0: path 0-1-2; graph 1: single bond 3-4 plus isolated node 5.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    ptr = torch.tensor([0, 3, 6], dtype=torch.long)
    graph_ids = torch.tensor([11, 22], dtype=torch.long)
    first = sample_per_graph_negatives(
        edge_index,
        ptr,
        graph_ids,
        easy_ratio=1.0,
        hard_ratio=1.0,
        hard_pool_ratio=5.0,
        seed=42,
    )
    second = sample_per_graph_negatives(
        edge_index,
        ptr,
        graph_ids,
        easy_ratio=1.0,
        hard_ratio=1.0,
        hard_pool_ratio=5.0,
        seed=42,
    )
    assert_valid_candidates(first, ptr)
    assert torch.equal(first.positive, second.positive)
    assert torch.equal(first.easy, second.easy)
    assert torch.equal(first.hard_pool, second.hard_pool)
    # Graph 0 contributes one easy edge; graph 1 contributes one easy edge and
    # one disjoint hard-pool edge.
    assert first.easy.shape[1] == 2
    assert first.hard_pool.shape[1] == 1


def test_complete_graph_has_no_invalid_negative():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    result = sample_per_graph_negatives(
        edge_index,
        torch.tensor([0, 2]),
        torch.tensor([7]),
        easy_ratio=10.0,
        hard_ratio=10.0,
        hard_pool_ratio=100.0,
        seed=1,
    )
    assert result.easy.shape == (2, 0)
    assert result.hard_pool.shape == (2, 0)


def test_cross_graph_positive_edge_is_rejected():
    with pytest.raises(ValueError, match="cross-graph positive edge"):
        sample_per_graph_negatives(
            torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
            torch.tensor([0, 2, 4], dtype=torch.long),
            torch.tensor([7, 8], dtype=torch.long),
            easy_ratio=1.0,
            hard_ratio=1.0,
            hard_pool_ratio=5.0,
            seed=1,
        )


def test_segmented_hard_negative_selection_preserves_selected_gradients():
    logits = torch.tensor([1.0, 3.0, 4.0, 2.0, 5.0], requires_grad=True)
    candidates = NegativeCandidates(
        positive=torch.empty((2, 0), dtype=torch.long),
        easy=torch.empty((2, 0), dtype=torch.long),
        hard_pool=torch.tensor([[0, 0, 2, 2, 3], [1, 2, 3, 4, 4]], dtype=torch.long),
        hard_counts=[1, 2],
        pool_graph_offsets=[0, 2, 5],
    )
    selected = select_hard_negative_logits(logits, candidates)
    assert selected.tolist() == [3.0, 5.0, 4.0]
    selected.sum().backward()
    assert logits.grad is not None
    assert logits.grad.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]

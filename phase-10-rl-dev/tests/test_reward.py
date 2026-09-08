from __future__ import annotations

import pytest
import torch

from igenvs_rl.reward import (
    binary_elite_desirability,
    elite_desirability,
    empirical_quantile,
    hybrid_desirability,
    normalized_advantages,
    percentile_desirability,
    tail_excess_desirability,
)


def test_lower_docking_score_has_higher_desirability() -> None:
    reference = [-9.0, -8.0, -7.0, -6.0]
    assert percentile_desirability(reference, -10.0) == 1.0
    assert percentile_desirability(reference, -7.5) == 0.5
    assert percentile_desirability(reference, -5.0) == 0.0


def test_hybrid_reward_improves_whole_distribution_and_extreme_tail() -> None:
    reference = [-10.0, -9.0, -8.0, -7.0, -6.0, -5.0]
    assert empirical_quantile(reference, 0.0) == -10.0
    assert empirical_quantile(reference, 1.0) == -5.0
    assert hybrid_desirability(reference, -8.0) > hybrid_desirability(reference, -7.0)
    assert tail_excess_desirability(reference, -12.0) > tail_excess_desirability(reference, -11.0)
    assert hybrid_desirability(reference, -12.0) > hybrid_desirability(reference, -10.0)


def test_tail_reward_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="tail_fraction"):
        tail_excess_desirability([-8.0, -7.0], -9.0, tail_fraction=0.0)
    with pytest.raises(ValueError, match="tail_weight"):
        hybrid_desirability([-8.0, -7.0], -9.0, tail_weight=-1.0)


def test_elite_reward_is_sparse_and_unsaturated() -> None:
    reference = [float(value) for value in range(-100, 0)]
    threshold = empirical_quantile(reference, 0.01)
    assert elite_desirability(reference, threshold + 0.001) == 0.0
    assert elite_desirability(reference, threshold) == pytest.approx(1.0)
    assert elite_desirability(reference, threshold - 2.0) > 1.0
    assert elite_desirability(reference, -120.0) > elite_desirability(reference, -110.0)


def test_elite_reward_rejects_nonelite_fraction() -> None:
    with pytest.raises(ValueError, match="elite_fraction"):
        elite_desirability([-8.0, -7.0], -9.0, elite_fraction=0.0)


def test_binary_elite_reward_does_not_favor_extreme_scores() -> None:
    reference = [float(value) for value in range(-100, 0)]
    threshold = empirical_quantile(reference, 0.01)
    assert binary_elite_desirability(reference, threshold + 0.001) == 0.0
    assert binary_elite_desirability(reference, threshold) == 1.0
    assert binary_elite_desirability(reference, -200.0) == 1.0


def test_normalized_advantages_respect_mask() -> None:
    rewards = torch.tensor([0.0, 0.25, 0.75, 1.0])
    mask = torch.tensor([False, True, True, False])
    advantages = normalized_advantages(rewards, mask)
    assert advantages.tolist() == pytest.approx([0.0, -1.0, 1.0, 0.0])


def test_normalized_advantages_handle_constant_rewards() -> None:
    rewards = torch.ones(3)
    mask = torch.ones(3, dtype=torch.bool)
    assert torch.equal(normalized_advantages(rewards, mask), torch.zeros(3))

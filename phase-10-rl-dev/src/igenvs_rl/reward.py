"""Small, engine-independent reward helpers."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from math import ceil, floor

import torch


def percentile_desirability(reference_scores: Sequence[float], score: float) -> float:
    """Map a lower-is-better score to its rank against a frozen reference."""
    if not reference_scores:
        raise ValueError("reference_scores must not be empty")
    ordered = sorted(float(value) for value in reference_scores)
    return 1.0 - bisect_right(ordered, float(score)) / len(ordered)


def empirical_quantile(reference_scores: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated quantile without requiring NumPy."""
    if not reference_scores:
        raise ValueError("reference_scores must not be empty")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in reference_scores)
    position = (len(ordered) - 1) * fraction
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def tail_excess_desirability(
    reference_scores: Sequence[float],
    score: float,
    *,
    tail_fraction: float = 0.10,
) -> float:
    """Reward improvement into and beyond the good-score reference tail."""
    if not 0.0 < tail_fraction <= 0.5:
        raise ValueError("tail_fraction must be in (0, 0.5]")
    threshold = empirical_quantile(reference_scores, tail_fraction)
    anchor = empirical_quantile(reference_scores, tail_fraction / 10.0)
    scale = max(threshold - anchor, 1e-6)
    return max(0.0, (threshold - float(score)) / scale)


def hybrid_desirability(
    reference_scores: Sequence[float],
    score: float,
    *,
    tail_fraction: float = 0.10,
    tail_weight: float = 1.0,
) -> float:
    """Combine whole-distribution rank improvement with unsaturated tail gain."""
    if tail_weight < 0.0:
        raise ValueError("tail_weight must be non-negative")
    return percentile_desirability(reference_scores, score) + tail_weight * tail_excess_desirability(
        reference_scores,
        score,
        tail_fraction=tail_fraction,
    )


def elite_desirability(
    reference_scores: Sequence[float],
    score: float,
    *,
    elite_fraction: float = 0.01,
) -> float:
    """Reward only scores inside the frozen reference's best tail.

    Crossing the elite threshold earns one unit. Better scores keep earning
    more, scaled by the distance between the elite and 10x-elite reference
    quantiles, so the objective does not saturate at the best observed base
    score.
    """
    if not 0.0 < elite_fraction <= 0.1:
        raise ValueError("elite_fraction must be in (0, 0.1]")
    threshold = empirical_quantile(reference_scores, elite_fraction)
    background = empirical_quantile(reference_scores, min(1.0, 10.0 * elite_fraction))
    scale = max(background - threshold, 1e-6)
    if float(score) > threshold:
        return 0.0
    return 1.0 + (threshold - float(score)) / scale


def binary_elite_desirability(
    reference_scores: Sequence[float],
    score: float,
    *,
    elite_fraction: float = 0.01,
) -> float:
    """Give equal reward to every molecule inside the frozen elite region.

    Unlike :func:`elite_desirability`, this deliberately does not reward ever
    more extreme docking scores. It therefore trains probability mass toward
    the target-specific elite region without encouraging exploitation of one
    unusually favorable score.
    """
    if not 0.0 < elite_fraction <= 0.1:
        raise ValueError("elite_fraction must be in (0, 0.1]")
    threshold = empirical_quantile(reference_scores, elite_fraction)
    return 1.0 if float(score) <= threshold else 0.0


def normalized_advantages(rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standardize selected rewards and leave excluded rows at zero."""
    if rewards.ndim != 1 or mask.shape != rewards.shape:
        raise ValueError("rewards and mask must be matching one-dimensional tensors")
    output = torch.zeros_like(rewards)
    selected = rewards[mask]
    if selected.numel() == 0:
        return output
    centered = selected - selected.mean()
    scale = selected.std(unbiased=False).clamp_min(1e-6)
    output[mask] = centered / scale
    return output

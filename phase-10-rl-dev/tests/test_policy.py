from __future__ import annotations

import torch

from igen3.model import CachedGPTLikeModel, GPTLikeModel
from igenvs_rl.policy import active_action_mask, causal_logits, sequence_statistics


def test_cached_and_full_logits_match() -> None:
    torch.manual_seed(7)
    model = GPTLikeModel(11, 16, 4, 2, 0.0, 8).eval()
    cached = CachedGPTLikeModel(11, 16, 4, 2, 0.0, 8).eval()
    cached.convert_from_original(model)
    tokens = torch.randint(0, 11, (3, 6))
    k_caches = [torch.empty(3, 4, 8, 4) for _ in range(2)]
    v_caches = [torch.empty(3, 4, 8, 4) for _ in range(2)]

    for position in range(tokens.shape[1]):
        full = causal_logits(model, tokens[:, : position + 1])[:, -1]
        step = cached.step(tokens[:, position], position, k_caches, v_caches)
        torch.testing.assert_close(step, full, atol=2e-5, rtol=2e-5)


def test_action_mask_includes_first_eos_only() -> None:
    actions = torch.tensor([[2, 3, 9, 9, 0], [1, 2, 3, 4, 5]])
    expected = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    assert torch.equal(active_action_mask(actions, eos_idx=9), expected)


def test_identical_prior_has_zero_kl() -> None:
    torch.manual_seed(11)
    policy = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval()
    prior = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval()
    prior.load_state_dict(policy.state_dict())
    tokens = torch.randint(0, 13, (4, 8))
    stats = sequence_statistics(policy, prior, tokens, eos_idx=12)
    torch.testing.assert_close(stats.kl_per_token, torch.zeros(4), atol=1e-7, rtol=0)

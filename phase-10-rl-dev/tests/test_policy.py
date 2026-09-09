from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from igen3.model import CachedGPTLikeModel, GPTLikeModel, LoadedGenerator
from igen3.generation import generate_de_novo_batch
from igenvs_rl import trainer
from igenvs_rl import policy as policy_module
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


def test_microbatched_objective_matches_full_batch_gradients() -> None:
    torch.manual_seed(29)
    policy = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval()
    prior = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval()
    tokens = torch.randint(0, 12, (7, 8))
    advantages = torch.tensor([-1.1, 0.3, 0.0, 1.7, -0.4, 0.8, -0.2])
    mask = torch.tensor([True, True, False, True, True, False, True])
    beta = 0.02

    statistics = sequence_statistics(policy, prior, tokens, eos_idx=12)
    full_policy_loss = -(
        advantages[mask] * statistics.sequence_log_probability[mask]
    ).mean()
    full_kl = statistics.kl_per_token.mean()
    full_loss = full_policy_loss + beta * full_kl
    full_loss.backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in policy.named_parameters()
    }

    policy.zero_grad(set_to_none=True)
    bundle = SimpleNamespace(
        policy=policy,
        prior=prior,
        vocab=SimpleNamespace(eos_idx=12),
        device=torch.device("cpu"),
    )
    result = trainer._backward_sequence_objective(
        bundle,
        tokens,
        advantages,
        mask,
        kl_beta=beta,
        temperature=1.0,
        top_k=None,
        microbatch_size=2,
    )
    assert abs(float(result["loss"]) - float(full_loss.detach())) < 1e-5
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            full_gradients[name],
            atol=2e-5,
            rtol=2e-5,
        )


def _sampling_bundle(device="cpu"):
    torch.manual_seed(43)
    model = GPTLikeModel(11, 16, 4, 2, 0.0, 8).eval().to(device)
    cached = CachedGPTLikeModel(11, 16, 4, 2, 0.0, 8).eval().to(device)
    cached.convert_from_original(model)
    vocab = SimpleNamespace(
        size=11, sos_idx=1, eos_idx=2, pad_idx=0,
        decode_rows=lambda rows: [" ".join(map(str, row)) for row in rows],
    )
    sampler = LoadedGenerator(
        spec=SimpleNamespace(seq_len=8), vocab=vocab, model=cached,
        device=torch.device(device), dtype=torch.float32, nhead=4, head_dim=4, num_layers=2,
    )
    return SimpleNamespace(sampler=sampler)


@pytest.mark.parametrize("microbatch", [1, 2, 3])
@pytest.mark.parametrize("temperature,top_k", [(1.0, None), (0.7, 5)])
def test_sampling_chunks_preserve_full_batch_draws_and_rng(microbatch, temperature, top_k):
    bundle = _sampling_bundle()
    torch.manual_seed(91)
    state = torch.get_rng_state()
    expected = generate_de_novo_batch(bundle.sampler, 7, temperature=temperature, top_k=top_k)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    actual = policy_module._sample_chunks(bundle, 7, microbatch, temperature, top_k)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_sampling_oom_retries_without_changing_draws(monkeypatch):
    bundle = _sampling_bundle()
    torch.manual_seed(61)
    state = torch.get_rng_state()
    expected = generate_de_novo_batch(bundle.sampler, 7, top_k=5)
    expected_rng = torch.get_rng_state()
    allocate = bundle.sampler.allocate_caches
    attempts = []
    def limited(batch_size, seq_len=None):
        attempts.append(batch_size)
        if batch_size > 2:
            torch.rand(3)  # A failed attempt may already have consumed randomness.
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return allocate(batch_size, seq_len)
    monkeypatch.setattr(bundle.sampler, "allocate_caches", limited)
    torch.set_rng_state(state)
    tokens, decoded = policy_module.sample_policy(bundle, 7, top_k=5)
    assert torch.equal(tokens, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert len(decoded) == 7
    assert attempts[:2] == [7, 3]


def test_sampling_does_not_retry_programming_errors(monkeypatch):
    bundle = _sampling_bundle()
    def broken(*args, **kwargs):
        raise RuntimeError("invalid model shape")
    monkeypatch.setattr(policy_module, "generate_de_novo_batch", broken)
    with pytest.raises(RuntimeError, match="invalid model shape"):
        policy_module.sample_policy(bundle, 7)


def test_backward_oom_retry_discards_partial_gradients(monkeypatch):
    torch.manual_seed(17)
    policy = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval()
    prior = GPTLikeModel(13, 16, 4, 2, 0.0, 8).eval()
    bundle = SimpleNamespace(policy=policy, prior=prior, vocab=SimpleNamespace(eos_idx=12), device=torch.device("cpu"))
    tokens = torch.randint(0, 12, (7, 8))
    advantages = torch.tensor([-1.1, 0.3, 0.0, 1.7, -0.4, 0.8, -0.2])
    mask = torch.tensor([True, True, False, True, True, False, True])
    kwargs = dict(kl_beta=0.02, temperature=1.0, top_k=None)
    trainer._backward_sequence_objective(bundle, tokens, advantages, mask, microbatch_size=7, **kwargs)
    expected = {name: parameter.grad.clone() for name, parameter in policy.named_parameters()}
    statistics = trainer.sequence_statistics
    calls = []
    def fail_after_first_piece(*args, **kwargs):
        calls.append(len(args[2]))
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return statistics(*args, **kwargs)
    monkeypatch.setattr(trainer, "sequence_statistics", fail_after_first_piece)
    monkeypatch.setattr(trainer, "SEQUENCE_MICROBATCH_MAX", 4)
    result = trainer._backward_sequence_objective_adaptive(bundle, tokens, advantages, mask, **kwargs)
    assert result["microbatch_size"] == 2
    assert calls[:3] == [4, 3, 2]
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], atol=2e-5, rtol=2e-5)

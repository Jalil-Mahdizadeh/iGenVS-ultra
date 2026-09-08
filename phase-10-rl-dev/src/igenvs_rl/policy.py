"""Trainable iGen3 policy with the existing cached iGen3 sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from igen3.generation import decode_token_batch, generate_de_novo_batch
from igen3.model import (
    CachedGPTLikeModel,
    GPTLikeModel,
    LoadedGenerator,
    configure_torch_for_inference,
    resolve_device,
)
from igen3.registry import ModelSpec, resolve_model
from igen3.tokenization import Vocabulary


@dataclass
class PolicyBundle:
    spec: ModelSpec
    vocab: Vocabulary
    policy: GPTLikeModel
    prior: GPTLikeModel
    sampler: LoadedGenerator
    device: torch.device


@dataclass
class SequenceStatistics:
    sequence_log_probability: torch.Tensor
    prior_sequence_log_probability: torch.Tensor
    kl_per_token: torch.Tensor
    entropy_per_token: torch.Tensor
    active_tokens: torch.Tensor


def causal_logits(model: GPTLikeModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Run the original iGen3 architecture with its causal attention mask."""
    batch_size, sequence_length = input_ids.shape
    positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
    positions = positions.expand(batch_size, sequence_length)
    hidden = model.token_emb(input_ids) + model.pos_emb(positions)
    attention_mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=input_ids.device,
    ).triu(1)
    hidden = model.transformer_encoder(hidden, mask=attention_mask)
    return model.fc_out(hidden)


def active_action_mask(actions: torch.Tensor, eos_idx: int) -> torch.Tensor:
    """Select actions up to and including the first EOS token."""
    eos = actions.eq(eos_idx)
    seen_before = eos.cumsum(dim=1) - eos.to(dtype=torch.long)
    return seen_before.eq(0)


def _sampling_log_probabilities(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    if top_k and top_k < scaled.size(-1):
        threshold = torch.topk(scaled, int(top_k), dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < threshold, torch.finfo(scaled.dtype).min)
    return F.log_softmax(scaled, dim=-1)


def sequence_statistics(
    policy: GPTLikeModel,
    prior: GPTLikeModel,
    token_ids: torch.Tensor,
    *,
    eos_idx: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> SequenceStatistics:
    """Evaluate sampled sequences under the policy and immutable base prior."""
    inputs = token_ids[:, :-1]
    actions = token_ids[:, 1:]
    active = active_action_mask(actions, eos_idx)

    policy_logits = causal_logits(policy, inputs)
    policy_behavior_logp = _sampling_log_probabilities(
        policy_logits,
        temperature=temperature,
        top_k=top_k,
    )
    policy_action_logp = policy_behavior_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        prior_logits = causal_logits(prior, inputs)
        prior_behavior_logp = _sampling_log_probabilities(
            prior_logits,
            temperature=temperature,
            top_k=top_k,
        )
        prior_action_logp = prior_behavior_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    active_float = active.to(dtype=policy_logits.dtype)
    sequence_logp = (policy_action_logp * active_float).sum(dim=1)
    prior_sequence_logp = (prior_action_logp * active_float).sum(dim=1)

    policy_full_logp = F.log_softmax(policy_logits, dim=-1)
    with torch.no_grad():
        prior_full_logp = F.log_softmax(prior_logits, dim=-1)
    policy_probability = policy_full_logp.exp()
    token_kl = (policy_probability * (policy_full_logp - prior_full_logp)).sum(dim=-1)
    token_entropy = -(policy_probability * policy_full_logp).sum(dim=-1)
    lengths = active_float.sum(dim=1).clamp_min(1.0)

    return SequenceStatistics(
        sequence_log_probability=sequence_logp,
        prior_sequence_log_probability=prior_sequence_logp,
        kl_per_token=(token_kl * active_float).sum(dim=1) / lengths,
        entropy_per_token=(token_entropy * active_float).sum(dim=1) / lengths,
        active_tokens=active,
    )


def _new_original_model(spec: ModelSpec, vocab: Vocabulary) -> GPTLikeModel:
    return GPTLikeModel(
        vocab.size,
        spec.d_model,
        spec.n_head,
        spec.num_layers,
        spec.dropout,
        spec.seq_len,
    )


def load_policy_bundle(
    *,
    model_root: Path,
    model_id: str = "base-isomeric",
    device_name: str = "auto",
    policy_state: dict[str, torch.Tensor] | None = None,
) -> PolicyBundle:
    configure_torch_for_inference()
    spec = resolve_model(model_id)
    device = resolve_device(device_name)
    vocab = Vocabulary.from_file(spec.vocab_path(model_root))
    base_state = torch.load(spec.weights_path(model_root), map_location="cpu")
    if isinstance(base_state, dict) and "state_dict" in base_state:
        base_state = base_state["state_dict"]

    prior = _new_original_model(spec, vocab)
    prior.load_state_dict(base_state, strict=True)
    prior.to(device=device, dtype=torch.float32).eval()
    prior.requires_grad_(False)

    policy = _new_original_model(spec, vocab)
    policy.load_state_dict(policy_state or base_state, strict=True)
    policy.to(device=device, dtype=torch.float32).eval()

    cached = CachedGPTLikeModel(
        vocab.size,
        spec.d_model,
        spec.n_head,
        spec.num_layers,
        spec.dropout,
        spec.seq_len,
    ).to(device=device, dtype=torch.float32)
    cached.eval()
    cached.convert_from_original(policy)
    sampler = LoadedGenerator(
        spec=spec,
        vocab=vocab,
        model=cached,
        device=device,
        dtype=torch.float32,
        nhead=spec.n_head,
        head_dim=spec.d_model // spec.n_head,
        num_layers=spec.num_layers,
    )
    return PolicyBundle(spec, vocab, policy, prior, sampler, device)


@torch.no_grad()
def synchronize_sampler(bundle: PolicyBundle) -> None:
    sampler_model = bundle.sampler.model
    if not isinstance(sampler_model, CachedGPTLikeModel):
        raise TypeError("RL synchronization requires the uncompiled cached iGen3 model")
    sampler_model.convert_from_original(bundle.policy)
    sampler_model.eval()


@torch.no_grad()
def sample_policy(
    bundle: PolicyBundle,
    batch_size: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> tuple[torch.Tensor, list[str]]:
    sampled_tokens = generate_de_novo_batch(
        bundle.sampler,
        batch_size,
        temperature=temperature,
        top_k=top_k,
    )
    # iGen3 samples under torch.inference_mode(). Clone after that context so
    # the token tensor can safely feed the gradient-tracked policy forward.
    tokens = sampled_tokens.clone()
    return tokens, decode_token_batch(bundle.sampler, tokens)

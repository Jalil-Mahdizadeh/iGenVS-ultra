"""Trainable iGen3 policy with the existing cached iGen3 sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from igen3.generation import decode_token_batch, generate_de_novo_batch, sample_next_token
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
    if batch_size <= 0:
        raise ValueError("sampling batch size must be positive")
    # Keep the released single-call path when it fits. On capacity failure,
    # retry the entire logical batch with smaller physical chunks: no draws
    # are lost, and a failed attempt never advances the checkpoint RNG stream.
    microbatch = batch_size
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    while True:
        try:
            sampled_tokens = _sample_chunks(bundle, batch_size, microbatch, temperature, top_k)
            break
        except RuntimeError as exc:
            capacity_error = isinstance(exc, torch.cuda.OutOfMemoryError) or any(
                marker in str(exc).lower()
                for marker in ("out of memory", "failed to allocate", "cublas_status_alloc_failed")
            )
            if not capacity_error:
                raise
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if microbatch == 1:
                raise RuntimeError(
                    "RL sampling is out of memory even for one sequence; free device memory or use a larger GPU. "
                    "The scientific batch and precision have not been reduced."
                ) from exc
        # Exit the exception handler before freeing cached allocations, so the
        # traceback cannot keep a failed chunk's tensors alive during retry.
        if cuda_rng is not None:
            torch.cuda.empty_cache()
        microbatch = max(1, microbatch // 2)
        print(f"[igenvs-rl] retrying {batch_size} policy draws in chunks of {microbatch}", flush=True)
    # iGen3 samples under torch.inference_mode(). Clone after that context so
    # the token tensor can safely feed the gradient-tracked policy forward.
    tokens = sampled_tokens.clone()
    return tokens, decode_token_batch(bundle.sampler, tokens)


@torch.inference_mode()
def _sample_chunks(bundle, batch_size: int, microbatch: int, temperature: float, top_k):
    if microbatch >= batch_size:
        return generate_de_novo_batch(
            bundle.sampler, batch_size, temperature=temperature, top_k=top_k,
        )
    generator = bundle.sampler
    device, vocab = generator.device, generator.vocab
    max_len = generator.spec.seq_len
    get_rng = (lambda: torch.cuda.get_rng_state(device)) if device.type == "cuda" else torch.get_rng_state
    set_rng = (lambda state: torch.cuda.set_rng_state(state, device)) if device.type == "cuda" else torch.set_rng_state
    # Sequence-major sub-batches would reorder the released token-major RNG
    # stream. Record each full-batch draw's state instead. Replaying a small
    # logits-sized buffer costs little memory and keeps the same Gumbel draws
    # for every (sequence, position), including already-finished sequences.
    logits = torch.zeros((batch_size, vocab.size), dtype=generator.dtype, device=device)
    states = [get_rng()]
    for _ in range(max_len - 1):
        torch.empty_like(logits).exponential_()
        states.append(get_rng())
    outputs = torch.full((batch_size, max_len), vocab.pad_idx, dtype=torch.long, device=device)
    outputs[:, 0] = vocab.sos_idx
    ends = []
    for begin in range(0, batch_size, microbatch):
        stop = min(batch_size, begin + microbatch)
        finished = torch.zeros(stop - begin, dtype=torch.bool, device=device)
        k_caches, v_caches = generator.allocate_caches(stop - begin, max_len)
        last_position = 0
        for position in range(max_len - 1):
            logits[begin:stop] = generator.model.step(
                outputs[begin:stop, position], position, k_caches, v_caches,
            )
            set_rng(states[position])
            next_tokens = sample_next_token(
                logits, temperature=temperature, do_sample=True, top_k=top_k,
            )[begin:stop]
            next_tokens = torch.where(finished, torch.full_like(next_tokens, vocab.eos_idx), next_tokens)
            outputs[begin:stop, position + 1] = next_tokens
            finished.logical_or_(next_tokens == vocab.eos_idx)
            last_position = position + 1
            if bool(finished.all()):
                break
        ends.append((begin, stop, last_position))
        del k_caches, v_caches
    last_position = max(end for _, _, end in ends)
    for begin, stop, end in ends:
        outputs[begin:stop, end + 1:last_position + 1] = vocab.eos_idx
    set_rng(states[last_position])
    return outputs

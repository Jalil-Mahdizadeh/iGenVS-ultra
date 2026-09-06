"""Frozen decoder primitives used by the public gMolAI inference CLI.

The architecture and generation routines mirror the hash-bound Step-2 and
Step-2d implementations.  User-facing code imports them from this small,
self-contained module so it does not depend on the archived study layout.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn


PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
BYTE_OFFSET = 3
ASCII_VALUES = 128
VOCAB_SIZE = BYTE_OFFSET + ASCII_VALUES


class ConditionalSmilesTransformer(nn.Module):
    """Autoregressive byte-SMILES decoder conditioned at every layer."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.vocab_size = int(config["vocab_size"])
        self.d_model = int(config["d_model"])
        self.maximum_positions = int(config["maximum_positions"])
        self.condition_memory_tokens = int(config["condition_memory_tokens"])
        self.token_embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=PAD_TOKEN
        )
        self.position_embedding = nn.Embedding(
            self.maximum_positions, self.d_model
        )
        condition_dimensions = int(config["condition_dimensions"])
        self.condition_memory = nn.Sequential(
            nn.Linear(condition_dimensions, self.d_model * 2),
            nn.SiLU(),
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(
                self.d_model * 2,
                self.condition_memory_tokens * self.d_model,
            ),
        )
        self.condition_bias = nn.Sequential(
            nn.Linear(condition_dimensions, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.memory_position = nn.Parameter(
            torch.empty(self.condition_memory_tokens, self.d_model)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(config["attention_heads"]),
            dim_feedforward=int(config["feedforward_dimensions"]),
            dropout=float(config["dropout"]),
            activation=str(config["activation"]),
            batch_first=True,
            norm_first=bool(config["norm_first"]),
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=int(config["decoder_layers"]),
            norm=nn.LayerNorm(self.d_model),
        )
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        causal = torch.triu(
            torch.ones(
                self.maximum_positions,
                self.maximum_positions,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD_TOKEN].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_position, mean=0.0, std=0.02)

    def encode_condition(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.condition_memory(condition).view(
            len(condition), self.condition_memory_tokens, self.d_model
        )
        memory = memory + self.memory_position.unsqueeze(0)
        bias = self.condition_bias(condition)
        return memory, bias

    def decode_prefix(
        self,
        input_tokens: torch.Tensor,
        memory: torch.Tensor,
        condition_bias: torch.Tensor,
    ) -> torch.Tensor:
        length = int(input_tokens.shape[1])
        if length > self.maximum_positions:
            raise ValueError("Decoder prefix exceeds positional capacity")
        positions = torch.arange(length, device=input_tokens.device)
        hidden = (
            self.token_embedding(input_tokens) * math.sqrt(self.d_model)
            + self.position_embedding(positions).unsqueeze(0)
            + condition_bias.unsqueeze(1)
        )
        decoded = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=self.causal_mask[:length, :length],
            tgt_key_padding_mask=input_tokens.eq(PAD_TOKEN),
        )
        return self.output(decoded)

    def forward(
        self, input_tokens: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        memory, bias = self.encode_condition(condition)
        return self.decode_prefix(input_tokens, memory, bias)

    @torch.inference_mode()
    def generate(
        self,
        condition: torch.Tensor,
        *,
        maximum_steps: int,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Return one greedy sequence for each conditioning vector."""
        if maximum_steps + 1 > self.maximum_positions:
            raise ValueError("Generation ceiling exceeds positional capacity")
        batch = len(condition)
        tokens = torch.full(
            (batch, maximum_steps + 1),
            PAD_TOKEN,
            dtype=torch.long,
            device=condition.device,
        )
        tokens[:, 0] = BOS_TOKEN
        finished = torch.zeros(batch, dtype=torch.bool, device=condition.device)
        memory, bias = self.encode_condition(condition)
        for step in range(maximum_steps):
            with torch.autocast(
                device_type=condition.device.type,
                dtype=autocast_dtype,
                enabled=condition.device.type == "cuda",
            ):
                logits = self.decode_prefix(
                    tokens[:, : step + 1], memory, bias
                )[:, -1]
            next_token = logits.float().argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, PAD_TOKEN),
                next_token,
            )
            tokens[:, step + 1] = next_token
            finished |= next_token.eq(EOS_TOKEN)
            if bool(finished.all()):
                return tokens[:, 1 : step + 2]
        return tokens[:, 1:]


@torch.inference_mode()
def generate_beam_pool(
    model: ConditionalSmilesTransformer,
    condition: torch.Tensor,
    *,
    maximum_steps: int,
    beam_width: int,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return every final beam hypothesis, score, and token length."""
    if beam_width < 2:
        raise ValueError("beam_width must be at least two")
    if maximum_steps + 1 > model.maximum_positions:
        raise ValueError("Generation ceiling exceeds positional capacity")
    batch = len(condition)
    device = condition.device
    tokens = torch.full(
        (batch, beam_width, maximum_steps + 1),
        PAD_TOKEN,
        dtype=torch.long,
        device=device,
    )
    tokens[:, :, 0] = BOS_TOKEN
    scores = torch.full(
        (batch, beam_width), -torch.inf, dtype=torch.float32, device=device
    )
    scores[:, 0] = 0.0
    finished = torch.zeros(
        (batch, beam_width), dtype=torch.bool, device=device
    )
    memory, bias = model.encode_condition(condition)
    memory = (
        memory.unsqueeze(1)
        .expand(-1, beam_width, -1, -1)
        .reshape(
            batch * beam_width,
            model.condition_memory_tokens,
            model.d_model,
        )
    )
    bias = (
        bias.unsqueeze(1)
        .expand(-1, beam_width, -1)
        .reshape(batch * beam_width, model.d_model)
    )
    for step in range(maximum_steps):
        prefix = tokens[:, :, : step + 1].reshape(
            batch * beam_width, step + 1
        )
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            logits = model.decode_prefix(prefix, memory, bias)[:, -1]
        log_probabilities = logits.float().log_softmax(dim=-1).view(
            batch, beam_width, model.vocab_size
        )
        log_probabilities[:, :, PAD_TOKEN] = -torch.inf
        log_probabilities[:, :, BOS_TOKEN] = -torch.inf
        if bool(finished.any()):
            log_probabilities = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(log_probabilities, -torch.inf),
                log_probabilities,
            )
            ended = finished.nonzero(as_tuple=False)
            log_probabilities[ended[:, 0], ended[:, 1], PAD_TOKEN] = 0.0
        candidates = scores.unsqueeze(-1) + log_probabilities
        next_scores, flat_indices = torch.topk(
            candidates.view(batch, -1),
            k=beam_width,
            dim=1,
            largest=True,
            sorted=True,
        )
        parents = torch.div(
            flat_indices, model.vocab_size, rounding_mode="floor"
        )
        next_tokens = flat_indices.remainder(model.vocab_size)
        gather = parents.unsqueeze(-1).expand(
            -1, -1, maximum_steps + 1
        )
        tokens = tokens.gather(1, gather)
        finished = finished.gather(1, parents)
        tokens[:, :, step + 1] = next_tokens
        finished |= next_tokens.eq(EOS_TOKEN)
        scores = next_scores
        if bool(finished.all()):
            break
    generated = tokens[:, :, 1:]
    eos = generated.eq(EOS_TOKEN)
    positions = torch.arange(
        1, maximum_steps + 1, device=device
    ).view(1, 1, -1)
    lengths = torch.where(
        eos,
        positions,
        torch.full_like(positions, maximum_steps + 1),
    ).amin(dim=-1).clamp_max(maximum_steps)
    return generated, scores, lengths


def _top_p_probabilities(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1]")
    if top_p >= 1.0:
        return logits.softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(
        logits, dim=-1, descending=True
    )
    sorted_probabilities = sorted_logits.softmax(dim=-1)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative > float(top_p)
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    return torch.zeros_like(sorted_logits).scatter(
        1, sorted_indices, sorted_logits.softmax(dim=-1)
    )


@torch.inference_mode()
def generate_seeded_sample_pool(
    model: ConditionalSmilesTransformer,
    condition: torch.Tensor,
    *,
    maximum_steps: int,
    draws: int,
    temperature: float,
    top_p: float,
    seeds: Sequence[int],
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample with one immutable RNG stream per conditioning vector."""
    if len(seeds) != len(condition):
        raise ValueError("One sample seed is required per conditioning vector")
    if draws < 1 or temperature <= 0.0:
        raise ValueError("Invalid stochastic generation parameters")
    if maximum_steps + 1 > model.maximum_positions:
        raise ValueError("Generation ceiling exceeds positional capacity")
    batch = len(condition)
    device = condition.device
    expanded = batch * int(draws)
    tokens = torch.full(
        (expanded, maximum_steps + 1),
        PAD_TOKEN,
        dtype=torch.long,
        device=device,
    )
    tokens[:, 0] = BOS_TOKEN
    finished = torch.zeros(expanded, dtype=torch.bool, device=device)
    scores = torch.zeros(expanded, dtype=torch.float32, device=device)
    memory, bias = model.encode_condition(condition)
    memory = memory.repeat_interleave(draws, dim=0)
    bias = bias.repeat_interleave(draws, dim=0)
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) % (2**63 - 1))
        generators.append(generator)
    for step in range(maximum_steps):
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            logits = model.decode_prefix(
                tokens[:, : step + 1], memory, bias
            )[:, -1]
        logits = logits.float()
        base_log_probabilities = logits.log_softmax(dim=-1)
        sampling_logits = logits / float(temperature)
        sampling_logits[:, PAD_TOKEN] = -torch.inf
        sampling_logits[:, BOS_TOKEN] = -torch.inf
        probabilities = _top_p_probabilities(
            sampling_logits, float(top_p)
        )
        pieces = []
        for position, generator in enumerate(generators):
            start = position * draws
            stop = start + draws
            pieces.append(
                torch.multinomial(
                    probabilities[start:stop],
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).squeeze(1)
            )
        sampled = torch.cat(pieces)
        selected_scores = base_log_probabilities.gather(
            1, sampled.unsqueeze(1)
        ).squeeze(1)
        sampled = torch.where(
            finished, torch.full_like(sampled, PAD_TOKEN), sampled
        )
        selected_scores = torch.where(
            finished, torch.zeros_like(selected_scores), selected_scores
        )
        tokens[:, step + 1] = sampled
        scores += selected_scores
        finished |= sampled.eq(EOS_TOKEN)
        if bool(finished.all()):
            break
    generated = tokens[:, 1:].view(batch, draws, maximum_steps)
    eos = generated.eq(EOS_TOKEN)
    positions = torch.arange(
        1, maximum_steps + 1, device=device
    ).view(1, 1, -1)
    lengths = torch.where(
        eos,
        positions,
        torch.full_like(positions, maximum_steps + 1),
    ).amin(dim=-1).clamp_max(maximum_steps)
    return generated, scores.view(batch, draws), lengths


def beam_order(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    length_penalty: float,
) -> torch.Tensor:
    normalizer = torch.pow(
        (5.0 + lengths.float()) / 6.0, float(length_penalty)
    )
    adjusted = scores.float() / normalizer
    return torch.argsort(adjusted, dim=1, descending=True, stable=True)


def proportional_merge(
    beam: list[tuple[str, int]],
    sample: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Deterministically interleave equal registered source quotas."""
    if len(beam) != len(sample):
        raise ValueError("Balanced merge requires equal source quotas")
    result: list[tuple[str, int]] = []
    for first, second in zip(beam, sample, strict=True):
        result.extend((first, second))
    return result


def decode_tokens(tokens: Sequence[int]) -> tuple[str, str]:
    """Decode one byte-token sequence and return ``(SMILES, error)``."""
    raw: list[int] = []
    saw_eos = False
    for value in tokens:
        value = int(value)
        if value == EOS_TOKEN:
            saw_eos = True
            break
        if value in {PAD_TOKEN, BOS_TOKEN}:
            return "", f"reserved_token_{value}"
        byte = value - BYTE_OFFSET
        if byte < 0 or byte >= ASCII_VALUES:
            return "", "out_of_range_token"
        raw.append(byte)
    if not saw_eos:
        return "", "missing_eos"
    try:
        return bytes(raw).decode("ascii", errors="strict"), ""
    except UnicodeDecodeError:
        return "", "non_ascii"


def decoder_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "ConditionalSmilesTransformer",
    "beam_order",
    "decode_tokens",
    "decoder_parameter_count",
    "generate_beam_pool",
    "generate_seeded_sample_pool",
    "proportional_merge",
]

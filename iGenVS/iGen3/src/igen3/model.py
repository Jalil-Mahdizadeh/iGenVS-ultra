"""Weight-compatible cached decoder for iGen3 generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import ModelSpec
from .tokenization import Vocabulary


def configure_torch_for_inference() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_dtype(device: torch.device, dtype_name: str = "auto") -> torch.dtype:
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError("dtype must be auto, float32, float16, or bfloat16") from exc


class GPTLikeModel(nn.Module):
    """Original architecture, used only for loading trained state dicts."""

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)


class CachedSelfAttention(nn.Module):
    """Self-attention layer modified to use an external KV cache."""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim * nhead != d_model:
            raise ValueError("d_model must be divisible by nhead")

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    def step(self, x_t: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, pos_idx: int) -> torch.Tensor:
        q = self._shape(self.q_proj(x_t))
        k_new = self._shape(self.k_proj(x_t))
        v_new = self._shape(self.v_proj(x_t))

        k_cache[:, :, pos_idx : pos_idx + 1, :].copy_(k_new)
        v_cache[:, :, pos_idx : pos_idx + 1, :].copy_(v_new)

        y = F.scaled_dot_product_attention(
            q,
            k_cache[:, :, : pos_idx + 1, :],
            v_cache[:, :, : pos_idx + 1, :],
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(x_t.size(0), 1, self.d_model)
        return self.out_proj(y)


class CachedEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CachedSelfAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def step(self, x_t: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, pos_idx: int) -> torch.Tensor:
        x_t = self.norm1(x_t + self.self_attn.step(x_t, k_cache, v_cache, pos_idx))
        x_t = self.norm2(x_t + self.linear2(self.dropout(F.relu(self.linear1(x_t)))))
        return x_t


class CachedGPTLikeModel(nn.Module):
    """Full inference architecture optimized for cached autoregressive decoding."""

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.layers = nn.ModuleList([CachedEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_ids", torch.arange(max_len), persistent=False)

    @torch.no_grad()
    def convert_from_original(self, orig: GPTLikeModel) -> None:
        self.token_emb.weight.copy_(orig.token_emb.weight)
        self.pos_emb.weight.copy_(orig.pos_emb.weight)
        self.fc_out.load_state_dict(orig.fc_out.state_dict())

        for cached, original in zip(self.layers, orig.transformer_encoder.layers):
            ipw = original.self_attn.in_proj_weight
            ipb = original.self_attn.in_proj_bias
            dim = ipw.shape[1]
            cached.self_attn.q_proj.weight.copy_(ipw[0:dim, :])
            cached.self_attn.q_proj.bias.copy_(ipb[0:dim])
            cached.self_attn.k_proj.weight.copy_(ipw[dim : 2 * dim, :])
            cached.self_attn.k_proj.bias.copy_(ipb[dim : 2 * dim])
            cached.self_attn.v_proj.weight.copy_(ipw[2 * dim : 3 * dim, :])
            cached.self_attn.v_proj.bias.copy_(ipb[2 * dim : 3 * dim])
            cached.self_attn.out_proj.load_state_dict(original.self_attn.out_proj.state_dict())
            cached.norm1.load_state_dict(original.norm1.state_dict())
            cached.norm2.load_state_dict(original.norm2.state_dict())
            cached.linear1.load_state_dict(original.linear1.state_dict())
            cached.linear2.load_state_dict(original.linear2.state_dict())

    def step(self, last_token_ids: torch.Tensor, pos_idx: int, k_caches: list[torch.Tensor], v_caches: list[torch.Tensor]) -> torch.Tensor:
        x = self.token_emb(last_token_ids) + self.pos_emb(self.pos_ids[pos_idx])
        x = x.unsqueeze(1)
        for layer_index, layer in enumerate(self.layers):
            x = layer.step(x, k_caches[layer_index], v_caches[layer_index], pos_idx)
        return self.fc_out(x).squeeze(1)


@dataclass
class LoadedGenerator:
    spec: ModelSpec
    vocab: Vocabulary
    model: nn.Module
    device: torch.device
    dtype: torch.dtype
    nhead: int
    head_dim: int
    num_layers: int

    def allocate_caches(self, batch_size: int, seq_len: int | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        seq_len = seq_len or self.spec.seq_len
        shape = (batch_size, self.nhead, seq_len, self.head_dim)
        k_caches = [torch.empty(shape, dtype=self.dtype, device=self.device) for _ in range(self.num_layers)]
        v_caches = [torch.empty(shape, dtype=self.dtype, device=self.device) for _ in range(self.num_layers)]
        return k_caches, v_caches


def load_generator(
    spec: ModelSpec,
    *,
    model_root: Path | None = None,
    device_name: str = "auto",
    dtype_name: str = "auto",
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
) -> LoadedGenerator:
    configure_torch_for_inference()
    device = resolve_device(device_name)
    dtype = resolve_dtype(device, dtype_name)
    vocab = Vocabulary.from_file(spec.vocab_path(model_root))

    original = GPTLikeModel(vocab.size, spec.d_model, spec.n_head, spec.num_layers, spec.dropout, spec.seq_len)
    state = torch.load(spec.weights_path(model_root), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    original.load_state_dict(state, strict=True)
    original.eval()

    cached = CachedGPTLikeModel(vocab.size, spec.d_model, spec.n_head, spec.num_layers, spec.dropout, spec.seq_len)
    cached.convert_from_original(original)
    cached.to(device=device, dtype=dtype).eval()
    del original, state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model: nn.Module = cached
    if compile_model and device.type == "cuda":
        try:
            model = torch.compile(cached, mode=compile_mode, fullgraph=True)
        except Exception as exc:  # pragma: no cover - depends on GPU/Triton runtime.
            print(f"torch.compile failed; continuing without compile: {exc}")
            model = cached

    return LoadedGenerator(
        spec=spec,
        vocab=vocab,
        model=model,
        device=device,
        dtype=dtype,
        nhead=spec.n_head,
        head_dim=spec.d_model // spec.n_head,
        num_layers=spec.num_layers,
    )


def maybe_compile_generator(generator: LoadedGenerator, *, enabled: bool, compile_mode: str = "reduce-overhead") -> LoadedGenerator:
    if not enabled or generator.device.type != "cuda":
        return generator
    try:
        generator.model = torch.compile(generator.model, mode=compile_mode, fullgraph=True)
    except Exception as exc:  # pragma: no cover - depends on GPU/Triton runtime.
        print(f"torch.compile failed; continuing without compile: {exc}")
    return generator

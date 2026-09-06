import os
import math
import pickle
from time import perf_counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------------------
# Fast path knobs
# --------------------------------------------------------------------------------------
# Set precision and disable determinism for speed
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = False

# Enable CUDA-specific optimizations if available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        # Prefer FlashAttention (SDPA) kernels for significant speedup
        print("Enabling SDPA kernel (FlashAttention)")
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception as e:
        print(f"Could not enable SDPA kernel: {e}")
# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
class Config:
    """Configuration for the generation task."""
    seq_len    = 112
    d_model    = 256
    n_head     = 8
    num_layers = 6
    dropout    = 0.10

    pad_token  = 'Z'
    sos_token  = 'G'
    eos_token  = 'E'

    # --- Generation Parameters ---
    # Target count (set once!)
    TOTAL_TO_GENERATE = 10_000_000

    # Batch size: None => autotune (recommended); or set an int, e.g. 20000
    batch_size = None

    # Sampling strategy
    temperature = 1.0
    do_sample   = True
    top_k       = 64      # set 0 or None to disable (slightly faster without)

    # Output file path
    out_path    = "batch_T1.0_10M_v3.txt"

config = Config()

# --------------------------------------------------------------------------------------
# Device & dtype
# --------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use bfloat16 for best performance on modern GPUs (A100, H100)
use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
DTYPE = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
print(f"Using device: {device} | dtype: {DTYPE}")

# --------------------------------------------------------------------------------------
# Load vocabulary
# --------------------------------------------------------------------------------------
try:
    with open("./model/vocab_NOISO.pkl", "rb") as f:
        vocab = pickle.load(f)
except FileNotFoundError:
    print("Error: vocab_transformer.pkl not found. Please ensure the vocabulary file is in the './model/' directory.")
    exit()

vocab_size = len(vocab)
print(f"Loaded vocab size = {vocab_size}")

char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_piece = [None] * vocab_size  # Fast decode table (string piece per token)

pad_idx = char_to_idx[config.pad_token]
sos_idx = char_to_idx[config.sos_token]
eos_idx = char_to_idx[config.eos_token]

# Build per-token text mapping once to avoid slow lookups during decoding
for ch, idx in char_to_idx.items():
    if ch in ('G', 'E', 'Z'): # remove SOS/EOS/PAD
        idx_to_piece[idx] = ""
    elif ch == 'X':           # 'Cl'
        idx_to_piece[idx] = "Cl"
    elif ch == 'Y':           # 'Br'
        idx_to_piece[idx] = "Br"
    else:
        idx_to_piece[idx] = ch

# --------------------------------------------------------------------------------------
# Original model (just to load weights)
# --------------------------------------------------------------------------------------
class GPTLikeModel(nn.Module):
    """The original model definition, used only for loading the state dict."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.fc_out = nn.Linear(d_model, vocab_size)

# --------------------------------------------------------------------------------------
# Cache-aware decoder (weight-compatible) for fast inference
# --------------------------------------------------------------------------------------
class CachedSelfAttention(nn.Module):
    """Self-attention layer modified to use a KV cache."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _shape(self, x):  # [B, 1, D] -> [B, H, 1, Hd]
        B, S, _ = x.shape
        return x.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

    def step(self, x_t, k_cache, v_cache, pos_idx):
        """A single forward step for one token."""
        q = self._shape(self.q_proj(x_t))
        k_new = self._shape(self.k_proj(x_t))
        v_new = self._shape(self.v_proj(x_t))

        # Write new key/value into the cache at the current position
        k_cache[:, :, pos_idx:pos_idx+1, :].copy_(k_new)
        v_cache[:, :, pos_idx:pos_idx+1, :].copy_(v_new)

        # Attend over the entire prefix [0...pos_idx]
        k_all = k_cache[:, :, :pos_idx+1, :]
        v_all = v_cache[:, :, :pos_idx+1, :]

        # Use efficient scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(x_t.size(0), 1, self.d_model)
        return self.out_proj(y)

class CachedEncoderLayer(nn.Module):
    """Transformer layer modified to use a KV cache."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = CachedSelfAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # Note: dropout is inactive in .eval() mode

    def step(self, x_t, k_cache, v_cache, pos_idx):
        """A single forward step for one token."""
        attn_out = self.self_attn.step(x_t, k_cache, v_cache, pos_idx)
        x_t = self.norm1(x_t + attn_out)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x_t))))
        x_t = self.norm2(x_t + ff_out)
        return x_t

class CachedGPTLikeModel(nn.Module):
    """The full model architecture optimized for fast cached decoding."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout, max_len):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.layers = nn.ModuleList([CachedEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.fc_out    = nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_ids", torch.arange(max_len), persistent=False)

    @torch.no_grad()
    def convert_from_original(self, orig: GPTLikeModel):
        """Copies weights from the original model to this cached version."""
        self.token_emb.weight.copy_(orig.token_emb.weight)
        self.pos_emb.weight.copy_(orig.pos_emb.weight)
        self.fc_out.load_state_dict(orig.fc_out.state_dict())

        for c, o in zip(self.layers, orig.transformer_encoder.layers):
            ipw, ipb = o.self_attn.in_proj_weight, o.self_attn.in_proj_bias
            D = ipw.shape[1]
            # De-interleave the fused QKV projection weights
            c.self_attn.q_proj.weight.copy_(ipw[0:D, :]);     c.self_attn.q_proj.bias.copy_(ipb[0:D])
            c.self_attn.k_proj.weight.copy_(ipw[D:2*D, :]);   c.self_attn.k_proj.bias.copy_(ipb[D:2*D])
            c.self_attn.v_proj.weight.copy_(ipw[2*D:3*D, :]); c.self_attn.v_proj.bias.copy_(ipb[2*D:3*D])
            c.self_attn.out_proj.load_state_dict(o.self_attn.out_proj.state_dict())
            c.norm1.load_state_dict(o.norm1.state_dict())
            c.norm2.load_state_dict(o.norm2.state_dict())
            c.linear1.load_state_dict(o.linear1.state_dict())
            c.linear2.load_state_dict(o.linear2.state_dict())

    # This is the core function that will be compiled for maximum speed
    def step(self, last_token_ids: torch.Tensor, pos_idx: int, k_caches, v_caches):
        """Forward pass for a single token, designed to be compiled."""
        x = self.token_emb(last_token_ids) + self.pos_emb(self.pos_ids[pos_idx])
        x = x.unsqueeze(1) # Add sequence dimension: [B, D] -> [B, 1, D]
        for l, layer in enumerate(self.layers):
            x = layer.step(x, k_caches[l], v_caches[l], pos_idx)
        return self.fc_out(x).squeeze(1)  # [B, V]

# --------------------------------------------------------------------------------------
# Load original weights and convert to cached model
# --------------------------------------------------------------------------------------
print("Loading and converting model weights...")
orig = GPTLikeModel(vocab_size, config.d_model, config.n_head, config.num_layers, config.dropout, config.seq_len)

try:
    state = torch.load("./model/iGen3.0_256_NOISO.pth", map_location="cpu")
except FileNotFoundError:
    print("Error: iGen_v3.0_1024.pth not found. Please ensure the model weights file is in the './model/' directory.")
    exit()

orig.load_state_dict(state, strict=True)
orig.eval()

model = CachedGPTLikeModel(vocab_size, config.d_model, config.n_head, config.num_layers, config.dropout, config.seq_len)
model.convert_from_original(orig)
model.to(device=device, dtype=DTYPE).eval()
del orig, state
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Converted to cached decoder.")

# --------------------------------------------------------------------------------------
# OPTIMIZATION: Compile the model with torch.compile
# This is the most important optimization. It fuses operations and removes Python
# overhead, which is critical in a token-by-token generation loop.
# --------------------------------------------------------------------------------------
if device.type == "cuda":
    print("Compiling model with torch.compile... (this may take a moment on first run)")
    # 'reduce-overhead' is great for inference, 'max-autotune' takes longer to compile but may be faster.
    # fullgraph=True is key to eliminating Python overhead in the generation loop.
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    print("Model compiled successfully.")

# --------------------------------------------------------------------------------------
# Fast top-k masking
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _top_k_mask_(logits, k: int):
    """In-place masking for top-k sampling."""
    if not k or k >= logits.size(-1):
        return
    # Find the k-th largest logit value
    kth_vals = torch.topk(logits, k).values[:, -1].unsqueeze(1)
    # Mask out everything smaller than the k-th value
    logits.masked_fill_(logits < kth_vals, torch.finfo(logits.dtype).min)

# --------------------------------------------------------------------------------------
# Generation with KV cache (returns LongTensor [B,T])
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def generate_batch(
    model: CachedGPTLikeModel,
    batch_size: int,
    max_len: int,
    temperature: float,
    do_sample: bool,
    top_k: int | None,
):
    """Generates a batch of sequences using the cached model."""
    B, T = batch_size, max_len

    # Start with SOS token for all sequences in the batch
    outputs = torch.full((B, T), pad_idx, dtype=torch.long, device=device)
    outputs[:, 0] = sos_idx
    # Keep track of which sequences have finished (hit EOS)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # Pre-allocate KV caches on the GPU to avoid reallocation in the loop
    H, Hd, L = model.nhead, model.head_dim, len(model.layers)
    k_caches = [torch.empty((B, H, T, Hd), dtype=DTYPE, device=device) for _ in range(L)]
    v_caches = [torch.empty((B, H, T, Hd), dtype=DTYPE, device=device) for _ in range(L)]

    for pos in range(T - 1):
        last_tok = outputs[:, pos]
        logits = model.step(last_tok, pos, k_caches, v_caches)  # [B, V]

        # Apply top-k and sampling
        if top_k:
            _top_k_mask_(logits, top_k)
        if do_sample:
            if temperature != 1.0:
                logits.div_(temperature)
            # Gumbel-Max trick for efficient, numerically stable sampling
            gumbel_noise = torch.empty_like(logits).exponential_().log_().neg_()
            next_tokens = torch.argmax(logits + gumbel_noise, dim=-1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)

        # For sequences that are already finished, force the next token to be EOS.
        # This prevents generating garbage after the molecule is complete.
        next_tokens = torch.where(finished, eos_idx, next_tokens)
        outputs[:, pos + 1] = next_tokens

        # Update finished mask
        finished.logical_or_(next_tokens == eos_idx)

        # Early exit if all sequences in the batch are finished
        if finished.all():
            break

    return outputs

# --------------------------------------------------------------------------------------
# OPTIMIZATION: Vectorized decoding and batched file writing
# --------------------------------------------------------------------------------------
def decode_and_write_batch(token_batch: torch.Tensor, fh):
    """
    Decodes a batch of tokens to strings and writes them to a file handle.
    This version uses vectorized numpy operations to find EOS positions faster.
    """
    # Move the whole batch to CPU once
    arr = token_batch.cpu().numpy()
    B, T = arr.shape

    # Vectorized search for the first EOS token in each row
    eos_mask = (arr == eos_idx)
    # If a row has an EOS, argmax finds its index. If not, we use the full length T.
    has_eos = np.any(eos_mask, axis=1)
    eos_indices = np.where(has_eos, np.argmax(eos_mask, axis=1), T)

    # Build all strings in a list comprehension
    lines_to_write = [
        "".join(idx_to_piece[token] for token in arr[b, :eos_indices[b]])
        for b in range(B)
    ]

    # Write all lines to the file in a single, efficient operation
    fh.write('\n'.join(lines_to_write))
    fh.write('\n')

# --------------------------------------------------------------------------------------
# Autotune batch size to maximize VRAM usage
# --------------------------------------------------------------------------------------
def autotune_batch_size(model: CachedGPTLikeModel, max_len: int, start_bs=30000, step=1000, max_bs=40000):
    """Dynamically finds the largest batch size that fits in VRAM."""
    if device.type != "cuda":
        return 4 # Small default for CPU
    
    print(f"Autotuning batch size (starting from {start_bs})...")
    best_bs = start_bs
    bs = start_bs
    H, Hd, L, T = model.nhead, model.head_dim, len(model.layers), max_len
    
    while bs <= max_bs:
        try:
            # Attempt to allocate memory for the largest tensors (KV caches)
            k_caches = [torch.empty((bs, H, T, Hd), dtype=DTYPE, device=device) for _ in range(L)]
            v_caches = [torch.empty((bs, H, T, Hd), dtype=DTYPE, device=device) for _ in range(L)]
            # Also test a single model step
            last_tok = torch.full((bs,), sos_idx, dtype=torch.long, device=device)
            _ = model.step(last_tok, 0, k_caches, v_caches)
            
            # If successful, free memory and try a larger size
            del k_caches, v_caches, last_tok
            torch.cuda.empty_cache()
            best_bs = bs
            bs += step
        except RuntimeError as e:
            # If an OOM error occurs, we've found our limit
            if "out of memory" in str(e).lower():
                print("OOM detected. Using previous successful batch size.")
                torch.cuda.empty_cache()
                break
            else: # Re-raise other errors
                raise e
    return best_bs

# --------------------------------------------------------------------------------------
# Main execution block
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Determine the optimal batch size
    if config.batch_size is None:
        bs = autotune_batch_size(model, config.seq_len)
        print(f"[Autotune] Using batch_size = {bs}")
    else:
        bs = int(config.batch_size)
        print(f"[Fixed] Using batch_size = {bs}")

    total = config.TOTAL_TO_GENERATE
    gen_batches = math.ceil(total / bs)
    print(f"Target total: {total:,} | Batch size: {bs:,} | Batches: {gen_batches}")

    # Warmup run: On the first call, a compiled function will run more slowly
    # due to compilation overhead. A small warmup call ensures subsequent
    # timed runs reflect true performance.
    print("Performing a small warmup run...")
    _ = generate_batch(model, batch_size=2, max_len=config.seq_len, temperature=config.temperature, do_sample=config.do_sample, top_k=config.top_k)
    print("Warmup complete.")

    # Prepare output file for streaming writes
    with open(config.out_path, "w", encoding="utf-8") as fh:
        start_time = perf_counter()
        generated_count = 0

        pbar = tqdm(range(gen_batches), desc="Generating SMILES")
        for _ in pbar:
            current_bs = min(bs, total - generated_count)
            if current_bs <= 0: break

            toks = generate_batch(
                model=model,
                batch_size=current_bs,
                max_len=config.seq_len,
                temperature=config.temperature,
                do_sample=config.do_sample,
                top_k=config.top_k,
            )
            decode_and_write_batch(toks, fh)
            generated_count += current_bs
            
            # Update progress bar with live throughput
            elapsed = perf_counter() - start_time
            throughput = generated_count / max(elapsed, 1e-9)
            pbar.set_postfix({"SMILES/s": f"{throughput:.1f}"})

    total_time = perf_counter() - start_time
    final_throughput = total / max(total_time, 1e-9)

    print("-" * 50)
    print(f"Successfully saved {generated_count:,} SMILES to: {config.out_path}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Final average throughput: {final_throughput:.1f} SMILES/s")
    print("-" * 50)



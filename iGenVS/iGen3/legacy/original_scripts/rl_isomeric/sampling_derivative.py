'''
# Optional HPC modules
# ml PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# ml scikit-learn/1.3.1-gfbf-2023a
# ml tqdm/4.66.1-GCCcore-12.3.0
# ml RDKit/2024.03.3-foss-2023a
'''

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
# Fast-path knobs (works great on Ada/A100)
# --------------------------------------------------------------------------------------
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic=False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        # Prefer FlashAttention SDPA kernels
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass

# --------------------------------------------------------------------------------------
# Step A: Configuration and constants
# --------------------------------------------------------------------------------------
class Config:
    seq_len   = 122
    d_model   = 256
    n_head    = 8
    num_layers= 6
    dropout   = 0.1

    pad_token = 'Z'
    sos_token = 'G'
    eos_token = 'E'

    # Generation controls
    temperature = 2.0
    do_sample   = True
    top_k       = 64        # set None/0 to disable
    batch_size  = 30000      # adjust or autotune if you like
    out_path    = "batch_DERIVATIVE_T2.0_v3.txt"  # streaming text output

config = Config()

# Device / dtype (BF16 on A100/H100; FP16 on most RTX; FP32 on CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
DTYPE = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
print(f"Using device: {device} | dtype: {DTYPE}")

# --------------------------------------------------------------------------------------
# Step B: Load vocabulary
# --------------------------------------------------------------------------------------
with open("./model/vocab_ISO.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
print("Loaded vocab size =", vocab_size)

char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

pad_idx = char_to_idx[config.pad_token]
sos_idx = char_to_idx[config.sos_token]
eos_idx = char_to_idx[config.eos_token]

# Fast decode table (per-token string piece; avoids str.translate per-whole-string)
idx_to_piece = [""] * vocab_size
for ch, idx in char_to_idx.items():
    if ch == 'G' or ch == 'E' or ch == 'Z':
        idx_to_piece[idx] = ""
    elif ch == 'X':
        idx_to_piece[idx] = "Cl"
    elif ch == 'Y':
        idx_to_piece[idx] = "Br"
    else:
        idx_to_piece[idx] = ch

# --------------------------------------------------------------------------------------
# Step C: Define the (original) GPT-like model (for loading trained weights)
# --------------------------------------------------------------------------------------
class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

# --------------------------------------------------------------------------------------
# Cache-aware decoder that is weight-compatible with TransformerEncoderLayer
# --------------------------------------------------------------------------------------
class CachedSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _shape(self, x):  # [B,1,D] -> [B,H,1,Hd]
        B, S, _ = x.shape
        return x.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

    def step(self, x_t, k_cache, v_cache, pos_idx):
        # x_t: [B,1,D]; caches: [B,H,T,Hd]
        q = self._shape(self.q_proj(x_t))
        k_new = self._shape(self.k_proj(x_t))
        v_new = self._shape(self.v_proj(x_t))

        # write KV for this position (in-place)
        k_cache[:, :, pos_idx:pos_idx+1, :].copy_(k_new)
        v_cache[:, :, pos_idx:pos_idx+1, :].copy_(v_new)

        # attend over prefix [0..pos_idx]
        k_all = k_cache[:, :, :pos_idx+1, :]
        v_all = v_cache[:, :, :pos_idx+1, :]

        y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=None, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(x_t.size(0), 1, self.d_model)
        return self.out_proj(y)

class CachedEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = CachedSelfAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 4*d_model)
        self.linear2 = nn.Linear(4*d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def step(self, x_t, k_cache, v_cache, pos_idx):
        x_t = self.norm1(x_t + self.self_attn.step(x_t, k_cache, v_cache, pos_idx))
        x_t = self.norm2(x_t + self.linear2(self.dropout(F.relu(self.linear1(x_t)))))
        return x_t

class CachedGPTLikeModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.layers    = nn.ModuleList([CachedEncoderLayer(d_model, nhead, dropout) for _ in range(num_layers)])
        self.fc_out    = nn.Linear(d_model, vocab_size)
        self.register_buffer("pos_ids", torch.arange(max_len).unsqueeze(0), persistent=False)

    @torch.no_grad()
    def convert_from_original(self, orig: GPTLikeModel):
        # embeddings & head
        self.token_emb.weight.copy_(orig.token_emb.weight)
        self.pos_emb.weight.copy_(orig.pos_emb.weight)
        self.fc_out.weight.copy_(orig.fc_out.weight)
        self.fc_out.bias.copy_(orig.fc_out.bias)
        # layers
        for c, o in zip(self.layers, orig.transformer_encoder.layers):
            ipw, ipb = o.self_attn.in_proj_weight, o.self_attn.in_proj_bias
            D = ipw.shape[1]
            c.self_attn.q_proj.weight.copy_(ipw[0:D, :]);      c.self_attn.q_proj.bias.copy_(ipb[0:D])
            c.self_attn.k_proj.weight.copy_(ipw[D:2*D, :]);    c.self_attn.k_proj.bias.copy_(ipb[D:2*D])
            c.self_attn.v_proj.weight.copy_(ipw[2*D:3*D, :]);  c.self_attn.v_proj.bias.copy_(ipb[2*D:3*D])
            c.self_attn.out_proj.weight.copy_(o.self_attn.out_proj.weight)
            c.self_attn.out_proj.bias.copy_(o.self_attn.out_proj.bias)

            c.norm1.weight.copy_(o.norm1.weight); c.norm1.bias.copy_(o.norm1.bias)
            c.norm2.weight.copy_(o.norm2.weight); c.norm2.bias.copy_(o.norm2.bias)
            c.linear1.weight.copy_(o.linear1.weight); c.linear1.bias.copy_(o.linear1.bias)
            c.linear2.weight.copy_(o.linear2.weight); c.linear2.bias.copy_(o.linear2.bias)

    @torch.inference_mode()
    def step(self, last_token_ids: torch.Tensor, pos_idx: int, k_caches, v_caches):
        """
        last_token_ids: [B] current token at position `pos_idx`
        pos_idx:       int (0-based)
        k_caches/v_caches: lists of [B,H,T,Hd]
        returns logits_t: [B, vocab_size]
        """
        B = last_token_ids.size(0)
        x = self.token_emb(last_token_ids) + self.pos_emb(self.pos_ids[:, pos_idx:pos_idx+1]).squeeze(0)
        x = x.view(B, 1, -1)
        for l, layer in enumerate(self.layers):
            x = layer.step(x, k_caches[l], v_caches[l], pos_idx)
        return self.fc_out(x).squeeze(1)

# --------------------------------------------------------------------------------------
# Step D: Load the trained model weights and convert to cached-decoder
# --------------------------------------------------------------------------------------
orig = GPTLikeModel(
    vocab_size=vocab_size,
    d_model=config.d_model,
    nhead=config.n_head,
    num_layers=config.num_layers,
    dropout=config.dropout,
    max_len=config.seq_len
)
state = torch.load("./model/iGen3.0_RL_qed-sa_best.pth", map_location="cpu")
orig.load_state_dict(state, strict=True)
orig.eval()

model = CachedGPTLikeModel(
    vocab_size=vocab_size,
    d_model=config.d_model,
    nhead=config.n_head,
    num_layers=config.num_layers,
    dropout=config.dropout,
    max_len=config.seq_len
)
model.convert_from_original(orig)
model.to(device=device, dtype=DTYPE).eval()
del orig, state
torch.cuda.empty_cache()
print("Model converted to cached decoder and moved to device/dtype.")

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _top_k_mask_(logits, k: int):
    if not k or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k).values[:, -1].unsqueeze(1)
    logits.masked_fill_(logits < kth, torch.finfo(logits.dtype).min)
    return logits

def tokenize_DERIVATIVE(smi: str) -> list[int]:
    # Ensure SOS present; strip spaces
    smi = smi.strip()
    if not smi or smi[0] != config.sos_token:
        smi = config.sos_token + smi
    # Convert chars to ids; truncate to seq_len
    toks = [char_to_idx.get(ch, pad_idx) for ch in smi][:config.seq_len]
    return toks

# --------------------------------------------------------------------------------------
# Step E: Batch generation FROM DERIVATIVES with KV-cache (fast)
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def generate_from_DERIVATIVES_batch_kvcache(
    model: CachedGPTLikeModel,
    DERIVATIVE_smiles_list: list[str],
    max_len: int = 112,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: int | None = 64,
    device: torch.device = device
):
    """
    Fast batched generation conditioned on per-sequence DERIVATIVE prefixes.
    We *force* the model to follow each given prefix by overriding the sampled token
    with the next prefix token until the prefix ends; after that, we sample/greedy.
    """
    # Tokenize DERIVATIVES and ensure SOS at start
    prefixes = [tokenize_DERIVATIVE(s) for s in DERIVATIVE_smiles_list]
    B = len(prefixes)
    T = max_len

    # outputs buffer
    outputs = torch.full((B, T), pad_idx, dtype=torch.long, device=device)

    # Fill position 0 with first token of each prefix (guaranteed SOS after tokenize_DERIVATIVE)
    for b, p in enumerate(prefixes):
        outputs[b, 0] = p[0]

    # Finished flags: if prefix already contains EOS or length >= max_len
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    prefix_lengths = torch.tensor([min(len(p), T) for p in prefixes], device=device)

    for b, p in enumerate(prefixes):
        # if EOS somewhere in prefix => truncate prefix at EOS and mark finished positions accordingly
        if eos_idx in p:
            epos = p.index(eos_idx)
            p[:] = p[:epos+1]
            prefix_lengths[b] = min(prefix_lengths[b], epos+1)

    # Preallocate KV caches: [L][B,H,T,Hd]
    H, Hd, L = model.nhead, model.head_dim, len(model.layers)
    k_caches = [torch.empty((B, H, T, Hd), dtype=DTYPE, device=device) for _ in range(L)]
    v_caches = [torch.empty((B, H, T, Hd), dtype=DTYPE, device=device) for _ in range(L)]

    # Main autoregressive loop (pos = 0..T-2; we write next token at pos+1)
    for pos in range(0, T-1):
        last_tok = outputs[:, pos]  # [B]
        logits = model.step(last_tok, pos, k_caches, v_caches)  # [B, V]

        # Decide next tokens
        next_tokens = None

        # If we're still inside each sequence's prefix, FORCE the next token to equal prefix[pos+1]
        # Build a mask of which rows have a prefix token at pos+1
        pos1 = pos + 1
        has_forced = (prefix_lengths > pos1)

        if has_forced.any():
            # Start from logits -> (optional) top-k -> (optional) sampling/greedy
            cand = logits
            if top_k:
                _top_k_mask_(cand, top_k)
            if do_sample:
                if temperature != 1.0:
                    cand = cand / temperature
                # Gumbel-max
                u = torch.rand_like(cand).clamp_(1e-6, 1-1e-6)
                g = -torch.log(-torch.log(u))
                sampled = torch.argmax(cand + g, dim=-1)
            else:
                sampled = torch.argmax(cand, dim=-1)

            # Now override with forced prefix tokens where available
            next_tokens = sampled
            forced_ids = torch.tensor(
                [prefixes[b][pos1] if has_forced[b].item() else 0 for b in range(B)],
                dtype=torch.long, device=device
            )
            next_tokens = torch.where(has_forced, forced_ids, next_tokens)
        else:
            # No forced prefix anywhere; normal sampling/greedy
            cand = logits
            if top_k:
                _top_k_mask_(cand, top_k)
            if do_sample:
                if temperature != 1.0:
                    cand = cand / temperature
                u = torch.rand_like(cand).clamp_(1e-6, 1-1e-6)
                g = -torch.log(-torch.log(u))
                next_tokens = torch.argmax(cand + g, dim=-1)
            else:
                next_tokens = torch.argmax(cand, dim=-1)

        # Respect finished rows: once EOS emitted (either by prefix or sampling), stick to EOS
        next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_idx), next_tokens)

        outputs[:, pos+1] = next_tokens
        finished |= (next_tokens == eos_idx)

        # If all finished, break
        if finished.all():
            break

    return outputs  # [B, <=T] (EOS may appear earlier per row)

# --------------------------------------------------------------------------------------
# Decoding: tokens -> SMILES strings (fast, per-token map, trims at EOS)
# --------------------------------------------------------------------------------------
def decode_batch(token_batch: torch.Tensor) -> list[str]:
    arr = token_batch.detach().cpu().numpy()  # [B, T]
    B, T = arr.shape
    out = []
    for b in tqdm(range(B)):
        seq = arr[b]
        # trim at first EOS if present
        eos_pos = np.where(seq == eos_idx)[0]
        end = int(eos_pos[0] + 1) if eos_pos.size > 0 else T
        # build string quickly
        s_list = []
        append = s_list.append
        for t in seq[:end]:
            append(idx_to_piece[t])
        out.append(''.join(s_list))
    return out

# --------------------------------------------------------------------------------------
# Step F: Example usage — DERIVATIVE generation + streaming write
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example DERIVATIVE SMILES (may or may not include 'G'; EOS in DERIVATIVE is respected)
    DERIVATIVES = ['N1(C(COC)=O)C(C)CN(CC1)C', 'N1(C(=O)COC)C(CN(C)CC1)C', 'N1(C)CCN(C(C1)C)C(=O)COC']*33
    '''
    DERIVATIVES = ['N1(C(COC)=O)C(C)CN(CC1)C',
 'N1(C(=O)COC)C(CN(C)CC1)C',
 'N1(C)CCN(C(C1)C)C(=O)COC',
 'C1N(C)CC(N(C(COC)=O)C1)C',
 'C1N(C)CCN(C(=O)COC)C1C',
 'C1(C)CN(C)CCN1C(=O)COC',
 'O=C(COC)N1C(C)CN(C)CC1',
 'C1C(N(CCN1C)C(=O)COC)C',
 'CN1CC(N(CC1)C(=O)COC)C',
 'C1CN(CC(C)N1C(=O)COC)C',
 'C1N(C(C)CN(C1)C)C(COC)=O',
 'C1C(C)N(CCN1C)C(=O)COC',
 'COCC(=O)N1CCN(CC1C)C',
 'O=C(COC)N1C(CN(C)CC1)C',
 'CN1CC(C)N(CC1)C(=O)COC',
 'C1N(C(C)CN(C)C1)C(=O)COC',
 'N1(C(=O)COC)CCN(CC1C)C',
 'C1N(C)CC(C)N(C(COC)=O)C1',
 'N1(CCN(C)CC1C)C(=O)COC',
 'N1(C)CC(C)N(CC1)C(=O)COC',
 'C1(N(CCN(C1)C)C(COC)=O)C',
 'C1C(N(C(=O)COC)CCN1C)C',
 'O(CC(N1C(C)CN(CC1)C)=O)C',
 'C(N1CCN(CC1C)C)(COC)=O',
 'C1N(C(CN(C1)C)C)C(=O)COC',
 'C(OC)C(=O)N1CCN(CC1C)C',
 'C(COC)(=O)N1CCN(C)CC1C',
 'N1(CCN(C(=O)COC)C(C1)C)C',
 'N1(C(C)CN(CC1)C)C(COC)=O',
 'C1N(C(CN(C)C1)C)C(=O)COC',
 'N1(CCN(CC1C)C)C(=O)COC',
 'CC1CN(C)CCN1C(=O)COC',
 'O=C(N1CCN(CC1C)C)COC',
 'C1N(C)CC(N(C1)C(COC)=O)C',
 'C1CN(C)CC(C)N1C(COC)=O',
 'N1(C)CCN(C(=O)COC)C(C1)C',
 'CC1CN(CCN1C(COC)=O)C',
 'COCC(N1CCN(C)CC1C)=O',
 'COCC(=O)N1C(C)CN(CC1)C',
 'CC1N(C(COC)=O)CCN(C)C1',
 'O(C)CC(=O)N1CCN(CC1C)C',
 'CN1CC(C)N(CC1)C(COC)=O',
 'N1(C(CN(C)CC1)C)C(=O)COC',
 'COCC(N1C(CN(CC1)C)C)=O',
 'N1(C)CC(C)N(C(=O)COC)CC1',
 'O(CC(=O)N1C(C)CN(CC1)C)C',
 'CC1N(CCN(C1)C)C(COC)=O',
 'C1CN(C)CC(N1C(COC)=O)C',
 'C1CN(C(CN1C)C)C(=O)COC',
 'COCC(=O)N1CCN(C)CC1C',
 'C(=O)(N1C(C)CN(C)CC1)COC',
 'C1(N(CCN(C)C1)C(COC)=O)C',
 'C(=O)(COC)N1CCN(CC1C)C',
 'O(CC(N1C(CN(C)CC1)C)=O)C',
 'O=C(N1CCN(C)CC1C)COC',
 'C1N(CCN(C1C)C(COC)=O)C',
 'N1(C(CN(CC1)C)C)C(COC)=O',
 'CC1N(CCN(C)C1)C(=O)COC',
 'C1N(C)CC(C)N(C1)C(COC)=O',
 'O=C(N1C(C)CN(CC1)C)COC',
 'C1C(C)N(CCN1C)C(COC)=O',
 'COCC(N1CCN(CC1C)C)=O',
 'C1C(C)N(C(=O)COC)CCN1C',
 'O(CC(=O)N1C(CN(C)CC1)C)C',
 'CN1CCN(C(C1)C)C(=O)COC',
 'C1(CN(C)CCN1C(=O)COC)C',
 'N1(CCN(C(=O)COC)C(C)C1)C',
 'N1(CCN(C)CC1C)C(COC)=O',
 'C1(CN(C)CCN1C(COC)=O)C',
 'CC1N(C(=O)COC)CCN(C1)C',
 'N1(C(CN(C)CC1)C)C(COC)=O',
 'N1(C)CC(N(C(COC)=O)CC1)C',
 'C(COC)(N1C(CN(C)CC1)C)=O',
 'N1(CCN(CC1C)C)C(COC)=O',
 'C1(C)N(C(=O)COC)CCN(C1)C',
 'C1(C)CN(C)CCN1C(COC)=O',
 'C(OC)C(N1CCN(C)CC1C)=O',
 'C(OC)C(N1CCN(CC1C)C)=O',
 'CN1CCN(C(COC)=O)C(C)C1',
 'C1(N(CCN(C1)C)C(=O)COC)C',
 'C1N(C(C)CN(C1)C)C(=O)COC',
 'C1N(C(COC)=O)C(CN(C)C1)C',
 'N1(CC(C)N(CC1)C(=O)COC)C',
 'COCC(N1C(C)CN(C)CC1)=O',
 'C1(C)CN(CCN1C(COC)=O)C',
 'C(OC)C(=O)N1CCN(C)CC1C',
 'C1N(C(=O)COC)C(C)CN(C)C1',
 'C1CN(CC(N1C(=O)COC)C)C',
 'C(C(=O)N1C(CN(CC1)C)C)OC',
 'C1CN(C(=O)COC)C(C)CN1C',
 'COCC(=O)N1C(C)CN(C)CC1',
 'C1N(CCN(C(COC)=O)C1C)C',
 'C(C(=O)N1CCN(C)CC1C)OC',
 'O=C(COC)N1CCN(C)CC1C',
 'N1(C)CCN(C(C)C1)C(=O)COC',
 'C(C(N1C(C)CN(CC1)C)=O)OC',
 'C(N1C(CN(CC1)C)C)(=O)COC',
 'O=C(COC)N1CCN(CC1C)C',
 'N1(C(=O)COC)CCN(C)CC1C',
 'C(=O)(N1CCN(C)CC1C)COC']
    '''

    # Build a batch of repeated DERIVATIVES to exercise throughput
    repeats = math.ceil(config.batch_size / len(DERIVATIVES))
    DERIVATIVES = (DERIVATIVES * repeats)[:config.batch_size]

    print(f"Generating from {len(DERIVATIVES)} DERIVATIVES...")

    t0 = perf_counter()
    tokens = generate_from_DERIVATIVES_batch_kvcache(
        model=model,
        DERIVATIVE_smiles_list=DERIVATIVES,
        max_len=config.seq_len,
        temperature=config.temperature,
        do_sample=config.do_sample,
        top_k=config.top_k,
        device=device
    )
    gen = decode_batch(tokens)
    dt = perf_counter() - t0
    print(f"Generated {len(gen)} molecules in {dt:.2f}s  ({len(gen)/max(dt,1e-6):.1f} SMILES/s)")

    # Streaming write (append)
    with open(config.out_path, "a", encoding="utf-8") as fh:
        fh.write('\n'.join(gen))
        fh.write('\n')
    print(f"Appended {len(gen)} lines to {config.out_path}")


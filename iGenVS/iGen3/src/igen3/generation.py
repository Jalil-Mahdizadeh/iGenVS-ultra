"""Generation loops shared by all iGen3 model variants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

from rdkit import Chem, RDLogger
import torch
from tqdm import tqdm

from .model import LoadedGenerator

RDLogger.DisableLog("rdApp.*")


@dataclass
class GenerationStats:
    requested: int
    generated: int
    candidates_generated: int
    seconds: float
    smiles_per_second: float
    candidate_smiles_per_second: float
    batch_size: int
    output_path: Path
    batch_trace: list[dict[str, float | int]]
    stopped_reason: str


def canonicalize_valid_smiles(smiles: str, *, isomeric_smiles: bool) -> str | None:
    smiles = smiles.strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric_smiles)


class ValidUniqueSmilesWriter:
    def __init__(
        self,
        *,
        fh,
        isomeric_smiles: bool,
        exclude_smiles: Iterable[str] | None = None,
    ) -> None:
        self.fh = fh
        self.isomeric_smiles = isomeric_smiles
        self.seen: set[str] = set()
        if exclude_smiles:
            for smiles in exclude_smiles:
                canonical = canonicalize_valid_smiles(smiles, isomeric_smiles=isomeric_smiles)
                if canonical:
                    self.seen.add(canonical)

    def accept_and_write(self, smiles: Iterable[str], *, limit: int) -> int:
        accepted: list[str] = []
        for smi in smiles:
            if len(accepted) >= limit:
                break
            canonical = canonicalize_valid_smiles(smi, isomeric_smiles=self.isomeric_smiles)
            if canonical is None or canonical in self.seen:
                continue
            self.seen.add(canonical)
            accepted.append(canonical)

        if accepted:
            self.fh.write("\n".join(accepted))
            self.fh.write("\n")
        return len(accepted)


@torch.no_grad()
def top_k_mask_(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    if not k or k >= logits.size(-1):
        return logits
    kth_vals = torch.topk(logits, int(k)).values[:, -1].unsqueeze(1)
    logits.masked_fill_(logits < kth_vals, torch.finfo(logits.dtype).min)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    do_sample: bool,
    top_k: int | None,
) -> torch.Tensor:
    if top_k:
        top_k_mask_(logits, top_k)
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    if temperature <= 0:
        raise ValueError("temperature must be > 0 when sampling")
    if temperature != 1.0:
        logits = logits / temperature
    gumbel_noise = torch.empty_like(logits).exponential_().log_().neg_()
    return torch.argmax(logits + gumbel_noise, dim=-1)


@torch.inference_mode()
def generate_de_novo_batch(
    generator: LoadedGenerator,
    batch_size: int,
    *,
    max_len: int | None = None,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: int | None = 64,
) -> torch.Tensor:
    vocab = generator.vocab
    device = generator.device
    max_len = max_len or generator.spec.seq_len

    outputs = torch.full((batch_size, max_len), vocab.pad_idx, dtype=torch.long, device=device)
    outputs[:, 0] = vocab.sos_idx
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    k_caches, v_caches = generator.allocate_caches(batch_size, max_len)

    for pos in range(max_len - 1):
        logits = generator.model.step(outputs[:, pos], pos, k_caches, v_caches)
        next_tokens = sample_next_token(logits, temperature=temperature, do_sample=do_sample, top_k=top_k)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, vocab.eos_idx), next_tokens)
        outputs[:, pos + 1] = next_tokens
        finished.logical_or_(next_tokens == vocab.eos_idx)
        if bool(finished.all()):
            break

    return outputs


def prepare_prefix_tensor(generator: LoadedGenerator, seed_smiles: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    vocab = generator.vocab
    prefixes = [vocab.encode_smiles(smi, add_sos=True, strict=True)[:max_len] for smi in seed_smiles]
    prefix_tensor = torch.full((len(prefixes), max_len), vocab.pad_idx, dtype=torch.long, device=generator.device)
    prefix_lengths = torch.empty(len(prefixes), dtype=torch.long, device=generator.device)

    for row_index, prefix in enumerate(prefixes):
        if vocab.eos_idx in prefix:
            prefix = prefix[: prefix.index(vocab.eos_idx) + 1]
        length = min(len(prefix), max_len)
        prefix_lengths[row_index] = length
        prefix_tensor[row_index, :length] = torch.tensor(prefix[:length], dtype=torch.long, device=generator.device)

    return prefix_tensor, prefix_lengths


@torch.inference_mode()
def generate_derivative_batch(
    generator: LoadedGenerator,
    seed_smiles: list[str],
    *,
    max_len: int | None = None,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_k: int | None = 64,
) -> torch.Tensor:
    if not seed_smiles:
        raise ValueError("seed_smiles must contain at least one seed")

    vocab = generator.vocab
    device = generator.device
    max_len = max_len or generator.spec.seq_len
    batch_size = len(seed_smiles)
    prefix_tensor, prefix_lengths = prepare_prefix_tensor(generator, seed_smiles, max_len)

    outputs = torch.full((batch_size, max_len), vocab.pad_idx, dtype=torch.long, device=device)
    outputs[:, 0] = prefix_tensor[:, 0]
    finished = outputs[:, 0] == vocab.eos_idx
    k_caches, v_caches = generator.allocate_caches(batch_size, max_len)

    for pos in range(max_len - 1):
        logits = generator.model.step(outputs[:, pos], pos, k_caches, v_caches)
        next_pos = pos + 1
        forced_mask = prefix_lengths > next_pos

        if bool(forced_mask.all()):
            next_tokens = prefix_tensor[:, next_pos]
        else:
            sampled = sample_next_token(logits, temperature=temperature, do_sample=do_sample, top_k=top_k)
            next_tokens = torch.where(forced_mask, prefix_tensor[:, next_pos], sampled)

        next_tokens = torch.where(finished, torch.full_like(next_tokens, vocab.eos_idx), next_tokens)
        outputs[:, next_pos] = next_tokens
        finished.logical_or_(next_tokens == vocab.eos_idx)
        if bool(finished.all()):
            break

    return outputs


def decode_token_batch(generator: LoadedGenerator, token_batch: torch.Tensor) -> list[str]:
    arr = token_batch.detach().cpu().numpy()
    return generator.vocab.decode_rows(arr)


def write_token_batch(generator: LoadedGenerator, token_batch: torch.Tensor, fh) -> None:
    smiles = decode_token_batch(generator, token_batch)
    fh.write("\n".join(smiles))
    fh.write("\n")


def estimate_batch_upper_bound(generator: LoadedGenerator, max_batch_size: int, safety_fraction: float = 0.50) -> int:
    if generator.device.type != "cuda":
        return min(32, max_batch_size)
    free_bytes, _ = torch.cuda.mem_get_info(generator.device)
    bytes_per_value = torch.empty((), dtype=generator.dtype).element_size()
    kv_bytes_per_sequence = (
        2
        * generator.num_layers
        * generator.nhead
        * generator.spec.seq_len
        * generator.head_dim
        * bytes_per_value
    )
    if kv_bytes_per_sequence <= 0:
        return min(1, max_batch_size)
    estimated = int((free_bytes * safety_fraction) // kv_bytes_per_sequence)
    return max(1, min(max_batch_size, estimated))


def autotune_batch_size(generator: LoadedGenerator, *, max_batch_size: int = 32768, min_batch_size: int = 1) -> int:
    """Binary-search the largest batch size that fits a one-step cache allocation."""
    if generator.device.type != "cuda":
        return min(32, max_batch_size)

    upper = estimate_batch_upper_bound(generator, max_batch_size)
    lower = min_batch_size
    best = 0

    while lower <= upper:
        candidate = (lower + upper) // 2
        try:
            k_caches, v_caches = generator.allocate_caches(candidate, generator.spec.seq_len)
            last_tok = torch.full((candidate,), generator.vocab.sos_idx, dtype=torch.long, device=generator.device)
            _ = generator.model.step(last_tok, 0, k_caches, v_caches)
            del k_caches, v_caches, last_tok
            torch.cuda.empty_cache()
            best = candidate
            lower = candidate + 1
        except RuntimeError as exc:
            message = str(exc).lower()
            allocation_failure = (
                "out of memory" in message
                or "cudacachingallocator" in message
                or "internal assert failed" in message
            )
            if not allocation_failure:
                raise
            torch.cuda.empty_cache()
            upper = candidate - 1

    if best < min_batch_size:
        raise RuntimeError("Could not find a CUDA batch size that fits memory")
    return best


def decode_smiles_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _candidate_limit(count: int, max_candidates: int | None, max_candidate_multiplier: float) -> int:
    if count <= 0:
        raise ValueError("count must be positive")
    if max_candidates is not None:
        if max_candidates < count:
            raise ValueError("max_candidates must be at least count")
        return max_candidates
    if max_candidate_multiplier < 1:
        raise ValueError("max_candidate_multiplier must be >= 1")
    return int(math.ceil(count * max_candidate_multiplier))


def _default_stagnation_limit(batch_size: int, stagnation_limit: int | None) -> int | None:
    if stagnation_limit is not None and stagnation_limit <= 0:
        return None
    return stagnation_limit if stagnation_limit is not None else max(10_000, batch_size * 10)


def write_de_novo_file(
    generator: LoadedGenerator,
    *,
    output_path: Path,
    count: int,
    batch_size: int,
    temperature: float,
    do_sample: bool,
    top_k: int | None,
    max_candidates: int | None = None,
    max_candidate_multiplier: float = 50.0,
    stagnation_limit: int | None = None,
    progress: bool = True,
) -> GenerationStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_limit = _candidate_limit(count, max_candidates, max_candidate_multiplier)
    stagnation_limit = _default_stagnation_limit(batch_size, stagnation_limit)
    accepted = 0
    candidates_generated = 0
    stalled_candidates = 0
    batch_index = 0
    stopped_reason = "target"
    start = perf_counter()

    iterator = None
    if progress:
        iterator = tqdm(total=count, desc=f"{generator.spec.model_id} de novo", unit="SMILES")

    with output_path.open("w", encoding="utf-8") as fh:
        writer = ValidUniqueSmilesWriter(fh=fh, isomeric_smiles=generator.spec.is_isomeric)
        batch_trace: list[dict[str, float | int]] = []
        while accepted < count and candidates_generated < candidate_limit:
            batch_index += 1
            current_bs = min(batch_size, candidate_limit - candidates_generated)
            if current_bs <= 0:
                break
            tokens = generate_de_novo_batch(
                generator,
                current_bs,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
            )
            smiles = decode_token_batch(generator, tokens)
            new_accepted = writer.accept_and_write(smiles, limit=count - accepted)
            accepted += new_accepted
            candidates_generated += current_bs
            stalled_candidates = 0 if new_accepted else stalled_candidates + current_bs
            elapsed = perf_counter() - start
            batch_trace.append(
                {
                    "batch_index": batch_index,
                    "generated": accepted,
                    "accepted": accepted,
                    "candidates_generated": candidates_generated,
                    "elapsed_seconds": elapsed,
                    "smiles_per_second": accepted / max(elapsed, 1e-9),
                    "candidate_smiles_per_second": candidates_generated / max(elapsed, 1e-9),
                }
            )
            if iterator is not None:
                iterator.update(new_accepted)
                iterator.set_postfix(
                    {
                        "candidates": f"{candidates_generated:,}",
                        "SMILES/s": f"{accepted / max(elapsed, 1e-9):.1f}",
                    }
                )
            if stagnation_limit is not None and stalled_candidates >= stagnation_limit:
                stopped_reason = "stagnation"
                break

    if accepted < count and stopped_reason == "target":
        stopped_reason = "max-candidates"
    if iterator is not None:
        iterator.close()

    seconds = perf_counter() - start
    return GenerationStats(
        requested=count,
        generated=accepted,
        candidates_generated=candidates_generated,
        seconds=seconds,
        smiles_per_second=accepted / max(seconds, 1e-9),
        candidate_smiles_per_second=candidates_generated / max(seconds, 1e-9),
        batch_size=batch_size,
        output_path=output_path,
        batch_trace=batch_trace,
        stopped_reason=stopped_reason,
    )


def expand_derivative_seed_schedule(seeds: list[str], samples_per_seed: int) -> list[str]:
    if not seeds:
        raise ValueError("seeds must contain at least one SMILES")
    if samples_per_seed <= 0:
        raise ValueError("samples_per_seed must be positive")
    return [seed for seed in seeds for _ in range(samples_per_seed)]


def _cyclic_seed_batch(seed_schedule: list[str], start: int, batch_size: int) -> list[str]:
    schedule_size = len(seed_schedule)
    return [seed_schedule[(start + offset) % schedule_size] for offset in range(batch_size)]


def write_derivative_file(
    generator: LoadedGenerator,
    *,
    seeds: list[str],
    output_path: Path,
    batch_size: int,
    samples_per_seed: int,
    count: int,
    temperature: float,
    do_sample: bool,
    top_k: int | None,
    exclude_seed_molecules: bool = True,
    max_candidates: int | None = None,
    max_candidate_multiplier: float = 50.0,
    stagnation_limit: int | None = None,
    progress: bool = True,
) -> GenerationStats:
    seed_schedule = expand_derivative_seed_schedule(seeds, samples_per_seed)
    candidate_limit = _candidate_limit(count, max_candidates, max_candidate_multiplier)
    stagnation_limit = _default_stagnation_limit(batch_size, stagnation_limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    candidates_generated = 0
    stalled_candidates = 0
    batch_index = 0
    seed_cursor = 0
    stopped_reason = "target"
    start = perf_counter()

    iterator = None
    if progress:
        iterator = tqdm(total=count, desc=f"{generator.spec.model_id} derivative", unit="SMILES")

    with output_path.open("w", encoding="utf-8") as fh:
        writer = ValidUniqueSmilesWriter(
            fh=fh,
            isomeric_smiles=generator.spec.is_isomeric,
            exclude_smiles=seeds if exclude_seed_molecules else None,
        )
        batch_trace: list[dict[str, float | int]] = []
        while accepted < count and candidates_generated < candidate_limit:
            batch_index += 1
            current_bs = min(batch_size, candidate_limit - candidates_generated)
            if current_bs <= 0:
                break
            batch = _cyclic_seed_batch(seed_schedule, seed_cursor, current_bs)
            seed_cursor += current_bs
            tokens = generate_derivative_batch(
                generator,
                batch,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
            )
            smiles = decode_token_batch(generator, tokens)
            new_accepted = writer.accept_and_write(smiles, limit=count - accepted)
            accepted += new_accepted
            candidates_generated += current_bs
            stalled_candidates = 0 if new_accepted else stalled_candidates + current_bs
            elapsed = perf_counter() - start
            batch_trace.append(
                {
                    "batch_index": batch_index,
                    "generated": accepted,
                    "accepted": accepted,
                    "candidates_generated": candidates_generated,
                    "elapsed_seconds": elapsed,
                    "smiles_per_second": accepted / max(elapsed, 1e-9),
                    "candidate_smiles_per_second": candidates_generated / max(elapsed, 1e-9),
                }
            )
            if iterator is not None:
                iterator.update(new_accepted)
                iterator.set_postfix(
                    {
                        "candidates": f"{candidates_generated:,}",
                        "SMILES/s": f"{accepted / max(elapsed, 1e-9):.1f}",
                    }
                )
            if stagnation_limit is not None and stalled_candidates >= stagnation_limit:
                stopped_reason = "stagnation"
                break

    if accepted < count and stopped_reason == "target":
        stopped_reason = "max-candidates"
    if iterator is not None:
        iterator.close()

    seconds = perf_counter() - start
    return GenerationStats(
        requested=count,
        generated=accepted,
        candidates_generated=candidates_generated,
        seconds=seconds,
        smiles_per_second=accepted / max(seconds, 1e-9),
        candidate_smiles_per_second=candidates_generated / max(seconds, 1e-9),
        batch_size=batch_size,
        output_path=output_path,
        batch_trace=batch_trace,
        stopped_reason=stopped_reason,
    )


def line_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def quick_decode_preview(generator: LoadedGenerator, token_batch: torch.Tensor, limit: int = 5) -> list[str]:
    arr = token_batch.detach().cpu().numpy()
    return generator.vocab.decode_rows(arr[:limit])


def unique_fraction(smiles: list[str]) -> float:
    if not smiles:
        return 0.0
    return len(set(smiles)) / len(smiles)

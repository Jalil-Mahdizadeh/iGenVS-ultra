"""Opt-in real-model CUDA portability check; run in the dedicated iGenVS SIF."""
import os
from pathlib import Path

import pytest
import torch

from igen3.generation import generate_de_novo_batch
from igenvs_rl.policy import load_policy_bundle, sample_policy


@pytest.mark.skipif(os.environ.get("IGENVS_RL_CUDA_TEST") != "1", reason="set IGENVS_RL_CUDA_TEST=1 in the iGenVS SIF with --nv")
def test_real_4096_draws_under_two_gib_allocator_budget():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    budget = 2 * 1024**3
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    try:
        bundle = load_policy_bundle(model_root=Path(__file__).resolve().parents[2] / "iGenVS/iGen3/models")
        torch.manual_seed(20260909)
        before = torch.cuda.get_rng_state(device)
        torch.cuda.reset_peak_memory_stats(device)
        tokens, smiles = sample_policy(bundle, 4096, temperature=1.0, top_k=64)
        after = torch.cuda.get_rng_state(device)
        peak = torch.cuda.max_memory_reserved(device)
        assert len(tokens) == len(smiles) == 4096
        assert peak <= budget
        # When enough physical memory exists, compare with the original sampler
        # using exactly the same seed, weights, precision, and 4,096 draws.
        torch.cuda.set_per_process_memory_fraction(1.0, device)
        if total >= 12 * 1024**3:
            torch.cuda.set_rng_state(before, device)
            expected = generate_de_novo_batch(bundle.sampler, 4096, temperature=1.0, top_k=64)
            assert torch.equal(after, torch.cuda.get_rng_state(device))
            assert torch.equal(tokens, expected)
        print(f"sampling draws=4096; allocator budget={budget}; peak reserved={peak}; full-batch RNG preserved")
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, device)

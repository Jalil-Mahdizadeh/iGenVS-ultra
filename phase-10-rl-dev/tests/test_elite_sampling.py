from __future__ import annotations

from types import SimpleNamespace

import torch

from igenvs_rl.oracle import FakeOracle
from igenvs_rl import trainer


def test_igen3_lipinski_gate_rejects_large_hydrocarbon() -> None:
    assert trainer._passes_igen3_lipinski("CCO")
    assert not trainer._passes_igen3_lipinski("C" * 40)


def test_candidate_chemistry_gate_combines_qed_and_formal_charge() -> None:
    _, qed, charge, csp3, aromatic_rings, passed = trainer._candidate_properties(
        "CCO",
        minimum_qed=0.0,
        maximum_absolute_formal_charge=1,
    )
    assert charge == 0
    assert csp3 == 1.0
    assert aromatic_rings == 0
    assert passed
    assert not trainer._candidate_properties(
        "CCO",
        minimum_qed=qed + 0.01,
        maximum_absolute_formal_charge=1,
    )[-1]
    assert not trainer._candidate_properties(
        "[NH3+]CC[NH3+]",
        minimum_qed=0.0,
        maximum_absolute_formal_charge=1,
    )[-1]
    assert not trainer._candidate_properties(
        "c1ccccc1",
        minimum_qed=0.0,
        maximum_absolute_formal_charge=1,
        minimum_fraction_csp3=0.05,
    )[-1]
    assert not trainer._candidate_properties(
        "c1ccccc1-c1ccccc1",
        minimum_qed=0.0,
        maximum_absolute_formal_charge=1,
        maximum_aromatic_rings=1,
    )[-1]


def test_repeated_elite_occurrences_share_one_cached_docking(tmp_path, monkeypatch) -> None:
    raw_smiles = ["CC", "CC", "O"]
    monkeypatch.setattr(
        trainer,
        "sample_policy",
        lambda bundle, count, temperature, top_k: (
            torch.zeros((count, 2), dtype=torch.long),
            raw_smiles,
        ),
    )
    bundle = SimpleNamespace(device=torch.device("cpu"))
    cache: dict[str, tuple[str, float]] = {}
    common = {
        "bundle": bundle,
        "oracle": FakeOracle(),
        "reference_scores": [-0.35, -0.30, -0.20],
        "count": 3,
        "job_dir": tmp_path,
        "temperature": 1.0,
        "top_k": 64,
        "previously_seen": set(),
        "score_cache": cache,
        "reward_mode": "elite",
        "tail_fraction": 0.1,
        "tail_weight": 1.0,
        "reward_seen_molecules": True,
        "elite_fraction": 0.01,
        "require_lipinski": False,
    }

    _, rewards, mask, rows, metrics = trainer._sample_and_score(
        **common,
        output_dir=tmp_path / "first",
    )
    assert rewards[0].item() == rewards[1].item() > 0.0
    assert mask.tolist() == [True, True, True]
    assert rows[1]["sample_status"] == "batch_repeat"
    assert metrics["oracle_unique_count"] == 2
    assert metrics["docked"] == 3
    assert metrics["elite_count"] == 2
    assert metrics["elite_unique_count"] == 1

    _, _, _, rows, metrics = trainer._sample_and_score(
        **common,
        output_dir=tmp_path / "second",
    )
    assert metrics["oracle_unique_count"] == 0
    assert metrics["cache_hit_count"] == 3
    assert all(row["score_source"] == "cache" for row in rows)
    assert len((tmp_path / "score-cache.csv").read_text().splitlines()) == 3

    _, _, mask, _, metrics = trainer._sample_and_score(
        **common,
        output_dir=tmp_path / "third",
        reward_occurrence_cap=1,
    )
    assert mask.tolist() == [True, False, True]
    assert metrics["gradient_capped_count"] == 1

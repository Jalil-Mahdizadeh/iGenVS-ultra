from __future__ import annotations

from pathlib import Path

from igenvs_rl.oracle import FakeOracle, IGenVSOracle, stable_molecule_id
from igenvs_rl import oracle


def test_fake_oracle_is_deterministic(tmp_path) -> None:
    molecules = {
        stable_molecule_id("CCO"): "CCO",
        stable_molecule_id("c1ccccc1"): "c1ccccc1",
    }
    first = FakeOracle().dock(molecules, tmp_path / "first")
    second = FakeOracle().dock(molecules, tmp_path / "second")
    assert {key: value.score for key, value in first.items()} == {
        key: value.score for key, value in second.items()
    }
    assert all(value.status == "success" for value in first.values())


def test_igenvs_oracle_command_accepts_effective_sparse_shard_count() -> None:
    oracle = IGenVSOracle(
        target=Path("target"),
        engine="unidock",
        scoring="vina",
        search_mode="fast",
        shards=4,
        prep_workers=16,
        validation_workers=8,
        seed=181129,
    )
    command = oracle._command(
        Path("input.csv"),
        Path("output"),
        1,
        shard_count=2,
    )
    assert command[command.index("--num-shards") + 1] == "2"
    assert command[command.index("--shard-index") + 1] == "1"


def test_oracle_honors_explicit_disable_masks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled visibility must not discover physical GPUs")
    monkeypatch.setattr(oracle.subprocess, "run", forbidden)
    for mask in ("", "-1", "NoDevFiles", "void"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        assert oracle._visible_gpu_tokens() == []

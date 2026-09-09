"""Opt-in real-weight smoke test with a tiny, explicitly synthetic oracle job."""
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from igenvs_rl import cli, trainer


@pytest.mark.skipif(os.environ.get("IGENVS_RL_REAL_MODEL_TEST") != "1", reason="set IGENVS_RL_REAL_MODEL_TEST=1 in the dedicated iGenVS SIF")
def test_real_weight_cli_checkpoint_recovery_is_idempotent(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    job = tmp_path / "synthetic-oracle-job"
    assert cli.main([
        "init", "--job-dir", str(job),
        "--target", str(repo / "phase-7-benchmark-docking/targets/1err"),
        "--model-root", str(repo / "iGenVS/iGen3/models"),
        "--oracle", "fake", "--reference-count", "4", "--batch-size", "8",
        "--evaluation-every", "0",
    ]) == 0
    assert cli.main(["train", "--job-dir", str(job), "--target-update", "1"]) == 0
    checkpoint = job / "checkpoints/latest.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    weights = job / "model-latest/base_isomeric/iGen3_base_isomeric_256d.pth"
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    weights.rename(weights.with_suffix(".interrupted"))
    trainer._publish_progress(job, 1, model_latest_exported=False)
    with (job / "history.csv").open("a") as handle:
        handle.write("partial")
    (job / "updates/update-0002").mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint repair/idempotent target must not sample")
    monkeypatch.setattr(trainer, "_sample_and_score", forbidden)
    assert cli.main(["recover", "--job-dir", str(job)]) == 0
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == weight_hash
    assert cli.main(["train", "--job-dir", str(job), "--target-update", "1"]) == 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    assert json.loads((job / "progress.json").read_text())["model_latest_exported"] is True
    with (job / "history.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_target.py"
SPEC = importlib.util.spec_from_file_location("rl_dev_run_target", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


RULE = {
    "consecutive_evaluations": 2,
    "minimum_qualified_elite_fraction": 0.9,
    "minimum_qualified_elite_unique": 200,
    "minimum_chemistry_fraction": 0.9,
    "maximum_top_molecule_fraction": 0.2,
    "maximum_positive_score_fraction": 0.01,
}


def _write(path: Path, rows: list[tuple[int, float, int, float, float, float]]) -> None:
    path.mkdir()
    fields = [
        "label",
        "qualified_elite_fraction",
        "qualified_elite_unique_count",
        "chemistry_fraction",
        "top_molecule_fraction",
        "positive_score_fraction",
    ]
    with (path / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for update, fraction, unique, chemistry, top, positive in rows:
            writer.writerow(
                {
                    "label": f"update-{update:04d}",
                    "qualified_elite_fraction": fraction,
                    "qualified_elite_unique_count": unique,
                    "chemistry_fraction": chemistry,
                    "top_molecule_fraction": top,
                    "positive_score_fraction": positive,
                }
            )


def test_two_complete_fresh_evaluations_pass(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    _write(
        stage,
        [
            (10, 0.91, 350, 0.95, 0.12, 0.002),
            (11, 0.93, 310, 0.96, 0.15, 0.001),
        ],
    )
    assert RUNNER._gate_check(stage, RULE, 11)["passed"]


def test_concentration_cannot_hide_collapse_or_bad_chemistry(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    _write(
        stage,
        [
            (10, 0.96, 300, 0.95, 0.25, 0.001),
            (11, 0.97, 300, 0.88, 0.10, 0.001),
        ],
    )
    assert not RUNNER._gate_check(stage, RULE, 11)["passed"]


@pytest.mark.parametrize("legacy,stopped", [(False, False), (False, True), (True, False), (True, True)])
def test_development_wrapper_recovers_before_stopping(tmp_path, monkeypatch, legacy, stopped):
    stage = json.loads((SCRIPT.parents[1] / "protocol.json").read_text())["stages"][2]
    (tmp_path / "history.csv").write_text("update,seconds\n10,1\n")
    if stopped:
        RUNNER._atomic_json(tmp_path / "stopping.json", {"rule": stage["adaptive_stopping"], "stopped_at_update": 10, "gate_met": True})
    calls = []
    def publish(update):
        RUNNER._atomic_json(tmp_path / "progress.json", {"status": "complete", "completed_updates": update, "model_latest_exported": True})
    def recover(*args):
        calls.append("recover")
        publish(0 if legacy else 10)
        return 0 if legacy else 10
    def train(**kwargs):
        calls.append("train")
        assert kwargs["requested_total"] == 10
        publish(10)
    monkeypatch.setattr(RUNNER, "_recover_stage", recover)
    monkeypatch.setattr(RUNNER, "_run_training_to", train)
    monkeypatch.setattr(RUNNER, "_gate_check", lambda *args: {"passed": True})
    result = RUNNER._run_adaptive_stage(
        wrapper=SCRIPT, stage_dir=tmp_path, stage=stage, timing_records=[], gpus=1,
        timing_path=tmp_path / "timing.json", target="fixture", stages=[stage],
    )
    assert result["gate_met"]
    assert calls == (["recover", "train"] if legacy else ["recover"])

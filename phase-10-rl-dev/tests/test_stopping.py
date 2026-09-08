from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


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

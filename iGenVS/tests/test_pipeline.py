from __future__ import annotations

import pytest

from igenvs.errors import ExternalToolError
from igenvs.pipeline import (
    _first_preparation_batch_size,
    _preparation_batches,
    _require_successful_docking,
    _result_rows,
    _failure_row,
)
from igenvs.records import DockingResult, PreparedLigand, PreparationFailure, ValidatedRecord


def _records(count: int):
    return iter(
        ValidatedRecord(str(index), "C", "C", index, 1, 0)
        for index in range(1, count + 1)
    )


def test_large_unidock_job_uses_ramp_then_steady_batch() -> None:
    first = _first_preparation_batch_size(
        engine="unidock",
        valid_records=20_000,
        batch_size=32_768,
        prep_workers=64,
        automatic=True,
    )
    assert first == 2_048
    batches = list(
        _preparation_batches(
            _records(20_000),
            batch_size=32_768,
            first_batch_size=first,
        )
    )
    assert [len(batch) for batch in batches] == [2_048, 17_952]
    assert [record.source_row for batch in batches for record in batch] == list(
        range(1, 20_001)
    )


def test_small_or_autodock_job_avoids_extra_ramp_invocation() -> None:
    assert (
        _first_preparation_batch_size(
            engine="unidock",
            valid_records=4_096,
            batch_size=32_768,
            prep_workers=64,
            automatic=True,
        )
        == 32_768
    )
    assert (
        _first_preparation_batch_size(
            engine="autodock-gpu",
            valid_records=20_000,
            batch_size=4_096,
            prep_workers=64,
            automatic=True,
        )
        == 4_096
    )
    assert (
        _first_preparation_batch_size(
            engine="unidock",
            valid_records=20_000,
            batch_size=32_768,
            prep_workers=64,
            automatic=False,
        )
        == 32_768
    )


def test_total_docking_engine_failure_is_not_success() -> None:
    with pytest.raises(ExternalToolError, match="zero successful ligands"):
        _require_successful_docking({"prepared": 3, "docked": 0})
    with pytest.raises(ExternalToolError, match="zero successful ligands"):
        _require_successful_docking(
            {"prepared": 0, "preparation_failed": 3, "docked": 0}
        )
    _require_successful_docking({"prepared": 3, "docked": 1})
    _require_successful_docking({"prepared": 0, "docked": 0})


def test_results_include_every_pose_score_and_one_row_per_failure(tmp_path) -> None:
    record = next(_records(1))
    ligand = PreparedLigand(record, tmp_path / "ligand.pdbqt", 1, 0, 0.1)
    result = DockingResult(ligand, "success", (-7.5, 2.0, 12.0), ligand.path, "", 0.5)
    rows = list(_result_rows(result, 1, "poses.pdbqt#1", engine="unidock", scoring="vina"))
    assert [row["pose_rank"] for row in rows] == [1, 2, 3]
    assert [row["docking_score"] for row in rows] == ["-7.500000", "2.000000", "12.000000"]
    assert all(row["num_poses"] == 3 and row["source_row"] == 1 for row in rows)
    assert all(row["pose_ref"] == "poses.pdbqt#1" for row in rows)
    failed = DockingResult(ligand, "invalid_output", (), None, "broken ring", 0.5)
    rows = list(_result_rows(failed, 1, "", engine="unidock", scoring="vina"))
    assert len(rows) == 1 and rows[0]["num_poses"] == 0
    assert rows[0]["pose_rank"] == rows[0]["docking_score"] == ""
    row = _failure_row(PreparationFailure(record, "preparation_failed", "embedding", 0.1),
                       1, engine="unidock", scoring="vina")
    assert row["pose_rank"] == row["docking_score"] == ""

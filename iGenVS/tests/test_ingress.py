from __future__ import annotations

from pathlib import Path

from igenvs.ingress import (
    _bounded_validation_results,
    inspect_prevalidated_library,
    iter_source_records,
    iter_validated_records,
    validate_library,
)
from igenvs.records import SourceRecord


def _write_library(path: Path) -> None:
    path.write_text(
        "ID,SMILES\n"
        "a,CCO\n"
        "b,OCC\n"
        "c,not_a_smiles\n"
        "d,C.C\n",
        encoding="utf-8",
    )


def test_csv_shards_are_complete_and_disjoint(tmp_path: Path) -> None:
    library = tmp_path / "library.csv"
    _write_library(library)
    shard_zero = list(iter_source_records(library, num_shards=2, shard_index=0))
    shard_one = list(iter_source_records(library, num_shards=2, shard_index=1))
    assert [record.molecule_id for record in shard_zero] == ["a", "c"]
    assert [record.molecule_id for record in shard_one] == ["b", "d"]
    assert {record.source_row for record in shard_zero}.isdisjoint(
        record.source_row for record in shard_one
    )


def test_validation_is_reproducible_on_rerun(tmp_path: Path) -> None:
    library = tmp_path / "library.csv"
    output = tmp_path / "validated"
    _write_library(library)

    first = validate_library(
        iter_source_records(library),
        output_dir=output,
        workers=1,
        fragment_policy="reject",
    )
    second = validate_library(
        iter_source_records(library),
        output_dir=output,
        workers=1,
        fragment_policy="reject",
    )
    expected = {"input": 4, "valid": 1, "rejected": 3, "duplicates": 1}
    assert {key: first[key] for key in expected} == expected
    assert {key: second[key] for key in expected} == expected


def test_validation_submission_is_windowed() -> None:
    class FakeExecutor:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def map(self, function, payloads, *, chunksize: int):
            values = list(payloads)
            self.calls.append((len(values), chunksize))
            return iter(values)

    payloads = [
        (SourceRecord(f"mol-{index}", "CCO", index), "reject")
        for index in range(5)
    ]
    executor = FakeExecutor()

    assert list(_bounded_validation_results(executor, payloads, window_size=2)) == payloads
    assert executor.calls == [(2, 64), (2, 64), (1, 64)]


def test_prevalidated_shards_are_complete_and_disjoint(tmp_path: Path) -> None:
    path = tmp_path / "validated.csv"
    path.write_text(
        "molecule_id,original_smiles,canonical_smiles,source_row,heavy_atoms,rotatable_bonds\n"
        "a,CC,CC,1,2,0\n"
        "b,CCC,CCC,2,3,0\n"
        "c,CO,CO,3,2,0\n",
        encoding="utf-8",
    )
    zero = list(iter_validated_records(path, num_shards=2, shard_index=0))
    one = list(iter_validated_records(path, num_shards=2, shard_index=1))
    assert [record.molecule_id for record in zero] == ["a", "c"]
    assert [record.molecule_id for record in one] == ["b"]
    assert inspect_prevalidated_library(path, num_shards=2, shard_index=0)["valid"] == 2

from __future__ import annotations

from igenvs.pipeline import _first_preparation_batch_size, _preparation_batches
from igenvs.records import ValidatedRecord


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

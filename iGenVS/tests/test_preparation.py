from __future__ import annotations

from time import perf_counter

from igenvs.preparation import (
    _prepare_one,
    ligand_basename,
    submit_preparation_batch,
)
from igenvs.records import PreparedLigand, PreparationFailure, ValidatedRecord


def _ethanol() -> ValidatedRecord:
    return ValidatedRecord("ethanol", "CCO", "CCO", 1, 3, 0)


def test_meeko_ligand_preparation(tmp_path) -> None:
    record = _ethanol()
    result = _prepare_one(record, tmp_path, 13, 300, 48)
    assert isinstance(result, PreparedLigand)
    assert result.path.is_file()
    assert result.atom_count > 0
    assert result.path.name == f"{ligand_basename(record)}.pdbqt"


def test_fast_preparation_mode(tmp_path) -> None:
    record = _ethanol()
    result = _prepare_one(record, tmp_path, 13, 300, 48, "fast")
    assert isinstance(result, PreparedLigand)
    assert result.path.is_file()
    assert result.atom_count > 0


def test_unidock_atom_limit_is_typed_failure(tmp_path) -> None:
    result = _prepare_one(_ethanol(), tmp_path, 13, 1, 48)
    assert isinstance(result, PreparationFailure)
    assert result.status == "unsupported_size"


def test_hard_molecule_embedding_has_a_native_time_budget(tmp_path) -> None:
    smiles = (
        "C[C@@H]1[C@H]2CC[C@@H](C)[C@@H]3CC[C@@H](C)[C@@H]4CC[C@](C)"
        "(OO[C@@]234)O[C@@H]1n1cc(SC(=O)c2ccccc2)nn1"
    )
    record = ValidatedRecord("hard", smiles, smiles, 1, 36, 3)
    started = perf_counter()
    result = _prepare_one(
        record,
        tmp_path,
        181129,
        300,
        48,
        embed_max_attempts=100_000,
        embed_timeout_seconds=1,
    )
    assert isinstance(result, PreparationFailure)
    assert result.status == "preparation_timeout"
    assert perf_counter() - started < 3.0


def test_complex_records_are_submitted_first_but_returned_in_source_order(tmp_path) -> None:
    class FakeExecutor:
        def __init__(self) -> None:
            self.submitted = []

        def submit(self, function, record, *args):
            self.submitted.append(record.molecule_id)
            return record.molecule_id

    records = [
        ValidatedRecord("simple", "CCCC", "CCCC", 1, 4, 1),
        ValidatedRecord("complex", "C[C@@H]1CC[C@@H]1O", "C[C@@H]1CC[C@@H]1O", 2, 7, 0),
    ]
    executor = FakeExecutor()
    futures = submit_preparation_batch(executor, records, output_dir=tmp_path, seed=1)
    assert executor.submitted == ["complex", "simple"]
    assert futures == ["simple", "complex"]

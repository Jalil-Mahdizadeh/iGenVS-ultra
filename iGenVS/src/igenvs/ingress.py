"""Streaming library ingestion and RDKit validation."""

from __future__ import annotations

import csv
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from .errors import InputError
from .hardware import auto_worker_count
from .records import SourceRecord, ValidatedRecord, ValidationFailure


VALIDATED_FIELDS = [
    "molecule_id",
    "original_smiles",
    "canonical_smiles",
    "source_row",
    "heavy_atoms",
    "rotatable_bonds",
]
REJECTED_FIELDS = ["molecule_id", "original_smiles", "source_row", "status", "error"]
VALIDATION_SUBMISSION_WINDOW = 8_192


def _delimiter_value(value: str | None, sample: str) -> str:
    if value and value.lower() != "auto":
        return "\t" if value == r"\t" else value
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def _casefold_column(fieldnames: list[str], requested: str | None, fallbacks: tuple[str, ...]) -> str | None:
    lookup = {name.casefold(): name for name in fieldnames}
    if requested:
        return lookup.get(requested.casefold())
    for candidate in fallbacks:
        if candidate in lookup:
            return lookup[candidate]
    return None


def iter_csv_records(
    path: Path,
    *,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    delimiter: str | None = "auto",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[SourceRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(65_536)
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=_delimiter_value(delimiter, sample))
        fields = list(reader.fieldnames or [])
        smi_key = _casefold_column(fields, smiles_column, ("smiles", "canonical_smiles"))
        if smi_key is None:
            raise InputError(f"SMILES column '{smiles_column}' not found in {path}; columns: {fields}")
        id_key = _casefold_column(fields, id_column, ("molecule_id", "id", "name"))
        if id_column and id_key is None:
            raise InputError(f"ID column '{id_column}' not found in {path}; columns: {fields}")
        for row_number, row in enumerate(reader, start=1):
            if (row_number - 1) % num_shards != shard_index:
                continue
            molecule_id = str(row.get(id_key, "") if id_key else "").strip()
            if not molecule_id:
                molecule_id = f"row_{row_number:012d}"
            yield SourceRecord(
                molecule_id=molecule_id,
                original_smiles=str(row.get(smi_key, "") or "").strip(),
                source_row=row_number,
            )


def iter_smi_records(
    path: Path,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[SourceRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if (row_number - 1) % num_shards != shard_index:
                continue
            fields = stripped.split()
            molecule_id = fields[1] if len(fields) > 1 else f"row_{row_number:012d}"
            yield SourceRecord(molecule_id=molecule_id, original_smiles=fields[0], source_row=row_number)


def iter_source_records(
    path: Path,
    *,
    input_format: str = "auto",
    smiles_column: str = "smiles",
    id_column: str | None = None,
    delimiter: str | None = "auto",
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[SourceRecord]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise InputError("shard_index must be in [0, num_shards)")
    if not path.is_file():
        raise InputError(f"input library does not exist: {path}")
    resolved_format = input_format
    if resolved_format == "auto":
        resolved_format = "csv" if path.suffix.lower() in {".csv", ".tsv"} else "smi"
    if resolved_format == "csv":
        yield from iter_csv_records(
            path,
            smiles_column=smiles_column,
            id_column=id_column,
            delimiter=delimiter,
            num_shards=num_shards,
            shard_index=shard_index,
        )
    elif resolved_format == "smi":
        yield from iter_smi_records(path, num_shards=num_shards, shard_index=shard_index)
    else:
        raise InputError(f"unsupported input format: {resolved_format}")


def _validate_one(payload: tuple[SourceRecord, str]) -> ValidatedRecord | ValidationFailure:
    record, fragment_policy = payload
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem.MolStandardize import rdMolStandardize

        if not record.original_smiles:
            return ValidationFailure(record.molecule_id, record.original_smiles, record.source_row, "invalid_smiles", "empty SMILES")
        mol = Chem.MolFromSmiles(record.original_smiles)
        if mol is None:
            return ValidationFailure(record.molecule_id, record.original_smiles, record.source_row, "invalid_smiles", "RDKit could not parse or sanitize SMILES")
        if len(Chem.GetMolFrags(mol)) > 1:
            if fragment_policy == "reject":
                return ValidationFailure(record.molecule_id, record.original_smiles, record.source_row, "multiple_fragments", "disconnected structures require --fragment-policy largest")
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        heavy_atoms = int(mol.GetNumHeavyAtoms())
        rotatable = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        if heavy_atoms == 0:
            return ValidationFailure(record.molecule_id, record.original_smiles, record.source_row, "invalid_smiles", "molecule has no heavy atoms")
        return ValidatedRecord(
            molecule_id=record.molecule_id,
            original_smiles=record.original_smiles,
            canonical_smiles=canonical,
            source_row=record.source_row,
            heavy_atoms=heavy_atoms,
            rotatable_bonds=rotatable,
        )
    except Exception as exc:
        return ValidationFailure(record.molecule_id, record.original_smiles, record.source_row, "validation_error", f"{type(exc).__name__}: {exc}")


def _dedup_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE IF NOT EXISTS ids (value TEXT PRIMARY KEY)")
    connection.execute("CREATE TABLE IF NOT EXISTS smiles (value TEXT PRIMARY KEY)")
    return connection


def _insert_unique(connection: sqlite3.Connection, table: str, value: str) -> bool:
    cursor = connection.execute(f"INSERT OR IGNORE INTO {table}(value) VALUES (?)", (value,))
    return cursor.rowcount == 1


def _bounded_validation_results(
    executor: ProcessPoolExecutor,
    payloads: Iterable[tuple[SourceRecord, str]],
    *,
    window_size: int = VALIDATION_SUBMISSION_WINDOW,
) -> Iterator[ValidatedRecord | ValidationFailure]:
    """Validate in windows because Executor.map eagerly queues inputs on Python 3.10."""
    if window_size <= 0:
        raise ValueError("validation submission window must be positive")
    iterator = iter(payloads)
    while window := list(islice(iterator, window_size)):
        yield from executor.map(_validate_one, window, chunksize=64)


def validate_library(
    records: Iterable[SourceRecord],
    *,
    output_dir: Path,
    workers: str | int = "auto",
    fragment_policy: str = "reject",
    deduplicate: bool = True,
) -> dict[str, int | str]:
    if fragment_policy not in {"reject", "largest"}:
        raise ValueError("fragment_policy must be 'reject' or 'largest'")
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_path = output_dir / "validated.csv"
    rejected_path = output_dir / "rejected.csv"
    database_path = output_dir / "dedup.sqlite3"
    database_path.unlink(missing_ok=True)
    connection = _dedup_connection(database_path)
    counts = {"input": 0, "valid": 0, "rejected": 0, "duplicates": 0}
    worker_count = auto_worker_count(workers)
    payloads = ((record, fragment_policy) for record in records)

    with valid_path.open("w", encoding="utf-8", newline="") as valid_handle, rejected_path.open(
        "w", encoding="utf-8", newline=""
    ) as rejected_handle:
        valid_writer = csv.DictWriter(valid_handle, fieldnames=VALIDATED_FIELDS)
        rejected_writer = csv.DictWriter(rejected_handle, fieldnames=REJECTED_FIELDS)
        valid_writer.writeheader()
        rejected_writer.writeheader()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for result in _bounded_validation_results(executor, payloads):
                counts["input"] += 1
                if isinstance(result, ValidationFailure):
                    rejected_writer.writerow(asdict(result))
                    counts["rejected"] += 1
                    continue
                if not _insert_unique(connection, "ids", result.molecule_id):
                    rejected_writer.writerow(
                        asdict(
                            ValidationFailure(
                                result.molecule_id,
                                result.original_smiles,
                                result.source_row,
                                "duplicate_id",
                                "molecule ID was already seen in this shard",
                            )
                        )
                    )
                    counts["rejected"] += 1
                    counts["duplicates"] += 1
                    continue
                if deduplicate and not _insert_unique(connection, "smiles", result.canonical_smiles):
                    rejected_writer.writerow(
                        asdict(
                            ValidationFailure(
                                result.molecule_id,
                                result.original_smiles,
                                result.source_row,
                                "duplicate_smiles",
                                "canonical isomeric SMILES was already seen in this shard",
                            )
                        )
                    )
                    counts["rejected"] += 1
                    counts["duplicates"] += 1
                    continue
                valid_writer.writerow(asdict(result))
                counts["valid"] += 1
                if counts["input"] % 10_000 == 0:
                    connection.commit()
    connection.commit()
    connection.close()
    return {
        **counts,
        "workers": worker_count,
        "validated_path": str(valid_path),
        "rejected_path": str(rejected_path),
        "dedup_database": str(database_path),
    }


def iter_validated_records(
    path: Path,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[ValidatedRecord]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise InputError("shard_index must be in [0, num_shards)")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(VALIDATED_FIELDS)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise InputError(
                f"prevalidated library is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            source_row = int(row["source_row"])
            if (source_row - 1) % num_shards != shard_index:
                continue
            yield ValidatedRecord(
                molecule_id=row["molecule_id"],
                original_smiles=row["original_smiles"],
                canonical_smiles=row["canonical_smiles"],
                source_row=source_row,
                heavy_atoms=int(row["heavy_atoms"]),
                rotatable_bonds=int(row["rotatable_bonds"]),
            )


def inspect_prevalidated_library(
    path: Path,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, int | str | bool]:
    """Verify and count a trusted validation artifact for one logical shard."""

    if not path.is_file():
        raise InputError(f"prevalidated library does not exist: {path}")
    count = sum(
        1
        for _ in iter_validated_records(
            path,
            num_shards=num_shards,
            shard_index=shard_index,
        )
    )
    return {
        "input": count,
        "valid": count,
        "rejected": 0,
        "duplicates": 0,
        "workers": 0,
        "validated_path": str(path.resolve()),
        "rejected_path": "",
        "dedup_database": "",
        "prevalidated": True,
    }


def chunked(records: Iterable[ValidatedRecord], size: int) -> Iterator[list[ValidatedRecord]]:
    batch: list[ValidatedRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .chem import CanonicalMolecule, canonicalize
from .util import atomic_write_json, atomic_write_jsonl, ensure_directory, read_jsonl_row, sha256_file


def _open_feather(path: str | Path):
    memory = pa.memory_map(str(path), "r")
    return memory, ipc.open_file(memory)


def _validate_arrow_schema(cfg: dict[str, Any], source: dict[str, Any], schema: pa.Schema) -> None:
    source_name = source.get("name", source.get("source", "unknown"))
    smiles_column = str(cfg["data"]["smiles_column"])
    descriptor_columns = [str(value) for value in cfg["data"]["descriptor_columns"]]
    expected = [smiles_column, *descriptor_columns]
    missing = [name for name in expected if schema.get_field_index(name) < 0]
    if missing:
        raise ValueError(f"{source_name} is missing Arrow columns: {missing}")
    if not (pa.types.is_string(schema.field(smiles_column).type) or pa.types.is_large_string(schema.field(smiles_column).type)):
        raise TypeError(f"{source_name} column {smiles_column!r} must be string")
    for name in descriptor_columns:
        dtype = schema.field(name).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise TypeError(f"{source_name} descriptor {name!r} is not numeric: {dtype}")


def prepare_tasks(cfg: dict[str, Any]) -> dict[str, Any]:
    work_dir = ensure_directory(cfg["paths"]["work_dir"])
    task_dir = ensure_directory(work_dir / "tasks")
    batches_per_task = int(cfg["data"]["record_batches_per_task"])
    tasks: list[dict[str, Any]] = []
    source_metadata = []
    task_id = 0
    for source in sorted(cfg["paths"]["sources"], key=lambda item: (int(item.get("priority", 0)), item["name"])):
        source_path = Path(source["path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        memory, reader = _open_feather(source_path)
        try:
            _validate_arrow_schema(cfg, source, reader.schema)
            num_batches = reader.num_record_batches
            source_metadata.append(
                {
                    "name": source["name"],
                    "path": str(source_path),
                    "priority": int(source.get("priority", 0)),
                    "expected_sha256": str(source.get("sha256", "")).lower() or None,
                    "size_bytes": source_path.stat().st_size,
                    "record_batches": num_batches,
                    "expected_rows": int(source["rows"]) if source.get("rows") is not None else None,
                    "arrow_schema": str(reader.schema),
                }
            )
            for start in range(0, num_batches, batches_per_task):
                tasks.append(
                    {
                        "task_id": task_id,
                        "source": source["name"],
                        "source_path": str(source_path),
                        "source_priority": int(source.get("priority", 0)),
                        "start_batch": start,
                        "end_batch": min(start + batches_per_task, num_batches),
                    }
                )
                task_id += 1
        finally:
            memory.close()
    task_path = task_dir / "canonicalize.jsonl"
    atomic_write_jsonl(task_path, tasks)
    result = {
        "schema_version": 1,
        "config_hash": cfg["_config_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "canonicalize_task_count": len(tasks),
        "hash_buckets": int(cfg["data"]["hash_buckets"]),
        "canonicalize_tasks": str(task_path),
        "sources": source_metadata,
    }
    atomic_write_json(work_dir / "prepared.json", result)
    return result


def verify_sources(cfg: dict[str, Any]) -> dict[str, Any]:
    results = []
    for source in cfg["paths"]["sources"]:
        expected = str(source.get("sha256", "")).lower()
        if not expected:
            raise ValueError(f"Source {source['name']} has no expected sha256 in the configuration")
        actual = sha256_file(source["path"])
        if actual.lower() != expected:
            raise ValueError(f"SHA-256 mismatch for {source['name']}: expected {expected}, got {actual}")
        results.append({"name": source["name"], "path": source["path"], "sha256": actual})
    result = {"schema_version": 1, "config_hash": cfg["_config_hash"], "sources": results}
    atomic_write_json(Path(cfg["paths"]["work_dir"]) / "verified_sources.json", result)
    return result


def _canonical_schema(descriptor_count: int) -> pa.Schema:
    fields = [
        pa.field("bucket", pa.int16(), nullable=False),
        pa.field("molecule_hash", pa.string(), nullable=False),
        pa.field("canonical_smiles", pa.string(), nullable=False),
        pa.field("nonisomeric_smiles", pa.string(), nullable=False),
        pa.field("scaffold", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_priority", pa.int16(), nullable=False),
        pa.field("source_batch", pa.int32(), nullable=False),
        pa.field("source_row", pa.int32(), nullable=False),
        pa.field("atom_count", pa.int16(), nullable=False),
        pa.field("bond_count", pa.int16(), nullable=False),
    ]
    fields.extend(pa.field(f"d{index:02d}", pa.float64(), nullable=False) for index in range(descriptor_count))
    return pa.schema(fields)


def _accepted_table(
    accepted: list[tuple[CanonicalMolecule, list[float], int]],
    task: dict[str, Any],
    descriptor_count: int,
) -> pa.Table:
    values: dict[str, list[Any]] = {field.name: [] for field in _canonical_schema(descriptor_count)}
    for molecule, descriptors, row_index in accepted:
        values["bucket"].append(molecule.bucket)
        values["molecule_hash"].append(molecule.molecule_hash)
        values["canonical_smiles"].append(molecule.smiles)
        values["nonisomeric_smiles"].append(molecule.nonisomeric_smiles)
        values["scaffold"].append(molecule.scaffold)
        values["split"].append(molecule.split)
        values["source"].append(task["source"])
        values["source_priority"].append(task["source_priority"])
        values["source_batch"].append(task["_current_batch"])
        values["source_row"].append(row_index)
        values["atom_count"].append(molecule.atom_count)
        values["bond_count"].append(molecule.bond_count)
        for index, descriptor in enumerate(descriptors):
            values[f"d{index:02d}"].append(descriptor)
    return pa.Table.from_pydict(values, schema=_canonical_schema(descriptor_count))


def canonicalize_task(cfg: dict[str, Any], task_index: int) -> dict[str, Any]:
    work_dir = Path(cfg["paths"]["work_dir"])
    task = read_jsonl_row(work_dir / "tasks" / "canonicalize.jsonl", task_index)
    output_dir = ensure_directory(work_dir / "canonical")
    output = output_dir / f"task-{task_index:06d}.parquet"
    done_path = output_dir / f"task-{task_index:06d}.done.json"
    if output.is_file() and done_path.is_file():
        import json

        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_hash") != cfg["_config_hash"]:
            raise RuntimeError(f"Stale canonicalization output for task {task_index}; use a new work_dir")
        return done

    data_cfg = cfg["data"]
    canonical_cfg = data_cfg["canonicalization"]
    descriptor_columns = [str(value) for value in data_cfg["descriptor_columns"]]
    descriptor_count = len(descriptor_columns)
    allowed = set(str(value) for value in canonical_cfg["allowed_elements"])
    counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[int] = Counter()
    output_schema = _canonical_schema(descriptor_count)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output_dir)
    os.close(file_descriptor)
    writer = pq.ParquetWriter(
        temporary_name,
        output_schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=["split", "source"],
        write_statistics=True,
    )
    pending_by_bucket: list[list[pa.Table]] = [
        [] for _ in range(int(data_cfg["hash_buckets"]))
    ]
    memory = None
    try:
        memory, reader = _open_feather(task["source_path"])
        _validate_arrow_schema(cfg, task, reader.schema)
        for batch_index in range(int(task["start_batch"]), int(task["end_batch"])):
            batch = reader.get_batch(batch_index)
            smiles_array = batch.column(batch.schema.get_field_index(str(data_cfg["smiles_column"])))
            descriptor_arrays = [
                batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
                for name in descriptor_columns
            ]
            accepted: list[tuple[CanonicalMolecule, list[float], int]] = []
            task["_current_batch"] = batch_index
            for row_index in range(batch.num_rows):
                counts["input_rows"] += 1
                if smiles_array[row_index].is_valid is False:
                    counts["reject_null_smiles"] += 1
                    continue
                descriptors = [float(array[row_index]) for array in descriptor_arrays]
                if not all(math.isfinite(value) for value in descriptors):
                    counts["reject_nonfinite_descriptor"] += 1
                    continue
                result = canonicalize(
                    str(smiles_array[row_index].as_py()),
                    isomeric_smiles=bool(canonical_cfg["isomeric_smiles"]),
                    fragment_policy=str(canonical_cfg["fragment_policy"]),
                    allowed_elements=allowed,
                    min_atoms=int(canonical_cfg["min_atoms"]),
                    max_atoms=int(canonical_cfg["max_atoms"]),
                    buckets=int(data_cfg["hash_buckets"]),
                    split_cfg=data_cfg["split"],
                )
                if not isinstance(result, CanonicalMolecule):
                    counts[f"reject_{result.reason}"] += 1
                    continue
                accepted.append((result, descriptors, row_index))
                counts["accepted_rows"] += 1
                split_counts[result.split] += 1
                bucket_counts[result.bucket] += 1
            if accepted:
                table = _accepted_table(accepted, task, descriptor_count)
                order = pc.sort_indices(table, sort_keys=[("bucket", "ascending"), ("canonical_smiles", "ascending")])
                table = table.take(order)
                bucket_values = table.column("bucket").to_numpy(zero_copy_only=False)
                boundaries = np.flatnonzero(np.diff(bucket_values)) + 1
                starts = np.concatenate(([0], boundaries))
                ends = np.concatenate((boundaries, [len(bucket_values)]))
                for start, end in zip(starts, ends):
                    bucket_value = int(bucket_values[int(start)])
                    pending_by_bucket[bucket_value].append(
                        table.slice(int(start), int(end - start))
                    )
        for bucket_tables in pending_by_bucket:
            if bucket_tables:
                combined = pa.concat_tables(bucket_tables)
                writer.write_table(combined, row_group_size=combined.num_rows)
        if counts["accepted_rows"] == 0:
            writer.write_table(pa.Table.from_batches([], schema=output_schema))
        writer.close()
        writer = None
        os.replace(temporary_name, output)
    except BaseException:
        if writer is not None:
            writer.close()
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    finally:
        if memory is not None:
            memory.close()

    done = {
        "schema_version": 1,
        "task_id": task_index,
        "config_hash": cfg["_config_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "output": str(output),
        "counts": dict(sorted(counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "nonempty_buckets": len(bucket_counts),
    }
    atomic_write_json(done_path, done)
    return done

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.dataset as ds

from .config import descriptor_names, object_hash
from .util import atomic_write_json, ensure_directory, runtime_versions


def _sql_path(path: str | Path) -> str:
    return str(path).replace("'", "''").replace("\\", "/")


def deduplicate_bucket(cfg: dict[str, Any], bucket: int, threads: int = 1) -> dict[str, Any]:
    bucket_count = int(cfg["data"]["hash_buckets"])
    if bucket < 0 or bucket >= bucket_count:
        raise IndexError(f"Bucket {bucket} is outside [0, {bucket_count})")
    work_dir = Path(cfg["paths"]["work_dir"])
    canonical_files = sorted((work_dir / "canonical").glob("task-*.parquet"))
    if not canonical_files:
        raise FileNotFoundError("No canonical task outputs found")
    output_dir = ensure_directory(work_dir / "deduplicated")
    conflict_dir = ensure_directory(work_dir / "conflicts")
    output = output_dir / f"bucket-{bucket:04d}.parquet"
    conflicts = conflict_dir / f"bucket-{bucket:04d}.parquet"
    done_path = output_dir / f"bucket-{bucket:04d}.done.json"
    if output.is_file() and conflicts.is_file() and done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_hash") != cfg["_config_hash"]:
            raise RuntimeError(f"Stale deduplication output for bucket {bucket}; use a new work_dir")
        return done

    descriptor_count = len(cfg["data"]["descriptor_columns"])
    descriptor_columns = [f"d{index:02d}" for index in range(descriptor_count)]
    dedup_cfg = cfg["data"]["deduplication"]
    atol, rtol = float(dedup_cfg["descriptor_atol"]), float(dedup_cfg["descriptor_rtol"])
    min_max = ",\n".join(
        f"min({name}) AS {name}_min, max({name}) AS {name}_max" for name in descriptor_columns
    )
    conflict_terms = [
        f"abs({name}_max - {name}_min) > {atol:.17g} + {rtol:.17g} * "
        f"greatest(abs({name}_max), abs({name}_min))"
        for name in descriptor_columns
    ]
    conflict_expression = " OR ".join(f"({term})" for term in conflict_terms)
    path_list = ",".join(f"'{_sql_path(path)}'" for path in canonical_files)
    output_tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    conflict_tmp = conflicts.with_name(f".{conflicts.name}.{os.getpid()}.tmp")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"PRAGMA threads={max(1, int(threads))}")
        connection.execute(
            f"""
            CREATE TEMP TABLE bucket_rows AS
            SELECT * FROM read_parquet([{path_list}]) WHERE bucket = ?
            """,
            [bucket],
        )
        input_rows = int(connection.execute("SELECT count(*) FROM bucket_rows").fetchone()[0])
        connection.execute(
            f"""
            CREATE TEMP TABLE molecule_groups AS
            SELECT canonical_smiles,
                   min(molecule_hash) AS molecule_hash,
                   count(*)::BIGINT AS duplicate_count,
                   string_agg(DISTINCT source, ',' ORDER BY source) AS sources,
                   {min_max}
            FROM bucket_rows
            GROUP BY canonical_smiles
            """
        )
        unique_groups = int(connection.execute("SELECT count(*) FROM molecule_groups").fetchone()[0])
        conflict_groups = int(
            connection.execute(f"SELECT count(*) FROM molecule_groups WHERE {conflict_expression}").fetchone()[0]
        )
        minmax_columns = ", ".join(
            f"{name}_min, {name}_max" for name in descriptor_columns
        )
        connection.execute(
            f"""
            COPY (
              SELECT canonical_smiles, molecule_hash, duplicate_count, sources, {minmax_columns}
              FROM molecule_groups
              WHERE {conflict_expression}
              ORDER BY canonical_smiles
            ) TO '{_sql_path(conflict_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        selected_columns = ", ".join(f"r.{name}" for name in descriptor_columns)
        connection.execute(
            f"""
            COPY (
              WITH ranked AS (
                SELECT *, row_number() OVER (
                  PARTITION BY canonical_smiles
                  ORDER BY source_priority, source, source_batch, source_row
                ) AS choice_rank
                FROM bucket_rows
              )
              SELECT r.molecule_hash, r.canonical_smiles, r.nonisomeric_smiles,
                     r.scaffold, r.split, r.atom_count, r.bond_count,
                     g.duplicate_count, g.sources, {selected_columns}
              FROM ranked r
              JOIN molecule_groups g USING (canonical_smiles)
              WHERE r.choice_rank = 1 AND NOT ({conflict_expression})
              ORDER BY r.canonical_smiles
            ) TO '{_sql_path(output_tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)
            """
        )
        output_rows = unique_groups - conflict_groups
    finally:
        connection.close()
    os.replace(output_tmp, output)
    os.replace(conflict_tmp, conflicts)
    done = {
        "schema_version": 1,
        "bucket": bucket,
        "config_hash": cfg["_config_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "input_rows": input_rows,
        "unique_groups": unique_groups,
        "duplicate_rows_removed": input_rows - unique_groups,
        "descriptor_conflict_groups_excluded": conflict_groups,
        "output_rows": output_rows,
        "output": str(output),
        "conflicts": str(conflicts),
    }
    atomic_write_json(done_path, done)
    return done


def finalize_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    work_dir = Path(cfg["paths"]["work_dir"])
    bucket_count = int(cfg["data"]["hash_buckets"])
    canonical_done = sorted((work_dir / "canonical").glob("task-*.done.json"))
    prepared = json.loads((work_dir / "prepared.json").read_text(encoding="utf-8"))
    verified_path = work_dir / "verified_sources.json"
    if not verified_path.is_file():
        raise RuntimeError("Missing verified_sources.json; run verify-inputs before finalization")
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    if verified.get("config_hash") != cfg["_config_hash"]:
        raise RuntimeError("Input verification belongs to a different configuration")
    expected_tasks = int(prepared["canonicalize_task_count"])
    if len(canonical_done) != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} canonical tasks, found {len(canonical_done)}")
    canonical_stats: dict[str, int] = {}
    for path in canonical_done:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["config_hash"] != cfg["_config_hash"]:
            raise RuntimeError(f"Stale canonical task metadata: {path}")
        for key, value in item["counts"].items():
            canonical_stats[key] = canonical_stats.get(key, 0) + int(value)
    expected_rows = sum(
        int(source["expected_rows"])
        for source in prepared["sources"]
        if source.get("expected_rows") is not None
    )
    if expected_rows and canonical_stats.get("input_rows", 0) != expected_rows:
        raise RuntimeError(
            f"Canonical task coverage mismatch: expected {expected_rows} source rows, "
            f"observed {canonical_stats.get('input_rows', 0)}"
        )

    dedup_done = []
    for bucket in range(bucket_count):
        path = work_dir / "deduplicated" / f"bucket-{bucket:04d}.done.json"
        if not path.is_file():
            raise RuntimeError(f"Missing deduplication task output: {path}")
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["config_hash"] != cfg["_config_hash"]:
            raise RuntimeError(f"Stale deduplication metadata: {path}")
        dedup_done.append(item)
    unique_groups = sum(int(item["unique_groups"]) for item in dedup_done)
    conflicts = sum(int(item["descriptor_conflict_groups_excluded"]) for item in dedup_done)
    conflict_fraction = conflicts / max(1, unique_groups)
    maximum = float(cfg["data"]["deduplication"]["max_conflict_fraction"])
    if conflict_fraction > maximum:
        raise RuntimeError(
            f"Descriptor conflict fraction {conflict_fraction:.6g} exceeds configured maximum {maximum:.6g}; "
            "inspect work/conflicts before proceeding"
        )

    parquet_files = sorted((work_dir / "deduplicated").glob("bucket-*.parquet"))
    dataset = ds.dataset([str(path) for path in parquet_files], format="parquet")
    split_counts = {}
    for split in ("train", "validation", "test"):
        split_counts[split] = int(dataset.count_rows(filter=ds.field("split") == split))
    manifest = {
        "schema_version": 1,
        "config_hash": cfg["_config_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "descriptor_names": descriptor_names(cfg),
        "deduplication_key": "canonical_isomeric_smiles",
        "split_method": cfg["data"]["split"]["method"],
        "source_metadata": prepared["sources"],
        "verified_source_hashes": verified["sources"],
        "canonicalization_counts": dict(sorted(canonical_stats.items())),
        "deduplication": {
            "input_eligible_rows": sum(int(item["input_rows"]) for item in dedup_done),
            "unique_groups_before_conflict_filter": unique_groups,
            "duplicate_rows_removed": sum(int(item["duplicate_rows_removed"]) for item in dedup_done),
            "descriptor_conflict_groups_excluded": conflicts,
            "descriptor_conflict_fraction": conflict_fraction,
            "rows_after_deduplication": sum(int(item["output_rows"]) for item in dedup_done),
        },
        "split_counts": split_counts,
        "parquet_files": [str(path) for path in parquet_files],
        "runtime": runtime_versions(),
    }
    manifest["manifest_hash"] = object_hash(manifest)
    atomic_write_json(work_dir / "dataset_manifest.json", manifest)
    return manifest


def fit_train_scaler(cfg: dict[str, Any], batch_size: int = 262144) -> dict[str, Any]:
    work_dir = Path(cfg["paths"]["work_dir"])
    manifest_path = work_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run finalize-data before fitting the scaler")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor_columns = [f"d{index:02d}" for index in range(len(cfg["data"]["descriptor_columns"]))]
    dataset = ds.dataset(manifest["parquet_files"], format="parquet")
    scanner = dataset.scanner(
        columns=descriptor_columns,
        filter=ds.field("split") == "train",
        batch_size=batch_size,
        use_threads=True,
    )
    count = 0
    mean = np.zeros(len(descriptor_columns), dtype=np.float64)
    m2 = np.zeros_like(mean)
    for batch in scanner.to_batches():
        values = np.column_stack([batch.column(index).to_numpy(zero_copy_only=False) for index in range(batch.num_columns)])
        if values.size == 0:
            continue
        if not np.isfinite(values).all():
            raise ValueError("Non-finite descriptor reached train-only scaler stage")
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0, dtype=np.float64)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0, dtype=np.float64)
        delta = batch_mean - mean
        combined = count + batch_count
        mean += delta * (batch_count / combined)
        m2 += batch_m2 + delta**2 * count * batch_count / combined
        count = combined
    if count == 0:
        raise RuntimeError("Training split is empty")
    variance = m2 / count
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    result = {
        "schema_version": 1,
        "config_hash": cfg["_config_hash"],
        "dataset_manifest_hash": manifest["manifest_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "descriptor_names": descriptor_names(cfg),
        "fitted_split": "train",
        "count": int(count),
        "mean": mean.tolist(),
        "variance": variance.tolist(),
        "scale": scale.tolist(),
    }
    result["scaler_hash"] = object_hash(result)
    atomic_write_json(work_dir / "descriptor_scaler.json", result)
    return result

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from rdkit import Chem

from .chem import featurize_molecule
from .config import object_hash
from .schema import feature_schema, validate_feature_schema
from .util import atomic_write_json, ensure_directory, runtime_versions


def _signed_graph_id(molecule_hash: str) -> int:
    value = int(molecule_hash[:16], 16)
    return value if value < 2**63 else value - 2**64


class _ShardBuffer:
    def __init__(self) -> None:
        self.x: list[torch.Tensor] = []
        self.edge_index: list[torch.Tensor] = []
        self.edge_attr: list[torch.Tensor] = []
        self.y: list[torch.Tensor] = []
        self.graph_ids: list[int] = []
        self.molecule_hashes: list[str] = []
        self.node_counts: list[int] = []
        self.edge_counts: list[int] = []

    def __len__(self) -> int:
        return len(self.y)

    def append(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        y: torch.Tensor,
        molecule_hash: str,
    ) -> None:
        self.x.append(x)
        self.edge_index.append(edge_index)
        self.edge_attr.append(edge_attr)
        self.y.append(y)
        self.graph_ids.append(_signed_graph_id(molecule_hash))
        self.molecule_hashes.append(molecule_hash)
        self.node_counts.append(int(x.shape[0]))
        self.edge_counts.append(int(edge_index.shape[1]))

    def pack(self, metadata: dict[str, Any]) -> dict[str, Any]:
        node_ptr = [0]
        edge_ptr = [0]
        shifted_edges = []
        for count in self.node_counts:
            node_ptr.append(node_ptr[-1] + count)
        for index, count in enumerate(self.edge_counts):
            edge_ptr.append(edge_ptr[-1] + count)
            shifted_edges.append(self.edge_index[index].to(torch.int64) + node_ptr[index])
        return {
            "metadata": metadata,
            "x": torch.cat(self.x, dim=0),
            "edge_index": torch.cat(shifted_edges, dim=1).to(torch.int32),
            "edge_attr": torch.cat(self.edge_attr, dim=0),
            "y": torch.stack(self.y, dim=0).to(torch.float32),
            "node_ptr": torch.tensor(node_ptr, dtype=torch.int64),
            "edge_ptr": torch.tensor(edge_ptr, dtype=torch.int64),
            "graph_ids": torch.tensor(self.graph_ids, dtype=torch.int64),
            "molecule_hashes": self.molecule_hashes,
        }

    def clear(self) -> None:
        self.__init__()


def _atomic_torch_save(value: Any, path: Path) -> None:
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def featurize_bucket(cfg: dict[str, Any], bucket: int) -> dict[str, Any]:
    bucket_count = int(cfg["data"]["hash_buckets"])
    if bucket < 0 or bucket >= bucket_count:
        raise IndexError(bucket)
    work_dir = Path(cfg["paths"]["work_dir"])
    input_path = work_dir / "deduplicated" / f"bucket-{bucket:04d}.parquet"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    graph_root = ensure_directory(work_dir / "graphs")
    final_dir = graph_root / f"bucket-{bucket:04d}"
    done_path = graph_root / f"bucket-{bucket:04d}.done.json"
    if final_dir.is_dir() and done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_hash") != cfg["_config_hash"]:
            raise RuntimeError(f"Stale graph output for bucket {bucket}; use a new work_dir")
        return done
    if final_dir.exists():
        raise RuntimeError(f"Incomplete graph directory already exists: {final_dir}")

    features_cfg = cfg["features"]
    schema = feature_schema(
        bool(features_cfg["include_atom_chirality"]),
        int(features_cfg["canonical_position_encoding_dim"]),
    )
    descriptor_count = len(cfg["data"]["descriptor_columns"])
    descriptor_columns = [f"d{index:02d}" for index in range(descriptor_count)]
    columns = ["molecule_hash", "canonical_smiles", "split", *descriptor_columns]
    graphs_per_shard = int(cfg["data"]["graph_shards"]["graphs_per_shard"])
    staging = graph_root / f".bucket-{bucket:04d}.{os.getpid()}.tmp"
    staging.mkdir(parents=True, exist_ok=False)
    buffers = {split: _ShardBuffer() for split in ("train", "validation", "test")}
    sequence = Counter()
    counts = Counter()
    failure_examples: list[dict[str, str]] = []
    shard_entries: list[dict[str, Any]] = []

    def flush(split: str) -> None:
        buffer = buffers[split]
        if not buffer:
            return
        relative = Path(split) / f"shard-{sequence[split]:05d}.pt"
        target = staging / relative
        metadata = {
            "schema_version": 1,
            "config_hash": cfg["_config_hash"],
            "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
            "feature_schema_hash": schema["hash"],
            "split": split,
            "bucket": bucket,
            "sequence": sequence[split],
            "graphs": len(buffer),
        }
        packed = buffer.pack(metadata)
        _atomic_torch_save(packed, target)
        shard_entries.append(
            {
                "path": str(Path(f"bucket-{bucket:04d}") / relative).replace("\\", "/"),
                "split": split,
                "bucket": bucket,
                "sequence": sequence[split],
                "graphs": len(buffer),
                "nodes": int(packed["node_ptr"][-1]),
                "directed_edges": int(packed["edge_ptr"][-1]),
                "size_bytes": target.stat().st_size,
            }
        )
        sequence[split] += 1
        buffer.clear()

    try:
        parquet = pq.ParquetFile(input_path)
        for batch in parquet.iter_batches(batch_size=graphs_per_shard, columns=columns):
            values = batch.to_pydict()
            for row in range(batch.num_rows):
                split = str(values["split"][row])
                if split not in buffers:
                    raise ValueError(f"Unknown split {split!r}")
                molecule_hash = str(values["molecule_hash"][row])
                mol = Chem.MolFromSmiles(str(values["canonical_smiles"][row]))
                if mol is None:
                    counts["graph_parse_failure"] += 1
                    if len(failure_examples) < 20:
                        failure_examples.append({"molecule_hash": molecule_hash, "reason": "parse_failure"})
                    continue
                try:
                    x_np, edge_index_np, edge_attr_np = featurize_molecule(
                        mol,
                        include_chirality=bool(features_cfg["include_atom_chirality"]),
                        position_dim=int(features_cfg["canonical_position_encoding_dim"]),
                    )
                except Exception as error:
                    counts["graph_feature_failure"] += 1
                    if len(failure_examples) < 20:
                        failure_examples.append(
                            {"molecule_hash": molecule_hash, "reason": f"{type(error).__name__}: {error}"}
                        )
                    continue
                x = torch.from_numpy(x_np)
                if int(features_cfg["canonical_position_encoding_dim"]) == 0:
                    x = x.to(torch.uint8)
                else:
                    x = x.to(torch.float16)
                edge_index = torch.from_numpy(edge_index_np).to(torch.int32)
                edge_attr = torch.from_numpy(edge_attr_np).to(torch.uint8)
                y = torch.tensor([values[name][row] for name in descriptor_columns], dtype=torch.float32)
                buffers[split].append(x, edge_index, edge_attr, y, molecule_hash)
                counts[f"graphs_{split}"] += 1
                counts["graphs_total"] += 1
                if len(buffers[split]) >= graphs_per_shard:
                    flush(split)
        for split in buffers:
            flush(split)
        os.replace(staging, final_dir)
    except BaseException:
        # Leave the uniquely named staging directory for forensic inspection.
        raise

    for entry in shard_entries:
        entry["path"] = str((graph_root / entry["path"]).resolve())
    done = {
        "schema_version": 1,
        "bucket": bucket,
        "config_hash": cfg["_config_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "feature_schema": schema,
        "counts": dict(sorted(counts.items())),
        "failure_examples": failure_examples,
        "shards": shard_entries,
    }
    atomic_write_json(done_path, done)
    return done


def finalize_graphs(cfg: dict[str, Any]) -> dict[str, Any]:
    work_dir = Path(cfg["paths"]["work_dir"])
    dataset_manifest = json.loads((work_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    scaler = json.loads((work_dir / "descriptor_scaler.json").read_text(encoding="utf-8"))
    bucket_count = int(cfg["data"]["hash_buckets"])
    shards = []
    counts = Counter()
    schema = None
    for bucket in range(bucket_count):
        done_path = work_dir / "graphs" / f"bucket-{bucket:04d}.done.json"
        if not done_path.is_file():
            raise RuntimeError(f"Missing graph task output: {done_path}")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done["config_hash"] != cfg["_config_hash"]:
            raise RuntimeError(f"Stale graph metadata: {done_path}")
        if schema is None:
            schema = done["feature_schema"]
        elif schema["hash"] != done["feature_schema"]["hash"]:
            raise RuntimeError("Graph tasks used different feature schemas")
        shards.extend(done["shards"])
        counts.update({key: int(value) for key, value in done["counts"].items()})
    failures = counts["graph_parse_failure"] + counts["graph_feature_failure"]
    if failures:
        raise RuntimeError(f"Graph creation had {failures} failures; inspect per-bucket done files")
    for split, expected in dataset_manifest["split_counts"].items():
        actual = counts[f"graphs_{split}"]
        if actual != int(expected):
            raise RuntimeError(f"Graph count mismatch for {split}: expected {expected}, got {actual}")
    shards.sort(key=lambda item: (item["split"], item["bucket"], item["sequence"]))
    result = {
        "schema_version": 1,
        "config_hash": cfg["_config_hash"],
        "dataset_manifest_hash": dataset_manifest["manifest_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "scaler_hash": scaler["scaler_hash"],
        "feature_schema": schema,
        "counts": dict(sorted(counts.items())),
        "shards": shards,
        "runtime": runtime_versions(),
    }
    validate_feature_schema(schema)
    result["graph_manifest_hash"] = object_hash(result)
    atomic_write_json(work_dir / "graph_manifest.json", result)
    return result

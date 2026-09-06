#!/usr/bin/env python3
"""Generate calibrated gMolAI molecular embeddings from a SMILES CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

try:
    import numpy as np
    from rdkit import rdBase
    import torch

    from gmolai_retrain.chem import Rejection, canonicalize
    from gmolai_retrain.fast_inference import (
        build_smiles_encoder,
        implementation_metadata,
    )
    from gmolai_retrain.model import MolecularRepresentationModel
    from gmolai_retrain.schema import feature_schema, validate_feature_schema
except ImportError as error:  # pragma: no cover - depends on the deployment environment
    raise SystemExit(
        "Missing inference dependency. Run inside the pinned project container or "
        "install requirements.lock and the package dependencies. "
        f"Original import error: {error}"
    ) from error


SCRIPT_VERSION = "2.0.0"
PUBLIC_EMBEDDING_DEFINITION = (
    "clean_graph_z_plus_mean_node_z_train_standardized_raw_blocks"
)
RAW_EMBEDDING_DEFINITION = "clean_graph_z_plus_mean_node_z_raw_blocks"
EXPECTED_ARTIFACT_SHA256 = {
    "representation-best.pt": (
        "02f49a2a94ddfc9dc780cc3d5f1a3df54306ae0fdc5d4b3767e3fd2e7f27b05e"
    ),
    "representation-calibrator.pt": (
        "5cbe3210b2fa6742b165c61e3562118553f567df13181d863776c9ca5527365b"
    ),
    "representation_selection.json": (
        "43f1f857576f10fd8aa7ed9276f9ce899ca90d011172225d04e8cff77a9333a1"
    ),
    "resolved_config.json": (
        "9ad8e4000b3dc0b7a2c3ef8631200fbfa301ef377fd7518293b8636964844628"
    ),
}
OUTPUT_STEM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class InferenceError(RuntimeError):
    """Raised when inputs or model artifacts violate the inference contract."""


@dataclass(slots=True)
class PendingMolecule:
    input_row: int
    input_id: str
    input_smiles: str
    canonical_smiles: str
    molecule_hash: str
    atom_count: int


@dataclass(slots=True)
class ModelBundle:
    model: MolecularRepresentationModel
    coordinate_mean: torch.Tensor
    coordinate_scale: torch.Tensor
    device: torch.device
    graph_dimensions: int
    mean_node_dimensions: int
    mean_node_weight: float
    embedding_dimensions: int
    selection: dict[str, Any]
    resolved_config: dict[str, Any]
    checkpoint: dict[str, Any]
    calibrator_metadata: dict[str, Any]
    artifact_hashes: dict[str, str]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InferenceError(f"Cannot read valid JSON object from {path}: {error}") from error
    if not isinstance(value, dict):
        raise InferenceError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceError(message)


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as error:
        raise InferenceError(f"Invalid device {requested!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise InferenceError("--device must be auto, cpu, cuda, or cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise InferenceError("CUDA was requested but torch.cuda.is_available() is false")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise InferenceError(
                f"CUDA device index {index} is outside 0..{torch.cuda.device_count() - 1}"
            )
        device = torch.device("cuda", index)
    return device


def validate_artifact_files(model_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    if not model_dir.is_dir():
        raise InferenceError(f"Model directory does not exist: {model_dir}")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, expected_hash in EXPECTED_ARTIFACT_SHA256.items():
        path = model_dir / name
        if not path.is_file():
            raise InferenceError(f"Required model artifact is missing: {path}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise InferenceError(
                f"Artifact hash mismatch for {name}: expected {expected_hash}, "
                f"observed {observed_hash}"
            )
        paths[name] = path
        hashes[name] = observed_hash
    return paths, hashes


def load_model_bundle(model_dir: Path, device: torch.device) -> ModelBundle:
    paths, hashes = validate_artifact_files(model_dir)
    selection = load_json_object(paths["representation_selection.json"])
    config = load_json_object(paths["resolved_config.json"])

    require(selection.get("schema_version") == 1, "Unsupported selection schema")
    require(
        selection.get("embedding_definition") == PUBLIC_EMBEDDING_DEFINITION,
        "Selection record does not name the public calibrated embedding",
    )
    require(
        selection.get("checkpoint_sha256") == hashes["representation-best.pt"],
        "Selection record and checkpoint hash disagree",
    )
    selection_calibrator = selection.get("calibrator")
    require(isinstance(selection_calibrator, dict), "Selection lacks calibrator metadata")
    require(
        selection_calibrator.get("promoted") == "representation-calibrator.pt",
        "Selection names an unexpected calibrator",
    )
    require(
        selection_calibrator.get("sha256")
        == hashes["representation-calibrator.pt"],
        "Selection record and calibrator hash disagree",
    )

    # Both torch artifacts are hash-verified before unpickling. Only load the
    # checkpoint shipped with this repository.
    try:
        checkpoint = torch.load(
            paths["representation-best.pt"], map_location="cpu", weights_only=False
        )
        calibrator = torch.load(
            paths["representation-calibrator.pt"],
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise InferenceError(f"Cannot deserialize packaged model artifacts: {error}") from error
    require(isinstance(checkpoint, dict), "Checkpoint payload is not a dictionary")
    require(isinstance(calibrator, dict), "Calibrator payload is not a dictionary")
    require(checkpoint.get("checkpoint_version") == 1, "Unsupported checkpoint version")
    require(
        checkpoint.get("training_implementation_version") == "5",
        "Checkpoint is not the corrected representation implementation",
    )
    require(
        config.get("model", {}).get("architecture") == "masked_graph_vicreg",
        "Resolved configuration is not the representation architecture",
    )

    for identity_key in (
        "config_hash",
        "training_plan_hash",
        "graph_manifest_hash",
    ):
        require(
            checkpoint.get(identity_key) == selection.get(identity_key),
            f"Checkpoint and selection disagree at {identity_key}",
        )
    require(
        config.get("_config_hash") == checkpoint.get("config_hash"),
        "Resolved configuration hash does not match the checkpoint",
    )
    require(
        config.get("_descriptor_schema_hash")
        == checkpoint.get("descriptor_schema_hash"),
        "Descriptor schema hash does not match the checkpoint",
    )
    require(
        int(checkpoint.get("global_step", -1))
        == int(selection.get("global_step", -2)),
        "Checkpoint and selection global steps disagree",
    )

    feature_cfg = config.get("features", {})
    schema = feature_schema(
        include_chirality=bool(feature_cfg.get("include_atom_chirality", True)),
        position_dim=int(feature_cfg.get("canonical_position_encoding_dim", 0)),
    )
    validate_feature_schema(schema)
    require(
        schema["hash"] == checkpoint.get("feature_schema_hash"),
        "Runtime feature schema does not match the checkpoint",
    )

    descriptor_columns = config.get("data", {}).get("descriptor_columns")
    require(
        isinstance(descriptor_columns, list) and len(descriptor_columns) == 13,
        "Resolved configuration has an invalid descriptor contract",
    )
    model = MolecularRepresentationModel(
        schema, descriptor_count=len(descriptor_columns), model_cfg=config["model"]
    )
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise InferenceError(f"Checkpoint model state is incompatible: {error}") from error
    model.to(device).eval()

    parameters = selection.get("embedding_parameters")
    require(isinstance(parameters, dict), "Selection lacks embedding parameters")
    graph_dimensions = int(parameters.get("graph_dimensions", 0))
    mean_node_dimensions = int(parameters.get("mean_node_dimensions", 0))
    mean_node_weight = float(parameters.get("mean_node_weight", float("nan")))
    embedding_dimensions = graph_dimensions + mean_node_dimensions
    require(
        graph_dimensions == model.graph_latent_dim
        and mean_node_dimensions == model.node_latent_dim,
        "Selection dimensions do not match the model",
    )
    require(
        embedding_dimensions == 384
        and math.isfinite(mean_node_weight)
        and mean_node_weight > 0,
        "Invalid public embedding dimensions or block weight",
    )
    require(
        parameters.get("coordinate_transform") == "train_mean_and_population_std",
        "Unsupported coordinate transform",
    )

    calibrator_metadata = calibrator.get("metadata")
    require(isinstance(calibrator_metadata, dict), "Calibrator lacks metadata")
    require(
        calibrator_metadata == selection_calibrator.get("metadata"),
        "Calibrator metadata differs from the promoted selection record",
    )
    expected_calibrator_identity = {
        "calibration_definition": "coordinate_mean_and_population_std",
        "source_embedding_definition": RAW_EMBEDDING_DEFINITION,
        "checkpoint_sha256": hashes["representation-best.pt"],
        "config_hash": checkpoint["config_hash"],
        "training_plan_hash": checkpoint["training_plan_hash"],
        "graph_manifest_hash": checkpoint["graph_manifest_hash"],
        "descriptor_schema_hash": checkpoint["descriptor_schema_hash"],
        "global_step": int(checkpoint["global_step"]),
        "split": "train",
        "dimensions": embedding_dimensions,
        "graphs": 100000,
        "sampled_source_buckets": 256,
        "sampling": "deterministic_hash_bucket_stratified_without_replacement",
    }
    for key, expected in expected_calibrator_identity.items():
        require(
            calibrator_metadata.get(key) == expected,
            f"Calibrator identity mismatch at {key}",
        )
    require(
        parameters.get("calibrator_sha256") == hashes["representation-calibrator.pt"],
        "Embedding parameters and calibrator hash disagree",
    )

    coordinate_mean = calibrator.get("coordinate_mean")
    coordinate_scale = calibrator.get("coordinate_scale")
    require(
        isinstance(coordinate_mean, torch.Tensor)
        and isinstance(coordinate_scale, torch.Tensor),
        "Calibrator coordinates are not tensors",
    )
    require(
        coordinate_mean.shape == (embedding_dimensions,)
        and coordinate_scale.shape == (embedding_dimensions,),
        "Calibrator tensor dimensions are invalid",
    )
    require(
        bool(torch.isfinite(coordinate_mean).all())
        and bool(torch.isfinite(coordinate_scale).all())
        and float(coordinate_scale.min()) > 1.0e-8,
        "Calibrator tensors are non-finite or degenerate",
    )

    return ModelBundle(
        model=model,
        coordinate_mean=coordinate_mean.float().to(device),
        coordinate_scale=coordinate_scale.float().to(device),
        device=device,
        graph_dimensions=graph_dimensions,
        mean_node_dimensions=mean_node_dimensions,
        mean_node_weight=mean_node_weight,
        embedding_dimensions=embedding_dimensions,
        selection=selection,
        resolved_config=config,
        checkpoint=checkpoint,
        calibrator_metadata=calibrator_metadata,
        artifact_hashes=hashes,
    )


def encode_batch(encoder: Any, records: list[PendingMolecule]) -> np.ndarray:
    if not records:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = encoder.encode(
        [record.canonical_smiles for record in records],
        atom_counts=[record.atom_count for record in records],
    )
    if embeddings.shape[0] != len(records):
        raise InferenceError(
            f"Model returned {embeddings.shape[0]} rows, expected {len(records)}"
        )
    if not np.isfinite(embeddings).all():
        raise InferenceError("Model produced non-finite embeddings")
    return np.asarray(embeddings, dtype=np.float32)


def temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def fsync_text_handle(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def canonicalize_input(
    raw_smiles: str, config: dict[str, Any]
) -> tuple[Any | None, str | None]:
    data_cfg = config["data"]
    policy = data_cfg["canonicalization"]
    value = canonicalize(
        raw_smiles,
        isomeric_smiles=bool(policy["isomeric_smiles"]),
        fragment_policy=str(policy["fragment_policy"]),
        allowed_elements={str(element) for element in policy["allowed_elements"]},
        min_atoms=int(policy["min_atoms"]),
        max_atoms=int(policy["max_atoms"]),
        buckets=int(data_cfg["hash_buckets"]),
        split_cfg=data_cfg["split"],
    )
    if isinstance(value, Rejection):
        return None, value.reason
    return value, None


def resolve_id_column(requested: str, headers: list[str]) -> str | None:
    if requested.lower() == "none":
        return None
    if requested.lower() == "auto":
        for candidate in ("molecule_id", "id"):
            if candidate in headers:
                return candidate
        return None
    if requested not in headers:
        raise InferenceError(f"ID column {requested!r} is absent from the input CSV")
    return requested


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise InferenceError(f"Input CSV does not exist: {input_path}")
    if not OUTPUT_STEM_PATTERN.fullmatch(args.output_stem):
        raise InferenceError(
            "--output-stem must start with an alphanumeric character and contain "
            "only letters, digits, dot, underscore, or hyphen"
        )
    if args.batch_size <= 0 or args.node_budget <= 0:
        raise InferenceError("--batch-size and --node-budget must be positive")
    if args.limit is not None and args.limit <= 0:
        raise InferenceError("--limit must be positive when supplied")
    if args.threads <= 0:
        raise InferenceError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"{args.output_stem}.csv"
    rejections_path = output_dir / f"{args.output_stem}.rejections.csv"
    metadata_path = output_dir / f"{args.output_stem}.metadata.json"
    for path in (embeddings_path, rejections_path, metadata_path):
        if path.exists() and not args.overwrite:
            raise InferenceError(f"Output exists; pass --overwrite to replace it: {path}")
    if input_path in {embeddings_path, rejections_path, metadata_path}:
        raise InferenceError("Input and output paths must differ")

    device = resolve_device(args.device)
    bundle = load_model_bundle(model_dir, device)
    encoder = build_smiles_encoder(
        args.backend,
        bundle.model,
        bundle.coordinate_mean,
        bundle.coordinate_scale,
        device=device,
        batch_size=args.batch_size,
        node_budget=args.node_budget,
        workers=args.workers,
        mean_node_weight=bundle.mean_node_weight,
        verify_rows=args.verify_rows,
    )
    backend_info = implementation_metadata(encoder)
    pipeline_batches = max(1, int(backend_info["workers"]))
    pipeline_graph_budget = args.batch_size * pipeline_batches
    pipeline_node_budget = args.node_budget * pipeline_batches
    input_hash = sha256_file(input_path)
    embeddings_temporary = temporary_path(embeddings_path)
    rejections_temporary = temporary_path(rejections_path)
    metadata_temporary = temporary_path(metadata_path)
    temporary_files = [
        embeddings_temporary,
        rejections_temporary,
        metadata_temporary,
    ]

    total_rows = 0
    accepted_rows = 0
    rejected_rows = 0
    pending_nodes = 0
    pending: list[PendingMolecule] = []
    rejection_reasons: Counter[str] = Counter()
    unique_hashes: set[str] = set()
    seen_ids: set[str] = set()
    duplicate_nonempty_ids = 0

    try:
        with (
            input_path.open("r", encoding="utf-8-sig", newline="") as input_handle,
            embeddings_temporary.open("w", encoding="utf-8", newline="") as output_handle,
            rejections_temporary.open("w", encoding="utf-8", newline="") as rejection_handle,
        ):
            reader = csv.DictReader(input_handle)
            headers = list(reader.fieldnames or [])
            if not headers:
                raise InferenceError("Input CSV is empty or lacks a header")
            if len(headers) != len(set(headers)):
                raise InferenceError("Input CSV contains duplicate column names")
            if args.smiles_column not in headers:
                raise InferenceError(
                    f"SMILES column {args.smiles_column!r} is absent from the input CSV"
                )
            id_column = resolve_id_column(args.id_column, headers)
            embedding_columns = [
                f"embedding_{index:03d}" for index in range(bundle.embedding_dimensions)
            ]
            output_writer = csv.writer(output_handle, lineterminator="\n")
            output_writer.writerow(
                [
                    "input_row",
                    "input_id",
                    "input_smiles",
                    "canonical_smiles",
                    "molecule_hash",
                    *embedding_columns,
                ]
            )
            rejection_writer = csv.writer(rejection_handle, lineterminator="\n")
            rejection_writer.writerow(
                ["input_row", "input_id", "input_smiles", "reason"]
            )

            def flush_pending() -> None:
                nonlocal pending, pending_nodes, accepted_rows
                if not pending:
                    return
                vectors = encode_batch(encoder, pending)
                for record, vector in zip(pending, vectors, strict=True):
                    output_writer.writerow(
                        [
                            record.input_row,
                            record.input_id,
                            record.input_smiles,
                            record.canonical_smiles,
                            record.molecule_hash,
                            *(format(float(value), ".9g") for value in vector),
                        ]
                    )
                accepted_rows += len(pending)
                pending = []
                pending_nodes = 0

            for input_row, row in enumerate(reader, start=1):
                if args.limit is not None and total_rows >= args.limit:
                    break
                total_rows += 1
                if None in row or any(value is None for value in row.values()):
                    raise InferenceError(
                        f"Malformed CSV record at input row {input_row}: field count differs "
                        "from the header"
                    )
                raw_smiles = str(row[args.smiles_column])
                input_id = str(row[id_column]) if id_column is not None else ""
                if input_id:
                    if input_id in seen_ids:
                        duplicate_nonempty_ids += 1
                    seen_ids.add(input_id)

                canonical, rejection_reason = canonicalize_input(
                    raw_smiles, bundle.resolved_config
                )
                if rejection_reason is not None:
                    rejected_rows += 1
                    rejection_reasons[rejection_reason] += 1
                    rejection_writer.writerow(
                        [input_row, input_id, raw_smiles, rejection_reason]
                    )
                    if args.invalid_policy == "error":
                        raise InferenceError(
                            f"Rejected molecule at input row {input_row}: {rejection_reason}"
                        )
                    continue
                assert canonical is not None
                if pending and (
                    len(pending) >= pipeline_graph_budget
                    or pending_nodes + canonical.atom_count > pipeline_node_budget
                ):
                    flush_pending()
                pending.append(
                    PendingMolecule(
                        input_row=input_row,
                        input_id=input_id,
                        input_smiles=raw_smiles,
                        canonical_smiles=canonical.smiles,
                        molecule_hash=canonical.molecule_hash,
                        atom_count=int(canonical.atom_count),
                    )
                )
                pending_nodes += int(canonical.atom_count)
                unique_hashes.add(canonical.molecule_hash)
            flush_pending()
            if total_rows == 0:
                raise InferenceError("Input CSV contains no data rows")
            if accepted_rows == 0:
                raise InferenceError("No input molecule passed the training-time policy")
            fsync_text_handle(output_handle)
            fsync_text_handle(rejection_handle)

        output_hashes = {
            embeddings_path.name: sha256_file(embeddings_temporary),
            rejections_path.name: sha256_file(rejections_temporary),
        }
        metadata = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(input_path),
                "sha256": input_hash,
                "smiles_column": args.smiles_column,
                "id_column": id_column,
                "limit": args.limit,
            },
            "rows": {
                "total": total_rows,
                "accepted": accepted_rows,
                "rejected": rejected_rows,
                "unique_accepted_molecules": len(unique_hashes),
                "duplicate_nonempty_ids": duplicate_nonempty_ids,
                "rejection_reasons": dict(sorted(rejection_reasons.items())),
            },
            "model": {
                "checkpoint": "representation-best.pt",
                "checkpoint_sha256": bundle.artifact_hashes[
                    "representation-best.pt"
                ],
                "calibrator": "representation-calibrator.pt",
                "calibrator_sha256": bundle.artifact_hashes[
                    "representation-calibrator.pt"
                ],
                "selection_sha256": bundle.artifact_hashes[
                    "representation_selection.json"
                ],
                "resolved_config_sha256": bundle.artifact_hashes[
                    "resolved_config.json"
                ],
                "global_step": int(bundle.checkpoint["global_step"]),
                "config_hash": bundle.checkpoint["config_hash"],
                "training_plan_hash": bundle.checkpoint["training_plan_hash"],
                "graph_manifest_hash": bundle.checkpoint["graph_manifest_hash"],
                "feature_schema_hash": bundle.checkpoint["feature_schema_hash"],
                "descriptor_schema_hash": bundle.checkpoint[
                    "descriptor_schema_hash"
                ],
                "scaler_hash": bundle.checkpoint["scaler_hash"],
                "selection_scope": bundle.selection.get("selection_scope"),
            },
            "embedding": {
                "definition": PUBLIC_EMBEDDING_DEFINITION,
                "dimensions": bundle.embedding_dimensions,
                "graph_dimensions": bundle.graph_dimensions,
                "mean_node_dimensions": bundle.mean_node_dimensions,
                "mean_node_weight": bundle.mean_node_weight,
                "coordinate_transform": "train_mean_and_population_std",
                "calibration_graphs": int(bundle.calibrator_metadata["graphs"]),
                "calibration_sampling_seed": bundle.calibrator_metadata[
                    "sampling_seed"
                ],
                "dtype": "float32",
                "csv_float_format": ".9g",
                "deterministic_clean_eval": True,
            },
            "canonicalization": bundle.resolved_config["data"]["canonicalization"],
            "execution": {
                "device": str(bundle.device),
                "batch_size": args.batch_size,
                "node_budget": args.node_budget,
                "backend": backend_info["backend"],
                "fast_inference_version": backend_info[
                    "fast_inference_version"
                ],
                "fast_graph_version": backend_info["fast_graph_version"],
                "workers": backend_info["workers"],
                "pipeline_batches": pipeline_batches,
                "verify_rows": args.verify_rows if args.backend == "verify" else 0,
                "threads": args.threads,
                "invalid_policy": args.invalid_policy,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
                "torch": torch.__version__,
                "torch_geometric": package_version("torch-geometric"),
            },
            "outputs": {
                embeddings_path.name: {
                    "sha256": output_hashes[embeddings_path.name],
                    "rows": accepted_rows,
                },
                rejections_path.name: {
                    "sha256": output_hashes[rejections_path.name],
                    "rows": rejected_rows,
                },
            },
        }
        with metadata_temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            fsync_text_handle(handle)

        # Metadata is the completion marker and is published last.
        os.replace(embeddings_temporary, embeddings_path)
        os.replace(rejections_temporary, rejections_path)
        os.replace(metadata_temporary, metadata_path)
        return {
            "accepted": accepted_rows,
            "rejected": rejected_rows,
            "dimensions": bundle.embedding_dimensions,
            "device": str(bundle.device),
            "backend": backend_info["backend"],
            "workers": backend_info["workers"],
            "embeddings": str(embeddings_path),
            "rejections": str(rejections_path),
            "metadata": str(metadata_path),
            "checkpoint_sha256": bundle.artifact_hashes["representation-best.pt"],
            "calibrator_sha256": bundle.artifact_hashes[
                "representation-calibrator.pt"
            ],
        }
    finally:
        encoder.close()
        for temporary in temporary_files:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the promoted 384-dimensional deterministic gMolAI vector "
            "for every accepted SMILES in a CSV file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "data" / "example_smiles.csv",
        help="Input CSV (default: inference/data/example_smiles.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=SCRIPT_DIR / "model",
        help="Directory containing the four packaged model artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output",
        help="Output directory (default: inference/output)",
    )
    parser.add_argument("--output-stem", default="embeddings")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument(
        "--id-column",
        default="auto",
        help="ID column, auto (molecule_id/id when present), or none",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:<index> (default: auto)",
    )
    parser.add_argument(
        "--backend",
        choices=("optimized", "reference", "verify"),
        default="optimized",
        help="Embedding backend (default: optimized)",
    )
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--node-budget", type=int, default=16384)
    parser.add_argument(
        "--workers",
        default="auto",
        help=(
            "RDKit preprocessing workers for the optimized backend; auto uses "
            "up to 48 within the Slurm/CPU-affinity allocation"
        ),
    )
    parser.add_argument(
        "--verify-rows",
        type=int,
        default=1024,
        help="Reference rows checked by --backend verify (default: 1024)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="PyTorch CPU threads (default: min(8, available CPUs))",
    )
    parser.add_argument(
        "--invalid-policy",
        choices=("report", "error"),
        default="report",
        help="Report rejected molecules or fail atomically on the first rejection",
    )
    parser.add_argument("--limit", type=int, help="Optional maximum input rows")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output set",
    )
    return parser


def main() -> int:
    try:
        summary = run_inference(build_parser().parse_args())
    except (InferenceError, OSError, csv.Error, RuntimeError, TypeError, ValueError) as error:
        print(f"Inference failed: {error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

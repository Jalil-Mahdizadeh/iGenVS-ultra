#!/usr/bin/env python3
"""Encode SMILES and generate candidates with the frozen gMolAI release."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
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
import sqlite3
import sys
import tempfile
import time
from typing import Any, Sequence


# Required by deterministic CUDA linear-algebra kernels.  Setting the default
# before importing torch keeps the CLI self-contained while respecting a value
# explicitly supplied by the caller.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

try:
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    import torch

    if __package__:
        from . import generate_embeddings as encoder_api
        from ._decoder import (
            ConditionalSmilesTransformer,
            beam_order,
            decode_tokens,
            decoder_parameter_count,
            generate_beam_pool,
            generate_seeded_sample_pool,
            proportional_merge,
        )
    else:
        import generate_embeddings as encoder_api
        from _decoder import (
            ConditionalSmilesTransformer,
            beam_order,
            decode_tokens,
            decoder_parameter_count,
            generate_beam_pool,
            generate_seeded_sample_pool,
            proportional_merge,
        )
except ImportError as error:  # pragma: no cover - deployment dependent
    raise SystemExit(
        "Missing inference dependency. Run this CLI inside the pinned gMolAI "
        "container. Original import error: "
        f"{error}"
    ) from error


RDLogger.DisableLog("rdApp.*")

SCRIPT_VERSION = "1.1.0"
EMBEDDING_SCHEMA_VERSION = 1
EMBEDDING_ARTIFACT_TYPE = "gmolai_release_hybrid_w3_embeddings"
EMBEDDING_SPACE = "released_hybrid_w3"
EMBEDDING_DIMENSIONS = 384
MEAN_NODE_WEIGHT = 3.0
DEFAULT_MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_INPUT = SCRIPT_DIR / "data" / "example_smiles.csv"
DEFAULT_EMBEDDINGS = SCRIPT_DIR / "output" / "embeddings.npz"
DEFAULT_CANDIDATE_DIR = SCRIPT_DIR / "output" / "candidates"

DECODER_FILENAME = "decoder_inference.pt"
EXPECTED_DECODER_SHA256 = (
    "8b4f8db04499083ea2e9d028eaaae18d629b34ce773608d8e2c80863e9121d47"
)
EXPECTED_TRAINING_DECODER_SHA256 = (
    "bb9623080ddaed070278c8abca39252e070c110a6611b3bd7a75caf6c37a41f6"
)
EXPECTED_STEP2_PROTOCOL_SHA256 = (
    "b036ff99c81d073dbdcd7287b039be21c4e13f6f17cc9cf2191938652c047e6b"
)
EXPECTED_STEP2_MANIFEST_SHA256 = (
    "717a16ea45f205561b47b079731ebff0b2c7d1d6b46e1ee1e00fb5c37076ed15"
)
EXPECTED_DECODER_PARAMETERS = 28_316_160
EXPECTED_DECODER_CONFIG: dict[str, Any] = {
    "activation": "gelu",
    "architecture": "cross_attention_conditional_transformer_smiles_v1",
    "attention_heads": 8,
    "condition_dimensions": 384,
    "condition_injection": [
        "cross_attention_memory",
        "additive_token_bias",
    ],
    "condition_memory_tokens": 4,
    "d_model": 512,
    "decoder_layers": 6,
    "dropout": 0.1,
    "feedforward_dimensions": 2048,
    "maximum_positions": 130,
    "norm_first": True,
    "vocab_size": 131,
}

STRATEGY_NAME = "hybrid_b500_s500_t120"
FROZEN_PROPOSAL_BUDGETS = (50, 100, 250, 500, 1000)
GLOBAL_SEED = 20_260_817
SAMPLING_PHASE = "final"
SAMPLE_POOL_NAME = "t120_p0995"
SAMPLE_DRAWS = 999
SAMPLE_TEMPERATURE = 1.2
SAMPLE_TOP_P = 0.995
BEAM_WIDTH = 1000
BEAM_HYPOTHESES = 500
SAMPLE_HYPOTHESES = 500
LENGTH_PENALTY = 0.0
MAXIMUM_SMILES_BYTES = 128
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_INCLUDE_CHIRALITY = False

EXPECTED_ARTIFACT_SHA256 = {
    **encoder_api.EXPECTED_ARTIFACT_SHA256,
    DECODER_FILENAME: EXPECTED_DECODER_SHA256,
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class ReleaseInferenceError(encoder_api.InferenceError):
    """Raised when the public release contract is violated."""


@dataclass(frozen=True, slots=True)
class EmbeddingBundle:
    embeddings: np.ndarray
    input_rows: np.ndarray
    input_ids: np.ndarray
    input_smiles: np.ndarray
    canonical_smiles: np.ndarray
    molecule_hashes: np.ndarray
    atom_counts: np.ndarray
    file_sha256: str


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_rank: int
    source_kind: str
    source_rank: int
    raw_smiles: str
    token_error: str
    decoder_log_probability: float
    generated_length: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseInferenceError(message)


class _DiskBackedEmbeddingStore:
    """Stage accepted rows and vectors without collection-sized Python objects."""

    _STRING_FIELDS = (
        "input_id",
        "input_smiles",
        "canonical_smiles",
        "molecule_hash",
    )

    def __init__(self, directory: Path, dimensions: int) -> None:
        require(dimensions > 0, "Embedding staging dimensions must be positive")
        self.directory = directory
        self.dimensions = dimensions
        self.accepted_count = 0
        self._finalized = False
        self._maximum_lengths = {name: 0 for name in self._STRING_FIELDS}
        self._vector_path = directory / "embeddings.float32"
        self._vector_handle = self._vector_path.open("w+b")
        self._database: sqlite3.Connection | None = sqlite3.connect(
            directory / "rows.sqlite3"
        )
        self._database.execute("PRAGMA journal_mode=OFF")
        self._database.execute("PRAGMA synchronous=OFF")
        self._database.execute("PRAGMA temp_store=FILE")
        self._database.executescript(
            """
            CREATE TABLE accepted_rows (
                position INTEGER PRIMARY KEY,
                input_row INTEGER NOT NULL,
                input_id TEXT NOT NULL,
                input_smiles TEXT NOT NULL,
                canonical_smiles TEXT NOT NULL,
                molecule_hash TEXT NOT NULL,
                atom_count INTEGER NOT NULL
            );
            CREATE TABLE accepted_hashes (
                molecule_hash TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE seen_input_ids (
                input_id TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            """
        )

    def observe_input_id(self, input_id: str) -> bool:
        """Record an input ID and return whether it was already present."""

        if not input_id:
            return False
        database = self._active_database()
        changes_before = database.total_changes
        database.execute(
            "INSERT OR IGNORE INTO seen_input_ids(input_id) VALUES (?)",
            (input_id,),
        )
        return database.total_changes == changes_before

    def append(
        self,
        records: Sequence[encoder_api.PendingMolecule],
        vectors: np.ndarray,
    ) -> None:
        require(not self._finalized, "Embedding staging is already finalized")
        require(bool(records), "Cannot stage an empty embedding batch")
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        require(
            matrix.shape == (len(records), self.dimensions),
            "Encoder returned an unexpected batch shape",
        )
        require(bool(np.isfinite(matrix).all()), "Embeddings contain non-finite values")
        matrix.tofile(self._vector_handle)

        database = self._active_database()
        rows: list[tuple[Any, ...]] = []
        hashes: list[tuple[str]] = []
        for offset, record in enumerate(records):
            position = self.accepted_count + offset
            rows.append(
                (
                    position,
                    int(record.input_row),
                    record.input_id,
                    record.input_smiles,
                    record.canonical_smiles,
                    record.molecule_hash,
                    int(record.atom_count),
                )
            )
            hashes.append((record.molecule_hash,))
            for name in self._STRING_FIELDS:
                self._maximum_lengths[name] = max(
                    self._maximum_lengths[name],
                    len(str(getattr(record, name))),
                )
        database.executemany(
            """
            INSERT INTO accepted_rows(
                position,
                input_row,
                input_id,
                input_smiles,
                canonical_smiles,
                molecule_hash,
                atom_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        database.executemany(
            "INSERT OR IGNORE INTO accepted_hashes(molecule_hash) VALUES (?)",
            hashes,
        )
        self.accepted_count += len(records)

    def materialize_arrays(self) -> tuple[dict[str, np.ndarray], int]:
        """Return disk-backed arrays ready for streaming into the final NPZ."""

        require(not self._finalized, "Embedding staging is already finalized")
        require(self.accepted_count > 0, "No accepted embeddings were staged")
        database = self._active_database()
        database.commit()
        self._vector_handle.flush()
        os.fsync(self._vector_handle.fileno())
        self._vector_handle.close()
        expected_bytes = (
            self.accepted_count
            * self.dimensions
            * np.dtype(np.float32).itemsize
        )
        require(
            self._vector_path.stat().st_size == expected_bytes,
            "Disk-backed embedding staging size is inconsistent",
        )

        arrays: dict[str, np.ndarray] = {
            "embeddings": np.memmap(
                self._vector_path,
                mode="r",
                dtype=np.float32,
                shape=(self.accepted_count, self.dimensions),
                order="C",
            ),
            "input_row": np.lib.format.open_memmap(
                self.directory / "input_row.npy",
                mode="w+",
                dtype=np.int64,
                shape=(self.accepted_count,),
            ),
            "atom_count": np.lib.format.open_memmap(
                self.directory / "atom_count.npy",
                mode="w+",
                dtype=np.int32,
                shape=(self.accepted_count,),
            ),
        }
        for name in self._STRING_FIELDS:
            arrays[name] = np.lib.format.open_memmap(
                self.directory / f"{name}.npy",
                mode="w+",
                dtype=np.dtype(f"<U{max(1, self._maximum_lengths[name])}"),
                shape=(self.accepted_count,),
            )

        cursor = database.execute(
            """
            SELECT
                input_row,
                input_id,
                input_smiles,
                canonical_smiles,
                molecule_hash,
                atom_count
            FROM accepted_rows
            ORDER BY position
            """
        )
        offset = 0
        while batch := cursor.fetchmany(8192):
            stop = offset + len(batch)
            columns = tuple(zip(*batch, strict=True))
            arrays["input_row"][offset:stop] = columns[0]
            arrays["input_id"][offset:stop] = columns[1]
            arrays["input_smiles"][offset:stop] = columns[2]
            arrays["canonical_smiles"][offset:stop] = columns[3]
            arrays["molecule_hash"][offset:stop] = columns[4]
            arrays["atom_count"][offset:stop] = columns[5]
            offset = stop
        require(
            offset == self.accepted_count,
            "Disk-backed row metadata is not aligned to embeddings",
        )
        for name, array in arrays.items():
            if name != "embeddings" and isinstance(array, np.memmap):
                array.flush()

        unique_row = database.execute(
            "SELECT COUNT(*) FROM accepted_hashes"
        ).fetchone()
        require(unique_row is not None, "Cannot count unique accepted molecules")
        unique_accepted = int(unique_row[0])
        database.close()
        self._database = None
        self._finalized = True
        return arrays, unique_accepted

    def _active_database(self) -> sqlite3.Connection:
        require(
            self._database is not None,
            "Embedding staging database is closed",
        )
        assert self._database is not None
        return self._database

    def close(self) -> None:
        if not self._vector_handle.closed:
            self._vector_handle.close()
        if self._database is not None:
            self._database.close()
            self._database = None


def _close_memmap_arrays(arrays: dict[str, np.ndarray]) -> None:
    for array in arrays.values():
        if not isinstance(array, np.memmap):
            continue
        try:
            array.flush()
        except ValueError:
            pass
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_sha256sums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ReleaseInferenceError(f"Checksum manifest is missing: {path}")
    records: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2:
            raise ReleaseInferenceError(
                f"Malformed checksum manifest line {line_number}: {raw!r}"
            )
        digest, name = pieces
        name = name.lstrip("*")
        if not SHA256_PATTERN.fullmatch(digest) or Path(name).name != name:
            raise ReleaseInferenceError(
                f"Unsafe checksum record at line {line_number}: {raw!r}"
            )
        if name in records:
            raise ReleaseInferenceError(
                f"Duplicate checksum manifest entry: {name}"
            )
        records[name] = digest
    return records


def validate_artifact_hashes(models_dir: Path) -> dict[str, str]:
    models_dir = models_dir.resolve()
    require(models_dir.is_dir(), f"Models directory does not exist: {models_dir}")
    manifest = parse_sha256sums(models_dir / "SHA256SUMS")
    require(
        manifest == EXPECTED_ARTIFACT_SHA256,
        "SHA256SUMS does not exactly describe the five release artifacts",
    )
    observed: dict[str, str] = {}
    for name, expected in EXPECTED_ARTIFACT_SHA256.items():
        path = models_dir / name
        require(path.is_file(), f"Required release artifact is missing: {path}")
        digest = encoder_api.sha256_file(path)
        require(
            digest == expected,
            f"Artifact hash mismatch for {name}: expected {expected}, observed {digest}",
        )
        observed[name] = digest
    return observed


def validate_hybrid_encoder(bundle: encoder_api.ModelBundle) -> None:
    require(
        bundle.embedding_dimensions == EMBEDDING_DIMENSIONS,
        "Released encoder must return 384 dimensions",
    )
    require(
        bundle.graph_dimensions == 256 and bundle.mean_node_dimensions == 128,
        "Released encoder block dimensions changed",
    )
    require(
        bundle.mean_node_weight == MEAN_NODE_WEIGHT,
        "Released encoder is not the frozen hybrid ×3 representation",
    )


def load_decoder(
    models_dir: Path,
    device: torch.device,
    artifact_hashes: dict[str, str] | None = None,
) -> tuple[ConditionalSmilesTransformer, dict[str, Any], dict[str, str]]:
    hashes = artifact_hashes or validate_artifact_hashes(models_dir)
    decoder_path = models_dir.resolve() / DECODER_FILENAME
    try:
        payload = torch.load(
            decoder_path, map_location=device, weights_only=False
        )
    except Exception as error:
        raise ReleaseInferenceError(
            f"Cannot deserialize packaged decoder: {error}"
        ) from error
    require(isinstance(payload, dict), "Decoder export is not a dictionary")
    require(payload.get("schema_version") == 1, "Unsupported decoder schema")
    require(
        payload.get("artifact_type")
        == "conditional_smiles_decoder_inference",
        "Artifact is not the compact frozen decoder export",
    )
    require(
        payload.get("embedding_space") == EMBEDDING_SPACE,
        "Decoder is not conditioned on released_hybrid_w3",
    )
    require(
        payload.get("condition_dimensions") == EMBEDDING_DIMENSIONS,
        "Decoder condition dimension changed",
    )
    require(
        payload.get("source_training_checkpoint_sha256")
        == EXPECTED_TRAINING_DECODER_SHA256,
        "Decoder export is not derived from the frozen Step-2 checkpoint",
    )
    require(
        payload.get("protocol_sha256") == EXPECTED_STEP2_PROTOCOL_SHA256,
        "Decoder Step-2 protocol identity changed",
    )
    require(
        payload.get("manifest_sha256") == EXPECTED_STEP2_MANIFEST_SHA256,
        "Decoder Step-2 input manifest identity changed",
    )
    require(
        payload.get("model_config") == EXPECTED_DECODER_CONFIG,
        "Decoder architecture/configuration changed",
    )
    frozen_inputs = payload.get("frozen_input_sha256")
    require(isinstance(frozen_inputs, dict), "Decoder lacks frozen input hashes")
    expected_frozen_inputs = {
        "checkpoint": hashes["representation-best.pt"],
        "packaged_checkpoint": hashes["representation-best.pt"],
        "calibrator": hashes["representation-calibrator.pt"],
        "packaged_calibrator": hashes["representation-calibrator.pt"],
        "representation_selection": hashes["representation_selection.json"],
        "resolved_config": hashes["resolved_config.json"],
        "packaged_resolved_config": hashes["resolved_config.json"],
        "inference_entrypoint": encoder_api.sha256_file(
            SCRIPT_DIR / "generate_embeddings.py"
        ),
    }
    for key, expected in expected_frozen_inputs.items():
        require(
            frozen_inputs.get(key) == expected,
            f"Decoder frozen input identity mismatch at {key}",
        )
    state = payload.get("model_state_dict")
    require(isinstance(state, dict), "Decoder export lacks a model state dictionary")
    model = ConditionalSmilesTransformer(payload["model_config"]).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ReleaseInferenceError(
            f"Decoder state is incompatible with the frozen architecture: {error}"
        ) from error
    require(
        decoder_parameter_count(model) == EXPECTED_DECODER_PARAMETERS,
        "Decoder parameter count changed",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        require(bool(torch.isfinite(parameter).all()), "Decoder has non-finite weights")
    return model, payload, hashes


def configure_encode_runtime(threads: int) -> None:
    require(threads > 0, "--threads must be positive")
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def configure_decode_runtime(threads: int) -> None:
    require(threads > 0, "--threads must be positive")
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def output_companions(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_suffix(".rejections.csv"),
        output_path.with_suffix(".metadata.json"),
    )


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        encoder_api.fsync_text_handle(handle)


def run_encode(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    models_dir = args.models_dir.resolve()
    output_path = args.output.resolve()
    require(input_path.is_file(), f"Input CSV does not exist: {input_path}")
    require(output_path.suffix.lower() == ".npz", "--output must end in .npz")
    require(args.batch_size > 0, "--batch-size must be positive")
    require(args.node_budget > 0, "--node-budget must be positive")
    require(args.verify_rows > 0, "--verify-rows must be positive")
    if args.limit is not None:
        require(args.limit > 0, "--limit must be positive")
    configure_encode_runtime(args.threads)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path, metadata_path = output_companions(output_path)
    outputs = (output_path, rejections_path, metadata_path)
    require(input_path not in outputs, "Input and output paths must differ")
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise ReleaseInferenceError(
                f"Output exists; pass --overwrite to replace it: {path}"
            )

    artifact_hashes = validate_artifact_hashes(models_dir)
    device = encoder_api.resolve_device(args.device)
    bundle = encoder_api.load_model_bundle(models_dir, device)
    validate_hybrid_encoder(bundle)
    encoder = encoder_api.build_smiles_encoder(
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
    backend_info = encoder_api.implementation_metadata(encoder)
    pipeline_batches = max(1, int(backend_info["workers"]))
    pipeline_graph_budget = args.batch_size * pipeline_batches
    pipeline_node_budget = args.node_budget * pipeline_batches

    total_rows = 0
    rejected_rows = 0
    pending_nodes = 0
    pending: list[encoder_api.PendingMolecule] = []
    rejection_reasons: Counter[str] = Counter()
    duplicate_nonempty_ids = 0
    id_column: str | None = None
    disk_arrays: dict[str, np.ndarray] = {}

    npz_temporary = encoder_api.temporary_path(output_path)
    rejection_temporary = encoder_api.temporary_path(rejections_path)
    metadata_temporary = encoder_api.temporary_path(metadata_path)
    temporaries = (npz_temporary, rejection_temporary, metadata_temporary)
    staging_directory = tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.stream-",
        dir=output_path.parent,
    )
    store = _DiskBackedEmbeddingStore(
        Path(staging_directory.name),
        EMBEDDING_DIMENSIONS,
    )
    encoder_open = True

    def close_encoder() -> None:
        nonlocal encoder_open
        if encoder_open:
            encoder_open = False
            encoder.close()

    def flush_pending() -> None:
        nonlocal pending, pending_nodes
        if not pending:
            return
        store.append(pending, encoder_api.encode_batch(encoder, pending))
        pending = []
        pending_nodes = 0

    try:
        try:
            with rejection_temporary.open(
                "w", encoding="utf-8", newline=""
            ) as rejection_handle:
                rejection_writer = csv.writer(
                    rejection_handle,
                    lineterminator="\n",
                )
                rejection_writer.writerow(
                    ["input_row", "input_id", "input_smiles", "reason"]
                )
                with input_path.open(
                    "r", encoding="utf-8-sig", newline=""
                ) as input_handle:
                    reader = csv.DictReader(input_handle)
                    headers = list(reader.fieldnames or [])
                    require(bool(headers), "Input CSV is empty or lacks a header")
                    require(
                        len(headers) == len(set(headers)),
                        "Input CSV contains duplicate column names",
                    )
                    require(
                        args.smiles_column in headers,
                        f"SMILES column {args.smiles_column!r} is absent from "
                        "the input CSV",
                    )
                    id_column = encoder_api.resolve_id_column(
                        args.id_column,
                        headers,
                    )
                    for input_row, row in enumerate(reader, start=1):
                        if (
                            args.limit is not None
                            and total_rows >= args.limit
                        ):
                            break
                        total_rows += 1
                        if None in row or any(
                            value is None for value in row.values()
                        ):
                            raise ReleaseInferenceError(
                                f"Malformed CSV record at input row {input_row}: "
                                "field count differs from the header"
                            )
                        raw_smiles = str(row[args.smiles_column])
                        input_id = (
                            str(row[id_column])
                            if id_column is not None
                            else ""
                        )
                        if store.observe_input_id(input_id):
                            duplicate_nonempty_ids += 1
                        canonical, reason = encoder_api.canonicalize_input(
                            raw_smiles,
                            bundle.resolved_config,
                        )
                        if reason is not None:
                            rejection_reasons[reason] += 1
                            rejected_rows += 1
                            rejection_writer.writerow(
                                (input_row, input_id, raw_smiles, reason)
                            )
                            if args.invalid_policy == "error":
                                raise ReleaseInferenceError(
                                    "Rejected molecule at input row "
                                    f"{input_row}: {reason}"
                                )
                            continue
                        assert canonical is not None
                        if pending and (
                            len(pending) >= pipeline_graph_budget
                            or pending_nodes + canonical.atom_count
                            > pipeline_node_budget
                        ):
                            flush_pending()
                        pending.append(
                            encoder_api.PendingMolecule(
                                input_row=input_row,
                                input_id=input_id,
                                input_smiles=raw_smiles,
                                canonical_smiles=canonical.smiles,
                                molecule_hash=canonical.molecule_hash,
                                atom_count=int(canonical.atom_count),
                            )
                        )
                        pending_nodes += int(canonical.atom_count)
                    flush_pending()
                encoder_api.fsync_text_handle(rejection_handle)
        finally:
            close_encoder()

        require(total_rows > 0, "Input CSV contains no data rows")
        require(
            store.accepted_count > 0,
            "No input molecule passed the release policy",
        )
        disk_arrays, unique_accepted_molecules = store.materialize_arrays()
        embeddings = disk_arrays["embeddings"]
        require(
            embeddings.shape
            == (store.accepted_count, EMBEDDING_DIMENSIONS),
            "Encoder returned an unexpected matrix shape",
        )

        created_utc = utc_now()
        input_sha256 = encoder_api.sha256_file(input_path)
        arrays = {
            "embeddings": disk_arrays["embeddings"],
            "input_row": disk_arrays["input_row"],
            "input_id": disk_arrays["input_id"],
            "input_smiles": disk_arrays["input_smiles"],
            "canonical_smiles": disk_arrays["canonical_smiles"],
            "molecule_hash": disk_arrays["molecule_hash"],
            "atom_count": disk_arrays["atom_count"],
            "schema_version": np.asarray(
                EMBEDDING_SCHEMA_VERSION,
                dtype=np.int64,
            ),
            "artifact_type": np.asarray(EMBEDDING_ARTIFACT_TYPE),
            "embedding_space": np.asarray(EMBEDDING_SPACE),
            "embedding_definition": np.asarray(
                encoder_api.PUBLIC_EMBEDDING_DEFINITION
            ),
            "embedding_dimensions": np.asarray(
                EMBEDDING_DIMENSIONS,
                dtype=np.int64,
            ),
            "mean_node_weight": np.asarray(
                MEAN_NODE_WEIGHT,
                dtype=np.float32,
            ),
            "encoder_checkpoint_sha256": np.asarray(
                artifact_hashes["representation-best.pt"]
            ),
            "calibrator_sha256": np.asarray(
                artifact_hashes["representation-calibrator.pt"]
            ),
            "selection_sha256": np.asarray(
                artifact_hashes["representation_selection.json"]
            ),
            "resolved_config_sha256": np.asarray(
                artifact_hashes["resolved_config.json"]
            ),
            "input_sha256": np.asarray(input_sha256),
            "created_utc": np.asarray(created_utc),
            "script_version": np.asarray(SCRIPT_VERSION),
        }

        with npz_temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())

        npz_hash = encoder_api.sha256_file(npz_temporary)
        rejection_hash = encoder_api.sha256_file(rejection_temporary)
        metadata = {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "artifact_type": EMBEDDING_ARTIFACT_TYPE,
            "script_version": SCRIPT_VERSION,
            "created_utc": created_utc,
            "input": {
                "path": str(input_path),
                "sha256": input_sha256,
                "smiles_column": args.smiles_column,
                "id_column": id_column,
                "limit": args.limit,
            },
            "rows": {
                "total": total_rows,
                "accepted": store.accepted_count,
                "rejected": rejected_rows,
                "unique_accepted_molecules": unique_accepted_molecules,
                "duplicate_nonempty_ids": duplicate_nonempty_ids,
                "rejection_reasons": dict(sorted(rejection_reasons.items())),
            },
            "embedding": {
                "space": EMBEDDING_SPACE,
                "definition": encoder_api.PUBLIC_EMBEDDING_DEFINITION,
                "dimensions": EMBEDDING_DIMENSIONS,
                "graph_dimensions": bundle.graph_dimensions,
                "mean_node_dimensions": bundle.mean_node_dimensions,
                "mean_node_weight": bundle.mean_node_weight,
                "dtype": "float32",
                "storage": "compressed_npz_without_pickle",
                "assembly": "disk_backed_streaming",
            },
            "artifacts": artifact_hashes,
            "canonicalization": bundle.resolved_config["data"][
                "canonicalization"
            ],
            "execution": {
                "device": str(device),
                "backend": backend_info["backend"],
                "batch_size": args.batch_size,
                "node_budget": args.node_budget,
                "workers": backend_info["workers"],
                "pipeline_batches": pipeline_batches,
                "verify_rows": (
                    args.verify_rows if args.backend == "verify" else 0
                ),
                "threads": args.threads,
                "invalid_policy": args.invalid_policy,
                "encoder_memory_contract": (
                    "active_pipeline_batches_plus_bounded_compression_buffers"
                ),
                "staging": "disk_backed_float32_and_sqlite",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
                "torch": torch.__version__,
                "torch_geometric": package_version("torch-geometric"),
            },
            "outputs": {
                output_path.name: {
                    "sha256": npz_hash,
                    "rows": store.accepted_count,
                    "dimensions": EMBEDDING_DIMENSIONS,
                },
                rejections_path.name: {
                    "sha256": rejection_hash,
                    "rows": rejected_rows,
                },
            },
        }
        write_json(metadata_temporary, metadata)
        os.replace(npz_temporary, output_path)
        os.replace(rejection_temporary, rejections_path)
        os.replace(metadata_temporary, metadata_path)
    finally:
        close_encoder()
        _close_memmap_arrays(disk_arrays)
        store.close()
        staging_directory.cleanup()
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)

    return {
        "accepted": store.accepted_count,
        "rejected": rejected_rows,
        "dimensions": EMBEDDING_DIMENSIONS,
        "embedding_space": EMBEDDING_SPACE,
        "embeddings": str(output_path),
        "rejections": str(rejections_path),
        "metadata": str(metadata_path),
        "device": str(device),
        "backend": backend_info["backend"],
    }


def scalar(npz: Any, key: str) -> Any:
    require(key in npz.files, f"Embedding bundle lacks {key!r}")
    value = np.asarray(npz[key])
    require(value.shape == (), f"Embedding bundle field {key!r} must be scalar")
    return value.item()


def load_embedding_bundle(path: Path) -> EmbeddingBundle:
    path = path.resolve()
    require(path.is_file(), f"Embedding bundle does not exist: {path}")
    require(path.suffix.lower() == ".npz", "Embedding bundle must be .npz")
    required_arrays = {
        "embeddings",
        "input_row",
        "input_id",
        "input_smiles",
        "canonical_smiles",
        "molecule_hash",
        "atom_count",
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            require(
                int(scalar(payload, "schema_version"))
                == EMBEDDING_SCHEMA_VERSION,
                "Unsupported embedding bundle schema",
            )
            require(
                str(scalar(payload, "artifact_type"))
                == EMBEDDING_ARTIFACT_TYPE,
                "Input is not a gMolAI release embedding bundle",
            )
            require(
                str(scalar(payload, "embedding_space")) == EMBEDDING_SPACE,
                "Embedding bundle is not released_hybrid_w3",
            )
            require(
                int(scalar(payload, "embedding_dimensions"))
                == EMBEDDING_DIMENSIONS,
                "Embedding bundle dimension identity changed",
            )
            require(
                float(scalar(payload, "mean_node_weight"))
                == MEAN_NODE_WEIGHT,
                "Embedding bundle is not hybrid ×3",
            )
            expected_identity = {
                "encoder_checkpoint_sha256": encoder_api.EXPECTED_ARTIFACT_SHA256[
                    "representation-best.pt"
                ],
                "calibrator_sha256": encoder_api.EXPECTED_ARTIFACT_SHA256[
                    "representation-calibrator.pt"
                ],
                "selection_sha256": encoder_api.EXPECTED_ARTIFACT_SHA256[
                    "representation_selection.json"
                ],
                "resolved_config_sha256": encoder_api.EXPECTED_ARTIFACT_SHA256[
                    "resolved_config.json"
                ],
                "embedding_definition": encoder_api.PUBLIC_EMBEDDING_DEFINITION,
            }
            for key, expected in expected_identity.items():
                require(
                    str(scalar(payload, key)) == expected,
                    f"Embedding bundle identity mismatch at {key}",
                )
            require(
                required_arrays.issubset(payload.files),
                "Embedding bundle lacks one or more row arrays",
            )
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            input_rows = np.asarray(payload["input_row"], dtype=np.int64)
            input_ids = np.asarray(payload["input_id"], dtype=str)
            input_smiles = np.asarray(payload["input_smiles"], dtype=str)
            canonical_smiles = np.asarray(payload["canonical_smiles"], dtype=str)
            molecule_hashes = np.asarray(payload["molecule_hash"], dtype=str)
            atom_counts = np.asarray(payload["atom_count"], dtype=np.int32)
    except (OSError, ValueError) as error:
        raise ReleaseInferenceError(
            f"Cannot read valid embedding bundle {path}: {error}"
        ) from error

    require(
        embeddings.ndim == 2
        and embeddings.shape[1] == EMBEDDING_DIMENSIONS,
        "Embedding matrix must have shape (rows, 384)",
    )
    rows = embeddings.shape[0]
    for name, values in (
        ("input_row", input_rows),
        ("input_id", input_ids),
        ("input_smiles", input_smiles),
        ("canonical_smiles", canonical_smiles),
        ("molecule_hash", molecule_hashes),
        ("atom_count", atom_counts),
    ):
        require(values.shape == (rows,), f"{name} is not aligned to embeddings")
    require(rows > 0, "Embedding bundle contains no rows")
    require(bool(np.isfinite(embeddings).all()), "Embedding matrix is non-finite")
    require(
        len(set(int(value) for value in input_rows)) == rows,
        "Embedding input_row values are not unique",
    )
    for index, (smiles, digest) in enumerate(
        zip(canonical_smiles, molecule_hashes, strict=True)
    ):
        expected = hashlib.sha256(str(smiles).encode("utf-8")).hexdigest()
        require(
            str(digest) == expected,
            f"Canonical identity hash mismatch at embedding row {index}",
        )
        require(
            Chem.MolFromSmiles(str(smiles)) is not None,
            f"Invalid canonical seed SMILES at embedding row {index}",
        )
    return EmbeddingBundle(
        embeddings=np.ascontiguousarray(embeddings),
        input_rows=input_rows,
        input_ids=input_ids,
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        molecule_hashes=molecule_hashes,
        atom_counts=atom_counts,
        file_sha256=encoder_api.sha256_file(path),
    )


def stable_digest(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(str(value) for value in parts).encode("utf-8")
    ).hexdigest()


def sample_seed(target_hash: str) -> int:
    return int(
        stable_digest(
            GLOBAL_SEED,
            SAMPLING_PHASE,
            SAMPLE_POOL_NAME,
            target_hash,
        )[:16],
        16,
    ) % (2**63 - 1)


def proposal_from_tokens(
    tokens: Sequence[int],
    score: float,
    length: int,
    source_kind: str,
    source_rank: int,
    proposal_rank: int = 0,
) -> Proposal:
    raw_smiles, token_error = decode_tokens(tokens)
    return Proposal(
        proposal_rank=proposal_rank,
        source_kind=source_kind,
        source_rank=source_rank,
        raw_smiles=raw_smiles,
        token_error=token_error,
        decoder_log_probability=float(score),
        generated_length=int(length),
    )


def generate_proposals(
    model: ConditionalSmilesTransformer,
    condition_vector: np.ndarray,
    target_hash: str,
    device: torch.device,
) -> list[Proposal]:
    condition = torch.as_tensor(
        condition_vector[None, :], dtype=torch.float32, device=device
    )
    greedy_tokens = model.generate(
        condition, maximum_steps=MAXIMUM_SMILES_BYTES
    ).cpu().numpy()[0]
    greedy = proposal_from_tokens(
        greedy_tokens,
        math.nan,
        len(greedy_tokens),
        "greedy",
        0,
    )

    beam_tokens, beam_scores, beam_lengths = generate_beam_pool(
        model,
        condition,
        maximum_steps=MAXIMUM_SMILES_BYTES,
        beam_width=BEAM_WIDTH,
    )
    order = beam_order(beam_scores, beam_lengths, LENGTH_PENALTY).cpu().numpy()[0]
    beam_tokens_np = beam_tokens.cpu().numpy()[0]
    beam_scores_np = beam_scores.cpu().numpy()[0]
    beam_lengths_np = beam_lengths.cpu().numpy()[0]
    beam_stream = [
        proposal_from_tokens(
            beam_tokens_np[int(source_index)],
            float(beam_scores_np[int(source_index)]),
            int(beam_lengths_np[int(source_index)]),
            "beam",
            int(source_index) + 1,
        )
        for source_index in order
    ]
    del beam_tokens, beam_scores, beam_lengths

    sample_tokens, sample_scores, sample_lengths = generate_seeded_sample_pool(
        model,
        condition,
        maximum_steps=MAXIMUM_SMILES_BYTES,
        draws=SAMPLE_DRAWS,
        temperature=SAMPLE_TEMPERATURE,
        top_p=SAMPLE_TOP_P,
        seeds=[sample_seed(target_hash)],
    )
    sample_tokens_np = sample_tokens.cpu().numpy()[0]
    sample_scores_np = sample_scores.cpu().numpy()[0]
    sample_lengths_np = sample_lengths.cpu().numpy()[0]
    sample_stream = [greedy]
    sample_stream.extend(
        proposal_from_tokens(
            sample_tokens_np[draw],
            float(sample_scores_np[draw]),
            int(sample_lengths_np[draw]),
            "sample",
            draw + 1,
        )
        for draw in range(SAMPLE_DRAWS)
    )
    del sample_tokens, sample_scores, sample_lengths, condition

    sources = proportional_merge(
        [("beam", index) for index in range(BEAM_HYPOTHESES)],
        [("sample", index) for index in range(SAMPLE_HYPOTHESES)],
    )
    stream = [
        beam_stream[index] if kind == "beam" else sample_stream[index]
        for kind, index in sources
    ]
    require(len(stream) == 1000, "Frozen hybrid strategy did not yield 1,000 slots")
    return [
        Proposal(
            proposal_rank=rank,
            source_kind=proposal.source_kind,
            source_rank=proposal.source_rank,
            raw_smiles=proposal.raw_smiles,
            token_error=proposal.token_error,
            decoder_log_probability=proposal.decoder_log_probability,
            generated_length=proposal.generated_length,
        )
        for rank, proposal in enumerate(stream, start=1)
    ]


def filter_candidates(
    proposals: Sequence[Proposal],
    *,
    seed_input_row: int,
    seed_id: str,
    seed_canonical_smiles: str,
    seed_hash: str,
    resolved_config: dict[str, Any],
    include_seed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        includeChirality=MORGAN_INCLUDE_CHIRALITY,
    )
    seed_molecule = Chem.MolFromSmiles(seed_canonical_smiles)
    require(seed_molecule is not None, "Seed canonical SMILES cannot be parsed")
    seed_fingerprint = fingerprint_generator.GetFingerprint(seed_molecule)

    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    token_errors: Counter[str] = Counter()
    chemistry_rejections: Counter[str] = Counter()
    accepted_proposals = 0
    duplicate_proposals = 0
    excluded_seed_identity = 0
    for proposal in proposals:
        if proposal.token_error:
            token_errors[proposal.token_error] += 1
            continue
        canonical, reason = encoder_api.canonicalize_input(
            proposal.raw_smiles, resolved_config
        )
        if reason is not None:
            chemistry_rejections[reason] += 1
            continue
        assert canonical is not None
        accepted_proposals += 1
        if canonical.molecule_hash in seen_hashes:
            duplicate_proposals += 1
            continue
        seen_hashes.add(canonical.molecule_hash)
        is_seed = canonical.molecule_hash == seed_hash
        if is_seed and not include_seed:
            excluded_seed_identity += 1
            continue
        molecule = Chem.MolFromSmiles(canonical.smiles)
        require(molecule is not None, "Accepted candidate cannot be reparsed")
        fingerprint = fingerprint_generator.GetFingerprint(molecule)
        similarity = float(
            DataStructs.TanimotoSimilarity(seed_fingerprint, fingerprint)
        )
        require(
            math.isfinite(similarity) and 0.0 <= similarity <= 1.0,
            "Morgan/Tanimoto similarity is invalid",
        )
        rows.append(
            {
                "seed_input_row": seed_input_row,
                "seed_id": seed_id,
                "seed_canonical_smiles": seed_canonical_smiles,
                "candidate_rank": len(rows) + 1,
                "proposal_rank": proposal.proposal_rank,
                "source_kind": proposal.source_kind,
                "source_rank": proposal.source_rank,
                "canonical_smiles": canonical.smiles,
                "molecule_hash": canonical.molecule_hash,
                "morgan_tanimoto": similarity,
                "is_seed_identity": is_seed,
                "cumulative_decoder_log_probability": (
                    proposal.decoder_log_probability
                ),
                "generated_length": proposal.generated_length,
            }
        )

    require(
        len({row["canonical_smiles"] for row in rows}) == len(rows),
        "Candidate canonical SMILES are not unique",
    )
    stats = {
        "raw_proposals": len(proposals),
        "token_decodable_proposals": len(proposals) - sum(token_errors.values()),
        "policy_accepted_proposals": accepted_proposals,
        "duplicate_or_redundant_proposals": duplicate_proposals,
        "excluded_seed_identity": excluded_seed_identity,
        "retained_unique_candidates": len(rows),
        "token_errors": dict(sorted(token_errors.items())),
        "chemistry_policy_rejections": dict(
            sorted(chemistry_rejections.items())
        ),
    }
    return rows, stats


def safe_seed_filename(input_row: int, input_id: str, molecule_hash: str) -> str:
    slug = SAFE_NAME_PATTERN.sub("-", input_id.strip()).strip("-._")
    if not slug:
        slug = "molecule"
    slug = slug[:48]
    return f"seed-{input_row:06d}-{slug}-{molecule_hash[:8]}.csv"


CANDIDATE_COLUMNS = (
    "seed_input_row",
    "seed_id",
    "seed_canonical_smiles",
    "candidate_rank",
    "proposal_rank",
    "source_kind",
    "source_rank",
    "canonical_smiles",
    "molecule_hash",
    "morgan_tanimoto",
    "is_seed_identity",
    "cumulative_decoder_log_probability",
    "generated_length",
)


def write_candidate_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["morgan_tanimoto"] = format(
                float(output["morgan_tanimoto"]), ".6f"
            )
            score = float(output["cumulative_decoder_log_probability"])
            output["cumulative_decoder_log_probability"] = (
                format(score, ".9g") if math.isfinite(score) else ""
            )
            writer.writerow(output)
        encoder_api.fsync_text_handle(handle)


def run_decode(args: argparse.Namespace) -> dict[str, Any]:
    embeddings_path = args.embeddings.resolve()
    models_dir = args.models_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.seed_limit is not None:
        require(args.seed_limit > 0, "--seed-limit must be positive")
    configure_decode_runtime(args.threads)
    artifact_hashes = validate_artifact_hashes(models_dir)
    embedding_bundle = load_embedding_bundle(embeddings_path)
    rows_to_generate = len(embedding_bundle.embeddings)
    if args.seed_limit is not None:
        rows_to_generate = min(rows_to_generate, args.seed_limit)
    require(rows_to_generate > 0, "No embedded seed was selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "generation.metadata.json"
    filenames = [
        safe_seed_filename(
            int(embedding_bundle.input_rows[index]),
            str(embedding_bundle.input_ids[index]),
            str(embedding_bundle.molecule_hashes[index]),
        )
        for index in range(rows_to_generate)
    ]
    require(len(filenames) == len(set(filenames)), "Candidate filenames collide")
    targets = [output_dir / name for name in filenames]
    for path in [*targets, metadata_path]:
        if path.exists() and not args.overwrite:
            raise ReleaseInferenceError(
                f"Output exists; pass --overwrite to replace it: {path}"
            )

    device = encoder_api.resolve_device(args.device)
    model, decoder_payload, _ = load_decoder(
        models_dir, device, artifact_hashes
    )
    resolved_config = encoder_api.load_json_object(
        models_dir / "resolved_config.json"
    )
    started = time.monotonic()
    seed_summaries: list[dict[str, Any]] = []
    output_records: dict[str, dict[str, Any]] = {}
    temporary_paths: list[Path] = []
    completed_pairs: list[tuple[Path, Path]] = []
    try:
        for index in range(rows_to_generate):
            seed_started = time.monotonic()
            proposals = generate_proposals(
                model,
                embedding_bundle.embeddings[index],
                str(embedding_bundle.molecule_hashes[index]),
                device,
            )[: args.proposal_budget]
            candidate_rows, stats = filter_candidates(
                proposals,
                seed_input_row=int(embedding_bundle.input_rows[index]),
                seed_id=str(embedding_bundle.input_ids[index]),
                seed_canonical_smiles=str(
                    embedding_bundle.canonical_smiles[index]
                ),
                seed_hash=str(embedding_bundle.molecule_hashes[index]),
                resolved_config=resolved_config,
                include_seed=args.include_seed,
            )
            target = targets[index]
            temporary = encoder_api.temporary_path(target)
            temporary_paths.append(temporary)
            write_candidate_csv(temporary, candidate_rows)
            output_hash = encoder_api.sha256_file(temporary)
            output_records[target.name] = {
                "sha256": output_hash,
                "rows": len(candidate_rows),
                "seed_input_row": int(embedding_bundle.input_rows[index]),
                "seed_id": str(embedding_bundle.input_ids[index]),
            }
            seed_summary = {
                "seed_input_row": int(embedding_bundle.input_rows[index]),
                "seed_id": str(embedding_bundle.input_ids[index]),
                "seed_canonical_smiles": str(
                    embedding_bundle.canonical_smiles[index]
                ),
                "seed_molecule_hash": str(
                    embedding_bundle.molecule_hashes[index]
                ),
                "sampling_seed": sample_seed(
                    str(embedding_bundle.molecule_hashes[index])
                ),
                "output": target.name,
                "wall_seconds": time.monotonic() - seed_started,
                **stats,
            }
            seed_summaries.append(seed_summary)
            completed_pairs.append((temporary, target))
            print(
                json.dumps(
                    {
                        "seed": index + 1,
                        "of": rows_to_generate,
                        "seed_id": seed_summary["seed_id"],
                        "raw_proposals": stats["raw_proposals"],
                        "retained_unique_candidates": stats[
                            "retained_unique_candidates"
                        ],
                        "output": str(target),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        metadata_temporary = encoder_api.temporary_path(metadata_path)
        temporary_paths.append(metadata_temporary)
        metadata = {
            "schema_version": 1,
            "artifact_type": "gmolai_frozen_candidate_library",
            "script_version": SCRIPT_VERSION,
            "created_utc": utc_now(),
            "embedding_input": {
                "path": str(embeddings_path),
                "sha256": embedding_bundle.file_sha256,
                "rows_available": len(embedding_bundle.embeddings),
                "rows_generated": rows_to_generate,
                "space": EMBEDDING_SPACE,
                "unperturbed": True,
            },
            "strategy": {
                "name": STRATEGY_NAME,
                "proposal_budget": args.proposal_budget,
                "proposal_budget_definition": (
                    "nested prefix of frozen raw decoder proposal slots"
                ),
                "beam_width": BEAM_WIDTH,
                "beam_hypotheses_in_full_stream": BEAM_HYPOTHESES,
                "length_penalty": LENGTH_PENALTY,
                "sample_hypotheses_in_full_stream": SAMPLE_HYPOTHESES,
                "sample_pool": SAMPLE_POOL_NAME,
                "sample_draws_materialized": SAMPLE_DRAWS,
                "temperature": SAMPLE_TEMPERATURE,
                "top_p": SAMPLE_TOP_P,
                "maximum_smiles_bytes": MAXIMUM_SMILES_BYTES,
                "global_seed": GLOBAL_SEED,
                "sampling_phase": SAMPLING_PHASE,
                "sampling_seed_definition": (
                    "SHA256(global_seed, phase, pool_name, molecule_hash) "
                    "truncated to 63 bits"
                ),
                "merge": "proportional_balanced",
                "target_chemistry_used_for_generation_or_ordering": False,
            },
            "candidate_policy": {
                "canonicalization": resolved_config["data"]["canonicalization"],
                "unique_identity": "canonical isomeric SMILES SHA-256",
                "first_occurrence_retained": True,
                "include_seed_identity": bool(args.include_seed),
                "similarity": {
                    "fingerprint": "Morgan",
                    "radius": MORGAN_RADIUS,
                    "bits": MORGAN_BITS,
                    "include_chirality": MORGAN_INCLUDE_CHIRALITY,
                    "metric": "Tanimoto",
                },
            },
            "artifacts": artifact_hashes,
            "decoder": {
                "artifact_type": decoder_payload["artifact_type"],
                "embedding_space": decoder_payload["embedding_space"],
                "condition_dimensions": decoder_payload[
                    "condition_dimensions"
                ],
                "parameters": EXPECTED_DECODER_PARAMETERS,
                "source_training_checkpoint_sha256": decoder_payload[
                    "source_training_checkpoint_sha256"
                ],
                "contains_optimizer_state": False,
                "contains_gmolai_parameters": False,
            },
            "execution": {
                "device": str(device),
                "threads": args.threads,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
                "torch": torch.__version__,
                "wall_seconds": time.monotonic() - started,
            },
            "seeds": seed_summaries,
            "outputs": output_records,
        }
        write_json(metadata_temporary, metadata)
        for temporary, target in completed_pairs:
            os.replace(temporary, target)
        os.replace(metadata_temporary, metadata_path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)

    return {
        "seeds": rows_to_generate,
        "proposal_budget_per_seed": args.proposal_budget,
        "retained_unique_candidates": sum(
            item["retained_unique_candidates"] for item in seed_summaries
        ),
        "strategy": STRATEGY_NAME,
        "output_dir": str(output_dir),
        "metadata": str(metadata_path),
        "device": str(device),
    }


def run_validate(args: argparse.Namespace) -> dict[str, Any]:
    models_dir = args.models_dir.resolve()
    configure_encode_runtime(args.threads)
    hashes = validate_artifact_hashes(models_dir)
    device = encoder_api.resolve_device(args.device)
    encoder_bundle = encoder_api.load_model_bundle(models_dir, device)
    validate_hybrid_encoder(encoder_bundle)
    decoder, payload, _ = load_decoder(models_dir, device, hashes)
    return {
        "status": "passed",
        "models_dir": str(models_dir),
        "artifacts": hashes,
        "encoder": {
            "embedding_space": EMBEDDING_SPACE,
            "dimensions": encoder_bundle.embedding_dimensions,
            "graph_dimensions": encoder_bundle.graph_dimensions,
            "mean_node_dimensions": encoder_bundle.mean_node_dimensions,
            "mean_node_weight": encoder_bundle.mean_node_weight,
            "checkpoint_global_step": int(
                encoder_bundle.checkpoint["global_step"]
            ),
            "calibrator_dimensions": int(
                encoder_bundle.coordinate_mean.numel()
            ),
        },
        "decoder": {
            "artifact_type": payload["artifact_type"],
            "embedding_space": payload["embedding_space"],
            "condition_dimensions": payload["condition_dimensions"],
            "parameters": decoder_parameter_count(decoder),
            "training_allowed": False,
            "contains_optimizer_state": False,
            "contains_gmolai_parameters": False,
        },
        "strategy": {
            "name": STRATEGY_NAME,
            "proposal_budgets": list(FROZEN_PROPOSAL_BUDGETS),
            "beam_hypotheses": BEAM_HYPOTHESES,
            "sample_hypotheses": SAMPLE_HYPOTHESES,
            "temperature": SAMPLE_TEMPERATURE,
            "top_p": SAMPLE_TOP_P,
            "maximum_smiles_bytes": MAXIMUM_SMILES_BYTES,
        },
        "device": str(device),
    }


def add_common_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Release artifact directory (default: inference/models)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:<index> (default: auto)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="PyTorch CPU threads (default: min(8, available CPUs))",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Public gMolAI release CLI: encode SMILES as calibrated hybrid ×3 "
            "vectors and decode unperturbed vectors with the frozen Step-2/2d "
            "candidate generator."
        )
    )
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser(
        "encode",
        help="Encode a SMILES CSV into one self-describing .npz bundle",
    )
    add_common_model_flags(encode)
    encode.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV (default: inference/data/example_smiles.csv)",
    )
    encode.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help="Output .npz bundle (default: inference/output/embeddings.npz)",
    )
    encode.add_argument("--smiles-column", default="smiles")
    encode.add_argument(
        "--id-column",
        default="auto",
        help="ID column, auto (molecule_id/id when present), or none",
    )
    encode.add_argument(
        "--backend",
        choices=("optimized", "reference", "verify"),
        default="optimized",
        help="Encoder backend (default: optimized)",
    )
    encode.add_argument("--batch-size", type=int, default=192)
    encode.add_argument("--node-budget", type=int, default=16_384)
    encode.add_argument(
        "--workers",
        default="auto",
        help="RDKit workers; auto respects Slurm/CPU affinity",
    )
    encode.add_argument(
        "--verify-rows",
        type=int,
        default=1024,
        help="Rows checked by the reference backend when --backend verify is used",
    )
    encode.add_argument(
        "--invalid-policy",
        choices=("report", "error"),
        default="report",
        help="Report rejected inputs or fail atomically on the first rejection",
    )
    encode.add_argument("--limit", type=int, help="Optional maximum input rows")
    encode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output bundle and its sidecars",
    )
    encode.set_defaults(handler=run_encode)

    decode = subparsers.add_parser(
        "decode",
        aliases=("generate",),
        help="Generate one valid/unique candidate CSV per embedded seed",
    )
    add_common_model_flags(decode)
    decode.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help="Input bundle from the encode command",
    )
    decode.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
        help="Candidate CSV directory (default: inference/output/candidates)",
    )
    decode.add_argument(
        "--proposal-budget",
        type=int,
        choices=FROZEN_PROPOSAL_BUDGETS,
        default=1000,
        help="Frozen raw-proposal prefix per seed (default: 1000)",
    )
    decode.add_argument(
        "--seed-limit",
        type=int,
        help="Generate only the first N embedded seeds",
    )
    decode.add_argument(
        "--include-seed",
        action="store_true",
        help="Retain the reconstructed seed identity if proposed",
    )
    decode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace colliding candidate CSVs and metadata",
    )
    decode.set_defaults(handler=run_decode)

    validate = subparsers.add_parser(
        "validate",
        help="Hash, deserialize, and cross-check every release artifact",
    )
    add_common_model_flags(validate)
    validate.set_defaults(device="cpu", handler=run_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = args.handler(args)
    except (
        ReleaseInferenceError,
        encoder_api.InferenceError,
        OSError,
        csv.Error,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        print(f"gMolAI inference failed: {error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

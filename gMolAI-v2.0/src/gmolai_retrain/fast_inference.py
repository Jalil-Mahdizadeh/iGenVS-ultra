from __future__ import annotations

"""Canonical speed-optimized gMolAI inference backends.

The optimized backend is inference-only.  ``MolecularRepresentationModel`` and
its ordinary ``encode`` method remain the training implementation and the
reference oracle used to qualify this equivalent execution path.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .chem import featurize_molecule
from .fast_graph import (
    FAST_GRAPH_VERSION,
    PackedBatch,
    initialize_worker,
    pack_smiles_task,
    resolve_worker_count,
    smiles_tasks,
)
from .model import MolecularRepresentationModel
from .schema import feature_schema, validate_feature_schema


FAST_INFERENCE_VERSION = "1"


def validate_optimized_model(model: nn.Module) -> MolecularRepresentationModel:
    if not isinstance(model, MolecularRepresentationModel):
        raise TypeError("Optimized inference requires MolecularRepresentationModel")
    expected = feature_schema(include_chirality=True, position_dim=0)
    validate_feature_schema(model.feature_schema)
    if model.feature_schema.get("hash") != expected["hash"]:
        raise ValueError("Model feature schema is not the promoted optimized contract")
    encoder = model.encoder
    required = (
        "input_projection",
        "convolutions",
        "normalizations",
        "output_projection",
        "output_normalization",
    )
    missing = [name for name in required if not hasattr(encoder, name)]
    if missing:
        raise TypeError(f"Unsupported representation encoder; missing {missing}")
    if len(encoder.convolutions) != len(encoder.normalizations):
        raise ValueError("Encoder convolution/normalization counts disagree")
    for convolution in encoder.convolutions:
        if not all(hasattr(convolution, name) for name in ("lin", "nn", "eps")):
            raise TypeError("Unsupported PyG GINEConv implementation")
    return model


class OptimizedRawCore(nn.Module):
    """Equivalent eval-only GINE/readout returning raw graph+mean-node blocks."""

    def __init__(self, model: MolecularRepresentationModel) -> None:
        super().__init__()
        model = validate_optimized_model(model)
        self.input_projection = model.encoder.input_projection
        self.convolutions = model.encoder.convolutions
        self.normalizations = model.encoder.normalizations
        self.output_projection = model.encoder.output_projection
        self.output_normalization = model.encoder.output_normalization
        self.graph_readout = model.graph_readout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(x)
        source = edge_index[0]
        destination = edge_index[1]
        for convolution, normalization in zip(
            self.convolutions, self.normalizations, strict=True
        ):
            edge_hidden = convolution.lin(edge_attr)
            messages = (hidden.index_select(0, source) + edge_hidden).relu()
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, destination, messages)
            update = convolution.nn(
                aggregate + (1.0 + convolution.eps) * hidden
            )
            hidden = F.silu(normalization(hidden + update))
        node_z = self.output_normalization(self.output_projection(hidden))

        total = global_add_pool(node_z, batch)
        maximum = global_max_pool(node_z, batch, size=total.shape[0])
        count = (
            torch.bincount(batch, minlength=total.shape[0])
            .to(node_z.dtype)
            .clamp_min(1)
        )
        mean = total / count.unsqueeze(-1)
        normalized_total = total / count.sqrt().unsqueeze(-1)
        size = count.log1p().unsqueeze(-1)
        graph_z = self.graph_readout(
            torch.cat((mean, normalized_total, maximum, size), dim=-1)
        )
        return torch.cat((graph_z, mean), dim=1)


def packed_to_device(
    packed: PackedBatch,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        value.to(device, non_blocking=non_blocking)
        for value in (
            torch.from_numpy(packed.x),
            torch.from_numpy(packed.edge_index),
            torch.from_numpy(packed.edge_attr),
            torch.from_numpy(packed.batch),
        )
    )


def calibrated_embedding_numpy(
    raw: np.ndarray,
    coordinate_mean: np.ndarray,
    coordinate_scale: np.ndarray,
    *,
    graph_dimensions: int,
    mean_node_weight: float,
) -> np.ndarray:
    if raw.ndim != 2 or raw.shape[1] != len(coordinate_mean):
        raise ValueError("Raw embedding and calibrator dimensions disagree")
    selected = (raw.astype(np.float32, copy=False) - coordinate_mean) / coordinate_scale
    selected[:, int(graph_dimensions) :] *= np.float32(mean_node_weight)
    return np.asarray(selected, dtype=np.float32)


def _calibrator_arrays(
    coordinate_mean: np.ndarray | torch.Tensor,
    coordinate_scale: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    def convert(value: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    mean = convert(coordinate_mean)
    scale = convert(coordinate_scale)
    if (
        mean.ndim != 1
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid coordinate calibrator")
    return mean, scale


class ReferenceSmilesEncoder:
    """Authoritative PyG implementation retained for audit and verification."""

    implementation = "reference_pyg"

    def __init__(
        self,
        model: MolecularRepresentationModel,
        coordinate_mean: np.ndarray | torch.Tensor,
        coordinate_scale: np.ndarray | torch.Tensor,
        *,
        device: torch.device,
        batch_size: int = 192,
        node_budget: int = 16384,
        mean_node_weight: float = 3.0,
    ) -> None:
        self.model = validate_optimized_model(model).eval()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.node_budget = int(node_budget)
        self.mean_node_weight = float(mean_node_weight)
        self.coordinate_mean, self.coordinate_scale = _calibrator_arrays(
            coordinate_mean, coordinate_scale
        )
        self.graph_dimensions = int(model.graph_latent_dim)
        self.embedding_dimensions = len(self.coordinate_mean)
        self.workers = 1

    @torch.inference_mode()
    def encode(
        self,
        values: Sequence[str],
        *,
        atom_counts: Sequence[int] | None = None,
    ) -> np.ndarray:
        matrix = np.empty(
            (len(values), self.embedding_dimensions), dtype=np.float32
        )
        for start, batch_values in smiles_tasks(
            values,
            batch_size=self.batch_size,
            node_budget=self.node_budget,
            atom_counts=atom_counts,
        ):
            graphs: list[Data] = []
            for value in batch_values:
                molecule = Chem.MolFromSmiles(value)
                if molecule is None:
                    raise ValueError(f"gMolAI could not parse canonical SMILES {value!r}")
                x, edge_index, edge_attr = featurize_molecule(
                    molecule, include_chirality=True, position_dim=0
                )
                graphs.append(
                    Data(
                        x=torch.from_numpy(x),
                        edge_index=torch.from_numpy(edge_index),
                        edge_attr=torch.from_numpy(edge_attr),
                    )
                )
            batch = Batch.from_data_list(graphs).to(self.device)
            node_z, graph_z = self.model.encode(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            raw = torch.cat(
                (graph_z, global_mean_pool(node_z, batch.batch)), dim=1
            )
            observed = calibrated_embedding_numpy(
                raw.detach().float().cpu().numpy(),
                self.coordinate_mean,
                self.coordinate_scale,
                graph_dimensions=self.graph_dimensions,
                mean_node_weight=self.mean_node_weight,
            )
            matrix[start : start + len(batch_values)] = observed
        return matrix

    def close(self) -> None:
        return None


class OptimizedSmilesEncoder:
    """Multiprocess RDKit + direct packed-array + equivalent GINE inference."""

    implementation = "optimized_gine_v1"

    def __init__(
        self,
        model: MolecularRepresentationModel,
        coordinate_mean: np.ndarray | torch.Tensor,
        coordinate_scale: np.ndarray | torch.Tensor,
        *,
        device: torch.device,
        batch_size: int = 192,
        node_budget: int = 16384,
        workers: int | str | None = "auto",
        mean_node_weight: float = 3.0,
    ) -> None:
        self.model = validate_optimized_model(model).eval()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.node_budget = int(node_budget)
        self.workers = resolve_worker_count(workers)
        self.mean_node_weight = float(mean_node_weight)
        self.coordinate_mean, self.coordinate_scale = _calibrator_arrays(
            coordinate_mean, coordinate_scale
        )
        self.graph_dimensions = int(model.graph_latent_dim)
        self.embedding_dimensions = len(self.coordinate_mean)
        if self.embedding_dimensions != int(
            model.graph_latent_dim + model.node_latent_dim
        ):
            raise ValueError("Model and calibrator embedding dimensions disagree")
        if self.batch_size <= 0 or self.node_budget <= 0:
            raise ValueError("batch_size and node_budget must be positive")
        if not np.isfinite(self.mean_node_weight) or self.mean_node_weight <= 0:
            raise ValueError("mean_node_weight must be finite and positive")
        self.core = OptimizedRawCore(model).to(self.device).eval()
        self._executor: ProcessPoolExecutor | None = None

    def _ensure_executor(self) -> ProcessPoolExecutor | None:
        if self.workers == 1:
            return None
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=initialize_worker,
            )
        return self._executor

    def warm_workers(self, values: Sequence[str]) -> None:
        if not values:
            return
        sample = tuple(values[: min(8, len(values))])
        executor = self._ensure_executor()
        if executor is not None:
            warm_tasks = [(0, sample) for _ in range(2 * self.workers)]
            list(executor.map(pack_smiles_task, warm_tasks, chunksize=1))
        packed = pack_smiles_task((0, sample))
        with torch.inference_mode():
            self.core(*packed_to_device(packed, self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def encode(
        self,
        values: Sequence[str],
        *,
        atom_counts: Sequence[int] | None = None,
    ) -> np.ndarray:
        matrix = np.empty(
            (len(values), self.embedding_dimensions), dtype=np.float32
        )
        if not values:
            return matrix
        tasks = smiles_tasks(
            values,
            batch_size=self.batch_size,
            node_budget=self.node_budget,
            atom_counts=atom_counts,
        )
        # One-batch inputs are faster inline and should not pay for dozens of
        # spawned workers. Sustained workloads retain the qualified pool.
        executor = (
            None if len(values) <= self.batch_size else self._ensure_executor()
        )
        packed_batches: Any
        if executor is None:
            packed_batches = map(pack_smiles_task, tasks)
        else:
            packed_batches = executor.map(pack_smiles_task, tasks, chunksize=1)
        for packed in packed_batches:
            raw = self.core(*packed_to_device(packed, self.device))
            observed = calibrated_embedding_numpy(
                raw.detach().float().cpu().numpy(),
                self.coordinate_mean,
                self.coordinate_scale,
                graph_dimensions=self.graph_dimensions,
                mean_node_weight=self.mean_node_weight,
            )
            matrix[
                packed.start : packed.start + packed.graph_count
            ] = observed
        if not np.isfinite(matrix).all():
            raise RuntimeError("Optimized encoder produced non-finite embeddings")
        return matrix

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> "OptimizedSmilesEncoder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def compare_embedding_matrices(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise ValueError("Embedding matrices have different shapes")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    relative = np.linalg.norm(delta, axis=1) / np.maximum(
        np.linalg.norm(reference64, axis=1), 1.0e-12
    )
    cosine = np.sum(reference64 * candidate64, axis=1) / np.maximum(
        np.linalg.norm(reference64, axis=1)
        * np.linalg.norm(candidate64, axis=1),
        1.0e-12,
    )
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "maximum_absolute_delta": float(np.max(np.abs(delta), initial=0.0)),
        "maximum_relative_l2": float(np.max(relative, initial=0.0)),
        "minimum_cosine": float(np.min(cosine, initial=1.0)),
    }


class VerifyingSmilesEncoder:
    """Return optimized vectors while checking a bounded reference sample."""

    implementation = "optimized_gine_v1_verified"

    def __init__(
        self,
        optimized: OptimizedSmilesEncoder,
        reference: ReferenceSmilesEncoder,
        *,
        verify_rows: int = 1024,
        minimum_cosine: float = 0.9999,
        maximum_relative_l2: float = 0.005,
    ) -> None:
        if verify_rows <= 0:
            raise ValueError("verify_rows must be positive")
        self.optimized = optimized
        self.reference = reference
        self.remaining = int(verify_rows)
        self.minimum_cosine = float(minimum_cosine)
        self.maximum_relative_l2 = float(maximum_relative_l2)
        self.workers = optimized.workers
        self.batch_size = optimized.batch_size
        self.node_budget = optimized.node_budget
        self.last_comparison: dict[str, float | bool] | None = None

    def encode(
        self,
        values: Sequence[str],
        *,
        atom_counts: Sequence[int] | None = None,
    ) -> np.ndarray:
        candidate = self.optimized.encode(values, atom_counts=atom_counts)
        count = min(self.remaining, len(values))
        if count:
            sample_counts = None if atom_counts is None else atom_counts[:count]
            reference = self.reference.encode(
                values[:count], atom_counts=sample_counts
            )
            comparison = compare_embedding_matrices(reference, candidate[:count])
            self.last_comparison = comparison
            if (
                float(comparison["minimum_cosine"]) < self.minimum_cosine
                or float(comparison["maximum_relative_l2"])
                > self.maximum_relative_l2
            ):
                raise RuntimeError(
                    "Optimized/reference embedding verification failed: "
                    f"{comparison}"
                )
            self.remaining -= count
        return candidate

    def close(self) -> None:
        self.optimized.close()
        self.reference.close()


def build_smiles_encoder(
    backend: str,
    model: MolecularRepresentationModel,
    coordinate_mean: np.ndarray | torch.Tensor,
    coordinate_scale: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    node_budget: int,
    workers: int | str | None,
    mean_node_weight: float,
    verify_rows: int = 1024,
) -> ReferenceSmilesEncoder | OptimizedSmilesEncoder | VerifyingSmilesEncoder:
    normalized = backend.strip().lower()
    common = {
        "device": device,
        "batch_size": batch_size,
        "node_budget": node_budget,
        "mean_node_weight": mean_node_weight,
    }
    if normalized == "reference":
        return ReferenceSmilesEncoder(
            model, coordinate_mean, coordinate_scale, **common
        )
    optimized = OptimizedSmilesEncoder(
        model,
        coordinate_mean,
        coordinate_scale,
        workers=workers,
        **common,
    )
    if normalized == "optimized":
        return optimized
    if normalized == "verify":
        reference = ReferenceSmilesEncoder(
            model, coordinate_mean, coordinate_scale, **common
        )
        return VerifyingSmilesEncoder(
            optimized, reference, verify_rows=verify_rows
        )
    optimized.close()
    raise ValueError(f"Unknown inference backend: {backend!r}")


def optimized_representation_blocks(
    model: MolecularRepresentationModel,
    packed: PackedBatch,
    device: torch.device,
    *,
    mean_node_weight: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unit hybrid, raw graph, and raw mean-node blocks for one batch."""

    raw = OptimizedRawCore(model).to(device).eval()(
        *packed_to_device(packed, device)
    )
    graph_dimensions = int(model.graph_latent_dim)
    graph_z = raw[:, :graph_dimensions]
    mean_node_z = raw[:, graph_dimensions:]
    unit_hybrid = torch.cat(
        (
            F.normalize(graph_z.float(), dim=-1),
            float(mean_node_weight) * F.normalize(mean_node_z.float(), dim=-1),
        ),
        dim=-1,
    )
    return unit_hybrid, graph_z, mean_node_z


def implementation_metadata(encoder: Any) -> dict[str, Any]:
    return {
        "backend": str(encoder.implementation),
        "fast_inference_version": FAST_INFERENCE_VERSION,
        "fast_graph_version": FAST_GRAPH_VERSION,
        "workers": int(getattr(encoder, "workers", 1)),
        "batch_size": int(getattr(encoder, "batch_size", 0)),
        "node_budget": int(getattr(encoder, "node_budget", 0)),
    }

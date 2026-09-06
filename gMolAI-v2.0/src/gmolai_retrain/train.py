from __future__ import annotations

import json
import math
import os
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import (
    atomic_copy,
    atomic_torch_save,
    build_checkpoint,
    restore_rng_state,
    validate_checkpoint,
)
from .config import object_hash, public_config
from .data import InfiniteGraphBatchIterator, Standardizer, finite_batches, load_graph_manifest
from .model import (
    MolecularRepresentationModel,
    MolecularVGAE,
    corrupt_graph_inputs,
    grouped_feature_loss,
    kl_divergence,
    nt_xent_loss,
    vicreg_terms,
)
from .negative_sampling import NegativeCandidates, sample_per_graph_negatives, select_hard_negative_logits
from .schema import validate_feature_schema
from .util import (
    atomic_write_json,
    ensure_directory,
    runtime_versions,
    seed_everything,
    sha256_file,
)


_SIGNAL_REQUESTED = False
LEGACY_TRAINING_IMPLEMENTATION_VERSION = "4"
REPRESENTATION_TRAINING_IMPLEMENTATION_VERSION = "5"


def _architecture(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("architecture", "vgae"))


def _implementation_version(cfg: dict[str, Any]) -> str:
    return (
        REPRESENTATION_TRAINING_IMPLEMENTATION_VERSION
        if _architecture(cfg) == "masked_graph_vicreg"
        else LEGACY_TRAINING_IMPLEMENTATION_VERSION
    )


def _training_seed(cfg: dict[str, Any]) -> int:
    """Return a plan-scoped seed without changing immutable graph identity."""
    return int(cfg.get("training", {}).get("seed", cfg["seed"]))


def _build_model(
    feature_schema: dict[str, Any], descriptor_count: int, cfg: dict[str, Any]
) -> MolecularVGAE | MolecularRepresentationModel:
    if _architecture(cfg) == "masked_graph_vicreg":
        return MolecularRepresentationModel(feature_schema, descriptor_count, cfg["model"])
    return MolecularVGAE(feature_schema, descriptor_count, cfg["model"])


def _initialize_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str,
    identity: dict[str, str],
) -> dict[str, Any]:
    """Load compatible encoder/decoder weights while resetting optimizer state."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    observed_sha256 = sha256_file(source)
    if observed_sha256 != str(expected_sha256).lower():
        raise RuntimeError(
            f"Initialization checkpoint hash mismatch: expected {expected_sha256}, "
            f"got {observed_sha256}"
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != 1:
        raise RuntimeError("Unsupported initialization checkpoint format")
    for key in (
        "config_hash",
        "graph_manifest_hash",
        "descriptor_schema_hash",
        "feature_schema_hash",
        "scaler_hash",
        "training_implementation_version",
    ):
        if checkpoint.get(key) != identity[key]:
            raise RuntimeError(f"Initialization checkpoint {key} mismatch")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    if any(not name.startswith("vicreg_projector.") for name in missing):
        raise RuntimeError(
            "Initialization checkpoint has unexpected missing model keys: "
            + ", ".join(incompatible.missing_keys)
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Initialization checkpoint has unexpected model keys: "
            + ", ".join(incompatible.unexpected_keys)
        )
    return {
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": observed_sha256,
        "source_global_step": int(checkpoint["global_step"]),
        "source_training_plan_hash": checkpoint["training_plan_hash"],
        "new_parameters": sorted(missing),
        "optimizer_state": "reset",
        "scheduler_state": "reset",
    }


def _request_stop(signum, frame) -> None:  # noqa: ARG001
    global _SIGNAL_REQUESTED
    _SIGNAL_REQUESTED = True


def _training_plan_hash(cfg: dict[str, Any]) -> str:
    return object_hash(
        {
            "model": cfg["model"],
            "objective": cfg["objective"],
            "training": cfg["training"],
        }
    )


def _distributed_context(allow_cpu: bool = False) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl", device_id=device)
    elif allow_cpu:
        device = torch.device("cpu")
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="gloo")
    else:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only for a smoke test")
    return rank, world_size, local_rank, device


def _all_reduce_mean(value: float, count: int, device: torch.device, world_size: int) -> float:
    pair = torch.tensor([value, float(count)], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float(pair[0] / pair[1].clamp_min(1.0))


def _sample_negative_candidates(
    batch_cpu,
    cfg: dict[str, Any],
    *,
    step: int,
    rank: int,
    training: bool,
) -> NegativeCandidates:
    objective = cfg["objective"]
    return sample_per_graph_negatives(
        batch_cpu.edge_index,
        batch_cpu.ptr,
        batch_cpu.graph_id,
        easy_ratio=float(objective["easy_negative_ratio"]),
        # Validation must be at least as adversarial as training.  Previously
        # both values were forced to zero during evaluation, so the reported
        # existence loss covered only easy random non-edges.
        hard_ratio=float(objective["hard_negative_ratio"]),
        hard_pool_ratio=float(objective["hard_pool_ratio"]),
        seed=_training_seed(cfg) + 104729 * step + rank,
    )


@dataclass
class _PreparedTrainingBatch:
    batch: Any
    candidates: NegativeCandidates
    start_state: dict[str, int]
    batch_build_seconds: float
    negative_sampling_seconds: float
    pin_memory_seconds: float


def _prepare_training_batch(
    iterator: InfiniteGraphBatchIterator,
    cfg: dict[str, Any],
    *,
    step: int,
    rank: int,
    pin_memory: bool,
    start_state: dict[str, int],
) -> _PreparedTrainingBatch:
    started = time.perf_counter()
    batch = iterator.next_batch()
    after_batch = time.perf_counter()
    candidates = _sample_negative_candidates(batch, cfg, step=step, rank=rank, training=True)
    after_candidates = time.perf_counter()
    if pin_memory:
        batch = batch.pin_memory()
        candidates = candidates.pin_memory()
    finished = time.perf_counter()
    return _PreparedTrainingBatch(
        batch=batch,
        candidates=candidates,
        start_state=start_state,
        batch_build_seconds=after_batch - started,
        negative_sampling_seconds=after_candidates - after_batch,
        pin_memory_seconds=finished - after_candidates,
    )


class _TrainingBatchPrefetcher:
    """Prepare one deterministic batch ahead without advancing checkpoint cursors."""

    def __init__(
        self,
        iterator: InfiniteGraphBatchIterator,
        cfg: dict[str, Any],
        *,
        start_step: int,
        stop_step: int,
        rank: int,
        pin_memory: bool,
    ) -> None:
        if start_step >= stop_step:
            raise ValueError("The prefetcher requires at least one remaining training step")
        self.iterator = iterator
        self.cfg = cfg
        self.rank = int(rank)
        self.pin_memory = bool(pin_memory)
        self.stop_step = int(stop_step)
        self.next_step = int(start_step)
        self._checkpoint_state = dict(iterator.state_dict())
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gmolai-input-r{rank}")
        self._future: Future[_PreparedTrainingBatch] | None = self._submit()

    def _submit(self) -> Future[_PreparedTrainingBatch]:
        start_state = dict(self.iterator.state_dict())
        return self._executor.submit(
            _prepare_training_batch,
            self.iterator,
            self.cfg,
            step=self.next_step,
            rank=self.rank,
            pin_memory=self.pin_memory,
            start_state=start_state,
        )

    def next(self, expected_step: int) -> _PreparedTrainingBatch:
        if expected_step != self.next_step or self._future is None:
            raise RuntimeError(
                f"Input prefetch cursor mismatch: expected step {expected_step}, prepared {self.next_step}"
            )
        prepared = self._future.result()
        self._checkpoint_state = dict(self.iterator.state_dict())
        self.next_step += 1
        self._future = self._submit() if self.next_step < self.stop_step else None
        return prepared

    def checkpoint_state(self) -> dict[str, int]:
        return dict(self._checkpoint_state)

    def close(self) -> None:
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._future = None


def _balanced_existence_loss(
    positive_logits: torch.Tensor, negative_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = torch.cat((positive_logits, negative_logits)).sum() * 0.0
    positive_loss = (
        F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
        if positive_logits.numel()
        else zero
    )
    negative_loss = (
        F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
        if negative_logits.numel()
        else zero
    )
    return positive_loss + negative_loss, positive_loss, negative_loss


def _legacy_losses_for_batch(
    model: MolecularVGAE | DistributedDataParallel,
    batch_cpu,
    device: torch.device,
    cfg: dict[str, Any],
    *,
    step: int,
    rank: int,
    training: bool,
    candidates: NegativeCandidates | None = None,
) -> dict[str, Any]:
    objective = cfg["objective"]
    if candidates is None:
        candidates = _sample_negative_candidates(
            batch_cpu,
            cfg,
            step=step,
            rank=rank,
            training=training,
        )
    batch = batch_cpu.to(device, non_blocking=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(_training_seed(cfg) + 1_000_003 * step + rank)
    corrupted = corrupt_graph_inputs(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        node_probability=float(objective["node_mask_probability"]),
        edge_feature_probability=float(objective["bond_feature_mask_probability"]),
        edge_dropout_probability=float(objective["bond_dropout_probability"]),
        generator=generator,
    )
    positive = corrupted.unique_positive_edge_index[:, corrupted.edge_drop_mask]
    if positive.shape[1] == 0:
        positive = corrupted.unique_positive_edge_index
    easy = candidates.easy.to(device, non_blocking=True)
    hard_pool = candidates.hard_pool.to(device, non_blocking=True)
    existence_edge_index = torch.cat((positive, easy, hard_pool), dim=1)
    (
        z,
        mu,
        logvar,
        node_logits,
        descriptor_prediction,
        existence_logits,
        edge_logits,
    ) = model(
        corrupted.x,
        corrupted.edge_index,
        corrupted.edge_attr,
        batch.batch,
        existence_edge_index,
        corrupted.unique_positive_edge_index,
        sample=training,
    )
    if existence_logits is None or edge_logits is None:
        raise RuntimeError("Training forward pass omitted edge-decoder outputs")
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    node_target = batch.x[:, : base_model.node_target_dim]
    if float(objective["node_mask_probability"]) > 0 and batch.num_nodes:
        node_loss = grouped_feature_loss(
            node_logits[corrupted.node_mask],
            node_target[corrupted.node_mask],
            base_model.feature_schema["node_groups"],
        )
    else:
        node_loss = node_logits.sum() * 0.0

    positive_count = positive.shape[1]
    easy_count = easy.shape[1]
    positive_logits = existence_logits[:positive_count]
    easy_logits = existence_logits[positive_count : positive_count + easy_count]
    pool_logits = existence_logits[positive_count + easy_count :]
    hard_logits = select_hard_negative_logits(pool_logits, candidates)
    negative_logits = torch.cat((easy_logits, hard_logits))
    existence_loss, positive_existence_loss, negative_existence_loss = _balanced_existence_loss(
        positive_logits, negative_logits
    )

    edge_corruption_enabled = (
        float(objective["bond_feature_mask_probability"]) > 0
        or float(objective["bond_dropout_probability"]) > 0
    )
    if edge_corruption_enabled and corrupted.unique_positive_edge_index.shape[1]:
        edge_loss = grouped_feature_loss(
            edge_logits[corrupted.edge_target_mask],
            corrupted.unique_positive_edge_attr[corrupted.edge_target_mask],
            base_model.feature_schema["edge_groups"],
        )
    else:
        edge_loss = edge_logits.sum() * 0.0
    descriptor_loss = F.mse_loss(descriptor_prediction, batch.y)
    descriptor_mae = F.l1_loss(descriptor_prediction, batch.y)
    kl = kl_divergence(mu, logvar)
    beta_progress = min(1.0, step / max(1, int(objective["kl_warmup_steps"]))) if training else 1.0
    beta = float(objective["kl_beta_max"]) * beta_progress
    total = (
        float(objective["node_weight"]) * node_loss
        + float(objective["edge_existence_weight"]) * existence_loss
        + float(objective["edge_feature_weight"]) * edge_loss
        + float(objective["descriptor_weight"]) * descriptor_loss
        + beta * kl
    )
    node_count = corrupted.node_mask.sum()
    edge_count = corrupted.edge_target_mask.sum()
    descriptor_count = batch.y.numel()
    return {
        "total": total,
        "node": node_loss,
        "existence": existence_loss,
        "edge_feature": edge_loss,
        "descriptor": descriptor_loss,
        "descriptor_mae": descriptor_mae,
        "kl": kl,
        "kl_beta": torch.tensor(beta, device=device),
        # These are batch metadata, not model outputs. Keeping them as Python
        # integers avoids two needless device synchronizations every step.
        "graphs": batch.num_graphs,
        "nodes": batch.num_nodes,
        "_descriptor_prediction": descriptor_prediction.detach(),
        "_descriptor_target": batch.y.detach(),
        "_mu": mu.detach(),
        "_logvar": logvar.detach(),
        "_metric_sums": {
            "node": (node_loss.detach() * node_count),
            "existence_positive": (positive_existence_loss.detach() * positive_logits.numel()),
            "existence_negative": (negative_existence_loss.detach() * negative_logits.numel()),
            "edge_feature": (edge_loss.detach() * edge_count),
            "descriptor": F.mse_loss(
                descriptor_prediction.detach(), batch.y.detach(), reduction="sum"
            ),
            "descriptor_mae": F.l1_loss(
                descriptor_prediction.detach(), batch.y.detach(), reduction="sum"
            ),
            "kl": (kl.detach() * mu.shape[0]),
        },
        "_metric_counts": {
            "node": node_count,
            "existence_positive": positive_logits.numel(),
            "existence_negative": negative_logits.numel(),
            "edge_feature": edge_count,
            "descriptor": descriptor_count,
            "descriptor_mae": descriptor_count,
            "kl": mu.shape[0],
        },
        "_node_logits": node_logits.detach(),
        "_node_target": node_target.detach(),
        "_node_mask": corrupted.node_mask,
        "_edge_logits": edge_logits.detach(),
        "_edge_target": corrupted.unique_positive_edge_attr.detach(),
        "_edge_mask": corrupted.edge_target_mask,
        "_existence_positive_logits": positive_logits.detach(),
        "_existence_easy_logits": easy_logits.detach(),
        "_existence_hard_logits": hard_logits.detach(),
    }


def _representation_losses_for_batch(
    model: MolecularRepresentationModel | DistributedDataParallel,
    batch_cpu,
    device: torch.device,
    cfg: dict[str, Any],
    *,
    step: int,
    rank: int,
    training: bool,
    candidates: NegativeCandidates | None = None,
) -> dict[str, Any]:
    objective = cfg["objective"]
    if candidates is None:
        candidates = _sample_negative_candidates(
            batch_cpu, cfg, step=step, rank=rank, training=training
        )
    batch = batch_cpu.to(device, non_blocking=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(_training_seed(cfg) + 1_000_003 * step + rank)
    corrupted = corrupt_graph_inputs(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        node_probability=float(objective["node_mask_probability"]),
        edge_feature_probability=float(objective["bond_feature_mask_probability"]),
        edge_dropout_probability=float(objective["bond_dropout_probability"]),
        generator=generator,
    )
    positive = corrupted.unique_positive_edge_index[:, corrupted.edge_drop_mask]
    if positive.shape[1] == 0:
        positive = corrupted.unique_positive_edge_index
    easy = candidates.easy.to(device, non_blocking=True)
    hard_pool = candidates.hard_pool.to(device, non_blocking=True)
    existence_edge_index = torch.cat((positive, easy, hard_pool), dim=1)
    outputs = model(
        corrupted.x,
        corrupted.edge_index,
        corrupted.edge_attr,
        batch.batch,
        existence_edge_index,
        corrupted.unique_positive_edge_index,
        # The clean graph is the stable downstream-inference anchor.  The
        # independently re-sampled masked view changes every optimizer step.
        view2_x=batch.x,
        view2_edge_index=batch.edge_index,
        view2_edge_attr=batch.edge_attr,
        contrastive_detach_node=bool(
            objective.get("contrastive_detach_node_gradient", False)
        ),
    )
    required = (
        "node_z",
        "mean_node_z",
        "mean_node_z2",
        "graph_z",
        "graph_z2",
        "regularization_z",
        "regularization_z2",
        "contrastive_z",
        "contrastive_z2",
        "node_logits",
        "descriptor_prediction",
        "descriptor_prediction2",
        "existence_logits",
        "edge_logits",
    )
    if any(outputs[name] is None for name in required):
        raise RuntimeError("Representation forward pass omitted a required output")
    node_logits = outputs["node_logits"]
    mean_node_z = outputs["mean_node_z"]
    clean_mean_node_z = outputs["mean_node_z2"]
    graph_z = outputs["graph_z"]
    clean_graph_z = outputs["graph_z2"]
    regularization_z = outputs["regularization_z"]
    clean_regularization_z = outputs["regularization_z2"]
    model_contrastive_z = outputs["contrastive_z"]
    model_clean_contrastive_z = outputs["contrastive_z2"]
    descriptor_prediction = outputs["descriptor_prediction"]
    clean_descriptor_prediction = outputs["descriptor_prediction2"]
    existence_logits = outputs["existence_logits"]
    edge_logits = outputs["edge_logits"]
    assert isinstance(node_logits, torch.Tensor)
    assert isinstance(mean_node_z, torch.Tensor)
    assert isinstance(clean_mean_node_z, torch.Tensor)
    assert isinstance(graph_z, torch.Tensor)
    assert isinstance(clean_graph_z, torch.Tensor)
    assert isinstance(regularization_z, torch.Tensor)
    assert isinstance(clean_regularization_z, torch.Tensor)
    assert isinstance(model_contrastive_z, torch.Tensor)
    assert isinstance(model_clean_contrastive_z, torch.Tensor)
    assert isinstance(descriptor_prediction, torch.Tensor)
    assert isinstance(clean_descriptor_prediction, torch.Tensor)
    assert isinstance(existence_logits, torch.Tensor)
    assert isinstance(edge_logits, torch.Tensor)

    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    node_target = batch.x[:, : base_model.node_target_dim]
    if float(objective["node_mask_probability"]) > 0 and batch.num_nodes:
        node_loss = grouped_feature_loss(
            node_logits[corrupted.node_mask],
            node_target[corrupted.node_mask],
            base_model.feature_schema["node_groups"],
        )
    else:
        node_loss = node_logits.sum() * 0.0

    positive_count = positive.shape[1]
    easy_count = easy.shape[1]
    positive_logits = existence_logits[:positive_count]
    easy_logits = existence_logits[positive_count : positive_count + easy_count]
    pool_logits = existence_logits[positive_count + easy_count :]
    hard_logits = select_hard_negative_logits(pool_logits, candidates)
    negative_logits = torch.cat((easy_logits, hard_logits))
    existence_loss, positive_existence_loss, negative_existence_loss = _balanced_existence_loss(
        positive_logits, negative_logits
    )

    edge_corruption_enabled = (
        float(objective["bond_feature_mask_probability"]) > 0
        or float(objective["bond_dropout_probability"]) > 0
    )
    if edge_corruption_enabled and corrupted.edge_target_mask.any():
        edge_loss = grouped_feature_loss(
            edge_logits[corrupted.edge_target_mask],
            corrupted.unique_positive_edge_attr[corrupted.edge_target_mask],
            base_model.feature_schema["edge_groups"],
        )
    else:
        edge_loss = edge_logits.sum() * 0.0
    descriptor_loss = 0.5 * (
        F.mse_loss(descriptor_prediction, batch.y)
        + F.mse_loss(clean_descriptor_prediction, batch.y)
    )
    descriptor_mae = 0.5 * (
        F.l1_loss(descriptor_prediction, batch.y)
        + F.l1_loss(clean_descriptor_prediction, batch.y)
    )
    invariance_space = str(objective.get("invariance_space", "graph_z"))
    if invariance_space == "projector":
        invariance_z = regularization_z
        clean_invariance_z = clean_regularization_z
    else:
        invariance_z = graph_z
        clean_invariance_z = clean_graph_z
    invariance = F.mse_loss(invariance_z, clean_invariance_z)
    _, variance, covariance = vicreg_terms(
        regularization_z,
        clean_regularization_z,
        variance_target=float(objective["variance_target"]),
    )
    contrastive_weight = float(objective.get("contrastive_weight", 0.0))
    if contrastive_weight > 0:
        contrastive_space = str(objective.get("contrastive_space", "graph_z"))
        if contrastive_space == "mean_node_z":
            contrastive_z = mean_node_z
            clean_contrastive_z = clean_mean_node_z
        elif bool(objective.get("contrastive_detach_node_gradient", False)):
            contrastive_z = model_contrastive_z
            clean_contrastive_z = model_clean_contrastive_z
        elif contrastive_space == "projector":
            contrastive_z = regularization_z
            clean_contrastive_z = clean_regularization_z
        else:
            contrastive_z = graph_z
            clean_contrastive_z = clean_graph_z
        contrastive = nt_xent_loss(
            contrastive_z,
            clean_contrastive_z,
            temperature=float(objective.get("contrastive_temperature", 0.1)),
        )
    else:
        contrastive = graph_z.sum() * 0.0
    total = (
        float(objective["node_weight"]) * node_loss
        + float(objective["edge_existence_weight"]) * existence_loss
        + float(objective["edge_feature_weight"]) * edge_loss
        + float(objective["descriptor_weight"]) * descriptor_loss
        + float(objective["invariance_weight"]) * invariance
        + float(objective["variance_weight"]) * variance
        + float(objective["covariance_weight"]) * covariance
        + contrastive_weight * contrastive
    )
    zero = total.detach() * 0.0
    node_count = corrupted.node_mask.sum()
    edge_count = corrupted.edge_target_mask.sum()
    descriptor_count = 2 * batch.y.numel()
    return {
        "total": total,
        "node": node_loss,
        "existence": existence_loss,
        "edge_feature": edge_loss,
        "descriptor": descriptor_loss,
        "descriptor_mae": descriptor_mae,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
        "contrastive": contrastive,
        "kl": zero,
        "kl_beta": zero,
        "graphs": batch.num_graphs,
        "nodes": batch.num_nodes,
        "_descriptor_prediction": clean_descriptor_prediction.detach(),
        "_descriptor_target": batch.y.detach(),
        "_graph_z": graph_z.detach(),
        "_graph_z_clean": clean_graph_z.detach(),
        "_regularization_z": regularization_z.detach(),
        "_regularization_z_clean": clean_regularization_z.detach(),
        "_contrastive_z": contrastive_z.detach() if contrastive_weight > 0 else None,
        "_contrastive_z_clean": (
            clean_contrastive_z.detach() if contrastive_weight > 0 else None
        ),
        "_metric_sums": {
            "node": node_loss.detach() * node_count,
            "existence_positive": positive_existence_loss.detach() * positive_logits.numel(),
            "existence_negative": negative_existence_loss.detach() * negative_logits.numel(),
            "edge_feature": edge_loss.detach() * edge_count,
            "descriptor": (
                F.mse_loss(descriptor_prediction.detach(), batch.y.detach(), reduction="sum")
                + F.mse_loss(clean_descriptor_prediction.detach(), batch.y.detach(), reduction="sum")
            ),
            "descriptor_mae": (
                F.l1_loss(descriptor_prediction.detach(), batch.y.detach(), reduction="sum")
                + F.l1_loss(clean_descriptor_prediction.detach(), batch.y.detach(), reduction="sum")
            ),
            "invariance": (
                invariance_z.detach() - clean_invariance_z.detach()
            ).square().sum(),
            "contrastive": contrastive.detach() * graph_z.shape[0],
        },
        "_metric_counts": {
            "node": node_count,
            "existence_positive": positive_logits.numel(),
            "existence_negative": negative_logits.numel(),
            "edge_feature": edge_count,
            "descriptor": descriptor_count,
            "descriptor_mae": descriptor_count,
            "invariance": invariance_z.numel(),
            "contrastive": graph_z.shape[0],
        },
        "_node_logits": node_logits.detach(),
        "_node_target": node_target.detach(),
        "_node_mask": corrupted.node_mask,
        "_edge_logits": edge_logits.detach(),
        "_edge_target": corrupted.unique_positive_edge_attr.detach(),
        "_edge_mask": corrupted.edge_target_mask,
        "_existence_positive_logits": positive_logits.detach(),
        "_existence_easy_logits": easy_logits.detach(),
        "_existence_hard_logits": hard_logits.detach(),
    }


def _losses_for_batch(
    model: MolecularVGAE | MolecularRepresentationModel | DistributedDataParallel,
    batch_cpu,
    device: torch.device,
    cfg: dict[str, Any],
    *,
    step: int,
    rank: int,
    training: bool,
    candidates: NegativeCandidates | None = None,
) -> dict[str, Any]:
    if _architecture(cfg) == "masked_graph_vicreg":
        return _representation_losses_for_batch(
            model,
            batch_cpu,
            device,
            cfg,
            step=step,
            rank=rank,
            training=training,
            candidates=candidates,
        )
    return _legacy_losses_for_batch(
        model,
        batch_cpu,
        device,
        cfg,
        step=step,
        rank=rank,
        training=training,
        candidates=candidates,
    )


def _group_confusions(groups: list[dict[str, Any]], device: torch.device) -> list[torch.Tensor]:
    result = []
    for group in groups:
        classes = 2 if group["kind"] == "binary" else len(group["values"]) + int(group.get("other", False))
        result.append(torch.zeros((classes, classes), dtype=torch.float64, device=device))
    return result


def _update_group_confusions(
    accumulators: list[torch.Tensor],
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    groups: list[dict[str, Any]],
    exact: torch.Tensor,
) -> None:
    logits, target = logits[mask], target[mask]
    if logits.shape[0] == 0:
        return
    row_correct = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    offset = 0
    for accumulator, group in zip(accumulators, groups):
        if group["kind"] == "binary":
            truth = (target[:, offset] >= 0.5).to(torch.long)
            predicted = (logits[:, offset] >= 0).to(torch.long)
            offset += 1
        else:
            width = len(group["values"]) + int(group.get("other", False))
            truth = target[:, offset : offset + width].argmax(dim=-1)
            predicted = logits[:, offset : offset + width].argmax(dim=-1)
            offset += width
        row_correct &= truth == predicted
        classes = accumulator.shape[0]
        accumulator += torch.bincount(
            truth * classes + predicted, minlength=classes * classes
        ).reshape(classes, classes)
    exact[0] += row_correct.sum()
    exact[1] += row_correct.numel()


def _finalize_group_confusions(
    accumulators: list[torch.Tensor], groups: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for confusion, group in zip(accumulators, groups):
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        true_positive = confusion.diag()
        precision = true_positive / predicted.clamp_min(1)
        recall = true_positive / support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1.0e-12)
        present = support > 0
        result[str(group["name"])] = {
            "accuracy": float(true_positive.sum() / confusion.sum().clamp_min(1)),
            "balanced_accuracy": float(recall[present].mean()) if bool(present.any()) else None,
            "macro_f1": float(f1[present].mean()) if bool(present.any()) else None,
            "examples": int(confusion.sum()),
            "classes_present": int(present.sum()),
            "classes_total": int(confusion.shape[0]),
        }
    return result


def _update_score_histogram(histogram: torch.Tensor, logits: torch.Tensor) -> None:
    if logits.numel() == 0:
        return
    bins = histogram.shape[0]
    indices = (logits.float().sigmoid() * (bins - 1)).to(torch.long).clamp(0, bins - 1)
    histogram += torch.bincount(indices, minlength=bins).to(histogram.dtype)


def _binary_histogram_metrics(positive: torch.Tensor, negative: torch.Tensor) -> dict[str, Any]:
    positives, negatives = positive.sum(), negative.sum()
    if positives <= 0 or negatives <= 0:
        return {"auroc": None, "average_precision": None, "positives": int(positives), "negatives": int(negatives)}
    true_positive = torch.cumsum(positive.flip(0), dim=0)
    false_positive = torch.cumsum(negative.flip(0), dim=0)
    recall = true_positive / positives
    false_positive_rate = false_positive / negatives
    recall_with_origin = torch.cat((recall.new_zeros(1), recall))
    fpr_with_origin = torch.cat((false_positive_rate.new_zeros(1), false_positive_rate))
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall_increment = recall_with_origin[1:] - recall_with_origin[:-1]
    return {
        "auroc": float(torch.trapz(recall_with_origin, fpr_with_origin)),
        "average_precision": float((precision * recall_increment).sum()),
        "positives": int(positives),
        "negatives": int(negatives),
        "histogram_bins": int(positive.shape[0]),
    }


def _covariance_diagnostics(sum_value: torch.Tensor, cross: torch.Tensor, count: int) -> dict[str, Any]:
    dimension = sum_value.shape[0]
    mean = sum_value / max(1, count)
    covariance = (cross - count * torch.outer(mean, mean)) / max(1, count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    eigen_sum = eigenvalues.sum().clamp_min(1.0e-30)
    probability = eigenvalues / eigen_sum
    positive_probability = probability[probability > 0]
    effective_rank = torch.exp(-(positive_probability * positive_probability.log()).sum())
    participation = eigen_sum.square() / eigenvalues.square().sum().clamp_min(1.0e-30)
    variance = covariance.diag().clamp_min(0)
    standard_deviation = variance.sqrt()
    return {
        "dimensions": int(dimension),
        "effective_rank": float(effective_rank),
        "effective_rank_ratio": float(effective_rank / dimension),
        "participation_ratio": float(participation),
        "participation_ratio_fraction": float(participation / dimension),
        "top_eigenvalue_fraction": float(eigenvalues[-1] / eigen_sum),
        "mean_coordinate_std": float(standard_deviation.mean()),
        "median_coordinate_std": float(standard_deviation.median()),
        "minimum_coordinate_std": float(standard_deviation.min()),
        "active_units": int((variance > 0.01).sum()),
        "active_variance_threshold": 0.01,
        "covariance": covariance,
    }


@torch.no_grad()
def evaluate(
    model: MolecularVGAE | MolecularRepresentationModel | DistributedDataParallel,
    manifest: dict[str, Any],
    standardizer: Standardizer,
    cfg: dict[str, Any],
    *,
    split: str,
    max_graphs: int,
    rank: int,
    world_size: int,
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    model.eval()
    representation = _architecture(cfg) == "masked_graph_vicreg"
    metric_names = [
        "node",
        "existence_positive",
        "existence_negative",
        "edge_feature",
        "descriptor",
        "descriptor_mae",
    ]
    metric_names.extend(("invariance", "contrastive") if representation else ("kl",))
    metric_sums = {name: torch.zeros((), dtype=torch.float64, device=device) for name in metric_names}
    metric_counts = {name: torch.zeros((), dtype=torch.float64, device=device) for name in metric_names}
    descriptor_count = len(cfg["data"]["descriptor_columns"])
    descriptor_accumulator = torch.zeros((6, descriptor_count), dtype=torch.float64, device=device)
    schema = manifest["feature_schema"]
    node_confusions = _group_confusions(schema["node_groups"], device)
    edge_confusions = _group_confusions(schema["edge_groups"], device)
    node_exact = torch.zeros(2, dtype=torch.float64, device=device)
    edge_exact = torch.zeros(2, dtype=torch.float64, device=device)
    score_histograms = {
        name: torch.zeros(4096, dtype=torch.float64, device=device)
        for name in ("positive", "easy_negative", "hard_negative")
    }
    graph_count = 0
    if representation:
        latent_dim = int(cfg["model"]["graph_latent_dim"])
        regularization_dim = (
            int(cfg["model"].get("vicreg_projector_dim", latent_dim))
            if bool(cfg["model"].get("vicreg_projector", False))
            else latent_dim
        )
        graph_sums = torch.zeros((2, latent_dim), dtype=torch.float64, device=device)
        graph_cross = torch.zeros((2, latent_dim, latent_dim), dtype=torch.float64, device=device)
        regularization_sums = torch.zeros((2, regularization_dim), dtype=torch.float64, device=device)
        regularization_cross = torch.zeros(
            (2, regularization_dim, regularization_dim), dtype=torch.float64, device=device
        )
        graph_cosine = torch.zeros(2, dtype=torch.float64, device=device)
    else:
        latent_dim = int(cfg["model"]["latent_dim"])
        latent_accumulator = torch.zeros((3, latent_dim), dtype=torch.float64, device=device)
        latent_nodes = 0

    for batch_index, batch in enumerate(
        finite_batches(
            manifest,
            standardizer,
            split=split,
            rank=rank,
            world_size=world_size,
            max_graphs=max_graphs,
            node_budget=int(cfg["training"]["node_budget_per_gpu"]),
            graph_budget=int(cfg["training"]["max_graphs_per_gpu"]),
            seed=int(cfg["seed"]),
        )
    ):
        evaluation_model = model.module if isinstance(model, DistributedDataParallel) else model
        values = _losses_for_batch(
            evaluation_model,
            batch,
            device,
            cfg,
            step=10_000_000 + batch_index,
            rank=rank,
            training=False,
        )
        for name in metric_names:
            metric_sums[name] += torch.as_tensor(values["_metric_sums"][name], device=device, dtype=torch.float64)
            metric_counts[name] += torch.as_tensor(values["_metric_counts"][name], device=device, dtype=torch.float64)
        prediction = values["_descriptor_prediction"].to(torch.float64)
        target = values["_descriptor_target"].to(torch.float64)
        descriptor_accumulator[0] += torch.abs(prediction - target).sum(dim=0)
        descriptor_accumulator[1] += prediction.sum(dim=0)
        descriptor_accumulator[2] += target.sum(dim=0)
        descriptor_accumulator[3] += prediction.square().sum(dim=0)
        descriptor_accumulator[4] += target.square().sum(dim=0)
        descriptor_accumulator[5] += (prediction * target).sum(dim=0)
        _update_group_confusions(
            node_confusions,
            values["_node_logits"],
            values["_node_target"],
            values["_node_mask"],
            schema["node_groups"],
            node_exact,
        )
        _update_group_confusions(
            edge_confusions,
            values["_edge_logits"],
            values["_edge_target"],
            values["_edge_mask"],
            schema["edge_groups"],
            edge_exact,
        )
        _update_score_histogram(score_histograms["positive"], values["_existence_positive_logits"])
        _update_score_histogram(score_histograms["easy_negative"], values["_existence_easy_logits"])
        _update_score_histogram(score_histograms["hard_negative"], values["_existence_hard_logits"])
        if representation:
            masked = values["_graph_z"].to(torch.float64)
            clean = values["_graph_z_clean"].to(torch.float64)
            for index, value in enumerate((masked, clean)):
                graph_sums[index] += value.sum(dim=0)
                graph_cross[index] += value.T @ value
            masked_regularization = values["_regularization_z"].to(torch.float64)
            clean_regularization = values["_regularization_z_clean"].to(torch.float64)
            for index, value in enumerate((masked_regularization, clean_regularization)):
                regularization_sums[index] += value.sum(dim=0)
                regularization_cross[index] += value.T @ value
            graph_cosine[0] += F.cosine_similarity(masked, clean, dim=-1).sum()
            graph_cosine[1] += masked.shape[0]
        else:
            mu = values["_mu"].to(torch.float64)
            logvar = values["_logvar"].to(torch.float64)
            latent_accumulator[0] += mu.sum(dim=0)
            latent_accumulator[1] += mu.square().sum(dim=0)
            latent_accumulator[2] += (-0.5 * (1.0 + logvar - mu.square() - logvar.exp())).sum(dim=0)
            latent_nodes += mu.shape[0]
        graph_count += int(values["graphs"])

    reducible = [
        *metric_sums.values(),
        *metric_counts.values(),
        descriptor_accumulator,
        *node_confusions,
        *edge_confusions,
        node_exact,
        edge_exact,
        *score_histograms.values(),
    ]
    if representation:
        reducible.extend(
            (graph_sums, graph_cross, regularization_sums, regularization_cross, graph_cosine)
        )
    else:
        reducible.append(latent_accumulator)
    total_graphs = torch.tensor(graph_count, dtype=torch.int64, device=device)
    total_latent_nodes = torch.tensor(0 if representation else latent_nodes, dtype=torch.int64, device=device)
    if world_size > 1:
        for value in reducible:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_graphs, op=dist.ReduceOp.SUM)
        if not representation:
            dist.all_reduce(total_latent_nodes, op=dist.ReduceOp.SUM)

    def mean_metric(name: str) -> float:
        return float(metric_sums[name] / metric_counts[name].clamp_min(1))

    result: dict[str, Any] = {
        "node": mean_metric("node"),
        "existence_positive": mean_metric("existence_positive"),
        "existence_negative": mean_metric("existence_negative"),
        "edge_feature": mean_metric("edge_feature"),
        "descriptor": mean_metric("descriptor"),
        "descriptor_mae": mean_metric("descriptor_mae"),
        "graphs": int(total_graphs),
        "step": int(step),
        "metric_denominators": {name: int(metric_counts[name]) for name in metric_names},
    }
    result["existence"] = result["existence_positive"] + result["existence_negative"]
    objective = cfg["objective"]
    if representation:
        result["invariance"] = mean_metric("invariance")
        result["contrastive"] = mean_metric("contrastive")
        masked_diagnostics = _covariance_diagnostics(graph_sums[0], graph_cross[0], int(total_graphs))
        clean_diagnostics = _covariance_diagnostics(graph_sums[1], graph_cross[1], int(total_graphs))
        masked_diagnostics.pop("covariance")
        clean_diagnostics.pop("covariance")
        masked_regularization_diagnostics = _covariance_diagnostics(
            regularization_sums[0], regularization_cross[0], int(total_graphs)
        )
        clean_regularization_diagnostics = _covariance_diagnostics(
            regularization_sums[1], regularization_cross[1], int(total_graphs)
        )
        variance_target = float(objective["variance_target"])
        variance_losses = []
        covariance_losses = []
        for diagnostics in (
            masked_regularization_diagnostics,
            clean_regularization_diagnostics,
        ):
            covariance = diagnostics.pop("covariance")
            std = covariance.diag().clamp_min(0).sqrt()
            variance_losses.append(F.relu(variance_target - std).mean())
            off_diagonal = (
                covariance.flatten()[:-1]
                .view(regularization_dim - 1, regularization_dim + 1)[:, 1:]
                .flatten()
            )
            covariance_losses.append(off_diagonal.square().sum() / regularization_dim)
        result["variance"] = float(torch.stack(variance_losses).mean())
        result["covariance"] = float(torch.stack(covariance_losses).mean())
        result["kl"] = 0.0
        result["latent_diagnostics"] = {
            "masked": masked_diagnostics,
            "clean": clean_diagnostics,
            "masked_clean_cosine": float(graph_cosine[0] / graph_cosine[1].clamp_min(1)),
            "regularization_space": (
                "projector" if bool(cfg["model"].get("vicreg_projector", False)) else "graph_z"
            ),
            "regularization_masked": masked_regularization_diagnostics,
            "regularization_clean": clean_regularization_diagnostics,
        }
        result["total"] = (
            float(objective["node_weight"]) * result["node"]
            + float(objective["edge_existence_weight"]) * result["existence"]
            + float(objective["edge_feature_weight"]) * result["edge_feature"]
            + float(objective["descriptor_weight"]) * result["descriptor"]
            + float(objective["invariance_weight"]) * result["invariance"]
            + float(objective["variance_weight"]) * result["variance"]
            + float(objective["covariance_weight"]) * result["covariance"]
            + float(objective.get("contrastive_weight", 0.0)) * result["contrastive"]
        )
        minimum_rank = float(objective.get("minimum_effective_rank_ratio", 0.25))
        rank_shortfall = max(0.0, minimum_rank - clean_diagnostics["effective_rank_ratio"])
        result["selection_score"] = result["total"] + float(
            objective.get("collapse_penalty_weight", 10.0)
        ) * rank_shortfall
    else:
        result["kl"] = mean_metric("kl")
        result["total"] = (
            float(objective["node_weight"]) * result["node"]
            + float(objective["edge_existence_weight"]) * result["existence"]
            + float(objective["edge_feature_weight"]) * result["edge_feature"]
            + float(objective["descriptor_weight"]) * result["descriptor"]
            + float(objective["kl_beta_max"]) * result["kl"]
        )
        result["selection_score"] = result["total"]
        latent_count = max(1, int(total_latent_nodes))
        latent_mean = latent_accumulator[0] / latent_count
        latent_variance = (latent_accumulator[1] / latent_count - latent_mean.square()).clamp_min(0)
        mean_kl_per_dimension = latent_accumulator[2] / latent_count
        result["latent_diagnostics"] = {
            "active_units": int((latent_variance > 0.01).sum()),
            "total_units": latent_dim,
            "variance_threshold": 0.01,
            "mean_variance": float(latent_variance.mean()),
            "median_variance": float(latent_variance.median()),
            "mean_kl_per_dimension": float(mean_kl_per_dimension.mean()),
            "median_kl_per_dimension": float(mean_kl_per_dimension.median()),
        }

    count = max(1, int(total_graphs))
    standardized_mae = descriptor_accumulator[0] / count
    raw_mae = standardized_mae * standardizer.scale.to(device=device, dtype=torch.float64)
    sum_prediction, sum_target = descriptor_accumulator[1], descriptor_accumulator[2]
    numerator = count * descriptor_accumulator[5] - sum_prediction * sum_target
    denominator = torch.sqrt(
        (count * descriptor_accumulator[3] - sum_prediction.square()).clamp_min(0)
        * (count * descriptor_accumulator[4] - sum_target.square()).clamp_min(0)
    )
    correlation = torch.where(
        denominator > 0, numerator / denominator, torch.full_like(numerator, float("nan"))
    )
    names = [str(item["name"]) for item in cfg["_descriptors"]["features"]]
    result["descriptor_standardized_mae"] = dict(zip(names, standardized_mae.cpu().tolist()))
    result["descriptor_raw_mae"] = dict(zip(names, raw_mae.cpu().tolist()))
    result["descriptor_pearson_r"] = dict(
        zip(names, [value if math.isfinite(value) else None for value in correlation.cpu().tolist()])
    )
    result["node_feature_metrics"] = _finalize_group_confusions(node_confusions, schema["node_groups"])
    result["edge_feature_metrics"] = _finalize_group_confusions(edge_confusions, schema["edge_groups"])
    result["node_exact_match"] = float(node_exact[0] / node_exact[1].clamp_min(1))
    result["edge_exact_match"] = float(edge_exact[0] / edge_exact[1].clamp_min(1))
    result["edge_existence_metrics"] = {
        "easy": _binary_histogram_metrics(score_histograms["positive"], score_histograms["easy_negative"]),
        "model_hard": _binary_histogram_metrics(score_histograms["positive"], score_histograms["hard_negative"]),
    }
    model.train()
    return result


def _scheduler(optimizer, training_cfg: dict[str, Any]):
    warmup = int(training_cfg["warmup_steps"])
    maximum = int(training_cfg["max_steps"])
    minimum_ratio = float(training_cfg["min_learning_rate"]) / float(training_cfg["learning_rate"])

    def multiplier(step: int) -> float:
        if step < warmup:
            return max(1e-8, (step + 1) / max(1, warmup))
        progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _log_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def train(cfg: dict[str, Any], allow_cpu: bool = False) -> int:
    rank, world_size, local_rank, device = _distributed_context(allow_cpu)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)
    seed_everything(_training_seed(cfg) + rank)
    work_dir = Path(cfg["paths"]["work_dir"])
    run_dir = ensure_directory(cfg["paths"]["run_dir"])
    graph_manifest_path = work_dir / "graph_manifest.json"
    scaler_path = work_dir / "descriptor_scaler.json"
    manifest = load_graph_manifest(graph_manifest_path)
    validate_feature_schema(manifest["feature_schema"])
    standardizer = Standardizer.load(scaler_path)
    if manifest["config_hash"] != cfg["_config_hash"]:
        raise RuntimeError("Graph manifest and training configuration differ")
    if standardizer.schema_hash != cfg["_descriptor_schema_hash"]:
        raise RuntimeError("Descriptor scaler schema does not match the configuration")
    if manifest["descriptor_schema_hash"] != standardizer.schema_hash:
        raise RuntimeError("Graph manifest and descriptor scaler schemas differ")
    if manifest["scaler_hash"] != standardizer.scaler_hash:
        raise RuntimeError("Graph manifest and descriptor scaler identities differ")
    identity = {
        "config_hash": cfg["_config_hash"],
        "graph_manifest_hash": manifest["graph_manifest_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "feature_schema_hash": manifest["feature_schema"]["hash"],
        "scaler_hash": standardizer.scaler_hash,
        "training_implementation_version": _implementation_version(cfg),
        "training_plan_hash": _training_plan_hash(cfg),
    }
    already_complete = torch.tensor(0, dtype=torch.int32, device=device)
    if rank == 0:
        atomic_write_json(run_dir / "resolved_config.json", public_config(cfg))
        atomic_write_json(run_dir / "identity.json", {**identity, "runtime": runtime_versions()})
        stop_file = run_dir / "REQUEST_CHECKPOINT"
        if stop_file.exists():
            stop_file.unlink()
        if (run_dir / "COMPLETE").exists():
            already_complete.fill_(1)
    if world_size > 1:
        dist.broadcast(already_complete, src=0)
    if bool(already_complete.item()):
        if rank == 0:
            print(f"Run is already complete: {run_dir}", flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return 0
    if world_size > 1:
        dist.barrier()

    model = _build_model(
        manifest["feature_schema"], len(cfg["data"]["descriptor_columns"]), cfg
    ).to(device)
    training_cfg = cfg["training"]
    optimizer_kwargs = {
        "lr": float(training_cfg["learning_rate"]),
        "weight_decay": float(training_cfg["weight_decay"]),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (RuntimeError, TypeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = _scheduler(optimizer, training_cfg)
    use_fp16 = str(training_cfg["precision"]).lower() == "fp16" and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if device.type == "cuda" else None
    global_step = 0
    best_validation = float("inf")
    data_state = None
    checkpoint_path = run_dir / "last.pt"
    initialization_path = training_cfg.get("initialize_from_checkpoint")
    if initialization_path and not checkpoint_path.is_file():
        initialization = _initialize_model_from_checkpoint(
            model,
            initialization_path,
            str(training_cfg["initialize_from_sha256"]),
            identity,
        )
        if rank == 0:
            atomic_write_json(run_dir / "initialization.json", initialization)
            print(
                "Initialized model weights from "
                f"{initialization['source_checkpoint']} at source step "
                f"{initialization['source_global_step']}; optimizer and scheduler reset",
                flush=True,
            )
    if str(training_cfg.get("resume", "auto")) == "auto" and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, identity, world_size)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if amp_scaler is not None and checkpoint.get("amp_scaler") is not None:
            amp_scaler.load_state_dict(checkpoint["amp_scaler"])
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])
        data_state = checkpoint["data_states"][rank]
        restore_rng_state(checkpoint["rng_states"][rank])
        if rank == 0:
            print(f"Resumed exactly from step {global_step}", flush=True)
    iterator = InfiniteGraphBatchIterator(
        manifest,
        standardizer,
        split="train",
        rank=rank,
        world_size=world_size,
        seed=_training_seed(cfg),
        node_budget=int(training_cfg["node_budget_per_gpu"]),
        graph_budget=int(training_cfg["max_graphs_per_gpu"]),
        state=data_state,
    )
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    model.train()
    max_steps = int(training_cfg["max_steps"])
    if global_step >= max_steps:
        if rank == 0:
            atomic_write_json(run_dir / "COMPLETE", {"global_step": global_step, **identity})
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return 0
    prefetcher = _TrainingBatchPrefetcher(
        iterator,
        cfg,
        start_step=global_step,
        stop_step=max_steps,
        rank=rank,
        pin_memory=device.type == "cuda",
    )
    precision = str(training_cfg["precision"]).lower()
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    autocast_enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    started = time.monotonic()
    last_log = started
    interval_graphs = 0
    interval_nodes = 0
    interval_steps = 0
    interval_input_wait = 0.0
    interval_batch_build = 0.0
    interval_negative_sampling = 0.0
    interval_pin_memory = 0.0
    interval_training = 0.0
    stop_poll_interval = min(10, max(1, int(training_cfg["log_every_steps"])))

    def checkpoint_now(is_best: bool = False) -> None:
        nonlocal best_validation
        base_model = model.module if isinstance(model, DistributedDataParallel) else model
        state = build_checkpoint(
            model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_scaler=amp_scaler,
            global_step=global_step,
            best_validation=best_validation,
            data_state=prefetcher.checkpoint_state(),
            rank=rank,
            world_size=world_size,
            identity=identity,
        )
        if rank == 0 and state is not None:
            atomic_torch_save(state, checkpoint_path)
            if is_best:
                atomic_copy(checkpoint_path, run_dir / "best.pt")
            retain_every = int(training_cfg.get("retain_every_steps", 0))
            if retain_every > 0 and global_step % retain_every == 0:
                retained_dir = ensure_directory(run_dir / "checkpoints")
                atomic_copy(
                    checkpoint_path,
                    retained_dir / f"step-{global_step:09d}.pt",
                )

    while global_step < max_steps:
        wait_started = time.perf_counter()
        prepared = prefetcher.next(global_step)
        interval_input_wait += time.perf_counter() - wait_started
        interval_batch_build += prepared.batch_build_seconds
        interval_negative_sampling += prepared.negative_sampling_seconds
        interval_pin_memory += prepared.pin_memory_seconds
        batch = prepared.batch
        training_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)
            if autocast_enabled
            else nullcontext()
        )
        with context:
            losses = _losses_for_batch(
                model,
                batch,
                device,
                cfg,
                step=global_step,
                rank=rank,
                training=True,
                candidates=prepared.candidates,
            )
        if amp_scaler is not None and use_fp16:
            amp_scaler.scale(losses["total"]).backward()
            amp_scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_cfg["gradient_clip_norm"]), error_if_nonfinite=True
            )
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_cfg["gradient_clip_norm"]), error_if_nonfinite=True
            )
            optimizer.step()
        scheduler.step()
        interval_training += time.perf_counter() - training_started
        global_step += 1
        interval_graphs += int(losses["graphs"])
        interval_nodes += int(losses["nodes"])
        interval_steps += 1

        if global_step % int(training_cfg["log_every_steps"]) == 0:
            now = time.monotonic()
            elapsed = max(1e-9, now - last_log)
            if rank == 0:
                event = {
                    "event": "train",
                    "step": global_step,
                    "loss": {
                        key: float(losses[key].detach())
                        for key in (
                            "total",
                            "node",
                            "existence",
                            "edge_feature",
                            "descriptor",
                            "descriptor_mae",
                            "invariance",
                            "variance",
                            "covariance",
                            "contrastive",
                            "kl",
                            "kl_beta",
                        )
                        if key in losses
                    },
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "graphs_per_second_rank0": interval_graphs / elapsed,
                    "nodes_per_second_rank0": interval_nodes / elapsed,
                    "elapsed_seconds": now - started,
                    "data_cycle_rank0": prepared.start_state["cycle"],
                    "input_pipeline_rank0": {
                        "foreground_wait_seconds_per_step": interval_input_wait / interval_steps,
                        "batch_build_seconds_per_step": interval_batch_build / interval_steps,
                        "negative_sampling_seconds_per_step": interval_negative_sampling / interval_steps,
                        "pin_memory_seconds_per_step": interval_pin_memory / interval_steps,
                        "foreground_training_seconds_per_step": interval_training / interval_steps,
                    },
                }
                print(json.dumps(event, sort_keys=True), flush=True)
                _log_json(run_dir / "metrics.jsonl", event)
            last_log, interval_graphs, interval_nodes, interval_steps = now, 0, 0, 0
            interval_input_wait = 0.0
            interval_batch_build = 0.0
            interval_negative_sampling = 0.0
            interval_pin_memory = 0.0
            interval_training = 0.0

        is_best = False
        if global_step % int(training_cfg["validate_every_steps"]) == 0:
            validation = evaluate(
                model,
                manifest,
                standardizer,
                cfg,
                split="validation",
                max_graphs=int(training_cfg["validation_max_graphs"]),
                rank=rank,
                world_size=world_size,
                device=device,
                step=global_step,
            )
            if rank == 0:
                is_best = validation["selection_score"] < best_validation
                if is_best:
                    best_validation = validation["selection_score"]
                event = {"event": "validation", **validation, "best": is_best}
                print(json.dumps(event, sort_keys=True), flush=True)
                _log_json(run_dir / "metrics.jsonl", event)
            if world_size > 1:
                state_tensor = torch.tensor(
                    [best_validation if rank == 0 else 0.0, float(is_best if rank == 0 else 0)], device=device
                )
                dist.broadcast(state_tensor, src=0)
                best_validation = float(state_tensor[0])
                is_best = bool(state_tensor[1].item())

        stop_requested = False
        if global_step % stop_poll_interval == 0 or global_step >= max_steps:
            local_stop = _SIGNAL_REQUESTED or (run_dir / "REQUEST_CHECKPOINT").exists()
            if world_size > 1:
                stop_tensor = torch.tensor(int(local_stop), device=device)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                stop_requested = bool(stop_tensor.item())
            else:
                stop_requested = local_stop
        should_checkpoint = (
            global_step % int(training_cfg["checkpoint_every_steps"]) == 0
            or is_best
            or stop_requested
        )
        if should_checkpoint:
            checkpoint_now(is_best=is_best)
        if stop_requested:
            if rank == 0:
                print(f"Checkpointed at step {global_step}; requesting Slurm requeue", flush=True)
            prefetcher.close()
            if world_size > 1:
                dist.barrier()
                dist.destroy_process_group()
            return 99

    prefetcher.close()
    checkpoint_now(is_best=False)
    if rank == 0:
        atomic_write_json(run_dir / "COMPLETE", {"global_step": global_step, **identity})
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def evaluate_saved(
    cfg: dict[str, Any],
    checkpoint_name: str = "best.pt",
    split: str = "test",
    allow_cpu: bool = False,
    max_graphs: int | None = None,
) -> dict[str, Any]:
    rank, world_size, local_rank, device = _distributed_context(allow_cpu)
    work_dir = Path(cfg["paths"]["work_dir"])
    run_dir = Path(cfg["paths"]["run_dir"])
    resolved_config_path = run_dir / "resolved_config.json"
    if resolved_config_path.is_file():
        resolved = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        if resolved.get("_config_hash") != cfg["_config_hash"]:
            raise RuntimeError("Saved run and requested graph configuration differ")
        missing_sections = [key for key in ("model", "objective", "training") if key not in resolved]
        if missing_sections:
            raise RuntimeError(
                f"Saved resolved configuration lacks sections: {', '.join(missing_sections)}"
            )
        cfg = {
            **cfg,
            "model": resolved["model"],
            "objective": resolved["objective"],
            "training": resolved["training"],
        }
    manifest = load_graph_manifest(work_dir / "graph_manifest.json")
    validate_feature_schema(manifest["feature_schema"])
    standardizer = Standardizer.load(work_dir / "descriptor_scaler.json")
    checkpoint_path = run_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = {
        "config_hash": cfg["_config_hash"],
        "graph_manifest_hash": manifest["graph_manifest_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "feature_schema_hash": manifest["feature_schema"]["hash"],
        "scaler_hash": standardizer.scaler_hash,
        "training_implementation_version": _implementation_version(cfg),
        "training_plan_hash": _training_plan_hash(cfg),
    }
    # Evaluation is deterministic and does not restore rank-local data/RNG
    # cursors, so it may safely use a different GPU count than training.
    validate_checkpoint(checkpoint, identity)
    model = _build_model(manifest["feature_schema"], len(cfg["data"]["descriptor_columns"]), cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    evaluation_limit = int(max_graphs or cfg["training"]["test_max_graphs"])
    result = evaluate(
        model,
        manifest,
        standardizer,
        cfg,
        split=split,
        max_graphs=evaluation_limit,
        rank=rank,
        world_size=world_size,
        device=device,
        step=int(checkpoint["global_step"]),
    )
    result["checkpoint"] = {
        "name": str(checkpoint_name),
        "sha256": sha256_file(checkpoint_path),
        "global_step": int(checkpoint["global_step"]),
        "config_hash": cfg["_config_hash"],
        "training_plan_hash": _training_plan_hash(cfg),
        "graph_manifest_hash": manifest["graph_manifest_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "feature_schema_hash": manifest["feature_schema"]["hash"],
        "scaler_hash": standardizer.scaler_hash,
        "training_implementation_version": _implementation_version(cfg),
    }
    result["evaluation"] = {
        "split": split,
        "maximum_graphs": evaluation_limit,
        "world_size": world_size,
    }
    if rank == 0:
        atomic_write_json(run_dir / f"{split}_metrics.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return result

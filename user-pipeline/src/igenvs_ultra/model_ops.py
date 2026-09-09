#!/usr/bin/env python3
"""GPU-side model operations for the user-facing iGenVS-ultra pipeline.

This module is intentionally self-contained so the lightweight host driver can
invoke it inside the released gMolAI container without installing extra host
dependencies.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import contextlib
import csv
import gc
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from rdkit import rdBase
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .fast_policy import canonicalize_policy_chunk, initialize_policy_worker
except ImportError:  # model_ops.py is also executed directly inside the SIF
    from fast_policy import canonicalize_policy_chunk, initialize_policy_worker


SEEDS = (260904, 260905, 260906)
QUANTILES = (0.005, 0.01, 0.02, 0.05)
ARCHITECTURE = "wide_mlp_rank_aux"
EMBEDDING_DIMENSION = 384
EPOCHS = 7
TRAIN_BATCH_SIZE = 8192
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_NORM_CLIP = 5.0
INFERENCE_BATCH_SIZE = 65_536


class ModelOperationError(RuntimeError):
    """A model-side contract or execution error."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def atomic_npz(path: Path, *, compressed: bool = True, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


class WideRankMLP(nn.Module):
    """Frozen Phase-5 winner: 384 -> 512 -> 128, classifier + rank heads."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(384, 512),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.classifier = nn.Linear(128, 1)
        self.rank = nn.Linear(128, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        hidden = self.trunk(x)
        return {
            "logit": self.classifier(hidden).squeeze(-1),
            "rank": self.rank(hidden).squeeze(-1),
        }


def make_model() -> WideRankMLP:
    model = WideRankMLP()
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != 263_042:
        raise ModelOperationError(f"frozen architecture parameter count changed: {count}")
    return model


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelOperationError("CUDA was requested but is not visible")
    return device


def weighted_rank_regression(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = torch.sigmoid(prediction.float())
    target = target.float()
    weight = 1.0 + 4.0 * target.pow(4)
    raw = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.05)
    return (raw * weight).sum() / weight.sum()


def training_loss(
    outputs: dict[str, Tensor], labels: Tensor, desirability: Tensor, positive_weight: Tensor
) -> Tensor:
    classification = F.binary_cross_entropy_with_logits(
        outputs["logit"].float(), labels.float(), pos_weight=positive_weight
    )
    ranking = weighted_rank_regression(outputs["rank"], desirability)
    return classification + 0.5 * ranking


def load_embedding_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ModelOperationError(f"embedding bundle does not exist: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        required = {"embeddings", "input_row", "input_id", "input_smiles"}
        missing = required.difference(bundle.files)
        if missing:
            raise ModelOperationError(f"embedding bundle misses: {sorted(missing)}")
        matrix = bundle["embeddings"].astype(np.float32, copy=True)
        rows = bundle["input_row"].copy()
        ids = bundle["input_id"].copy()
        smiles = bundle["input_smiles"].copy()
        if "embedding_space" in bundle.files and str(np.asarray(bundle["embedding_space"]).item()) != "released_hybrid_w3":
            raise ModelOperationError(f"wrong embedding space: {path}")
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
        raise ModelOperationError(f"wrong embedding shape {matrix.shape}: {path}")
    if len(rows) != len(matrix) or len(ids) != len(matrix) or len(smiles) != len(matrix):
        raise ModelOperationError(f"embedding identity arrays are misaligned: {path}")
    if not np.isfinite(matrix).all():
        raise ModelOperationError(f"non-finite embedding: {path}")
    return matrix, rows, ids, smiles


def load_standardizer(assets: Path) -> tuple[np.ndarray, np.ndarray, Path, str]:
    path = assets / "phase-5-head-selection/artifacts/input-standardizer.npz"
    with np.load(path, allow_pickle=False) as bundle:
        mean = bundle["mean"].astype(np.float32)
        std = bundle["std"].astype(np.float32)
    if (
        mean.shape != (EMBEDDING_DIMENSION,)
        or std.shape != (EMBEDDING_DIMENSION,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std < 1e-6)
    ):
        raise ModelOperationError("the frozen input standardizer is invalid")
    return mean, std, path, sha256(path)


def load_aligned_scores(path: Path, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(len(ids), np.nan, dtype=np.float64)
    statuses = np.full(len(ids), "", dtype="U64")
    seen = np.zeros(len(ids), dtype=bool)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_row", "molecule_id", "status", "docking_score"}
        if not required.issubset(reader.fieldnames or []):
            raise ModelOperationError(f"score file has wrong schema: {path}")
        for row in reader:
            index = int(row["source_row"]) - 1
            if index < 0 or index >= len(ids) or seen[index]:
                raise ModelOperationError(f"invalid or duplicate source_row in {path}: {index + 1}")
            if str(ids[index]) != row["molecule_id"]:
                raise ModelOperationError(f"score/embedding identity mismatch at row {index + 1}")
            status = row["status"]
            if status == "success":
                value = float(row["docking_score"])
                if not math.isfinite(value):
                    raise ModelOperationError(f"non-finite successful docking at row {index + 1}")
                scores[index] = value
            elif row["docking_score"].strip():
                raise ModelOperationError(f"failed docking has an imputed score at row {index + 1}")
            statuses[index] = status
            seen[index] = True
    if not seen.all():
        raise ModelOperationError(f"score file lacks {int((~seen).sum())} terminal rows: {path}")
    if not np.array_equal(np.isfinite(scores), statuses == "success"):
        raise ModelOperationError(f"score/status contract failed: {path}")
    return scores, statuses


def frozen_desirability(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    count = len(reference)
    if count < 2:
        raise ModelOperationError("score reference requires at least two finite rows")
    output = np.empty(len(scores), dtype=np.float32)
    left = np.searchsorted(reference, scores, side="left")
    right = np.searchsorted(reference, scores, side="right")
    for index, score in enumerate(scores):
        lo, hi = int(left[index]), int(right[index])
        if hi > lo:
            position = 0.5 * (lo + hi - 1)
        elif lo <= 0:
            position = 0.0
        elif lo >= count:
            position = float(count - 1)
        else:
            lower, upper = float(reference[lo - 1]), float(reference[lo])
            fraction = 0.5 if upper == lower else (float(score) - lower) / (upper - lower)
            position = (lo - 1) + fraction
        output[index] = np.float32(1.0 - position / (count - 1))
    np.clip(output, 0.0, 1.0, out=output)
    return output


def stage_directory(job: Path, stage: str) -> Path:
    if stage == "initial":
        return job / "models/initial"
    if stage.startswith("round-") and stage[6:].isdigit() and 1 <= int(stage[6:]) <= 5:
        return job / "models" / stage
    raise ModelOperationError(f"invalid model stage: {stage}")


def stage_number(stage: str) -> int:
    return 0 if stage == "initial" else int(stage.split("-", 1)[1])


def update_final_pointer(job: Path, stage: str, manifest_path: Path) -> None:
    final_path = job / "models/final.json"
    if final_path.is_file():
        current = json.loads(final_path.read_text(encoding="utf-8"))
        if stage_number(current["stage"]) > stage_number(stage):
            return
    atomic_json(
        final_path,
        {
            "schema_version": 1,
            "status": "complete",
            "stage": stage,
            "ensemble_manifest": str(manifest_path.relative_to(job)),
            "ensemble_manifest_sha256": sha256(manifest_path),
        },
    )


def previous_stage(stage: str) -> Optional[str]:
    if stage == "initial":
        return None
    number = int(stage.split("-", 1)[1])
    return "initial" if number == 1 else f"round-{number - 1}"


def protocol_identity(job: Path) -> tuple[dict[str, Any], str]:
    path = job / "fit-config.json"
    if not path.is_file():
        raise ModelOperationError(f"missing fit configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8")), sha256(path)


def model_from_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("candidate") != ARCHITECTURE:
        raise ModelOperationError(f"checkpoint architecture mismatch: {path}")
    model = make_model()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def resolve_member(job: Path, member: dict[str, Any]) -> Path:
    path = job / member["checkpoint"]
    if not path.is_file() or sha256(path) != member["checkpoint_sha256"]:
        raise ModelOperationError(f"checkpoint missing or changed: {path}")
    return path


def predict_members(
    matrix: np.ndarray, manifest: dict[str, Any], job: Path, device: torch.device
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    loaded = load_member_predictors(manifest, job, device)
    try:
        return predict_loaded_members(matrix, loaded, device)
    finally:
        loaded.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_member_predictors(
    manifest: dict[str, Any], job: Path, device: torch.device
) -> list[tuple[dict[str, Any], nn.Module]]:
    """Load and verify an ensemble once for persistent screening."""
    loaded: list[tuple[dict[str, Any], nn.Module]] = []
    for member in manifest["members"]:
        checkpoint_path = resolve_member(job, member)
        model, checkpoint = model_from_checkpoint(checkpoint_path, device)
        if int(checkpoint["seed"]) != int(member["seed"]):
            raise ModelOperationError(f"checkpoint seed mismatch: {checkpoint_path}")
        loaded.append((dict(member), model))
    return loaded


def predict_loaded_members(
    matrix: np.ndarray,
    loaded: list[tuple[dict[str, Any], nn.Module]],
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Score with a resident ensemble while preserving the released math."""
    if len(matrix) == 0:
        records = [
            {
                "seed": int(member["seed"]),
                "checkpoint": member["checkpoint"],
                "checkpoint_sha256": member["checkpoint_sha256"],
                "inference_seconds": 0.0,
            }
            for member, _ in loaded
        ]
        return np.empty((len(loaded), 0), dtype=np.float32), records, 0.0
    probabilities = []
    records = []
    total_seconds = 0.0
    for member, model in loaded:
        pieces = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            for begin in range(0, len(matrix), INFERENCE_BATCH_SIZE):
                host = np.ascontiguousarray(matrix[begin : begin + INFERENCE_BATCH_SIZE], dtype=np.float32)
                batch = torch.from_numpy(host).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(batch)["logit"]
                pieces.append(torch.sigmoid(logits.float()).cpu().numpy())
                del batch
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        total_seconds += seconds
        probabilities.append(np.concatenate(pieces).astype(np.float32, copy=False))
        records.append(
            {
                "seed": int(member["seed"]),
                "checkpoint": member["checkpoint"],
                "checkpoint_sha256": member["checkpoint_sha256"],
                "inference_seconds": seconds,
            }
        )
        del pieces
    return np.stack(probabilities), records, total_seconds


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def ensemble_scores(member_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = member_probabilities.astype(np.float64).mean(axis=0)
    mutual_information = binary_entropy(mean) - binary_entropy(member_probabilities).mean(axis=0)
    return mean, np.maximum(mutual_information, 0.0)


def ranking_metrics(predictions: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ModelOperationError("evaluation population has degenerate labels")
    order = np.lexsort((ids.astype(str), -predictions))
    ranked = labels[order].astype(np.int64)
    cumulative = np.cumsum(ranked)
    output: dict[str, Any] = {"population": len(labels), "positives": positives}
    for fraction in (0.001, 0.005, 0.01, 0.02, 0.05):
        count = max(1, int(math.ceil(fraction * len(labels))))
        hits = int(cumulative[count - 1])
        output[f"recovery_at_{fraction:g}"] = hits / positives
        output[f"ef_at_{fraction:g}"] = (hits / count) / (positives / len(labels))
    required = int(math.ceil(0.80 * positives))
    output["fraction_for_80pct_recovery"] = (
        int(np.searchsorted(cumulative, required, side="left")) + 1
    ) / len(labels)
    output["average_precision"] = float(average_precision_score(labels.astype(np.int8), predictions))
    output["roc_auc"] = float(roc_auc_score(labels.astype(np.int8), predictions))
    return output


def train_member(
    output: Path,
    target_name: str,
    stage: str,
    seed: int,
    matrix: Tensor,
    labels: Tensor,
    desirability: Tensor,
    positive_weight: Tensor,
    thresholds: dict[str, float],
    protocol_hash: str,
    standardizer_hash: str,
    population: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = output / f"seed-{seed}.pt"
    metadata_path = output / f"seed-{seed}.json"
    if checkpoint_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("checkpoint_sha256") == sha256(checkpoint_path):
            return metadata
    configure_seed(seed)
    model = make_model().to(matrix.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(labels), device=matrix.device)
        total = 0.0
        batches = 0
        for begin in range(0, len(labels), TRAIN_BATCH_SIZE):
            positions = permutation[begin : begin + TRAIN_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=matrix.device.type,
                dtype=torch.bfloat16,
                enabled=matrix.device.type == "cuda",
            ):
                outputs = model(matrix.index_select(0, positions))
                loss = training_loss(
                    outputs,
                    labels.index_select(0, positions),
                    desirability.index_select(0, positions),
                    positive_weight,
                )
            if not torch.isfinite(loss):
                raise ModelOperationError(f"non-finite loss: {stage}/seed-{seed}/epoch-{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM_CLIP)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        mean_loss = total / batches
        history.append({"epoch": epoch, "training_loss": mean_loss})
        print(f"[head] {stage} seed={seed} epoch={epoch}/{EPOCHS} loss={mean_loss:.6f}", flush=True)
    checkpoint = {
        "schema_version": 1,
        "artifact_type": "igenvs_ultra_user_target_head_member",
        "candidate": ARCHITECTURE,
        "target": target_name,
        "stage": stage,
        "seed": seed,
        "epochs": EPOCHS,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "parameter_count": 263_042,
        "thresholds": thresholds,
        "protocol_sha256": protocol_hash,
        "standardizer_sha256": standardizer_hash,
        "training_population": population,
        "restart_policy": "from_scratch",
        "evaluation_labels_used": False,
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "target": target_name,
        "stage": stage,
        "seed": seed,
        "epochs": EPOCHS,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "parameter_count": 263_042,
        "training_rows": int(len(labels)),
        "training_positives": int(labels.sum().item()),
        "positive_weight": float(positive_weight.item()),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    atomic_json(metadata_path, metadata)
    del model, optimizer
    if matrix.device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def run_train(args: argparse.Namespace) -> None:
    job = args.job_dir.resolve()
    assets = args.assets_dir.resolve()
    output = stage_directory(job, args.stage)
    manifest_path = output / "ensemble-manifest.json"
    if manifest_path.is_file() and (output / "validation-metrics.json").is_file():
        update_final_pointer(job, args.stage, manifest_path)
        print(f"[head] {args.stage} already complete", flush=True)
        return
    config, protocol_hash = protocol_identity(job)
    target_name = config["target_name"]
    device = resolve_device(args.device)
    mean, std, standardizer_path, standardizer_hash = load_standardizer(assets)

    original_path = assets / "phase-1-udrl/embeddings/UDRL-train-embeddings.npz"
    original_embeddings, original_rows, original_ids, _ = load_embedding_bundle(original_path)
    if not np.array_equal(original_rows, np.arange(1, len(original_rows) + 1)):
        raise ModelOperationError("UDRL-train embedding source rows are not consecutive")
    original_scores, original_status = load_aligned_scores(
        job / "docking/UDRL-train/scores.csv", original_ids
    )
    original_finite = np.isfinite(original_scores)
    finite_scores = original_scores[original_finite]
    if len(finite_scores) < 100:
        raise ModelOperationError("too few successful UDRL-train dockings")
    thresholds = {
        f"Q{quantile:g}": float(np.quantile(finite_scores, quantile, method="linear"))
        for quantile in QUANTILES
    }
    reference = np.sort(finite_scores)
    reference_path = job / "models/score-reference.npz"
    if not reference_path.exists():
        atomic_npz(
            reference_path,
            compressed=False,
            sorted_finite_udrl_train_scores=reference,
            quantile_names=np.asarray(list(thresholds)),
            quantile_values=np.asarray(list(thresholds.values()), dtype=np.float64),
            source_score_sha256=np.asarray(sha256(job / "docking/UDRL-train/scores.csv")),
        )

    matrices = [original_embeddings[original_finite]]
    all_scores = [finite_scores]
    desirabilities = [
        (1.0 - (rankdata(finite_scores, method="average") - 1.0) / (len(finite_scores) - 1.0)).astype(np.float32)
    ]
    cumulative = []
    if args.stage != "initial":
        final_round = int(args.stage.split("-", 1)[1])
        for round_number in range(1, final_round + 1):
            selected_path = job / f"al/round-{round_number}/acquisition/selected-embeddings.npz"
            selected_matrix, _, selected_ids, _ = load_embedding_bundle(selected_path)
            selected_scores, selected_status = load_aligned_scores(
                job / f"al/round-{round_number}/docking/scores.csv", selected_ids
            )
            finite = np.isfinite(selected_scores)
            matrices.append(selected_matrix[finite])
            all_scores.append(selected_scores[finite])
            desirabilities.append(frozen_desirability(selected_scores[finite], reference))
            cumulative.append(
                {
                    "round": round_number,
                    "selected": int(len(selected_ids)),
                    "successful": int(finite.sum()),
                    "failed": int((~finite).sum()),
                    "score_sha256": sha256(job / f"al/round-{round_number}/docking/scores.csv"),
                    "selected_embeddings_sha256": sha256(selected_path),
                }
            )
            del selected_matrix, selected_scores, selected_status

    matrix_np = np.concatenate(matrices).astype(np.float32, copy=False)
    scores_np = np.concatenate(all_scores)
    desirability_np = np.concatenate(desirabilities).astype(np.float32, copy=False)
    matrix_np -= mean[None, :]
    matrix_np /= std[None, :]
    if not np.isfinite(matrix_np).all() or not np.isfinite(desirability_np).all():
        raise ModelOperationError("non-finite cumulative training values")
    labels_np = (scores_np <= thresholds["Q0.01"]).astype(np.float32)
    positives = int(labels_np.sum())
    if positives == 0 or positives == len(labels_np):
        raise ModelOperationError("degenerate cumulative labels")
    matrix = torch.from_numpy(np.ascontiguousarray(matrix_np)).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    desirability = torch.from_numpy(desirability_np).to(device)
    positive_weight = torch.tensor((len(labels_np) - positives) / positives, device=device)
    population = {
        "original_rows": int(len(original_ids)),
        "original_successful": int(original_finite.sum()),
        "original_failed": int((~original_finite).sum()),
        "acquired_rounds": cumulative,
        "finite_rows": int(len(labels_np)),
        "positive_rows": positives,
        "negative_rows": int(len(labels_np) - positives),
    }
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    members = []
    for seed in SEEDS:
        metadata = train_member(
            output,
            target_name,
            args.stage,
            seed,
            matrix,
            labels,
            desirability,
            positive_weight,
            thresholds,
            protocol_hash,
            standardizer_hash,
            population,
        )
        members.append(
            {
                "seed": seed,
                "epochs": EPOCHS,
                "checkpoint": str(Path(metadata["checkpoint"]).relative_to(job)),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "target": target_name,
        "stage": args.stage,
        "winner": ARCHITECTURE,
        "architecture": "384-512-128_shared_trunk_with_top1_logit_and_rank_outputs_dropout_0.10",
        "parameter_count_per_member": 263_042,
        "embedding_space": "released_hybrid_w3",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "ensemble_size": 3,
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "fixed_thresholds": thresholds,
        "score_reference": str(reference_path.relative_to(job)),
        "score_reference_sha256": sha256(reference_path),
        "standardizer": str(standardizer_path),
        "standardizer_sha256": standardizer_hash,
        "protocol_sha256": protocol_hash,
        "restart_policy": "from_scratch",
        "training_population": population,
        "members": members,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(manifest_path, manifest)

    del matrix, labels, desirability, positive_weight, matrix_np, original_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()

    valid_embeddings, valid_rows, valid_ids, _ = load_embedding_bundle(
        assets / "phase-1-udrl/embeddings/UDRL-valid-embeddings.npz"
    )
    if not np.array_equal(valid_rows, np.arange(1, len(valid_rows) + 1)):
        raise ModelOperationError("UDRL-valid embedding source rows are not consecutive")
    valid_scores, valid_status = load_aligned_scores(
        job / "docking/UDRL-valid/scores.csv", valid_ids
    )
    valid_embeddings -= mean[None, :]
    valid_embeddings /= std[None, :]
    member_predictions, inference_records, inference_seconds = predict_members(
        valid_embeddings, manifest, job, device
    )
    probabilities, _ = ensemble_scores(member_predictions)
    positive = np.isfinite(valid_scores) & (valid_scores <= thresholds["Q0.01"])
    metrics = ranking_metrics(probabilities, positive, valid_ids)
    finite = np.isfinite(valid_scores)
    metrics["finite_scores"] = int(finite.sum())
    metrics["docking_failures"] = int((~finite).sum())
    metrics["spearman_finite_scores"] = float(
        spearmanr(probabilities[finite], -valid_scores[finite]).statistic
    )
    metrics["inference_seconds_member_sum"] = inference_seconds
    validation = {
        "schema_version": 1,
        "status": "complete",
        "target": target_name,
        "stage": args.stage,
        "population": "UDRL-valid",
        "fixed_positive_threshold": thresholds["Q0.01"],
        "metrics": metrics,
        "members": inference_records,
        "evaluation_labels_used_for_training_or_model_selection": False,
    }
    atomic_json(output / "validation-metrics.json", validation)
    update_final_pointer(job, args.stage, manifest_path)
    print(
        f"[head] {args.stage} complete: rows={len(labels_np):,}, "
        f"UDRL-valid Recall@1%={metrics['recovery_at_0.01']:.4f}",
        flush=True,
    )


def stable_top(scores: np.ndarray, eligible: np.ndarray, ids: np.ndarray, count: int) -> np.ndarray:
    candidates = np.flatnonzero(eligible)
    if len(candidates) < count:
        raise ModelOperationError(f"only {len(candidates)} eligible rows for quota {count}")
    order = np.lexsort((ids[candidates].astype(str), -scores[candidates]))
    return candidates[order[:count]]


def reduced_pool_maxmin(
    normalized: Tensor,
    anchors: np.ndarray,
    eligible: np.ndarray,
    pool_size: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) < pool_size:
        raise ModelOperationError("diversity candidate population is too small")
    rng = np.random.Generator(np.random.PCG64(seed))
    pool = rng.choice(eligible_indices, size=pool_size, replace=False)
    pool.sort()
    device = normalized.device
    pool_tensor = torch.from_numpy(pool).to(device=device, dtype=torch.long)
    anchor_tensor = torch.from_numpy(anchors).to(device=device, dtype=torch.long)
    candidates = normalized.index_select(0, pool_tensor)
    references = normalized.index_select(0, anchor_tensor)
    maximum_similarity = torch.full((pool_size,), -torch.inf, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(references), 512):
            similarity = candidates @ references[begin : begin + 512].T
            torch.maximum(maximum_similarity, similarity.max(dim=1).values, out=maximum_similarity)
        positions = []
        distances = []
        for iteration in range(count):
            position = int(torch.argmin(maximum_similarity).item())
            positions.append(position)
            distances.append(1.0 - float(maximum_similarity[position].item()))
            similarity = candidates @ candidates[position]
            torch.maximum(maximum_similarity, similarity, out=maximum_similarity)
            maximum_similarity[position] = torch.inf
            if (iteration + 1) % 1000 == 0:
                print(f"[AL] diversity {iteration + 1:,}/{count:,}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    selected = pool[np.asarray(positions, dtype=np.int64)]
    return selected, np.asarray(distances), {
        "seed": seed,
        "eligible_rows": int(len(eligible_indices)),
        "candidate_pool_rows": pool_size,
        "selected_rows": count,
        "elapsed_seconds": time.perf_counter() - started,
        "minimum_recorded_selection_distance": float(min(distances)),
        "maximum_recorded_selection_distance": float(max(distances)),
    }


def run_acquire(args: argparse.Namespace) -> None:
    job = args.job_dir.resolve()
    assets = args.assets_dir.resolve()
    round_number = args.round
    output = job / f"al/round-{round_number}/acquisition"
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        print(f"[AL] round {round_number} acquisition already complete", flush=True)
        return
    source_stage = "initial" if round_number == 1 else f"round-{round_number - 1}"
    source_path = stage_directory(job, source_stage) / "ensemble-manifest.json"
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    al_path = assets / f"phase-2-al-sets/embeddings/AL-set-{round_number}-embeddings.npz"
    embeddings, source_rows, ids, smiles = load_embedding_bundle(al_path)
    if embeddings.shape != (1_000_000, EMBEDDING_DIMENSION):
        raise ModelOperationError(f"AL-set-{round_number} must contain exactly 1,000,000 embeddings")
    if not np.array_equal(source_rows, np.arange(1, 1_000_001)):
        raise ModelOperationError("AL source rows are not consecutive")
    mean, std, _, standardizer_hash = load_standardizer(assets)
    embeddings -= mean[None, :]
    embeddings /= std[None, :]
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise ModelOperationError("the released approximate MaxMin acquisition requires a CUDA GPU")
    member_predictions, member_records, inference_seconds = predict_members(
        embeddings, source_manifest, job, device
    )
    probability, uncertainty = ensemble_scores(member_predictions)
    matrix = torch.from_numpy(np.ascontiguousarray(embeddings)).to(device)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if torch.any(norms == 0):
        raise ModelOperationError("zero-norm standardized AL embedding")
    normalized = F.normalize(matrix, p=2.0, dim=1)

    eligible = np.ones(len(ids), dtype=bool)
    exploitation = stable_top(probability, eligible, ids, 15_000)
    eligible[exploitation] = False
    uncertainty_indices = stable_top(uncertainty, eligible, ids, 7_500)
    eligible[uncertainty_indices] = False
    anchors = np.concatenate((exploitation, uncertainty_indices))
    diversity_seed = 26_096_000 + 100 * round_number
    diversity, distances, diversity_details = reduced_pool_maxmin(
        normalized, anchors, eligible, 200_000, 7_500, diversity_seed
    )
    selected = np.concatenate((exploitation, uncertainty_indices, diversity))
    if len(selected) != 30_000 or len(np.unique(selected)) != 30_000:
        raise ModelOperationError("AL selection quota/disjointness failure")
    categories = np.asarray(
        ["exploitation"] * 15_000 + ["uncertainty"] * 7_500 + ["diversity"] * 7_500
    )
    category_ranks = np.concatenate(
        (np.arange(1, 15_001), np.arange(1, 7_501), np.arange(1, 7_501))
    )
    diversity_values = np.full(30_000, np.nan, dtype=np.float64)
    diversity_values[22_500:] = distances
    fields = [
        "selection_order",
        "al_source_row",
        "molecule_id",
        "smiles",
        "acquisition_category",
        "category_rank",
        "ensemble_probability",
        "ensemble_mutual_information",
        "diversity_min_cosine_distance",
        "diversity_candidate_pool_size",
    ]

    def selected_rows() -> Iterable[dict[str, Any]]:
        for position, index in enumerate(selected):
            category = str(categories[position])
            yield {
                "selection_order": position + 1,
                "al_source_row": int(source_rows[index]),
                "molecule_id": str(ids[index]),
                "smiles": str(smiles[index]),
                "acquisition_category": category,
                "category_rank": int(category_ranks[position]),
                "ensemble_probability": f"{probability[index]:.12g}",
                "ensemble_mutual_information": f"{uncertainty[index]:.12g}",
                "diversity_min_cosine_distance": (
                    "" if category != "diversity" else f"{diversity_values[position]:.12g}"
                ),
                "diversity_candidate_pool_size": "" if category != "diversity" else 200_000,
            }

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "selected.csv"
    atomic_csv(csv_path, selected_rows(), fields)
    selected_embedding_path = output / "selected-embeddings.npz"
    # Reload the selected rows from the released bundle so cumulative fitting
    # receives the exact stored float32 vectors, not a standardize/invert
    # round-trip.
    with np.load(al_path, allow_pickle=False) as released_bundle:
        raw_embeddings = released_bundle["embeddings"][selected].astype(np.float32, copy=True)
    atomic_npz(
        selected_embedding_path,
        embeddings=raw_embeddings.astype(np.float32, copy=False),
        input_row=np.arange(1, 30_001, dtype=np.int64),
        input_id=ids[selected],
        input_smiles=smiles[selected],
        al_source_row=source_rows[selected],
        acquisition_category=categories,
        ensemble_probability=probability[selected],
        ensemble_mutual_information=uncertainty[selected],
        embedding_space=np.asarray("released_hybrid_w3"),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "round": round_number,
        "source_stage": source_stage,
        "source_ensemble_manifest": str(source_path.relative_to(job)),
        "source_ensemble_manifest_sha256": sha256(source_path),
        "al_embeddings": str(al_path),
        "al_embeddings_sha256": sha256(al_path),
        "standardizer_sha256": standardizer_hash,
        "selection": str(csv_path.relative_to(job)),
        "selection_sha256": sha256(csv_path),
        "selected_embeddings": str(selected_embedding_path.relative_to(job)),
        "selected_embeddings_sha256": sha256(selected_embedding_path),
        "counts": {"total": 30_000, "exploitation": 15_000, "uncertainty": 7_500, "diversity": 7_500},
        "members": member_records,
        "inference_seconds_member_sum": inference_seconds,
        "diversity": diversity_details,
    }
    atomic_json(manifest_path, manifest)
    print(f"[AL] round {round_number} acquisition complete", flush=True)


def import_gmolai(gmolai_dir: Path) -> Any:
    inference = gmolai_dir / "inference"
    script = inference / "gmolai.py"
    if not script.is_file():
        raise ModelOperationError(f"gMolAI release inference script is missing: {script}")
    sys.path.insert(0, str(inference))
    return importlib.import_module("gmolai")


def build_encoder_resources(gmol: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Load the frozen encoder and its CPU pools for reuse across stream blocks."""
    gmol.configure_encode_runtime(args.encoder_threads)
    models_dir = args.gmolai_models_dir.resolve()
    artifact_hashes = gmol.validate_artifact_hashes(models_dir)
    device = gmol.encoder_api.resolve_device(args.device)
    bundle = gmol.encoder_api.load_model_bundle(models_dir, device)
    gmol.validate_hybrid_encoder(bundle)
    requested_batch = getattr(args, "encoder_batch_size", 512)
    # The portable auto planner starts from the largest batch already
    # representation-qualified by gMolAI.  A first-block calibration below
    # may select another qualified point on different hardware.
    initial_batch = 512 if str(requested_batch).lower() == "auto" else int(requested_batch)
    encoder = gmol.encoder_api.build_smiles_encoder(
        args.encoder_backend,
        bundle.model,
        bundle.coordinate_mean,
        bundle.coordinate_scale,
        device=device,
        batch_size=initial_batch,
        node_budget=args.encoder_node_budget,
        workers=args.encoder_workers,
        mean_node_weight=bundle.mean_node_weight,
        verify_rows=args.encoder_verify_rows,
    )
    backend_info = gmol.encoder_api.implementation_metadata(encoder)
    worker_count = max(1, int(backend_info["workers"]))
    policy_executor = None
    if worker_count > 1:
        policy_executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_policy_worker,
        )
    return {
        "artifact_hashes": artifact_hashes,
        "device": device,
        "bundle": bundle,
        "encoder": encoder,
        "backend_info": backend_info,
        "worker_count": worker_count,
        "policy_executor": policy_executor,
        "selected_batch_size": initial_batch,
        "calibration": {
            "source": "explicit" if str(requested_batch).lower() != "auto" else "pending",
            "selected_batch_size": initial_batch,
        },
    }


def close_encoder_resources(resources: dict[str, Any]) -> None:
    executor = resources.get("policy_executor")
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
        resources["policy_executor"] = None
    encoder = resources.get("encoder")
    if encoder is not None:
        encoder.close()


def _encoder_hardware_identity(resources: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = resources["device"]
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware = {
            "device_type": "cuda",
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
        }
    else:
        hardware = {
            "device_type": device.type,
            "machine": os.uname().machine,
            "cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        }
    allowed = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set(range(int(os.cpu_count() or 1)))
    )
    cpu_models: set[str] = set()
    physical_cores: set[tuple[str, str]] = set()
    try:
        for block in Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n"):
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    name, value = line.split(":", 1)
                    fields[name.strip()] = value.strip()
            if "processor" not in fields or int(fields["processor"]) not in allowed:
                continue
            description = fields.get("model name") or fields.get("Model") or ""
            if not description:
                description = ":".join(
                    filter(
                        None,
                        (fields.get("CPU implementer"), fields.get("CPU part")),
                    )
                )
            if description:
                cpu_models.add(description)
            physical_cores.add(
                (
                    fields.get("physical id", "0"),
                    fields.get("core id", fields["processor"]),
                )
            )
    except (OSError, ValueError):
        pass
    backend = resources["backend_info"]
    return {
        "calibration_version": 3,
        "hardware": hardware,
        "cpu": {
            "architecture": platform.machine(),
            "models": sorted(cpu_models),
            "affinity_cpus": len(allowed),
            "physical_cores": len(physical_cores) or len(allowed),
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rdkit": rdBase.rdkitVersion,
        "backend": args.encoder_backend,
        "backend_versions": {
            "fast_inference": backend.get("fast_inference_version"),
            "fast_graph": backend.get("fast_graph_version"),
        },
        "node_budget": int(args.encoder_node_budget),
        "workers": int(resources["worker_count"]),
        "threads": int(args.encoder_threads),
        "verify_rows": int(args.encoder_verify_rows),
        "model_artifacts": resources["artifact_hashes"],
    }


def _encoder_profile_path(args: argparse.Namespace, identity: dict[str, Any]) -> tuple[Path | None, str]:
    key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    root = getattr(args, "profile_cache", None)
    return ((Path(root).resolve() / f"encoder-{key}.json") if root else None, key)


def calibrate_encoder_batch(
    resources: dict[str, Any],
    args: argparse.Namespace,
    canonical_smiles: list[str],
    atom_counts: list[int],
) -> None:
    """Select the fastest qualified gMolAI batch on the current hardware."""
    if str(getattr(args, "encoder_batch_size", "auto")).lower() != "auto":
        return
    sustained = len(canonical_smiles) >= 32_768
    previous_source = resources["calibration"].get("source")
    # A tiny first stream block uses a cheap workload-local measurement.  If a
    # later block is large enough to represent steady state, promote it to a
    # reusable hardware profile instead of pinning that tiny-run choice.
    if previous_source != "pending" and not (
        sustained and previous_source == "workload_calibration"
    ):
        return
    identity = _encoder_hardware_identity(resources, args)
    profile_path, key = _encoder_profile_path(args, identity)
    if sustained and profile_path is not None and profile_path.is_file():
        try:
            cached = json.loads(profile_path.read_text(encoding="utf-8"))
            selected = int(cached["selected_batch_size"])
            if selected in {64, 128, 192, 256, 512}:
                resources["encoder"].batch_size = selected
                resources["selected_batch_size"] = selected
                resources["backend_info"]["batch_size"] = selected
                resources["calibration"] = {**cached, "source": "cache", "key": key}
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # These are the released representation-qualified points.  Do not silently
    # explore larger numerical batches merely because a GPU has more memory.
    sample_count = min(len(canonical_smiles), 49_152 if sustained else 4096)
    sample_values = canonical_smiles[:sample_count]
    sample_counts = atom_counts[:sample_count]
    encoder = resources["encoder"]
    candidates = [64, 128, 192, 256, 512]
    if sample_count < 512:
        candidates = sorted({min(candidate, sample_count) for candidate in candidates if sample_count})
    measurements = []
    for candidate in candidates:
        encoder.batch_size = candidate
        try:
            # One short warm call moves lazy kernels and worker creation out of
            # the measured pass for this candidate.
            warm_count = min(candidate, sample_count)
            _ = encoder.encode(sample_values[:warm_count], atom_counts=sample_counts[:warm_count])
            if resources["device"].type == "cuda":
                torch.cuda.synchronize(resources["device"])
            started = time.perf_counter()
            _ = encoder.encode(sample_values, atom_counts=sample_counts)
            if resources["device"].type == "cuda":
                torch.cuda.synchronize(resources["device"])
            elapsed = time.perf_counter() - started
            measurements.append(
                {
                    "batch_size": candidate,
                    "rows": sample_count,
                    "seconds": elapsed,
                    "rows_per_second": sample_count / max(elapsed, 1.0e-9),
                    "status": "complete",
                }
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if resources["device"].type == "cuda":
                torch.cuda.empty_cache()
            measurements.append(
                {
                    "batch_size": candidate,
                    "rows": sample_count,
                    "status": "out_of_memory",
                }
            )
    successful = [item for item in measurements if item["status"] == "complete"]
    if not successful:
        raise ModelOperationError("no qualified gMolAI encoder batch fits this device")
    selected = int(
        max(successful, key=lambda item: (item["rows_per_second"], item["batch_size"]))[
            "batch_size"
        ]
    )
    encoder.batch_size = selected
    resources["selected_batch_size"] = selected
    resources["backend_info"]["batch_size"] = selected
    profile = {
        "schema_version": 1,
        "key": key,
        "identity": identity,
        "selected_batch_size": selected,
        "measurements": measurements,
        "created_at_unix": time.time(),
    }
    resources["calibration"] = {
        **profile,
        "source": "sustained_calibration" if sustained else "workload_calibration",
    }
    if sustained and profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = profile_path.with_name(f"{profile_path.name}.{os.getpid()}.partial")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(profile, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, profile_path)


def encode_ephemeral(
    gmol: Any,
    args: argparse.Namespace,
    input_path: Path,
    metadata_path: Path,
    rejection_path: Path,
    *,
    resources: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], dict[str, Any]]:
    """Encode into host memory without creating an immediately deleted NPZ.

    Policy validation retains gMol's acceptance and canonical-SMILES contract,
    but parallelizes it and omits training-only scaffold/split calculations.
    The promoted gMol encoder and its exact batch boundaries are unchanged.
    """
    started_total = time.perf_counter()
    owns_resources = resources is None
    resources = resources or build_encoder_resources(gmol, args)
    artifact_hashes = resources["artifact_hashes"]
    device = resources["device"]
    bundle = resources["bundle"]
    encoder = resources["encoder"]
    backend_info = resources["backend_info"]
    worker_count = int(resources["worker_count"])

    source_rows: list[dict[str, str]] = []
    raw_rows: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    duplicate_nonempty_ids = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ModelOperationError("screening batch is empty or lacks a header")
        if len(headers) != len(set(headers)):
            raise ModelOperationError("screening batch has duplicate column names")
        for required in ("molecule_id", "smiles"):
            if required not in headers:
                raise ModelOperationError(f"screening batch lacks {required!r}")
        for input_row, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ModelOperationError(
                    f"malformed screening CSV record at input row {input_row}"
                )
            normalized = {str(key): str(value) for key, value in row.items()}
            source_rows.append(normalized)
            molecule_id = normalized["molecule_id"]
            if molecule_id:
                if molecule_id in seen_ids:
                    duplicate_nonempty_ids += 1
                seen_ids.add(molecule_id)
            raw_rows.append((molecule_id, normalized["smiles"]))
    if not raw_rows:
        raise ModelOperationError("screening batch contains no data rows")

    policy = bundle.resolved_config["data"]["canonicalization"]

    def policy_chunks() -> Iterable[tuple[int, list[tuple[str, str]], dict[str, Any]]]:
        for start in range(0, len(raw_rows), 512):
            yield start, raw_rows[start : start + 512], policy

    policy_started = time.perf_counter()
    groups: Iterable[list[tuple[int, str, str, str, int, Optional[str]]]]
    policy_executor = resources.get("policy_executor")
    if worker_count == 1:
        groups = map(canonicalize_policy_chunk, policy_chunks())
        temporary_executor = None
    elif policy_executor is not None:
        groups = policy_executor.map(canonicalize_policy_chunk, policy_chunks(), chunksize=1)
        temporary_executor = None
    else:
        temporary_executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_policy_worker,
        )
        groups = temporary_executor.map(canonicalize_policy_chunk, policy_chunks(), chunksize=1)

    accepted_input_rows: list[int] = []
    accepted_ids: list[str] = []
    canonical_smiles: list[str] = []
    atom_counts: list[int] = []
    unique_hashes: set[str] = set()
    rejection_reasons: Counter[str] = Counter()
    rejection_count = 0
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_temporary = rejection_path.with_suffix(rejection_path.suffix + ".partial")
    try:
        with rejection_temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["input_row", "input_id", "input_smiles", "reason"])
            expected_index = 0
            for group in groups:
                for index, molecule_id, canonical, molecule_hash, atom_count, reason in group:
                    if index != expected_index:
                        raise ModelOperationError("parallel gMol policy validation changed row order")
                    expected_index += 1
                    if reason is not None:
                        rejection_count += 1
                        rejection_reasons[reason] += 1
                        writer.writerow([index + 1, molecule_id, raw_rows[index][1], reason])
                        continue
                    accepted_input_rows.append(index + 1)
                    accepted_ids.append(molecule_id)
                    canonical_smiles.append(canonical)
                    atom_counts.append(atom_count)
                    unique_hashes.add(molecule_hash)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if temporary_executor is not None:
            temporary_executor.shutdown(wait=True, cancel_futures=True)
    os.replace(rejection_temporary, rejection_path)
    policy_seconds = time.perf_counter() - policy_started
    if canonical_smiles:
        calibrate_encoder_batch(resources, args, canonical_smiles, atom_counts)
        backend_info = resources["backend_info"]
        encoding_started = time.perf_counter()
        try:
            while True:
                try:
                    matrix = encoder.encode(canonical_smiles, atom_counts=atom_counts)
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    recoverable = any(
                        marker in message
                        for marker in (
                            "out of memory",
                            "failed to allocate",
                            "cublas_status_alloc_failed",
                            "launch out of resources",
                        )
                    )
                    selected_batch = int(resources["selected_batch_size"])
                    smaller = [
                        candidate
                        for candidate in (64, 128, 192, 256, 512)
                        if candidate < selected_batch
                    ]
                    if (
                        str(getattr(args, "encoder_batch_size", "auto")).lower()
                        != "auto"
                        or not recoverable
                        or not smaller
                    ):
                        raise
                    replacement = max(smaller)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    encoder.batch_size = replacement
                    resources["selected_batch_size"] = replacement
                    resources["backend_info"]["batch_size"] = replacement
                    calibration = resources["calibration"]
                    fallbacks = list(calibration.get("runtime_fallbacks", []))
                    fallbacks.append(
                        {
                            "from_batch_size": selected_batch,
                            "to_batch_size": replacement,
                            "error": str(exc).splitlines()[0],
                        }
                    )
                    resources["calibration"] = {
                        **calibration,
                        "source": "runtime_capacity_fallback",
                        "selected_batch_size": replacement,
                        "runtime_fallbacks": fallbacks,
                    }
        finally:
            if owns_resources:
                close_encoder_resources(resources)
        encoding_seconds = time.perf_counter() - encoding_started
    else:
        matrix = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        encoding_seconds = 0.0
        if owns_resources:
            close_encoder_resources(resources)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (len(canonical_smiles), EMBEDDING_DIMENSION):
        raise ModelOperationError(f"encoder returned unexpected shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ModelOperationError("encoder returned non-finite embeddings")

    total_seconds = time.perf_counter() - started_total
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "artifact_type": "igenvs_ultra_ephemeral_gmol_encoding",
        "input": {"path": str(input_path), "sha256": sha256(input_path)},
        "rows": {
            "total": len(source_rows),
            "accepted": len(canonical_smiles),
            "rejected": rejection_count,
            "unique_accepted_molecules": len(unique_hashes),
            "duplicate_nonempty_ids": duplicate_nonempty_ids,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "embedding": {
            "space": "released_hybrid_w3",
            "dimensions": EMBEDDING_DIMENSION,
            "dtype": "float32",
            "storage": "ephemeral_host_memory_not_persisted",
        },
        "artifacts": artifact_hashes,
        "canonicalization": policy,
        "execution": {
            "device": str(device),
            "backend": backend_info["backend"],
            "batch_size": int(resources["selected_batch_size"]),
            "node_budget": args.encoder_node_budget,
            "workers": worker_count,
            "verify_rows": args.encoder_verify_rows if args.encoder_backend == "verify" else 0,
            "threads": args.encoder_threads,
            "policy_validation": "parallel_inference_only_equivalent_v1",
            "policy_seconds": policy_seconds,
            "encoding_seconds": encoding_seconds,
            "elapsed_seconds": total_seconds,
            "rows_per_second": len(canonical_smiles) / total_seconds,
            "batch_calibration": resources["calibration"],
        },
        "rejections": {"path": str(rejection_path), "sha256": sha256(rejection_path)},
    }
    atomic_json(metadata_path, metadata)
    result = {
        "accepted": len(canonical_smiles),
        "rejected": rejection_count,
        "dimensions": EMBEDDING_DIMENSION,
        "embedding_space": "released_hybrid_w3",
        "embeddings": None,
        "rejections": str(rejection_path),
        "metadata": str(metadata_path),
        "device": str(device),
        "backend": backend_info["backend"],
        "workers": worker_count,
        "policy_seconds": policy_seconds,
        "encoding_seconds": encoding_seconds,
        "elapsed_seconds": total_seconds,
        "rows_per_second": len(canonical_smiles) / total_seconds,
        "batch_size": int(resources["selected_batch_size"]),
        "batch_calibration": resources["calibration"],
        "materialization_avoided": "compressed_npz_write_read_delete",
    }
    return (
        matrix,
        np.asarray(accepted_input_rows, dtype=np.int64),
        np.asarray(accepted_ids),
        np.asarray(canonical_smiles),
        source_rows,
        result,
    )


class PersistentScoreEngine:
    """Resident gMolAI encoder, preprocessing pools, and target ensemble."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.keep_embeddings:
            raise ModelOperationError(
                "persistent score workers currently require ephemeral embeddings"
            )
        self.job = args.job_dir.resolve()
        self.model_manifest_path = args.model_manifest.resolve()
        self.model_manifest = json.loads(
            self.model_manifest_path.read_text(encoding="utf-8")
        )
        self.mean, self.std, _, standardizer_hash = load_standardizer(
            args.assets_dir.resolve()
        )
        if self.model_manifest.get("standardizer_sha256") != standardizer_hash:
            raise ModelOperationError("model and frozen standardizer hashes differ")
        self.gmol = import_gmolai(args.gmolai_dir.resolve())
        self.encoder_resources = build_encoder_resources(self.gmol, args)
        self.device = resolve_device(args.device)
        if self.device != self.encoder_resources["device"]:
            raise ModelOperationError("gMolAI and target heads resolved different devices")
        self.loaded_members = load_member_predictors(
            self.model_manifest, self.job, self.device
        )
        self._warm(args)

    def _warm(self, args: argparse.Namespace) -> None:
        """Create lazy pools and kernels before the first measured score request."""
        sample = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCOC"]
        policy = self.encoder_resources["bundle"].resolved_config["data"]["canonicalization"]
        executor = self.encoder_resources.get("policy_executor")
        if executor is not None:
            tasks = [
                (0, [(str(index), value) for index, value in enumerate(sample)], policy)
                for _ in range(int(self.encoder_resources["worker_count"]))
            ]
            list(executor.map(canonicalize_policy_chunk, tasks, chunksize=1))
        encoder = self.encoder_resources["encoder"]
        if hasattr(encoder, "warm_workers"):
            encoder.warm_workers(sample)
        with torch.inference_mode():
            zeros = torch.zeros((len(sample), EMBEDDING_DIMENSION), device=self.device)
            for _, model in self.loaded_members:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    _ = model(zeros)["logit"]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def close(self) -> None:
        close_encoder_resources(self.encoder_resources)
        self.loaded_members.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def run_score(
    args: argparse.Namespace,
    *,
    persistent: Optional[PersistentScoreEngine] = None,
) -> dict[str, Any]:
    score_started = time.perf_counter()
    job = args.job_dir.resolve()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    score_manifest_path = output_path.with_suffix(".manifest.json")
    if score_manifest_path.is_file() and output_path.is_file():
        existing = json.loads(score_manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("output_sha256") == sha256(output_path):
            print(f"[score] already complete: {output_path.name}", flush=True)
            return existing
    if persistent is None:
        model_manifest_path = args.model_manifest.resolve()
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        mean, std, _, standardizer_hash = load_standardizer(args.assets_dir.resolve())
        if model_manifest.get("standardizer_sha256") != standardizer_hash:
            raise ModelOperationError("model and frozen standardizer hashes differ")
        gmol = import_gmolai(args.gmolai_dir.resolve())
    else:
        if job != persistent.job or args.model_manifest.resolve() != persistent.model_manifest_path:
            raise ModelOperationError("persistent score request changed its job or model")
        model_manifest_path = persistent.model_manifest_path
        model_manifest = persistent.model_manifest
        mean, std = persistent.mean, persistent.std
        gmol = persistent.gmol
    embedding_path = output_path.with_suffix(".embeddings.npz")
    rejection_path = embedding_path.with_suffix(".rejections.csv")
    metadata_path = embedding_path.with_suffix(".metadata.json")
    encode_args = argparse.Namespace(
        input=input_path,
        models_dir=args.gmolai_models_dir.resolve(),
        output=embedding_path,
        smiles_column="smiles",
        id_column="molecule_id",
        backend=args.encoder_backend,
        batch_size=(
            512
            if str(args.encoder_batch_size).lower() == "auto"
            else int(args.encoder_batch_size)
        ),
        node_budget=args.encoder_node_budget,
        workers=args.encoder_workers,
        verify_rows=args.encoder_verify_rows,
        invalid_policy="report",
        limit=None,
        # A completed score manifest is checked above. Overwrite only the
        # batch-local encoder files left by an interrupted scoring attempt.
        overwrite=True,
        device=args.device,
        threads=args.encoder_threads,
    )
    print(f"[score] encoding {input_path.name}", flush=True)
    if args.keep_embeddings:
        encoding_started = time.perf_counter()
        encode_result = gmol.run_encode(encode_args)
        encode_result["elapsed_seconds"] = time.perf_counter() - encoding_started
        if int(encode_result["accepted"]) > 0:
            encode_result["rows_per_second"] = (
                int(encode_result["accepted"]) / float(encode_result["elapsed_seconds"])
            )
        source_rows = []
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_rows.append(row)
        with np.load(embedding_path, allow_pickle=False) as bundle:
            matrix = bundle["embeddings"].astype(np.float32, copy=True)
            input_rows = bundle["input_row"].copy()
            ids = bundle["input_id"].copy()
            canonical = bundle["canonical_smiles"].copy()
    else:
        matrix, input_rows, ids, canonical, source_rows, encode_result = encode_ephemeral(
            gmol,
            args,
            input_path,
            metadata_path,
            rejection_path,
            resources=persistent.encoder_resources if persistent is not None else None,
        )
    if persistent is None:
        del gmol
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not np.all((input_rows >= 1) & (input_rows <= len(source_rows))):
        raise ModelOperationError("gMolAI input-row alignment is invalid")
    for offset, input_row in enumerate(input_rows):
        if str(ids[offset]) != source_rows[int(input_row) - 1]["molecule_id"]:
            raise ModelOperationError("gMolAI ID alignment is invalid")
    matrix -= mean[None, :]
    matrix /= std[None, :]
    if not np.isfinite(matrix).all():
        raise ModelOperationError("non-finite standardized screening embedding")
    device = persistent.device if persistent is not None else resolve_device(args.device)
    if persistent is None:
        member_predictions, members, inference_seconds = predict_members(
            matrix, model_manifest, job, device
        )
    else:
        member_predictions, members, inference_seconds = predict_loaded_members(
            matrix, persistent.loaded_members, device
        )
    probabilities, uncertainty = ensemble_scores(member_predictions)
    if args.save_policy == "threshold":
        keep = probabilities >= args.score_threshold
    else:
        keep = np.ones(len(probabilities), dtype=bool)
    fields = [
        "molecule_id",
        "smiles",
        "original_smiles",
        "source_kind",
        "source_batch",
        "source_row",
        "target",
        "model_stage",
        *[f"member_probability_{seed}" for seed in SEEDS],
        "ensemble_probability",
        "ensemble_mutual_information",
    ]

    def output_rows() -> Iterable[dict[str, Any]]:
        for index in np.flatnonzero(keep):
            source = source_rows[int(input_rows[index]) - 1]
            row: dict[str, Any] = {
                "molecule_id": str(ids[index]),
                "smiles": str(canonical[index]),
                "original_smiles": source.get("original_smiles", source.get("smiles", "")),
                "source_kind": source.get("source_kind", "external"),
                "source_batch": source.get("source_batch", ""),
                "source_row": source.get("source_row", ""),
                "target": model_manifest["target"],
                "model_stage": model_manifest["stage"],
                "ensemble_probability": f"{probabilities[index]:.12g}",
                "ensemble_mutual_information": f"{uncertainty[index]:.12g}",
            }
            for member_index, seed in enumerate(SEEDS):
                row[f"member_probability_{seed}"] = f"{member_predictions[member_index, index]:.12g}"
            yield row

    retained = atomic_csv(output_path, output_rows(), fields)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "input_rows": len(source_rows),
        "encoded_rows": int(len(matrix)),
        "encoder_rejected_rows": int(len(source_rows) - len(matrix)),
        "retained_rows": retained,
        "save_policy": args.save_policy,
        "score_threshold": args.score_threshold if args.save_policy == "threshold" else None,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": sha256(model_manifest_path),
        "members": members,
        "inference_seconds_member_sum": inference_seconds,
        "encoder": encode_result,
        "encoder_rejections": str(rejection_path),
        "encoder_rejections_sha256": sha256(rejection_path),
        "encoder_metadata": str(metadata_path),
        "embeddings_retained": bool(args.keep_embeddings),
        "persistent_worker": persistent is not None,
        "score_wall_seconds": time.perf_counter() - score_started,
    }
    if not args.keep_embeddings:
        embedding_path.unlink(missing_ok=True)
    atomic_json(score_manifest_path, manifest)
    print(
        f"[score] complete: encoded={len(matrix):,}, retained={retained:,}",
        flush=True,
    )
    return manifest


def run_self_test(_: argparse.Namespace) -> None:
    model = make_model()
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((3, EMBEDDING_DIMENSION), dtype=torch.float32))
    if output["logit"].shape != (3,) or output["rank"].shape != (3,):
        raise ModelOperationError("model output schema changed")
    print("MODEL_OPS_SELF_TEST=passed")


def _worker_emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def run_score_worker(args: argparse.Namespace) -> None:
    """Serve many score blocks while all expensive state remains resident."""
    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        engine = PersistentScoreEngine(args)
    _worker_emit(
        {
            "event": "ready",
            "protocol_version": 1,
            "worker": "gMolAI+target-heads",
            "startup_seconds": time.perf_counter() - started,
            "device": str(engine.device),
            "encoder_workers": int(engine.encoder_resources["worker_count"]),
            "encoder_batch_size": int(engine.encoder_resources["selected_batch_size"]),
        }
    )
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                request_id = request.get("request_id")
                if request.get("command") == "shutdown":
                    _worker_emit({"event": "stopped", "request_id": request_id})
                    return
                if request.get("command") != "score":
                    raise ModelOperationError("unknown persistent score-worker command")
                request_args = argparse.Namespace(**vars(args))
                request_args.input = Path(request["input"])
                request_args.output = Path(request["output"])
                with contextlib.redirect_stdout(sys.stderr):
                    manifest = run_score(request_args, persistent=engine)
                _worker_emit(
                    {
                        "event": "result",
                        "request_id": request_id,
                        "result": {
                            "manifest": str(
                                request_args.output.resolve().with_suffix(".manifest.json")
                            ),
                            "input_rows": int(manifest["input_rows"]),
                            "encoded_rows": int(manifest["encoded_rows"]),
                            "retained_rows": int(manifest["retained_rows"]),
                            "encoder_batch_size": int(
                                engine.encoder_resources["selected_batch_size"]
                            ),
                            "batch_calibration": engine.encoder_resources["calibration"],
                        },
                    }
                )
            except BaseException as exc:
                _worker_emit(
                    {
                        "event": "error",
                        "request_id": request.get("request_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            engine.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal iGenVS-ultra model operations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--job-dir", type=Path, required=True)
    train.add_argument("--assets-dir", type=Path, required=True)
    train.add_argument("--stage", required=True)
    train.add_argument("--device", default="auto")
    train.set_defaults(handler=run_train)

    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--job-dir", type=Path, required=True)
    acquire.add_argument("--assets-dir", type=Path, required=True)
    acquire.add_argument("--round", type=int, choices=range(1, 6), required=True)
    acquire.add_argument("--device", default="auto")
    acquire.set_defaults(handler=run_acquire)

    score = subparsers.add_parser("score")
    score.add_argument("--job-dir", type=Path, required=True)
    score.add_argument("--assets-dir", type=Path, required=True)
    score.add_argument("--gmolai-dir", type=Path, required=True)
    score.add_argument("--gmolai-models-dir", type=Path, required=True)
    score.add_argument("--model-manifest", type=Path, required=True)
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--save-policy", choices=("all", "threshold"), required=True)
    score.add_argument("--score-threshold", type=float)
    score.add_argument("--keep-embeddings", action="store_true")
    score.add_argument("--device", default="auto")
    score.add_argument("--encoder-backend", choices=("optimized", "reference", "verify"), default="optimized")
    score.add_argument("--encoder-batch-size", default="auto")
    score.add_argument("--encoder-node-budget", type=int, default=16_384)
    score.add_argument("--encoder-workers", default="auto")
    score.add_argument("--encoder-verify-rows", type=int, default=1024)
    score.add_argument("--encoder-threads", type=int, default=8)
    score.add_argument("--profile-cache", type=Path)
    score.set_defaults(handler=run_score)

    worker = subparsers.add_parser("score-worker")
    worker.add_argument("--job-dir", type=Path, required=True)
    worker.add_argument("--assets-dir", type=Path, required=True)
    worker.add_argument("--gmolai-dir", type=Path, required=True)
    worker.add_argument("--gmolai-models-dir", type=Path, required=True)
    worker.add_argument("--model-manifest", type=Path, required=True)
    worker.add_argument("--save-policy", choices=("all", "threshold"), required=True)
    worker.add_argument("--score-threshold", type=float)
    worker.add_argument("--keep-embeddings", action="store_true")
    worker.add_argument("--device", default="auto")
    worker.add_argument(
        "--encoder-backend",
        choices=("optimized", "reference", "verify"),
        default="optimized",
    )
    worker.add_argument("--encoder-batch-size", default="auto")
    worker.add_argument("--encoder-node-budget", type=int, default=16_384)
    worker.add_argument("--encoder-workers", default="auto")
    worker.add_argument("--encoder-verify-rows", type=int, default=1024)
    worker.add_argument("--encoder-threads", type=int, default=8)
    worker.add_argument("--profile-cache", type=Path)
    worker.set_defaults(handler=run_score_worker)

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(handler=run_self_test)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "save_policy", None) == "threshold":
        if args.score_threshold is None or not 0.0 <= args.score_threshold <= 1.0:
            raise ModelOperationError("threshold policy requires --score-threshold in [0, 1]")
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

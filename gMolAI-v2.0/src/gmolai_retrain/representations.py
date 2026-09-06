from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .checkpoint import atomic_copy, atomic_torch_save, validate_checkpoint
from .data import Standardizer, finite_batches, load_graph_manifest
from .fast_inference import FAST_INFERENCE_VERSION, OptimizedRawCore
from .model import MolecularRepresentationModel, MolecularVGAE
from .schema import validate_feature_schema
from .train import (
    _architecture,
    _build_model,
    _distributed_context,
    _implementation_version,
    _training_plan_hash,
)
from .util import atomic_write_json


_EMBEDDING_DEFINITIONS = {
    "auto",
    "graph_z",
    "mean_node_z",
    "projector_z",
    "hybrid",
    "raw_hybrid",
    "standardized_raw_hybrid",
}

_RAW_HYBRID_DEFINITION = "clean_graph_z_plus_mean_node_z_raw_blocks"
_STANDARDIZED_RAW_HYBRID_DEFINITION = (
    "clean_graph_z_plus_mean_node_z_train_standardized_raw_blocks"
)
_STRATIFIED_SAMPLING = (
    "deterministic_hash_bucket_stratified_without_replacement"
)


def reweight_hybrid_embeddings(
    source: str | Path,
    destination: str | Path,
    *,
    mean_node_weight: float,
) -> dict[str, Any]:
    """Atomically recalibrate a saved hybrid vector without re-encoding graphs."""
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        raise ValueError("Hybrid reweighting requires a distinct output path")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not math.isfinite(mean_node_weight) or mean_node_weight <= 0:
        raise ValueError("mean_node_weight must be finite and positive")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("embedding_definition") not in {
        "clean_graph_z_plus_mean_node_z_unit_blocks",
        _STANDARDIZED_RAW_HYBRID_DEFINITION,
    }:
        raise ValueError(
            "Only saved canonical or train-standardized hybrid embeddings can be reweighted"
        )
    parameters = metadata.get("embedding_parameters", {})
    graph_dimensions = int(parameters.get("graph_dimensions", 0))
    old_weight = float(parameters.get("mean_node_weight", 0.0))
    embeddings = payload.get("embeddings")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or graph_dimensions <= 0
        or graph_dimensions >= embeddings.shape[1]
        or old_weight <= 0
    ):
        raise ValueError("Saved hybrid embedding payload has invalid block metadata")
    result = copy.deepcopy(payload)
    result["embeddings"] = embeddings.clone()
    result["embeddings"][:, graph_dimensions:] *= float(mean_node_weight) / old_weight
    result["metadata"]["embedding_parameters"]["mean_node_weight"] = float(
        mean_node_weight
    )
    atomic_torch_save(result, destination_path)
    return {**result["metadata"], "output": str(destination_path)}

_REPRESENTATION_PROMOTION_GATES = {
    "embedding_effective_rank": ("minimum", 25.0),
    # A single extreme Kier-shape value can dominate mean R2 while leaving
    # every robust error statistic and the other topology targets unchanged.
    # Require both broad average quality and a strong median, plus an absolute
    # standardized-error ceiling, instead of letting one squared-error outlier
    # decide whether an otherwise stable representation is promotable.
    "held_out_topology_mean_r2": ("minimum", 0.90),
    "held_out_topology_median_r2": ("minimum", 0.95),
    "held_out_topology_mean_standardized_mae": ("maximum", 0.15),
    "scaffold_disjoint_topology_mean_r2": ("minimum", 0.95),
    "morgan_recall_at_10": ("minimum", 0.18),
    # Global Morgan-distance imitation is a sanity check, not the objective;
    # local chemical retrieval and scaffold organization are measured directly.
    "cosine_tanimoto_spearman": ("minimum", 0.35),
    "neighbor_mean_tanimoto": ("minimum", 0.20),
    "neighbor_tanimoto_enrichment": ("minimum", 1.70),
    "scaffold_neighbor_purity_enrichment": ("minimum", 25.0),
}

_DOWNSTREAM_PROMOTION_GATES = {
    "bace": ("roc_auc", "minimum", 0.82),
    "bbbp": ("roc_auc", "minimum", 0.87),
    # Ten repeated scaffold splits include substantially harder ESOL partitions;
    # 0.80 remains a strong floor relative to the pinned Morgan baseline (~1.60).
    "esol": ("rmse", "maximum", 0.80),
    "freesolv": ("rmse", "maximum", 1.30),
    "lipophilicity": ("rmse", "maximum", 0.85),
}

_DOWNSTREAM_MINIMUM_MOLECULES = {
    "bace": 1400,
    "bbbp": 1800,
    "esol": 1000,
    "freesolv": 600,
    "lipophilicity": 4000,
}

_DOWNSTREAM_DIAGNOSTIC_FEATURES = {
    "molecule_embedding",
    "morgan_radius2_2048",
    "unit_graph_z",
    "unit_mean_node_z",
    "graph_z",
    "mean_node_z",
    "raw_graph_z_plus_mean_node_z",
}


def _resolve_embedding_definition(
    requested: str, *, representation_model: bool
) -> str:
    """Resolve ``auto`` without changing the legacy checkpoint contract."""
    if requested not in _EMBEDDING_DEFINITIONS:
        raise ValueError(
            "embedding_definition must be auto, graph_z, mean_node_z, projector_z, "
            "hybrid, raw_hybrid, or standardized_raw_hybrid"
        )
    if representation_model:
        # The calibrated hybrid is the public v5 molecule representation. Keep
        # graph_z and mean_node_z available for diagnostics and ablations.
        return "hybrid" if requested == "auto" else requested
    if requested not in {"auto", "graph_z"}:
        raise ValueError("Alternative embedding definitions require the representation model")
    return "graph_z"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_embedding_calibrator(
    source: str | Path,
    destination: str | Path,
    *,
    minimum_graphs: int = 10_000,
) -> dict[str, Any]:
    """Fit immutable coordinate statistics on a stratified pretraining-train export."""
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        raise ValueError("Calibrator output must differ from its source embeddings")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if minimum_graphs <= 0:
        raise ValueError("minimum_graphs must be positive")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    embeddings = payload.get("embeddings")
    if metadata.get("embedding_definition") != _RAW_HYBRID_DEFINITION:
        raise ValueError("Calibrator requires a raw_hybrid embedding export")
    if metadata.get("split") != "train":
        raise ValueError("Calibrator must be fitted only on the pretraining train split")
    if metadata.get("sampling") != _STRATIFIED_SAMPLING:
        raise ValueError("Calibrator source must use stratified hash-bucket sampling")
    if int(metadata.get("sampled_source_buckets", 0)) != 256:
        raise ValueError("Calibrator source must cover all 256 source buckets")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] < minimum_graphs
        or embeddings.shape[1] <= 0
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("Calibrator source embeddings are invalid or too small")
    values = embeddings.double()
    coordinate_mean = values.mean(dim=0)
    coordinate_scale = values.std(dim=0, unbiased=False)
    if (
        not torch.isfinite(coordinate_mean).all()
        or not torch.isfinite(coordinate_scale).all()
        or float(coordinate_scale.min()) <= 1.0e-8
    ):
        raise ValueError("Raw latent coordinates are constant or non-finite")
    identity_keys = (
        "checkpoint",
        "checkpoint_sha256",
        "global_step",
        "config_hash",
        "training_plan_hash",
        "graph_manifest_hash",
        "descriptor_schema_hash",
    )
    calibration_metadata = {
        "schema_version": 1,
        "calibration_definition": "coordinate_mean_and_population_std",
        "source_embedding_definition": _RAW_HYBRID_DEFINITION,
        "source_embedding_sha256": _file_sha256(source_path),
        "split": "train",
        "graphs": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "sampling": metadata.get("sampling"),
        "sampling_seed": metadata.get("sampling_seed"),
        "sampled_source_buckets": metadata.get("sampled_source_buckets"),
        **{key: metadata.get(key) for key in identity_keys},
    }
    artifact = {
        "metadata": calibration_metadata,
        "coordinate_mean": coordinate_mean.float(),
        "coordinate_scale": coordinate_scale.float(),
    }
    atomic_torch_save(artifact, destination_path)
    return {
        **calibration_metadata,
        "minimum_coordinate_scale": float(coordinate_scale.min()),
        "maximum_coordinate_scale": float(coordinate_scale.max()),
        "output": str(destination_path),
        "sha256": _file_sha256(destination_path),
    }


def _load_embedding_calibrator(
    calibrator: str | Path,
    *,
    expected: dict[str, Any],
    dimensions: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]:
    path = Path(calibrator).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    metadata = artifact.get("metadata", {})
    mean = artifact.get("coordinate_mean")
    scale = artifact.get("coordinate_scale")
    if metadata.get("calibration_definition") != "coordinate_mean_and_population_std":
        raise ValueError("Unsupported embedding calibrator definition")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Embedding calibrator does not match checkpoint at {key}")
    if (
        metadata.get("split") != "train"
        or metadata.get("sampling") != _STRATIFIED_SAMPLING
        or int(metadata.get("sampled_source_buckets", 0)) != 256
        or int(metadata.get("graphs", 0)) < 10_000
        or int(metadata.get("dimensions", 0)) != dimensions
        or not isinstance(mean, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or mean.shape != (dimensions,)
        or scale.shape != (dimensions,)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(scale).all()
        or float(scale.min()) <= 1.0e-8
    ):
        raise ValueError("Embedding calibrator payload is invalid")
    return mean.float(), scale.float(), metadata, _file_sha256(path)


def _calibrator_expected_identity(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "global_step": int(checkpoint["global_step"]),
        "config_hash": cfg["_config_hash"],
        "training_plan_hash": _training_plan_hash(cfg),
        "graph_manifest_hash": manifest["graph_manifest_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
    }


def _check_gate(name: str, value: Any, direction: str, threshold: float) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Promotion gate {name} is missing or non-numeric")
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(f"Promotion gate {name} is non-finite")
    passed = observed >= threshold if direction == "minimum" else observed <= threshold
    if not passed:
        comparator = ">=" if direction == "minimum" else "<="
        raise ValueError(
            f"Promotion gate {name} failed: {observed:.6g} must be {comparator} {threshold:.6g}"
        )
    return observed


def _validate_promotion_quality(
    probe: dict[str, Any], downstream: dict[str, Any]
) -> dict[str, Any]:
    if downstream.get("selected_only") is True:
        raise ValueError(
            "Promotion requires the full downstream diagnostic-baseline panel"
        )
    protocol_observed = {
        "validation_embeddings": probe.get("checkpoint_metadata", {}).get("graphs"),
        "probe_train_graphs": probe.get("held_out_linear_probe", {}).get(
            "train_graphs"
        ),
        "probe_validation_graphs": probe.get("held_out_linear_probe", {}).get(
            "validation_graphs"
        ),
        "similarity_graphs": probe.get("similarity", {}).get("graphs"),
        "similarity_available_graphs": probe.get("similarity", {}).get(
            "available_graphs"
        ),
        "clustering_graphs": probe.get("clustering", {}).get("graphs"),
        "clustering_sampled_graphs": probe.get("clustering", {}).get(
            "sampled_graphs"
        ),
        "clustering_kmeans_repetitions": probe.get("clustering", {}).get(
            "kmeans_repetitions"
        ),
        "train_sampled_source_buckets": probe.get(
            "train_embedding_metadata", {}
        ).get("sampled_source_buckets"),
        "validation_sampled_source_buckets": probe.get(
            "checkpoint_metadata", {}
        ).get("sampled_source_buckets"),
        "calibration_graphs": probe.get("checkpoint_metadata", {})
        .get("embedding_parameters", {})
        .get("calibration_graphs"),
    }
    protocol_minimums = {
        "validation_embeddings": 50000,
        "probe_train_graphs": 10000,
        "probe_validation_graphs": 50000,
        "similarity_graphs": 5000,
        "similarity_available_graphs": 50000,
        "clustering_graphs": 10000,
        "clustering_sampled_graphs": 50000,
        "clustering_kmeans_repetitions": 5,
        "train_sampled_source_buckets": 256,
        "validation_sampled_source_buckets": 256,
        "calibration_graphs": 100000,
    }
    protocol_report = {}
    for name, minimum in protocol_minimums.items():
        observed = _check_gate(
            f"protocol.{name}", protocol_observed.get(name), "minimum", minimum
        )
        protocol_report[name] = {"observed": int(observed), "minimum": minimum}
    expected_sampling = "seeded_without_replacement_across_export"
    if probe.get("similarity", {}).get("sampling") != expected_sampling:
        raise ValueError("Similarity probe did not use unbiased export-wide sampling")
    if probe.get("clustering", {}).get("sampling") != expected_sampling:
        raise ValueError("Clustering probe did not use unbiased export-wide sampling")
    expected_export_sampling = (
        "deterministic_hash_bucket_stratified_without_replacement"
    )
    if (
        probe.get("train_embedding_metadata", {}).get("sampling")
        != expected_export_sampling
    ):
        raise ValueError("Probe train export did not use hash-bucket-stratified sampling")
    if (
        probe.get("checkpoint_metadata", {}).get("sampling")
        != expected_export_sampling
    ):
        raise ValueError(
            "Probe validation export did not use hash-bucket-stratified sampling"
        )

    similarity = probe.get("similarity", {})
    observed_representation = {
        "embedding_effective_rank": probe.get("embedding_diagnostics", {}).get(
            "effective_rank"
        ),
        "held_out_topology_mean_r2": probe.get("held_out_linear_probe", {}).get(
            "mean_r2"
        ),
        "held_out_topology_median_r2": probe.get("held_out_linear_probe", {}).get(
            "median_r2"
        ),
        "held_out_topology_mean_standardized_mae": probe.get(
            "held_out_linear_probe", {}
        ).get("mean_standardized_mae"),
        "scaffold_disjoint_topology_mean_r2": probe.get(
            "scaffold_disjoint_linear_probe", {}
        ).get("mean_r2"),
        "morgan_recall_at_10": similarity.get("latent_to_morgan_recall_at_10"),
        "cosine_tanimoto_spearman": similarity.get(
            "latent_cosine_vs_morgan_spearman"
        ),
        "neighbor_mean_tanimoto": similarity.get(
            "latent_neighbor_mean_tanimoto"
        ),
        "neighbor_tanimoto_enrichment": similarity.get(
            "neighbor_tanimoto_enrichment"
        ),
        "scaffold_neighbor_purity_enrichment": similarity.get(
            "scaffold_purity_enrichment"
        ),
    }
    representation_report = {}
    for name, (direction, threshold) in _REPRESENTATION_PROMOTION_GATES.items():
        observed = _check_gate(
            name, observed_representation.get(name), direction, threshold
        )
        representation_report[name] = {
            "observed": observed,
            "direction": direction,
            "threshold": threshold,
        }

    clustering = probe.get("clustering", {})
    # These legacy keys name historical artifacts; probes.py uses ordinary
    # Euclidean KMeans after row L2 normalization, not spherical K-means.
    latent_ari = clustering.get("latent_spherical_kmeans", {}).get(
        "adjusted_rand_index"
    )
    morgan_ari = clustering.get("morgan_spherical_kmeans", {}).get(
        "adjusted_rand_index"
    )
    latent_nmi = clustering.get("latent_spherical_kmeans", {}).get(
        "normalized_mutual_information"
    )
    morgan_nmi = clustering.get("morgan_spherical_kmeans", {}).get(
        "normalized_mutual_information"
    )
    if not isinstance(morgan_ari, (int, float)):
        raise ValueError("Promotion gate Morgan clustering ARI is missing")
    observed_ari = _check_gate(
        "latent_clustering_ari_vs_morgan",
        latent_ari,
        "minimum",
        float(morgan_ari),
    )
    if not isinstance(morgan_nmi, (int, float)):
        raise ValueError("Promotion gate Morgan clustering NMI is missing")
    observed_nmi = _check_gate(
        "latent_clustering_nmi_noninferiority",
        latent_nmi,
        "minimum",
        float(morgan_nmi) - 0.03,
    )

    datasets = downstream.get("datasets", {})
    downstream_report = {}
    for dataset_name, (metric, direction, threshold) in _DOWNSTREAM_PROMOTION_GATES.items():
        dataset = datasets.get(dataset_name, {})
        feature_results = dataset.get("feature_results", {})
        missing_features = sorted(
            _DOWNSTREAM_DIAGNOSTIC_FEATURES - set(feature_results)
        )
        if missing_features:
            raise ValueError(
                f"{dataset_name} lacks diagnostic feature results: "
                + ", ".join(missing_features)
            )
        for feature_name in sorted(_DOWNSTREAM_DIAGNOSTIC_FEATURES):
            feature_value = (
                feature_results.get(feature_name, {})
                .get("summary", {})
                .get(metric, {})
                .get("mean")
            )
            if not isinstance(feature_value, (int, float)) or not math.isfinite(
                float(feature_value)
            ):
                raise ValueError(
                    f"{dataset_name}.{feature_name}.{metric} is missing or non-finite"
                )
        scaffold_splits = _check_gate(
            f"{dataset_name}.scaffold_splits",
            dataset.get("scaffold_splits"),
            "minimum",
            10,
        )
        molecules = _check_gate(
            f"{dataset_name}.molecules",
            dataset.get("preparation", {}).get("molecules"),
            "minimum",
            _DOWNSTREAM_MINIMUM_MOLECULES[dataset_name],
        )
        value = (
            feature_results
            .get("molecule_embedding", {})
            .get("summary", {})
            .get(metric, {})
            .get("mean")
        )
        observed = _check_gate(
            f"{dataset_name}.{metric}", value, direction, threshold
        )
        downstream_report[dataset_name] = {
            "metric": metric,
            "observed": observed,
            "direction": direction,
            "threshold": threshold,
            "scaffold_splits": int(scaffold_splits),
            "molecules": int(molecules),
            "diagnostic_features": sorted(_DOWNSTREAM_DIAGNOSTIC_FEATURES),
        }
    return {
        "protocol": protocol_report,
        "representation": representation_report,
        "clustering": {
            "adjusted_rand_index": {
                "observed": observed_ari,
                "minimum": float(morgan_ari),
            },
            "normalized_mutual_information": {
                "observed": observed_nmi,
                "minimum": float(morgan_nmi) - 0.03,
                "morgan": float(morgan_nmi),
                "noninferiority_margin": 0.03,
            },
        },
        "downstream": downstream_report,
    }


def resolve_run_configuration(cfg: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(cfg["paths"]["run_dir"])
    resolved_path = run_dir / "resolved_config.json"
    if not resolved_path.is_file():
        return cfg
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if resolved.get("_config_hash") != cfg["_config_hash"]:
        raise RuntimeError("Saved run and requested graph configuration differ")
    missing = [key for key in ("model", "objective", "training") if key not in resolved]
    if missing:
        raise RuntimeError(f"Saved resolved configuration lacks sections: {', '.join(missing)}")
    return {
        **cfg,
        "model": resolved["model"],
        "objective": resolved["objective"],
        "training": resolved["training"],
    }


def _automatic_checkpoint_name(cfg: dict[str, Any]) -> str:
    resolved = resolve_run_configuration(cfg)
    if _architecture(resolved) != "masked_graph_vicreg":
        return "best.pt"
    promoted = Path(resolved["paths"]["run_dir"]) / "representation-best.pt"
    if not promoted.is_file():
        raise FileNotFoundError(
            "No promoted representation-best.pt exists; validate and promote a retained "
            "checkpoint, or name an experimental checkpoint explicitly"
        )
    return promoted.name


def _automatic_representation_calibrator(cfg: dict[str, Any]) -> Path:
    resolved = resolve_run_configuration(cfg)
    run_dir = Path(resolved["paths"]["run_dir"])
    selection_path = run_dir / "representation_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            "Promoted representation lacks representation_selection.json"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("embedding_definition") != _STANDARDIZED_RAW_HYBRID_DEFINITION:
        raise ValueError("Promoted representation uses an unsupported embedding definition")
    metadata = selection.get("calibrator", {})
    name = metadata.get("promoted")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Promoted representation names an invalid calibrator")
    path = run_dir / name
    if not path.is_file() or _file_sha256(path) != metadata.get("sha256"):
        raise ValueError("Promoted representation calibrator is missing or changed")
    return path


def load_saved_model(
    cfg: dict[str, Any], checkpoint_name: str, device: torch.device
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Standardizer,
    MolecularVGAE | MolecularRepresentationModel,
    dict[str, Any],
]:
    cfg = resolve_run_configuration(cfg)
    work_dir = Path(cfg["paths"]["work_dir"])
    run_dir = Path(cfg["paths"]["run_dir"])
    manifest = load_graph_manifest(work_dir / "graph_manifest.json")
    validate_feature_schema(manifest["feature_schema"])
    standardizer = Standardizer.load(work_dir / "descriptor_scaler.json")
    checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu", weights_only=False)
    identity = {
        "config_hash": cfg["_config_hash"],
        "graph_manifest_hash": manifest["graph_manifest_hash"],
        "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        "feature_schema_hash": manifest["feature_schema"]["hash"],
        "scaler_hash": standardizer.scaler_hash,
        "training_implementation_version": _implementation_version(cfg),
        "training_plan_hash": _training_plan_hash(cfg),
    }
    validate_checkpoint(checkpoint, identity)
    model = _build_model(
        manifest["feature_schema"], len(cfg["data"]["descriptor_columns"]), cfg
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return cfg, manifest, standardizer, model, checkpoint


@torch.no_grad()
def export_embeddings(
    cfg: dict[str, Any],
    *,
    checkpoint_name: str = "auto",
    split: str = "test",
    max_graphs: int | None = None,
    skip_graphs: int = 0,
    output: str | Path | None = None,
    embedding_definition: str = "auto",
    mean_node_weight: float = 3.0,
    calibrator: str | Path | None = None,
    sampling_seed: int | None = None,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    rank, world_size, _, device = _distributed_context(allow_cpu)
    if world_size != 1:
        raise RuntimeError("Embedding export currently requires one process; run without torchrun")
    automatic_checkpoint = checkpoint_name == "auto"
    if automatic_checkpoint:
        checkpoint_name = _automatic_checkpoint_name(cfg)
    cfg, manifest, standardizer, model, checkpoint = load_saved_model(
        cfg, checkpoint_name, device
    )
    checkpoint_path = Path(cfg["paths"]["run_dir"]) / checkpoint_name
    requested_definition = embedding_definition
    if (
        automatic_checkpoint
        and embedding_definition == "auto"
        and isinstance(model, MolecularRepresentationModel)
    ):
        requested_definition = "standardized_raw_hybrid"
        if calibrator is None:
            calibrator = _automatic_representation_calibrator(cfg)
    selected_definition = _resolve_embedding_definition(
        requested_definition,
        representation_model=isinstance(model, MolecularRepresentationModel),
    )
    calibration_mean = None
    calibration_scale = None
    calibration_metadata: dict[str, Any] | None = None
    calibration_sha256 = None
    if selected_definition == "standardized_raw_hybrid":
        if not isinstance(model, MolecularRepresentationModel):
            raise ValueError("Standardized raw embeddings require the representation model")
        if calibrator is None:
            raise ValueError("standardized_raw_hybrid requires --calibrator")
        if not math.isfinite(mean_node_weight) or mean_node_weight <= 0:
            raise ValueError("mean_node_weight must be finite and positive")
        raw_dimensions = int(model.graph_latent_dim + model.node_latent_dim)
        calibration_mean, calibration_scale, calibration_metadata, calibration_sha256 = (
            _load_embedding_calibrator(
                calibrator,
                expected=_calibrator_expected_identity(
                    cfg, manifest, checkpoint_path, checkpoint
                ),
                dimensions=raw_dimensions,
            )
        )
        calibration_mean = calibration_mean.to(device)
        calibration_scale = calibration_scale.to(device)
    limit = int(max_graphs or cfg["training"]["test_max_graphs"])
    offset = int(skip_graphs)
    resolved_sampling_seed = (
        int(cfg["seed"]) if sampling_seed is None else int(sampling_seed)
    )
    if limit <= 0 or offset < 0:
        raise ValueError("max_graphs must be positive and skip_graphs must be non-negative")
    embeddings: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    graph_ids: list[torch.Tensor] = []
    source_buckets: list[torch.Tensor] = []
    molecule_hashes: list[str] = []
    optimized_core = (
        OptimizedRawCore(model).to(device).eval()
        if isinstance(model, MolecularRepresentationModel)
        else None
    )
    for batch in finite_batches(
        manifest,
        standardizer,
        split=split,
        rank=rank,
        world_size=world_size,
        max_graphs=limit,
        node_budget=int(cfg["training"]["node_budget_per_gpu"]),
        graph_budget=int(cfg["training"]["max_graphs_per_gpu"]),
        seed=resolved_sampling_seed,
        skip_graphs=offset,
    ):
        device_batch = batch.to(device, non_blocking=True)
        if isinstance(model, MolecularRepresentationModel):
            assert optimized_core is not None
            raw_hybrid = optimized_core(
                device_batch.x,
                device_batch.edge_index,
                device_batch.edge_attr,
                device_batch.batch,
            )
            learned_graph_z = raw_hybrid[:, : int(model.graph_latent_dim)]
            mean_node_z = raw_hybrid[:, int(model.graph_latent_dim) :]
            if selected_definition == "graph_z":
                graph_z = learned_graph_z
                definition = "clean_graph_z"
            elif selected_definition == "mean_node_z":
                graph_z = mean_node_z
                definition = "clean_mean_node_z"
            elif selected_definition == "projector_z":
                if model.vicreg_projector is None:
                    raise ValueError(
                        "projector_z requires a checkpoint trained with vicreg_projector"
                    )
                graph_z = model.vicreg_projector(learned_graph_z)
                definition = "clean_contrastive_projector_z"
            elif selected_definition in {"raw_hybrid", "standardized_raw_hybrid"}:
                if selected_definition == "standardized_raw_hybrid":
                    assert calibration_mean is not None and calibration_scale is not None
                    graph_z = model.apply_molecule_calibration(
                        raw_hybrid, calibration_mean, calibration_scale
                    )
                    graph_z[:, int(model.graph_latent_dim) :] *= float(
                        mean_node_weight
                    )
                    definition = _STANDARDIZED_RAW_HYBRID_DEFINITION
                else:
                    graph_z = raw_hybrid
                    definition = _RAW_HYBRID_DEFINITION
            else:
                graph_z = torch.cat(
                    (
                        F.normalize(learned_graph_z.float(), dim=-1),
                        float(mean_node_weight)
                        * F.normalize(mean_node_z.float(), dim=-1),
                    ),
                    dim=-1,
                )
                definition = "clean_graph_z_plus_mean_node_z_unit_blocks"
        else:
            _, mu, _ = model.encode(
                device_batch.x,
                device_batch.edge_index,
                device_batch.edge_attr,
                sample=False,
            )
            graph_z = global_mean_pool(mu, device_batch.batch)
            definition = "legacy_mean_node_posterior_mu"
        embeddings.append(graph_z.float().cpu())
        targets.append(batch.y.float().cpu())
        graph_ids.append(batch.graph_id.view(-1).cpu())
        source_buckets.append(batch.source_bucket.view(-1).cpu())
        molecule_hashes.extend(str(value) for value in batch.molecule_hash)
    embedding_tensor = torch.cat(embeddings) if embeddings else torch.empty((0, 0))
    target_tensor = torch.cat(targets) if targets else torch.empty((0, len(cfg["data"]["descriptor_columns"])))
    graph_id_tensor = torch.cat(graph_ids) if graph_ids else torch.empty(0, dtype=torch.long)
    bucket_tensor = torch.cat(source_buckets) if source_buckets else torch.empty(0, dtype=torch.int16)
    if embedding_tensor.shape[0] == 0:
        raise RuntimeError("Embedding export produced no graphs")
    if not (embedding_tensor.shape[0] == target_tensor.shape[0] == len(molecule_hashes)):
        raise RuntimeError("Embedding export lost graph alignment")
    destination = Path(output) if output else Path(cfg["paths"]["run_dir"]) / f"{split}-{Path(checkpoint_name).stem}-embeddings.pt"
    payload = {
        "metadata": {
            "schema_version": 1,
            "architecture": _architecture(cfg),
            "inference_backend": (
                "optimized_gine_v1"
                if isinstance(model, MolecularRepresentationModel)
                else "reference_legacy_vgae"
            ),
            "fast_inference_version": (
                FAST_INFERENCE_VERSION
                if isinstance(model, MolecularRepresentationModel)
                else None
            ),
            "embedding_definition": definition,
            "embedding_parameters": (
                {
                    "mean_node_weight": float(mean_node_weight),
                    "graph_dimensions": int(model.graph_latent_dim),
                    "mean_node_dimensions": int(model.node_latent_dim),
                }
                if selected_definition == "hybrid"
                else (
                    {
                        "graph_dimensions": int(model.graph_latent_dim),
                        "mean_node_dimensions": int(model.node_latent_dim),
                        "block_normalization": "none",
                    }
                    if selected_definition == "raw_hybrid"
                    else (
                        {
                            "graph_dimensions": int(model.graph_latent_dim),
                            "mean_node_dimensions": int(model.node_latent_dim),
                            "mean_node_weight": float(mean_node_weight),
                            "coordinate_transform": "train_mean_and_population_std",
                            "calibrator_sha256": calibration_sha256,
                            "calibration_graphs": int(
                                (calibration_metadata or {}).get("graphs", 0)
                            ),
                            "calibration_sampling_seed": (
                                calibration_metadata or {}
                            ).get("sampling_seed"),
                        }
                        if selected_definition == "standardized_raw_hybrid"
                        else {}
                    )
                )
            ),
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "global_step": int(checkpoint["global_step"]),
            "split": split,
            "graph_offset": offset,
            "graphs": int(embedding_tensor.shape[0]),
            "sampling": _STRATIFIED_SAMPLING,
            "sampling_seed": resolved_sampling_seed,
            "sampled_source_buckets": int(torch.unique(bucket_tensor).numel()),
            "population_source_buckets": int(
                len(
                    {
                        int(entry.get("bucket", -1))
                        for entry in manifest["shards"]
                        if entry["split"] == split
                    }
                )
            ),
            "population_graphs": int(
                sum(
                    int(entry["graphs"])
                    for entry in manifest["shards"]
                    if entry["split"] == split
                )
            ),
            "dimensions": int(embedding_tensor.shape[1]) if embedding_tensor.ndim == 2 else 0,
            "config_hash": cfg["_config_hash"],
            "training_plan_hash": _training_plan_hash(cfg),
            "graph_manifest_hash": manifest["graph_manifest_hash"],
            "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
        },
        "embeddings": embedding_tensor,
        "standardized_descriptor_targets": target_tensor,
        "graph_ids": graph_id_tensor,
        "source_buckets": bucket_tensor,
        "molecule_hashes": molecule_hashes,
    }
    atomic_torch_save(payload, destination)
    return {**payload["metadata"], "output": str(destination.resolve())}


def promote_representation_checkpoint(
    cfg: dict[str, Any],
    *,
    checkpoint_name: str,
    calibrator: str | Path,
    representation_probe: str | Path,
    downstream_benchmark: str | Path,
    destination_name: str = "representation-best.pt",
    calibrator_destination_name: str = "representation-calibrator.pt",
) -> dict[str, Any]:
    """Promote a validated retained milestone to the canonical downstream artifact."""
    if Path(destination_name).name != destination_name or not destination_name.endswith(".pt"):
        raise ValueError("destination_name must be a plain .pt filename")
    if (
        Path(calibrator_destination_name).name != calibrator_destination_name
        or not calibrator_destination_name.endswith(".pt")
    ):
        raise ValueError("calibrator_destination_name must be a plain .pt filename")
    checkpoint_relative = Path(checkpoint_name)
    step_text = checkpoint_relative.stem.removeprefix("step-")
    if (
        checkpoint_relative.parent != Path("checkpoints")
        or not checkpoint_relative.name.startswith("step-")
        or len(step_text) != 9
        or not step_text.isdigit()
    ):
        raise ValueError(
            "Only immutable checkpoints/step-NNNNNNNNN.pt milestones can be promoted"
        )
    run_dir = Path(cfg["paths"]["run_dir"]).resolve()
    source = (run_dir / checkpoint_name).resolve()
    if not source.is_relative_to(run_dir) or not source.is_file():
        raise FileNotFoundError(source)
    destination = run_dir / destination_name
    if source == destination:
        raise ValueError("Source checkpoint is already the promotion destination")

    cfg, manifest, _, _, checkpoint = load_saved_model(
        cfg, checkpoint_name, torch.device("cpu")
    )
    expected = _calibrator_expected_identity(cfg, manifest, source, checkpoint)
    calibrator_path = Path(calibrator).resolve()
    probe_path = Path(representation_probe).resolve()
    downstream_path = Path(downstream_benchmark).resolve()
    for path in (calibrator_path, probe_path, downstream_path):
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise FileNotFoundError(path)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    downstream = json.loads(downstream_path.read_text(encoding="utf-8"))
    probe_train_checkpoint = probe.get("train_embedding_metadata", {})
    probe_checkpoint = probe.get("checkpoint_metadata", {})
    downstream_checkpoint = downstream.get("checkpoint", {})
    _, _, calibration_metadata, calibrator_sha256 = _load_embedding_calibrator(
        calibrator_path,
        expected=expected,
        dimensions=int(
            cfg["model"]["graph_latent_dim"] + cfg["model"]["node_latent_dim"]
        ),
    )
    for label, metadata in (
        ("representation probe train export", probe_train_checkpoint),
        ("representation probe", probe_checkpoint),
        ("downstream benchmark", downstream_checkpoint),
    ):
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"{label} does not match checkpoint at {key}")
    if probe_train_checkpoint.get("checkpoint") != checkpoint_name:
        raise ValueError("Representation probe train export names a different checkpoint")
    if probe_checkpoint.get("checkpoint") != checkpoint_name:
        raise ValueError("Representation probe names a different checkpoint")
    if downstream_checkpoint.get("name") != checkpoint_name:
        raise ValueError("Downstream benchmark names a different checkpoint")
    canonical_definition = _STANDARDIZED_RAW_HYBRID_DEFINITION
    if probe_train_checkpoint.get("embedding_definition") != canonical_definition:
        raise ValueError(
            "Representation probe train export did not use the calibrated raw vector"
        )
    if probe_checkpoint.get("embedding_definition") != canonical_definition:
        raise ValueError("Representation probe did not evaluate the calibrated raw vector")
    if downstream_checkpoint.get("embedding_definition") != canonical_definition:
        raise ValueError("Downstream benchmark did not evaluate the calibrated raw vector")
    expected_parameters = {
        "graph_dimensions": int(cfg["model"]["graph_latent_dim"]),
        "mean_node_dimensions": int(cfg["model"]["node_latent_dim"]),
        "mean_node_weight": 3.0,
        "coordinate_transform": "train_mean_and_population_std",
        "calibrator_sha256": calibrator_sha256,
        "calibration_graphs": int(calibration_metadata["graphs"]),
        "calibration_sampling_seed": calibration_metadata["sampling_seed"],
    }
    if probe_train_checkpoint.get("embedding_parameters") != expected_parameters:
        raise ValueError(
            "Representation probe train export used a different calibrator"
        )
    if probe_checkpoint.get("embedding_parameters") != expected_parameters:
        raise ValueError("Representation probe used a different calibrator")
    if downstream_checkpoint.get("embedding_parameters") != expected_parameters:
        raise ValueError("Downstream benchmark used a different calibrator")
    expected_dimensions = (
        expected_parameters["graph_dimensions"]
        + expected_parameters["mean_node_dimensions"]
    )
    if probe_train_checkpoint.get("dimensions") != expected_dimensions:
        raise ValueError("Representation probe train export has unexpected dimensions")
    if probe_checkpoint.get("dimensions") != expected_dimensions:
        raise ValueError("Representation probe has unexpected embedding dimensions")
    if downstream_checkpoint.get("embedding_dimensions") != expected_dimensions:
        raise ValueError("Downstream benchmark has unexpected embedding dimensions")
    if not probe.get("clustering", {}).get("available", False):
        raise ValueError("Representation probe lacks an available clustering score")
    promotion_gates = _validate_promotion_quality(probe, downstream)

    calibrator_destination = run_dir / calibrator_destination_name
    if calibrator_path == calibrator_destination:
        raise ValueError("Calibrator source is already the promotion destination")
    atomic_copy(calibrator_path, calibrator_destination)
    if _file_sha256(calibrator_destination) != calibrator_sha256:
        raise RuntimeError("Promoted calibrator hash differs from retained source")
    # The checkpoint is copied last so its presence remains the promotion marker.
    atomic_copy(source, destination)
    source_hash = expected["checkpoint_sha256"]
    destination_hash = _file_sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError("Promoted checkpoint hash differs from retained source")
    selection = {
        "schema_version": 1,
        "selection_scope": (
            "scaffold validation and external development benchmarks; internal test not consulted"
        ),
        "source_checkpoint": checkpoint_name,
        "promoted_checkpoint": destination_name,
        "global_step": expected["global_step"],
        "checkpoint_sha256": destination_hash,
        "config_hash": expected["config_hash"],
        "training_plan_hash": expected["training_plan_hash"],
        "graph_manifest_hash": expected["graph_manifest_hash"],
        "embedding_definition": canonical_definition,
        "embedding_parameters": probe_checkpoint.get("embedding_parameters", {}),
        "calibrator": {
            "source": str(calibrator_path),
            "promoted": calibrator_destination_name,
            "sha256": calibrator_sha256,
            "metadata": calibration_metadata,
        },
        "promotion_gates": promotion_gates,
        "representation_probe": {
            "path": str(probe_path),
            "sha256": _file_sha256(probe_path),
            "embedding_diagnostics": probe.get("embedding_diagnostics"),
            "embedding_block_diagnostics": probe.get("embedding_block_diagnostics"),
            "held_out_linear_probe": probe.get("held_out_linear_probe"),
            "scaffold_disjoint_linear_probe": probe.get(
                "scaffold_disjoint_linear_probe"
            ),
            "similarity": probe.get("similarity"),
            "clustering": probe.get("clustering"),
        },
        "downstream_benchmark": {
            "path": str(downstream_path),
            "sha256": _file_sha256(downstream_path),
            "datasets": {
                name: {
                    "task": value.get("task"),
                    "molecules": value.get("preparation", {}).get("molecules"),
                    "molecule_embedding": value.get("feature_results", {}).get(
                        "molecule_embedding", {}
                    ).get("summary"),
                    "morgan_radius2_2048": value.get("feature_results", {}).get(
                        "morgan_radius2_2048", {}
                    ).get("summary"),
                }
                for name, value in downstream.get("datasets", {}).items()
            },
        },
    }
    atomic_write_json(run_dir / "representation_selection.json", selection)
    return selection

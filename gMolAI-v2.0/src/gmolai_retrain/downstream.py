from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from .chem import CanonicalMolecule, canonicalize, featurize_molecule
from .fast_graph import pack_molecules
from .fast_inference import (
    FAST_INFERENCE_VERSION,
    OptimizedRawCore,
    packed_to_device,
)
from .model import MolecularRepresentationModel
from .representations import (
    _RAW_HYBRID_DEFINITION,
    _STANDARDIZED_RAW_HYBRID_DEFINITION,
    _calibrator_expected_identity,
    _file_sha256,
    _load_embedding_calibrator,
    _resolve_embedding_definition,
    load_saved_model,
)
from .train import _distributed_context, _training_plan_hash
from .util import atomic_write_json


MOLECULENET_DATASETS: dict[str, dict[str, Any]] = {
    "esol": {
        "filename": "delaney-processed.csv",
        "sha256": "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189",
        "url": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
        "smiles_column": "smiles",
        "target_column": "measured log solubility in mols per litre",
        "task": "regression",
    },
    "freesolv": {
        "filename": "SAMPL.csv",
        "sha256": "ab5895d914ee87cb563bd7b9611e869527bba45bec6b014d34dc495a0f9dcb72",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "smiles_column": "smiles",
        "target_column": "expt",
        "task": "regression",
    },
    "lipophilicity": {
        "filename": "Lipophilicity.csv",
        "sha256": "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "smiles_column": "smiles",
        "target_column": "exp",
        "task": "regression",
    },
    "bbbp": {
        "filename": "BBBP.csv",
        "sha256": "d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smiles_column": "smiles",
        "target_column": "p_np",
        "task": "classification",
    },
    "bace": {
        "filename": "bace.csv",
        "sha256": "f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "smiles_column": "mol",
        "target_column": "Class",
        "task": "classification",
    },
    "hiv": {
        "filename": "HIV.csv",
        "sha256": "9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "smiles_column": "smiles",
        "target_column": "HIV_active",
        "task": "classification",
    },
}

DEFAULT_MOLECULENET_DATASETS = (
    "esol",
    "freesolv",
    "lipophilicity",
    "bbbp",
    "bace",
)


@dataclass(frozen=True)
class PreparedMoleculeNetDataset:
    """Canonical, deduplicated downstream rows in their deterministic order."""

    molecules: list[Chem.Mol]
    targets: np.ndarray
    scaffold_groups: np.ndarray
    canonical_smiles: tuple[str, ...]
    molecule_hashes: tuple[str, ...]
    source_buckets: np.ndarray
    preparation: dict[str, Any]


def _resolve_dataset_names(requested: list[str] | tuple[str, ...] | None) -> list[str]:
    if requested is None:
        return list(DEFAULT_MOLECULENET_DATASETS)
    names = list(dict.fromkeys(str(name).lower() for name in requested))
    if not names:
        raise ValueError("At least one downstream dataset must be requested")
    unknown = sorted(set(names) - set(MOLECULENET_DATASETS))
    if unknown:
        raise ValueError(f"Unknown MoleculeNet datasets: {', '.join(unknown)}")
    return names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_dataset_records(
    path: Path, spec: dict[str, Any], cfg: dict[str, Any]
) -> PreparedMoleculeNetDataset:
    observed_hash = _sha256(path)
    if observed_hash != spec["sha256"]:
        raise RuntimeError(
            f"MoleculeNet source hash mismatch for {path.name}: {observed_hash}"
        )
    frame = pd.read_csv(path)
    required = {spec["smiles_column"], spec["target_column"]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} lacks columns: {', '.join(missing)}")

    policy = cfg["data"]["canonicalization"]
    grouped: dict[str, tuple[CanonicalMolecule, Chem.Mol, list[float]]] = {}
    rejections: Counter[str] = Counter()
    valid_rows = 0
    for raw_smiles, raw_target in zip(
        frame[spec["smiles_column"]], frame[spec["target_column"]]
    ):
        try:
            target = float(raw_target)
        except (TypeError, ValueError):
            rejections["invalid_target"] += 1
            continue
        if not np.isfinite(target):
            rejections["invalid_target"] += 1
            continue
        canonical = canonicalize(
            str(raw_smiles),
            isomeric_smiles=bool(policy["isomeric_smiles"]),
            fragment_policy=str(policy["fragment_policy"]),
            allowed_elements=set(policy["allowed_elements"]),
            min_atoms=int(policy["min_atoms"]),
            max_atoms=int(policy["max_atoms"]),
            buckets=int(cfg["data"]["hash_buckets"]),
            split_cfg=cfg["data"]["split"],
        )
        if not isinstance(canonical, CanonicalMolecule):
            rejections[canonical.reason] += 1
            continue
        molecule = Chem.MolFromSmiles(canonical.smiles)
        if molecule is None:
            rejections["canonical_reparse_failure"] += 1
            continue
        valid_rows += 1
        existing = grouped.get(canonical.smiles)
        if existing is None:
            grouped[canonical.smiles] = (canonical, molecule, [target])
        else:
            existing[2].append(target)

    molecules: list[Chem.Mol] = []
    targets: list[float] = []
    scaffold_groups: list[str] = []
    canonical_smiles: list[str] = []
    molecule_hashes: list[str] = []
    source_buckets: list[int] = []
    conflicting_duplicates = 0
    duplicate_rows = 0
    for smiles in sorted(grouped):
        canonical, molecule, values = grouped[smiles]
        duplicate_rows += len(values) - 1
        if spec["task"] == "classification":
            labels = {int(round(value)) for value in values}
            if any(value not in {0, 1} for value in labels) or len(labels) != 1:
                conflicting_duplicates += len(values)
                continue
            target = float(next(iter(labels)))
        else:
            target = float(np.mean(values))
        molecules.append(molecule)
        targets.append(target)
        canonical_smiles.append(canonical.smiles)
        molecule_hashes.append(canonical.molecule_hash)
        source_buckets.append(int(canonical.bucket))
        scaffold_groups.append(
            f"SCAFFOLD:{canonical.scaffold}"
            if canonical.scaffold
            else f"ACYCLIC:{canonical.nonisomeric_smiles}"
        )

    if len(molecules) < 50:
        raise RuntimeError(f"Too few usable molecules in {path.name}: {len(molecules)}")
    target_array = np.asarray(targets, dtype=np.float64)
    if spec["task"] == "classification" and set(np.unique(target_array)) != {0.0, 1.0}:
        raise RuntimeError(f"{path.name} does not retain both binary classes")
    report = {
        "raw_rows": int(len(frame)),
        "valid_rows_before_deduplication": int(valid_rows),
        "molecules": int(len(molecules)),
        "scaffold_groups": int(len(set(scaffold_groups))),
        "duplicate_rows_collapsed": int(duplicate_rows),
        "conflicting_duplicate_rows_rejected": int(conflicting_duplicates),
        "rejections": dict(sorted(rejections.items())),
        "target_mean": float(target_array.mean()),
        "target_std": float(target_array.std()),
        "positive_fraction": (
            float(target_array.mean()) if spec["task"] == "classification" else None
        ),
    }
    return PreparedMoleculeNetDataset(
        molecules=molecules,
        targets=target_array,
        scaffold_groups=np.asarray(scaffold_groups, dtype=object),
        canonical_smiles=tuple(canonical_smiles),
        molecule_hashes=tuple(molecule_hashes),
        source_buckets=np.asarray(source_buckets, dtype=np.int16),
        preparation=report,
    )


def _prepare_dataset(
    path: Path, spec: dict[str, Any], cfg: dict[str, Any]
) -> tuple[list[Chem.Mol], np.ndarray, np.ndarray, dict[str, Any]]:
    """Backward-compatible tuple view used by the pretrained-model benchmark."""

    prepared = _prepare_dataset_records(path, spec, cfg)
    return (
        prepared.molecules,
        prepared.targets,
        prepared.scaffold_groups,
        prepared.preparation,
    )


@torch.no_grad()
def _encode_molecules(
    model: torch.nn.Module,
    molecules: list[Chem.Mol],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    include_chirality = bool(cfg["features"]["include_atom_chirality"])
    position_dim = int(cfg["features"]["canonical_position_encoding_dim"])
    if isinstance(model, MolecularRepresentationModel):
        if not include_chirality or position_dim != 0:
            raise ValueError(
                "Optimized downstream inference requires the promoted feature contract"
            )
        core = OptimizedRawCore(model).to(device).eval()
        chunks: list[torch.Tensor] = []
        raw_graph_chunks: list[torch.Tensor] = []
        raw_mean_node_chunks: list[torch.Tensor] = []
        pending: list[Chem.Mol] = []
        pending_nodes = 0

        def flush_representation() -> None:
            nonlocal pending, pending_nodes
            if not pending:
                return
            packed = pack_molecules(pending)
            raw = core(*packed_to_device(packed, device))
            graph_z = raw[:, : int(model.graph_latent_dim)]
            mean_node_z = raw[:, int(model.graph_latent_dim) :]
            embeddings = torch.cat(
                (
                    F.normalize(graph_z.float(), dim=-1),
                    3.0 * F.normalize(mean_node_z.float(), dim=-1),
                ),
                dim=-1,
            )
            chunks.append(embeddings.float().cpu())
            raw_graph_chunks.append(graph_z.float().cpu())
            raw_mean_node_chunks.append(mean_node_z.float().cpu())
            pending = []
            pending_nodes = 0

        for molecule in molecules:
            atoms = int(molecule.GetNumAtoms())
            if pending and (len(pending) >= 512 or pending_nodes + atoms > 16384):
                flush_representation()
            pending.append(molecule)
            pending_nodes += atoms
        flush_representation()
        result = torch.cat(chunks).numpy()
        raw_graph_z = torch.cat(raw_graph_chunks).numpy()
        raw_mean_node_z = torch.cat(raw_mean_node_chunks).numpy()
        if (
            result.shape[0] != len(molecules)
            or not np.isfinite(result).all()
            or not np.isfinite(raw_graph_z).all()
            or not np.isfinite(raw_mean_node_z).all()
        ):
            raise RuntimeError(
                "Optimized downstream embedding export lost rows or produced non-finite values"
            )
        return result, {
            "graph_dimensions": int(model.graph_latent_dim),
            "mean_node_dimensions": int(model.node_latent_dim),
            "raw_graph_z": raw_graph_z,
            "raw_mean_node_z": raw_mean_node_z,
        }

    chunks: list[torch.Tensor] = []
    graphs: list[Data] = []
    node_count = 0

    def flush() -> None:
        nonlocal graphs, node_count
        if not graphs:
            return
        batch = Batch.from_data_list(graphs).to(device)
        _, mu, _ = model.encode(
            batch.x, batch.edge_index, batch.edge_attr, sample=False
        )
        embeddings = global_mean_pool(mu, batch.batch)
        chunks.append(embeddings.float().cpu())
        graphs = []
        node_count = 0

    for molecule in molecules:
        x, edge_index, edge_attr = featurize_molecule(
            molecule,
            include_chirality=include_chirality,
            position_dim=position_dim,
        )
        if graphs and (len(graphs) >= 512 or node_count + len(x) > 16384):
            flush()
        graphs.append(
            Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr),
            )
        )
        node_count += len(x)
    flush()
    result = torch.cat(chunks).numpy()
    if result.shape[0] != len(molecules) or not np.isfinite(result).all():
        raise RuntimeError("Downstream embedding export lost rows or produced non-finite values")
    blocks = {"graph_dimensions": int(result.shape[1]), "mean_node_dimensions": 0}
    return result, blocks


def _morgan_features(molecules: list[Chem.Mol]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.stack(
        [generator.GetFingerprintAsNumPy(molecule) for molecule in molecules]
    ).astype(np.float32)


def _select_representation_embedding(
    selected_definition: str,
    unit_hybrid: np.ndarray,
    blocks: dict[str, Any],
    *,
    calibration_mean: np.ndarray | None = None,
    calibration_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Select the public vector without depending on diagnostic feature exports."""
    raw_graph_z = blocks["raw_graph_z"]
    raw_mean_node_z = blocks["raw_mean_node_z"]
    raw_hybrid = np.concatenate((raw_graph_z, raw_mean_node_z), axis=1)
    if selected_definition == "hybrid":
        return unit_hybrid
    if selected_definition == "raw_hybrid":
        return raw_hybrid
    if selected_definition == "standardized_raw_hybrid":
        if calibration_mean is None or calibration_scale is None:
            raise ValueError("standardized_raw_hybrid requires calibration statistics")
        selected = (raw_hybrid - calibration_mean) / calibration_scale
        selected[:, int(blocks["graph_dimensions"]) :] *= 3.0
        return selected
    if selected_definition == "graph_z":
        return raw_graph_z
    if selected_definition == "mean_node_z":
        return raw_mean_node_z
    raise ValueError("Downstream benchmarking does not support projector_z")


def _scaffold_splits(
    groups: np.ndarray,
    targets: np.ndarray,
    *,
    task: str,
    count: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    indices = np.arange(len(groups))
    result: list[tuple[np.ndarray, np.ndarray, int]] = []
    attempts = 0
    while len(result) < count and attempts < count * 50:
        split_seed = int(seed + 104729 * attempts)
        # In GroupShuffleSplit, test_size is a fraction of scaffold groups,
        # not molecules; realized molecule fractions vary with group sizes.
        outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=split_seed)
        train, test = next(outer.split(indices, targets, groups))
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Scaffold split leaked a group across partitions")
        if task == "classification" and any(
            len(np.unique(targets[subset])) != 2 for subset in (train, test)
        ):
            attempts += 1
            continue
        try:
            inner_folds = _inner_group_folds(
                train, groups, targets, task=task, seed=split_seed + 1
            )
        except ValueError:
            attempts += 1
            continue
        if task == "classification" and any(
            len(np.unique(targets[subset])) != 2
            for fold in inner_folds
            for subset in fold
        ):
            attempts += 1
            continue
        result.append((train, test, split_seed))
        attempts += 1
    if len(result) != count:
        raise RuntimeError(f"Could construct only {len(result)}/{count} valid scaffold splits")
    return result


def _inner_group_folds(
    train: np.ndarray,
    groups: np.ndarray,
    targets: np.ndarray,
    *,
    task: str,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if task == "classification":
        splitter = StratifiedGroupKFold(
            n_splits=3, shuffle=True, random_state=int(seed)
        )
    else:
        splitter = GroupKFold(n_splits=3)
    return [
        (train[fit], train[validation])
        for fit, validation in splitter.split(
            train, targets[train], groups=groups[train]
        )
    ]


def _regression_probe(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray, int]],
) -> list[dict[str, Any]]:
    results = []
    for train, test, split_seed in splits:
        inner_folds = _inner_group_folds(
            train, groups, targets, task="regression", seed=split_seed + 1
        )
        best_alpha, best_score = None, float("inf")
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            fold_scores = []
            for fit, validation in inner_folds:
                x_scaler = StandardScaler().fit(features[fit])
                y_mean = float(targets[fit].mean())
                y_std = max(1.0e-12, float(targets[fit].std()))
                model = Ridge(alpha=alpha, solver="lsqr")
                model.fit(
                    x_scaler.transform(features[fit]),
                    (targets[fit] - y_mean) / y_std,
                )
                prediction = (
                    model.predict(x_scaler.transform(features[validation]))
                    * y_std
                    + y_mean
                )
                fold_scores.append(
                    float(mean_squared_error(targets[validation], prediction) ** 0.5)
                )
            score = float(np.mean(fold_scores))
            if score < best_score:
                best_alpha, best_score = alpha, score

        x_scaler = StandardScaler().fit(features[train])
        y_mean = float(targets[train].mean())
        y_std = max(1.0e-12, float(targets[train].std()))
        model = Ridge(alpha=float(best_alpha), solver="lsqr")
        model.fit(
            x_scaler.transform(features[train]), (targets[train] - y_mean) / y_std
        )
        prediction = model.predict(x_scaler.transform(features[test])) * y_std + y_mean
        correlation = spearmanr(targets[test], prediction).statistic
        results.append(
            {
                "train": int(len(train)),
                "test": int(len(test)),
                "inner_scaffold_folds": 3,
                "outer_seed": int(split_seed),
                "ridge_alpha": float(best_alpha),
                "rmse": float(mean_squared_error(targets[test], prediction) ** 0.5),
                "normalized_rmse": float(
                    mean_squared_error(targets[test], prediction) ** 0.5 / y_std
                ),
                "mae": float(mean_absolute_error(targets[test], prediction)),
                "r2": float(r2_score(targets[test], prediction)),
                "spearman": float(correlation) if np.isfinite(correlation) else None,
            }
        )
    return results


def _classification_probe(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray, int]],
) -> list[dict[str, Any]]:
    labels = targets.astype(np.int64)
    results = []
    for train, test, split_seed in splits:
        inner_folds = _inner_group_folds(
            train, groups, targets, task="classification", seed=split_seed + 1
        )
        best_c, best_score = None, -float("inf")
        for c_value in (0.01, 0.1, 1.0, 10.0):
            fold_scores = []
            for fit, validation in inner_folds:
                x_scaler = StandardScaler().fit(features[fit])
                model = LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=0,
                )
                model.fit(x_scaler.transform(features[fit]), labels[fit])
                probability = model.predict_proba(
                    x_scaler.transform(features[validation])
                )[:, 1]
                fold_scores.append(
                    float(roc_auc_score(labels[validation], probability))
                )
            score = float(np.mean(fold_scores))
            if score > best_score:
                best_c, best_score = c_value, score

        x_scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(
            C=float(best_c),
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=0,
        )
        model.fit(x_scaler.transform(features[train]), labels[train])
        probability = model.predict_proba(x_scaler.transform(features[test]))[:, 1]
        prediction = (probability >= 0.5).astype(np.int64)
        results.append(
            {
                "train": int(len(train)),
                "test": int(len(test)),
                "inner_scaffold_folds": 3,
                "outer_seed": int(split_seed),
                "logistic_c": float(best_c),
                "test_positive_fraction": float(labels[test].mean()),
                "roc_auc": float(roc_auc_score(labels[test], probability)),
                "average_precision": float(
                    average_precision_score(labels[test], probability)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[test], prediction)
                ),
            }
        )
    return results


def _summarize(per_split: list[dict[str, Any]], task: str) -> dict[str, Any]:
    metric_names = (
        ("rmse", "normalized_rmse", "mae", "r2", "spearman")
        if task == "regression"
        else ("roc_auc", "average_precision", "balanced_accuracy")
    )
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray(
            [item[name] for item in per_split if item.get(name) is not None],
            dtype=np.float64,
        )
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    return {"summary": summary, "per_split": per_split}


def benchmark_moleculenet(
    cfg: dict[str, Any],
    *,
    checkpoint_name: str,
    datasets_dir: str | Path,
    output: str | Path,
    scaffold_splits: int = 5,
    dataset_names: list[str] | tuple[str, ...] | None = None,
    embedding_definition: str = "auto",
    calibrator: str | Path | None = None,
    selected_only: bool = False,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    if scaffold_splits <= 0:
        raise ValueError("scaffold_splits must be positive")
    rank, world_size, _, device = _distributed_context(allow_cpu)
    if rank != 0 or world_size != 1:
        raise RuntimeError("Downstream benchmarking requires one process")
    cfg, manifest, standardizer, model, checkpoint = load_saved_model(
        cfg, checkpoint_name, device
    )
    is_representation_model = isinstance(model, MolecularRepresentationModel)
    selected_definition = _resolve_embedding_definition(
        embedding_definition, representation_model=is_representation_model
    )
    calibration_mean = None
    calibration_scale = None
    calibration_metadata: dict[str, Any] | None = None
    calibration_sha256 = None
    checkpoint_path = Path(cfg["paths"]["run_dir"]) / checkpoint_name
    if selected_definition == "standardized_raw_hybrid":
        if calibrator is None:
            raise ValueError("standardized_raw_hybrid requires a calibrator")
        dimensions = int(model.graph_latent_dim + model.node_latent_dim)
        calibration_mean, calibration_scale, calibration_metadata, calibration_sha256 = (
            _load_embedding_calibrator(
                calibrator,
                expected=_calibrator_expected_identity(
                    cfg, manifest, checkpoint_path, checkpoint
                ),
                dimensions=dimensions,
            )
        )
        calibration_mean = calibration_mean.numpy()
        calibration_scale = calibration_scale.numpy()
    source_dir = Path(datasets_dir)
    datasets: dict[str, Any] = {}
    embedding_dimensions: int | None = None
    selected_names = _resolve_dataset_names(dataset_names)
    for name in selected_names:
        spec = MOLECULENET_DATASETS[name]
        path = source_dir / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        molecules, targets, groups, preparation = _prepare_dataset(path, spec, cfg)
        embeddings, blocks = _encode_molecules(model, molecules, cfg, device)
        features: dict[str, np.ndarray] = {}
        if not selected_only:
            features["morgan_radius2_2048"] = _morgan_features(molecules)
        if is_representation_model:
            graph_dimensions = blocks["graph_dimensions"]
            raw_hybrid = np.concatenate(
                (blocks["raw_graph_z"], blocks["raw_mean_node_z"]), axis=1
            )
            if not selected_only:
                features["unit_graph_z"] = embeddings[:, :graph_dimensions]
                features["unit_mean_node_z"] = embeddings[:, graph_dimensions:] / 3.0
                features["graph_z"] = blocks["raw_graph_z"]
                features["mean_node_z"] = blocks["raw_mean_node_z"]
                features["raw_graph_z_plus_mean_node_z"] = raw_hybrid
            selected_embeddings = _select_representation_embedding(
                selected_definition,
                embeddings,
                blocks,
                calibration_mean=calibration_mean,
                calibration_scale=calibration_scale,
            )
        else:
            selected_embeddings = embeddings
        features = {"molecule_embedding": selected_embeddings, **features}
        if embedding_dimensions is None:
            embedding_dimensions = int(selected_embeddings.shape[1])
        elif embedding_dimensions != int(selected_embeddings.shape[1]):
            raise RuntimeError("Model emitted inconsistent embedding dimensions")
        splits = _scaffold_splits(
            groups,
            targets,
            task=spec["task"],
            count=scaffold_splits,
            seed=int(cfg["seed"]),
        )
        feature_results = {}
        for feature_name, values in features.items():
            per_split = (
                _regression_probe(values, targets, groups, splits)
                if spec["task"] == "regression"
                else _classification_probe(values, targets, groups, splits)
            )
            feature_results[feature_name] = _summarize(per_split, spec["task"])
        datasets[name] = {
            "source": {
                "filename": spec["filename"],
                "sha256": spec["sha256"],
                "url": spec["url"],
                "smiles_column": spec["smiles_column"],
                "target_column": spec["target_column"],
            },
            "task": spec["task"],
            "preparation": preparation,
            "scaffold_splits": int(scaffold_splits),
            "feature_results": feature_results,
        }

    result = {
        "schema_version": 1,
        "benchmark": "MoleculeNet frozen scaffold-group linear probes",
        "datasets_requested": selected_names,
        "selected_only": bool(selected_only),
        "checkpoint": {
            "name": checkpoint_name,
            "checkpoint_sha256": _file_sha256(
                Path(cfg["paths"]["run_dir"]) / checkpoint_name
            ),
            "global_step": int(checkpoint["global_step"]),
            "architecture": cfg["model"].get("architecture", "legacy_vgae"),
            "inference_backend": (
                "optimized_gine_v1"
                if is_representation_model
                else "reference_legacy_vgae"
            ),
            "fast_inference_version": (
                FAST_INFERENCE_VERSION if is_representation_model else None
            ),
            "config_hash": cfg["_config_hash"],
            "training_plan_hash": _training_plan_hash(cfg),
            "graph_manifest_hash": manifest["graph_manifest_hash"],
            "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
            "feature_schema_hash": manifest["feature_schema"]["hash"],
            "scaler_hash": standardizer.scaler_hash,
            "embedding_definition": (
                _STANDARDIZED_RAW_HYBRID_DEFINITION
                if selected_definition == "standardized_raw_hybrid"
                else (
                    _RAW_HYBRID_DEFINITION
                    if selected_definition == "raw_hybrid"
                    else (
                        "clean_graph_z_plus_mean_node_z_unit_blocks"
                        if selected_definition == "hybrid"
                        else (
                            f"clean_{selected_definition}"
                            if is_representation_model
                            else "legacy_mean_node_posterior_mu"
                        )
                    )
                )
            ),
            "embedding_dimensions": int(embedding_dimensions or 0),
            "embedding_parameters": (
                (
                    {
                        "graph_dimensions": int(model.graph_latent_dim),
                        "mean_node_dimensions": int(model.node_latent_dim),
                        "mean_node_weight": 3.0,
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
                    else (
                        {
                            "graph_dimensions": int(model.graph_latent_dim),
                            "mean_node_dimensions": int(model.node_latent_dim),
                            "block_normalization": "none",
                        }
                        if selected_definition == "raw_hybrid"
                        else (
                            {
                                "mean_node_weight": 3.0,
                                "graph_dimensions": int(model.graph_latent_dim),
                                "mean_node_dimensions": int(model.node_latent_dim),
                            }
                            if selected_definition == "hybrid"
                            else {}
                        )
                    )
                )
                if is_representation_model
                else {}
            ),
        },
        "datasets": datasets,
    }
    atomic_write_json(output, result)
    return result

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import GraphDescriptors, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from .util import atomic_write_json


def _load_embedding_payload(path: str | Path) -> dict[str, Any]:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"metadata", "embeddings", "molecule_hashes", "source_buckets"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Embedding file lacks fields: {', '.join(missing)}")
    rows = int(value["embeddings"].shape[0])
    if rows != len(value["molecule_hashes"]) or rows != int(value["source_buckets"].numel()):
        raise ValueError("Embedding rows, molecule hashes, and source buckets are misaligned")
    if len(set(value["molecule_hashes"])) != rows:
        raise ValueError("Embedding probe input contains duplicate molecule hashes")
    return value


def _chemical_records(
    payload: dict[str, Any], work_dir: Path
) -> list[tuple[str, str]]:
    dataset_manifest_path = work_dir / "dataset_manifest.json"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(dataset_manifest_path)
    dataset_manifest_hash = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )["manifest_hash"]
    cache_path = work_dir / "probe_chemical_records_cache_v1.json"
    cached_records: dict[str, list[str]] = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("dataset_manifest_hash") == dataset_manifest_hash:
            cached_records = dict(cached.get("records", {}))

    by_bucket: dict[int, list[str]] = {}
    for molecule_hash, bucket in zip(
        payload["molecule_hashes"], payload["source_buckets"].view(-1).tolist()
    ):
        value = str(molecule_hash)
        if value not in cached_records:
            by_bucket.setdefault(int(bucket), []).append(value)
    records: dict[str, tuple[str, str]] = {
        key: (str(value[0]), str(value[1]))
        for key, value in cached_records.items()
    }
    connection = duckdb.connect(":memory:")
    added = 0
    try:
        for bucket, hashes in sorted(by_bucket.items()):
            parquet_path = work_dir / "deduplicated" / f"bucket-{bucket:04d}.parquet"
            if not parquet_path.is_file():
                raise FileNotFoundError(parquet_path)
            connection.register("wanted", pa.table({"molecule_hash": hashes}))
            rows = connection.execute(
                """
                SELECT d.molecule_hash, d.canonical_smiles, d.scaffold
                FROM read_parquet(?) AS d
                INNER JOIN wanted AS w USING (molecule_hash)
                """,
                [str(parquet_path)],
            ).fetchall()
            connection.unregister("wanted")
            for molecule_hash, smiles, scaffold in rows:
                records[str(molecule_hash)] = (str(smiles), str(scaffold or ""))
                added += 1
    finally:
        connection.close()
    missing = [value for value in payload["molecule_hashes"] if value not in records]
    if missing:
        raise RuntimeError(f"Failed to join {len(missing)} embedding rows to canonical molecules")
    if added:
        atomic_write_json(
            cache_path,
            {
                "schema_version": 1,
                "dataset_manifest_hash": dataset_manifest_hash,
                "records": {
                    key: [value[0], value[1]] for key, value in sorted(records.items())
                },
            },
        )
    return [records[value] for value in payload["molecule_hashes"]]


HELD_OUT_LABELS = (
    "bertz_complexity",
    "fraction_csp3",
    "ring_count",
    "aromatic_ring_count",
    "aliphatic_ring_count",
    "saturated_ring_count",
    "rotatable_bond_count",
    "hall_kier_alpha",
    "kappa1",
    "kappa2",
    "kappa3",
    "chi0v",
    "chi1v",
)


def _held_out_values(molecule: Chem.Mol) -> list[float]:
    return [
        float(GraphDescriptors.BertzCT(molecule)),
        float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        float(rdMolDescriptors.CalcNumRings(molecule)),
        float(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        float(rdMolDescriptors.CalcNumAliphaticRings(molecule)),
        float(rdMolDescriptors.CalcNumSaturatedRings(molecule)),
        float(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        float(GraphDescriptors.HallKierAlpha(molecule)),
        float(GraphDescriptors.Kappa1(molecule)),
        float(GraphDescriptors.Kappa2(molecule)),
        float(GraphDescriptors.Kappa3(molecule)),
        float(GraphDescriptors.Chi0v(molecule)),
        float(GraphDescriptors.Chi1v(molecule)),
    ]


def _molecules_and_labels(records: list[tuple[str, str]]) -> tuple[list[Chem.Mol], np.ndarray]:
    molecules = []
    labels = []
    for smiles, _ in records:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Canonical SMILES failed to parse during probe: {smiles}")
        values = _held_out_values(molecule)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite held-out descriptor for {smiles}")
        molecules.append(molecule)
        labels.append(values)
    return molecules, np.asarray(labels, dtype=np.float64)


def _ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> dict[str, Any]:
    x_scaler = StandardScaler().fit(train_x)
    y_scaler = StandardScaler().fit(train_y)
    standardized_train_y = y_scaler.transform(train_y)
    standardized_validation_y = y_scaler.transform(validation_y)
    predictor = Ridge(alpha=10.0, solver="lsqr")
    predictor.fit(x_scaler.transform(train_x), standardized_train_y)
    prediction = predictor.predict(x_scaler.transform(validation_x))
    per_label_r2 = r2_score(
        standardized_validation_y, prediction, multioutput="raw_values"
    )
    per_label_mae = np.abs(prediction - standardized_validation_y).mean(axis=0)
    return {
        "train_graphs": int(train_x.shape[0]),
        "validation_graphs": int(validation_x.shape[0]),
        "ridge_alpha": 10.0,
        "mean_r2": float(np.mean(per_label_r2)),
        "median_r2": float(np.median(per_label_r2)),
        "mean_standardized_mae": float(np.mean(per_label_mae)),
        "r2": dict(zip(HELD_OUT_LABELS, per_label_r2.tolist())),
        "standardized_mae": dict(zip(HELD_OUT_LABELS, per_label_mae.tolist())),
    }


def _embedding_diagnostics(embeddings: np.ndarray) -> dict[str, Any]:
    centered = embeddings.astype(np.float64) - embeddings.mean(axis=0, dtype=np.float64)
    covariance = centered.T @ centered / max(1, embeddings.shape[0] - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0, None)
    probability = eigenvalues / max(1.0e-30, eigenvalues.sum())
    positive = probability[probability > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum()))
    participation = float(eigenvalues.sum() ** 2 / max(1.0e-30, np.square(eigenvalues).sum()))
    std = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return {
        "graphs": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / embeddings.shape[1],
        "participation_ratio": participation,
        "participation_ratio_fraction": participation / embeddings.shape[1],
        "top_eigenvalue_fraction": float(eigenvalues[-1] / max(1.0e-30, eigenvalues.sum())),
        "median_coordinate_std": float(np.median(std)),
        "minimum_coordinate_std": float(np.min(std)),
    }


def _sample_indices(size: int, maximum: int, seed: int) -> np.ndarray:
    count = min(int(maximum), int(size))
    if count <= 0:
        raise ValueError("Probe sampling requires at least one graph")
    if count == size:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(size, size=count, replace=False))


def _similarity_probe(
    embeddings: np.ndarray,
    molecules: list[Chem.Mol],
    scaffolds: list[str],
    *,
    maximum_graphs: int,
    seed: int,
) -> dict[str, Any]:
    indices = _sample_indices(embeddings.shape[0], maximum_graphs, seed)
    count = len(indices)
    embeddings = embeddings[indices].astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1.0e-12, None)
    latent_similarity = normalized @ normalized.T
    np.fill_diagonal(latent_similarity, -np.inf)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(molecules[index]) for index in indices]
    fingerprint_similarity = np.empty((count, count), dtype=np.float32)
    for index, fingerprint in enumerate(fingerprints):
        fingerprint_similarity[index] = DataStructs.BulkTanimotoSimilarity(
            fingerprint, fingerprints
        )
    np.fill_diagonal(fingerprint_similarity, -np.inf)
    neighbors = min(10, count - 1)
    if neighbors <= 0:
        raise ValueError("Similarity probe requires at least two molecules")
    latent_neighbors = np.argpartition(latent_similarity, -neighbors, axis=1)[:, -neighbors:]
    fingerprint_neighbors = np.argpartition(
        fingerprint_similarity, -neighbors, axis=1
    )[:, -neighbors:]
    overlap = [
        len(set(latent_neighbors[index]).intersection(fingerprint_neighbors[index])) / neighbors
        for index in range(count)
    ]
    latent_neighbor_tanimoto = fingerprint_similarity[
        np.arange(count)[:, None], latent_neighbors
    ]
    rng = np.random.default_rng(seed)
    pair_count = min(200_000, count * (count - 1) // 2)
    first = rng.integers(0, count, size=pair_count)
    second = rng.integers(0, count - 1, size=pair_count)
    second += second >= first
    latent_pairs = latent_similarity[first, second]
    fingerprint_pairs = fingerprint_similarity[first, second]
    correlation = spearmanr(latent_pairs, fingerprint_pairs).statistic
    scaffold_values = np.asarray([scaffolds[index] for index in indices], dtype=object)
    valid_scaffold = scaffold_values != ""
    scaffold_purity_values = []
    scaffold_random_baselines = []
    for index in np.flatnonzero(valid_scaffold):
        scaffold_purity_values.append(
            np.mean(scaffold_values[latent_neighbors[index]] == scaffold_values[index])
        )
        scaffold_random_baselines.append(
            (np.count_nonzero(scaffold_values == scaffold_values[index]) - 1)
            / max(1, count - 1)
        )
    scaffold_purity = (
        float(np.mean(scaffold_purity_values)) if scaffold_purity_values else None
    )
    scaffold_random = (
        float(np.mean(scaffold_random_baselines)) if scaffold_random_baselines else None
    )
    return {
        "graphs": int(count),
        "available_graphs": int(len(molecules)),
        "sampling": "seeded_without_replacement_across_export",
        "sampling_seed": int(seed),
        "neighbors": int(neighbors),
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "latent_to_morgan_recall_at_10": float(np.mean(overlap)),
        "latent_neighbor_mean_tanimoto": float(latent_neighbor_tanimoto.mean()),
        "random_pair_mean_tanimoto": float(fingerprint_pairs.mean()),
        "neighbor_tanimoto_enrichment": float(
            latent_neighbor_tanimoto.mean() / max(1.0e-12, fingerprint_pairs.mean())
        ),
        "latent_cosine_vs_morgan_spearman": (
            float(correlation) if np.isfinite(correlation) else None
        ),
        "nonempty_scaffold_queries": int(len(scaffold_purity_values)),
        "scaffold_neighbor_purity_at_10": scaffold_purity,
        "random_scaffold_neighbor_purity": scaffold_random,
        "scaffold_purity_enrichment": (
            scaffold_purity / max(1.0e-12, scaffold_random)
            if scaffold_purity is not None and scaffold_random is not None
            else None
        ),
    }


def _scaffold_clustering_probe(
    embeddings: np.ndarray,
    molecules: list[Chem.Mol],
    scaffolds: list[str],
    *,
    maximum_graphs: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate L2-normalized inputs with standard Euclidean K-means.

    Centroids are not constrained or renormalized, so this is not spherical
    K-means. Historical artifact keys are retained for schema compatibility.
    """
    indices = _sample_indices(embeddings.shape[0], maximum_graphs, seed + 17)
    count = len(indices)
    sampled_embeddings = embeddings[indices]
    sampled_molecules = [molecules[index] for index in indices]
    scaffold_values = np.asarray([scaffolds[index] for index in indices], dtype=object)
    frequencies = Counter(value for value in scaffold_values if value)
    recurring = sorted(
        (value for value, frequency in frequencies.items() if frequency >= 5),
        key=lambda value: (-frequencies[value], value),
    )[:32]
    if len(recurring) < 2:
        return {
            "available": False,
            "reason": "fewer than two scaffolds occur at least five times",
        }
    selected = np.flatnonzero(np.isin(scaffold_values, recurring))
    labels_by_scaffold = {value: index for index, value in enumerate(recurring)}
    labels = np.asarray(
        [labels_by_scaffold[str(scaffold_values[index])] for index in selected],
        dtype=np.int64,
    )
    cluster_count = len(recurring)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan = np.stack(
        [generator.GetFingerprintAsNumPy(sampled_molecules[index]) for index in selected]
    ).astype(np.float64)

    kmeans_seeds = [int(seed + offset) for offset in range(5)]

    def cluster_once(features: np.ndarray, random_state: int) -> dict[str, float]:
        features = features.astype(np.float64)
        features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1.0e-12, None)
        assignments = KMeans(
            n_clusters=cluster_count,
            n_init=20,
            max_iter=500,
            random_state=int(random_state),
        ).fit_predict(features)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            labels, assignments
        )
        return {
            "adjusted_rand_index": float(adjusted_rand_score(labels, assignments)),
            "normalized_mutual_information": float(
                normalized_mutual_info_score(labels, assignments)
            ),
            "homogeneity": float(homogeneity),
            "completeness": float(completeness),
            "v_measure": float(v_measure),
        }

    def cluster(features: np.ndarray) -> dict[str, Any]:
        repetitions = [cluster_once(features, value) for value in kmeans_seeds]
        metric_names = tuple(repetitions[0])
        result: dict[str, Any] = {
            name: float(np.mean([item[name] for item in repetitions]))
            for name in metric_names
        }
        result.update(
            {
                f"{name}_std": float(
                    np.std([item[name] for item in repetitions])
                )
                for name in metric_names
            }
        )
        result["per_seed"] = repetitions
        return result

    return {
        "available": True,
        "available_graphs": int(embeddings.shape[0]),
        "sampled_graphs": int(count),
        "sampling": "seeded_without_replacement_across_export",
        "sampling_seed": int(seed + 17),
        "graphs": int(len(selected)),
        "scaffold_clusters": int(cluster_count),
        "minimum_scaffold_frequency": 5,
        "maximum_scaffold_clusters": 32,
        "kmeans_repetitions": len(kmeans_seeds),
        "kmeans_seeds": kmeans_seeds,
        "kmeans_n_init_per_repetition": 20,
        # Legacy field names are immutable artifact-schema labels. Both values
        # come from standard KMeans on row-L2-normalized inputs.
        "latent_spherical_kmeans": cluster(sampled_embeddings[selected]),
        "morgan_spherical_kmeans": cluster(morgan),
    }


def run_representation_probes(
    *,
    train_embeddings: str | Path,
    validation_embeddings: str | Path,
    work_dir: str | Path,
    output: str | Path,
    similarity_graphs: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    train = _load_embedding_payload(train_embeddings)
    validation = _load_embedding_payload(validation_embeddings)
    if train["embeddings"].shape[1] != validation["embeddings"].shape[1]:
        raise ValueError("Train and validation embedding dimensions differ")
    for key in (
        "config_hash",
        "training_plan_hash",
        "checkpoint",
        "global_step",
        "embedding_definition",
        "embedding_parameters",
        "checkpoint_sha256",
        "graph_manifest_hash",
        "descriptor_schema_hash",
        "sampling",
        "sampling_seed",
    ):
        if train["metadata"].get(key) != validation["metadata"].get(key):
            raise ValueError(f"Train and validation embedding metadata differ at {key}")
    work_dir = Path(work_dir)
    train_records = _chemical_records(train, work_dir)
    validation_records = _chemical_records(validation, work_dir)
    _, train_labels = _molecules_and_labels(train_records)
    validation_molecules, validation_labels = _molecules_and_labels(validation_records)
    train_x = train["embeddings"].numpy()
    validation_x = validation["embeddings"].numpy()
    latent_probe = _ridge_probe(train_x, train_labels, validation_x, validation_labels)
    training_scaffolds = {scaffold for _, scaffold in train_records if scaffold}
    novel_scaffold_mask = np.asarray(
        [bool(scaffold) and scaffold not in training_scaffolds for _, scaffold in validation_records]
    )
    if int(novel_scaffold_mask.sum()) < 2:
        raise RuntimeError("Too few validation molecules have scaffolds absent from the probe train set")
    scaffold_disjoint_probe = _ridge_probe(
        train_x,
        train_labels,
        validation_x[novel_scaffold_mask],
        validation_labels[novel_scaffold_mask],
    )
    descriptor_baseline = _ridge_probe(
        train["standardized_descriptor_targets"].numpy(),
        train_labels,
        validation["standardized_descriptor_targets"].numpy(),
        validation_labels,
    )
    embedding_parameters = validation["metadata"].get("embedding_parameters", {})
    block_diagnostics = None
    if validation["metadata"].get("embedding_definition") == (
        "clean_graph_z_plus_mean_node_z_unit_blocks"
    ):
        graph_dimensions = int(embedding_parameters["graph_dimensions"])
        mean_node_weight = float(embedding_parameters["mean_node_weight"])
        if graph_dimensions <= 0 or graph_dimensions >= validation_x.shape[1]:
            raise ValueError("Hybrid embedding metadata has an invalid graph block width")
        block_diagnostics = {
            "graph_z": _embedding_diagnostics(validation_x[:, :graph_dimensions]),
            "mean_node_z_unweighted": _embedding_diagnostics(
                validation_x[:, graph_dimensions:] / mean_node_weight
            ),
        }
    elif validation["metadata"].get("embedding_definition") in {
        "clean_graph_z_plus_mean_node_z_raw_blocks",
        "clean_graph_z_plus_mean_node_z_train_standardized_raw_blocks",
    }:
        graph_dimensions = int(embedding_parameters["graph_dimensions"])
        if graph_dimensions <= 0 or graph_dimensions >= validation_x.shape[1]:
            raise ValueError("Raw hybrid metadata has an invalid graph block width")
        block_diagnostics = {
            "graph_z": _embedding_diagnostics(validation_x[:, :graph_dimensions]),
            "mean_node_z": _embedding_diagnostics(validation_x[:, graph_dimensions:]),
        }
    result = {
        "schema_version": 1,
        "train_embedding_metadata": train["metadata"],
        "checkpoint_metadata": validation["metadata"],
        "embedding_diagnostics": _embedding_diagnostics(validation_x),
        "embedding_block_diagnostics": block_diagnostics,
        "held_out_linear_probe": latent_probe,
        "scaffold_disjoint_linear_probe": scaffold_disjoint_probe,
        "scaffold_disjoint_validation_fraction": float(novel_scaffold_mask.mean()),
        "training_descriptor_baseline": descriptor_baseline,
        "latent_minus_descriptor_baseline_mean_r2": (
            latent_probe["mean_r2"] - descriptor_baseline["mean_r2"]
        ),
        "similarity": _similarity_probe(
            validation_x,
            validation_molecules,
            [scaffold for _, scaffold in validation_records],
            maximum_graphs=similarity_graphs,
            seed=seed,
        ),
        "clustering": _scaffold_clustering_probe(
            validation_x,
            validation_molecules,
            [scaffold for _, scaffold in validation_records],
            maximum_graphs=validation_x.shape[0],
            seed=seed,
        ),
    }
    atomic_write_json(output, result)
    return result

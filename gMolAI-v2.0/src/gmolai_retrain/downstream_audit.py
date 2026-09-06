from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import sklearn
from rdkit import __version__ as rdkit_version
from rdkit.Chem import Descriptors

from .config import descriptor_names
from .downstream import (
    DEFAULT_MOLECULENET_DATASETS,
    MOLECULENET_DATASETS,
    PreparedMoleculeNetDataset,
    _classification_probe,
    _inner_group_folds,
    _prepare_dataset_records,
    _regression_probe,
    _resolve_dataset_names,
    _scaffold_splits,
    _sha256,
    _summarize,
)
from .util import atomic_write_csv, atomic_write_json, runtime_versions


OVERLAP_AUDIT_DATASETS = (
    "bace",
    "bbbp",
    "esol",
    "freesolv",
    "lipophilicity",
    "hiv",
)

_DESCRIPTOR_GENERATORS = {
    "qed": Descriptors.qed,
    "MolWt": Descriptors.MolWt,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "BalabanJ": Descriptors.BalabanJ,
    "LabuteASA": Descriptors.LabuteASA,
    "TPSA": Descriptors.TPSA,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
}


def _identity_digest(values: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dataset_source(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": spec["filename"],
        "sha256": spec["sha256"],
        "url": spec["url"],
        "smiles_column": spec["smiles_column"],
        "target_column": spec["target_column"],
    }


def _load_dataset_manifest(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(cfg["paths"]["work_dir"]) / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != cfg["_config_hash"]:
        raise RuntimeError("Dataset manifest belongs to a different configuration")
    if manifest.get("descriptor_schema_hash") != cfg["_descriptor_schema_hash"]:
        raise RuntimeError("Dataset manifest descriptor schema does not match configuration")
    expected = int(manifest["deduplication"]["rows_after_deduplication"])
    observed = sum(int(value) for value in manifest["split_counts"].values())
    if expected != observed:
        raise RuntimeError(
            f"Dataset manifest split counts sum to {observed}, expected {expected}"
        )
    return path, manifest


def _prepare_selected(
    cfg: dict[str, Any], datasets_dir: str | Path, names: list[str]
) -> dict[str, PreparedMoleculeNetDataset]:
    source_dir = Path(datasets_dir)
    prepared: dict[str, PreparedMoleculeNetDataset] = {}
    for name in names:
        spec = MOLECULENET_DATASETS[name]
        path = source_dir / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        prepared[name] = _prepare_dataset_records(path, spec, cfg)
    return prepared


def _parquet_files_by_bucket(
    manifest: dict[str, Any], bucket_count: int
) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw_path in manifest["parquet_files"]:
        path = Path(raw_path)
        try:
            bucket = int(path.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Cannot infer bucket from {path}") from error
        if bucket in result:
            raise RuntimeError(f"Duplicate Parquet file for bucket {bucket}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[bucket] = path
    expected = set(range(bucket_count))
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise RuntimeError(f"Parquet bucket mismatch: missing={missing}, extra={extra}")
    return result


def _join_pretraining_rows(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    prepared: dict[str, PreparedMoleculeNetDataset],
    *,
    include_descriptors: bool,
) -> dict[str, list[dict[str, Any] | None]]:
    """Bucket-pruned exact SHA-256 join against the immutable deduplicated corpus."""

    bucket_count = int(cfg["data"]["hash_buckets"])
    parquet_by_bucket = _parquet_files_by_bucket(manifest, bucket_count)
    wanted: dict[int, dict[str, list[Any]]] = {}
    matches: dict[str, list[dict[str, Any] | None]] = {
        name: [None] * len(dataset.molecule_hashes)
        for name, dataset in prepared.items()
    }
    for name, dataset in prepared.items():
        for index, (molecule_hash, smiles, bucket) in enumerate(
            zip(
                dataset.molecule_hashes,
                dataset.canonical_smiles,
                dataset.source_buckets.tolist(),
            )
        ):
            columns = wanted.setdefault(
                int(bucket),
                {
                    "dataset": [],
                    "dataset_index": [],
                    "molecule_hash": [],
                    "canonical_smiles": [],
                },
            )
            columns["dataset"].append(name)
            columns["dataset_index"].append(index)
            columns["molecule_hash"].append(molecule_hash)
            columns["canonical_smiles"].append(smiles)

    descriptor_columns = [f"d{index:02d}" for index in range(len(descriptor_names(cfg)))]
    descriptor_select = (
        ", " + ", ".join(f"d.{column}" for column in descriptor_columns)
        if include_descriptors
        else ""
    )
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("PRAGMA threads=8")
        for bucket, columns in sorted(wanted.items()):
            connection.register("wanted", pa.table(columns))
            rows = connection.execute(
                f"""
                SELECT w.dataset, w.dataset_index, w.molecule_hash,
                       w.canonical_smiles, d.canonical_smiles, d.split
                       {descriptor_select}
                FROM read_parquet(?) AS d
                INNER JOIN wanted AS w USING (molecule_hash)
                """,
                [str(parquet_by_bucket[bucket])],
            ).fetchall()
            connection.unregister("wanted")
            for row in rows:
                name = str(row[0])
                index = int(row[1])
                if matches[name][index] is not None:
                    raise RuntimeError(
                        f"Pretraining corpus returned duplicate identity for {name}[{index}]"
                    )
                if str(row[3]) != str(row[4]):
                    raise RuntimeError(
                        f"SHA-256 identity collision or canonicalization mismatch for {row[2]}"
                    )
                match: dict[str, Any] = {
                    "molecule_hash": str(row[2]),
                    "canonical_smiles": str(row[4]),
                    "split": str(row[5]),
                }
                if include_descriptors:
                    match["descriptors"] = [float(value) for value in row[6:]]
                matches[name][index] = match
    finally:
        connection.close()
    return matches


def audit_pretraining_overlap(
    cfg: dict[str, Any],
    *,
    datasets_dir: str | Path,
    output: str | Path,
    summary_csv: str | Path,
    dataset_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    names = (
        list(OVERLAP_AUDIT_DATASETS)
        if dataset_names is None
        else _resolve_dataset_names(dataset_names)
    )
    manifest_path, manifest = _load_dataset_manifest(cfg)
    prepared = _prepare_selected(cfg, datasets_dir, names)
    matches = _join_pretraining_rows(
        cfg, manifest, prepared, include_descriptors=False
    )

    datasets: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for name in names:
        dataset = prepared[name]
        split_hashes: dict[str, list[str]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        overlap_hashes: list[str] = []
        for molecule_hash, match in zip(dataset.molecule_hashes, matches[name]):
            if match is None:
                continue
            split = str(match["split"])
            if split not in split_hashes:
                raise RuntimeError(f"Unexpected pretraining split {split!r}")
            split_hashes[split].append(molecule_hash)
            overlap_hashes.append(molecule_hash)
        accepted = len(dataset.molecule_hashes)
        overlap = len(overlap_hashes)
        absent = accepted - overlap
        split_counts = {key: len(value) for key, value in split_hashes.items()}
        datasets[name] = {
            "source": _dataset_source(MOLECULENET_DATASETS[name]),
            "task": MOLECULENET_DATASETS[name]["task"],
            "preparation": dataset.preparation,
            "accepted_identity_set_sha256": _identity_digest(dataset.molecule_hashes),
            "overlap_identity_set_sha256": _identity_digest(overlap_hashes),
            "overlap": {
                "pretraining_corpus": overlap,
                "pretraining_corpus_fraction": overlap / accepted,
                "not_in_pretraining_corpus": absent,
                "not_in_pretraining_corpus_fraction": absent / accepted,
                "by_pretraining_split": split_counts,
                "by_pretraining_split_fraction": {
                    key: value / accepted for key, value in split_counts.items()
                },
                "identity_rule": "SHA-256 of canonical isomeric SMILES",
            },
        }
        csv_rows.append(
            {
                "dataset": name,
                "raw_rows": dataset.preparation["raw_rows"],
                "accepted_unique_molecules": accepted,
                "pretraining_corpus_overlap": overlap,
                "pretraining_corpus_overlap_percent": 100.0 * overlap / accepted,
                "pretraining_train_overlap": split_counts["train"],
                "pretraining_validation_overlap": split_counts["validation"],
                "pretraining_test_overlap": split_counts["test"],
                "not_in_pretraining_corpus": absent,
            }
        )

    result = {
        "schema_version": 1,
        "audit": "Exact canonical-molecule overlap with immutable pretraining corpus",
        "pretrained_model_executed": False,
        "datasets_requested": names,
        "identity_rule": "SHA-256 of canonical isomeric SMILES after the pinned downstream policy",
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
            "graphs": int(manifest["deduplication"]["rows_after_deduplication"]),
            "split_counts": manifest["split_counts"],
            "config_hash": manifest["config_hash"],
            "descriptor_schema_hash": manifest["descriptor_schema_hash"],
        },
        "runtime": {
            **runtime_versions(),
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit_version,
        },
        "datasets": datasets,
    }
    atomic_write_json(output, result)
    atomic_write_csv(summary_csv, csv_rows)
    return result


def _descriptor_matrix(
    cfg: dict[str, Any], molecules: list[Any]
) -> tuple[list[str], np.ndarray]:
    names = descriptor_names(cfg)
    missing = [name for name in names if name not in _DESCRIPTOR_GENERATORS]
    if missing:
        raise RuntimeError(f"No frozen RDKit generator mapping for: {', '.join(missing)}")
    values = np.asarray(
        [
            [float(_DESCRIPTOR_GENERATORS[name](molecule)) for name in names]
            for molecule in molecules
        ],
        dtype=np.float64,
    )
    if values.shape != (len(molecules), len(names)) or not np.isfinite(values).all():
        raise RuntimeError("Descriptor-only feature calculation produced invalid values")
    return names, values


def _indices_digest(dataset: PreparedMoleculeNetDataset, indices: np.ndarray) -> str:
    return _identity_digest([dataset.molecule_hashes[int(index)] for index in indices])


def _groups_digest(dataset: PreparedMoleculeNetDataset, indices: np.ndarray) -> str:
    return _identity_digest(
        list(set(str(value) for value in dataset.scaffold_groups[indices].tolist()))
    )


def _split_identity_manifest(
    dataset: PreparedMoleculeNetDataset,
    splits: list[tuple[np.ndarray, np.ndarray, int]],
    *,
    task: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    total_groups = len(set(str(value) for value in dataset.scaffold_groups.tolist()))
    for train, test, outer_seed in splits:
        train_groups = set(str(value) for value in dataset.scaffold_groups[train].tolist())
        test_groups = set(str(value) for value in dataset.scaffold_groups[test].tolist())
        if train_groups & test_groups:
            raise RuntimeError("Generated outer split leaks scaffold groups")
        inner = _inner_group_folds(
            train,
            dataset.scaffold_groups,
            dataset.targets,
            task=task,
            seed=int(outer_seed) + 1,
        )
        inner_manifest = []
        for fold, (fit, validation) in enumerate(inner):
            fit_groups = set(
                str(value) for value in dataset.scaffold_groups[fit].tolist()
            )
            validation_groups = set(
                str(value) for value in dataset.scaffold_groups[validation].tolist()
            )
            if fit_groups & validation_groups:
                raise RuntimeError("Generated inner split leaks scaffold groups")
            inner_manifest.append(
                {
                    "fold": fold,
                    "fit_molecules": int(len(fit)),
                    "validation_molecules": int(len(validation)),
                    "fit_scaffold_groups": int(len(fit_groups)),
                    "validation_scaffold_groups": int(len(validation_groups)),
                    "fit_identity_set_sha256": _indices_digest(dataset, fit),
                    "validation_identity_set_sha256": _indices_digest(
                        dataset, validation
                    ),
                }
            )
        result.append(
            {
                "outer_seed": int(outer_seed),
                "train_molecules": int(len(train)),
                "test_molecules": int(len(test)),
                "train_scaffold_groups": int(len(train_groups)),
                "test_scaffold_groups": int(len(test_groups)),
                "test_scaffold_group_fraction": len(test_groups) / total_groups,
                "test_molecule_fraction": len(test) / len(dataset.molecules),
                "train_identity_set_sha256": _indices_digest(dataset, train),
                "test_identity_set_sha256": _indices_digest(dataset, test),
                "train_scaffold_set_sha256": _groups_digest(dataset, train),
                "test_scaffold_set_sha256": _groups_digest(dataset, test),
                "inner_folds": inner_manifest,
            }
        )
    return result


def _validate_reference_dataset(
    name: str,
    dataset: PreparedMoleculeNetDataset,
    reference: dict[str, Any],
    splits: list[tuple[np.ndarray, np.ndarray, int]],
) -> None:
    reference_dataset = reference["datasets"].get(name)
    if reference_dataset is None:
        raise RuntimeError(f"Reference benchmark lacks dataset {name}")
    for key in (
        "molecules",
        "scaffold_groups",
        "duplicate_rows_collapsed",
        "conflicting_duplicate_rows_rejected",
    ):
        if reference_dataset["preparation"].get(key) != dataset.preparation.get(key):
            raise RuntimeError(f"{name} preparation mismatch for {key}")
    if int(reference_dataset["scaffold_splits"]) != len(splits):
        raise RuntimeError(f"{name} scaffold split count changed")
    expected = [
        (int(seed), int(len(train)), int(len(test))) for train, test, seed in splits
    ]
    for feature in ("molecule_embedding", "morgan_radius2_2048"):
        per_split = reference_dataset["feature_results"][feature]["per_split"]
        observed = [
            (int(item["outer_seed"]), int(item["train"]), int(item["test"]))
            for item in per_split
        ]
        if observed != expected:
            raise RuntimeError(
                f"{name} reconstructed splits do not match reference feature {feature}"
            )


def benchmark_descriptor_control(
    cfg: dict[str, Any],
    *,
    datasets_dir: str | Path,
    reference_benchmark: str | Path,
    output: str | Path,
    summary_csv: str | Path,
    dataset_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    names = (
        list(DEFAULT_MOLECULENET_DATASETS)
        if dataset_names is None
        else _resolve_dataset_names(dataset_names)
    )
    reference_path = Path(reference_benchmark)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if int(reference.get("schema_version", 0)) != 1:
        raise RuntimeError("Unsupported reference downstream benchmark schema")
    if int(reference["checkpoint"]["global_step"]) != 10_000:
        raise RuntimeError("Descriptor control must bind to the selected 10k benchmark")

    manifest_path, manifest = _load_dataset_manifest(cfg)
    prepared = _prepare_selected(cfg, datasets_dir, names)
    corpus_rows = _join_pretraining_rows(
        cfg, manifest, prepared, include_descriptors=True
    )
    datasets: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    descriptor_feature_names = descriptor_names(cfg)
    atol = float(cfg["data"]["deduplication"]["descriptor_atol"])
    rtol = float(cfg["data"]["deduplication"]["descriptor_rtol"])

    for name in names:
        spec = MOLECULENET_DATASETS[name]
        dataset = prepared[name]
        split_count = int(reference["datasets"][name]["scaffold_splits"])
        splits = _scaffold_splits(
            dataset.scaffold_groups,
            dataset.targets,
            task=spec["task"],
            count=split_count,
            seed=int(cfg["seed"]),
        )
        _validate_reference_dataset(name, dataset, reference, splits)

        names_from_generators, recomputed = _descriptor_matrix(cfg, dataset.molecules)
        if names_from_generators != descriptor_feature_names:
            raise RuntimeError("Descriptor feature order changed during calculation")
        values = recomputed.copy()
        stored_mask = np.zeros(len(dataset.molecules), dtype=bool)
        stored = np.full_like(recomputed, np.nan)
        for index, row in enumerate(corpus_rows[name]):
            if row is None:
                continue
            stored[index] = np.asarray(row["descriptors"], dtype=np.float64)
            stored_mask[index] = True
        if not np.isfinite(stored[stored_mask]).all():
            raise RuntimeError(f"{name} stored descriptor rows contain non-finite values")
        values[stored_mask] = stored[stored_mask]
        differences = np.abs(stored[stored_mask] - recomputed[stored_mask])
        close = np.isclose(
            stored[stored_mask], recomputed[stored_mask], rtol=rtol, atol=atol
        )
        mismatch_values = int(close.size - np.count_nonzero(close))
        mismatch_molecules = (
            int(np.count_nonzero(~np.all(close, axis=1))) if close.size else 0
        )
        descriptor_per_split = (
            _regression_probe(
                values,
                dataset.targets,
                dataset.scaffold_groups,
                splits,
            )
            if spec["task"] == "regression"
            else _classification_probe(
                values,
                dataset.targets,
                dataset.scaffold_groups,
                splits,
            )
        )
        descriptor_result = _summarize(descriptor_per_split, spec["task"])
        reference_features = reference["datasets"][name]["feature_results"]
        feature_results = {
            "molecule_embedding": copy.deepcopy(
                reference_features["molecule_embedding"]
            ),
            "morgan_radius2_2048": copy.deepcopy(
                reference_features["morgan_radius2_2048"]
            ),
            "auxiliary_descriptors_13": descriptor_result,
        }
        metric = "roc_auc" if spec["task"] == "classification" else "rmse"
        gmolai_mean = float(feature_results["molecule_embedding"]["summary"][metric]["mean"])
        morgan_mean = float(feature_results["morgan_radius2_2048"]["summary"][metric]["mean"])
        descriptor_mean = float(descriptor_result["summary"][metric]["mean"])
        favorable_gmolai_minus_descriptor = (
            gmolai_mean - descriptor_mean
            if spec["task"] == "classification"
            else descriptor_mean - gmolai_mean
        )
        favorable_descriptor_minus_morgan = (
            descriptor_mean - morgan_mean
            if spec["task"] == "classification"
            else morgan_mean - descriptor_mean
        )
        captured_fraction = None
        if spec["task"] == "regression" and morgan_mean != gmolai_mean:
            captured_fraction = (
                (morgan_mean - descriptor_mean) / (morgan_mean - gmolai_mean)
            )
        datasets[name] = {
            "source": _dataset_source(spec),
            "task": spec["task"],
            "preparation": dataset.preparation,
            "split_reconstruction": {
                "status": "exact_match_to_reference_seeds_and_counts",
                "outer_semantics": (
                    "approximately 80/20 percent of scaffold groups; realized molecule "
                    "fractions vary with group size"
                ),
                "identity_manifest": _split_identity_manifest(
                    dataset, splits, task=spec["task"]
                ),
            },
            "descriptor_features": {
                "names": descriptor_feature_names,
                "dimensions": len(descriptor_feature_names),
                "stored_pretraining_rows": int(np.count_nonzero(stored_mask)),
                "pinned_rdkit_recomputed_rows": int(
                    len(stored_mask) - np.count_nonzero(stored_mask)
                ),
                "stored_vs_recomputed_compared_molecules": int(
                    np.count_nonzero(stored_mask)
                ),
                "stored_vs_recomputed_mismatch_molecules": mismatch_molecules,
                "stored_vs_recomputed_mismatch_values": mismatch_values,
                "stored_vs_recomputed_max_absolute_difference": (
                    float(differences.max()) if differences.size else None
                ),
                "comparison_atol": atol,
                "comparison_rtol": rtol,
                "feature_source_rule": (
                    "immutable pretraining descriptor row on exact identity match; "
                    "otherwise the same frozen definitions recomputed with pinned RDKit"
                ),
            },
            "feature_results": feature_results,
            "comparison": {
                "primary_metric": metric,
                "favorable_gmolai_minus_descriptor": favorable_gmolai_minus_descriptor,
                "favorable_descriptor_minus_morgan": favorable_descriptor_minus_morgan,
                "regression_descriptor_fraction_of_gmolai_vs_morgan_mean_gain": captured_fraction,
                "interpretation_constraint": (
                    "Descriptive paired-split comparison only; this is not a causal "
                    "decomposition of representation performance."
                ),
            },
        }
        csv_rows.append(
            {
                "dataset": name,
                "molecules": len(dataset.molecules),
                "scaffold_groups": dataset.preparation["scaffold_groups"],
                "primary_metric": metric,
                "gmolai_mean": gmolai_mean,
                "gmolai_std": feature_results["molecule_embedding"]["summary"][metric]["std"],
                "morgan_mean": morgan_mean,
                "morgan_std": feature_results["morgan_radius2_2048"]["summary"][metric]["std"],
                "descriptor_13_mean": descriptor_mean,
                "descriptor_13_std": descriptor_result["summary"][metric]["std"],
                "favorable_gmolai_minus_descriptor": favorable_gmolai_minus_descriptor,
                "favorable_descriptor_minus_morgan": favorable_descriptor_minus_morgan,
                "regression_descriptor_fraction_of_gmolai_vs_morgan_mean_gain": captured_fraction,
                "stored_descriptor_rows": int(np.count_nonzero(stored_mask)),
                "recomputed_descriptor_rows": int(
                    len(stored_mask) - np.count_nonzero(stored_mask)
                ),
            }
        )

    result = {
        "schema_version": 1,
        "benchmark": "Frozen 13-auxiliary-descriptor downstream control",
        "pretrained_model_executed": False,
        "datasets_requested": names,
        "reference_benchmark": {
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
            "checkpoint": copy.deepcopy(reference["checkpoint"]),
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
            "graphs": int(manifest["deduplication"]["rows_after_deduplication"]),
            "descriptor_schema_hash": manifest["descriptor_schema_hash"],
        },
        "protocol": {
            "canonicalization_and_deduplication": "identical downstream preparation helper",
            "outer_splits": "reconstructed and validated against the frozen 10k artifact",
            "inner_cv": "identical three-fold grouped assignments",
            "feature_scaling": "StandardScaler fitted independently within each fold",
            "models_and_search": "identical Ridge/logistic models and hyperparameter grids",
            "metrics": "identical task-specific metrics and population standard deviations",
        },
        "runtime": {
            **runtime_versions(),
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit_version,
        },
        "datasets": datasets,
    }
    atomic_write_json(output, result)
    atomic_write_csv(summary_csv, csv_rows)
    return result

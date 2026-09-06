#!/usr/bin/env python3
"""Reproduce the Uni-Dock versus AutoDock-GPU correlation analysis.

The full result tables are intentionally Git-ignored. Run this script from a
checkout that retains the local matched benchmark outputs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "autodock-gpu-v1.6-vs-unidock-v1.2.0-gh200-20260830"
)

RUNS = {
    "one_gpu_20k": {
        "label": "Matched one-GH200 20,000-input run",
        "unidock": ["raw/unidock-fast-20k-one-gpu/results.csv"],
        "autodock_gpu": [
            "raw/adgpu-fast-20k-one-gpu-covered-optimized/results.csv"
        ],
    },
    "four_gpu_80k": {
        "label": "Matched four-GH200 80,000-input run",
        "unidock": ["raw/unidock-fast-80k-4gpu/shard-*/results.csv"],
        "autodock_gpu": [
            "raw/adgpu-fast-80k-4gpu-covered/shard-*/results.csv"
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Matched engine benchmark directory containing raw result tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON here instead of stdout.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_patterns(benchmark_dir: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(item) for item in glob.glob(str(benchmark_dir / pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"No result tables matched beneath {benchmark_dir}: {list(patterns)}"
        )
    return paths


def load_results(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], Counter]:
    rows: dict[str, dict[str, Any]] = {}
    statuses: Counter = Counter()
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                molecule_id = row["molecule_id"]
                if molecule_id in rows:
                    raise ValueError(f"Duplicate molecule_id: {molecule_id}")
                status = row["status"]
                statuses[status] += 1
                score_text = row["docking_score"].strip()
                score = float(score_text) if score_text else math.nan
                rows[molecule_id] = {
                    "canonical_smiles": row["canonical_smiles"],
                    "heavy_atoms": float(row["heavy_atoms"]),
                    "rotatable_bonds": float(row["rotatable_bonds"]),
                    "score": score,
                    "status": status,
                }
    return rows, statuses


def statistic(result: Any) -> float:
    return float(result.statistic)


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray,
    *,
    ranked: bool,
) -> float:
    if ranked:
        x = stats.rankdata(x, method="average")
        y = stats.rankdata(y, method="average")
        covariates = np.column_stack(
            [
                stats.rankdata(covariates[:, column], method="average")
                for column in range(covariates.shape[1])
            ]
        )
    design = np.column_stack([np.ones(len(x)), covariates])
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return statistic(stats.pearsonr(residual_x, residual_y))


def score_summary(scores: np.ndarray) -> dict[str, Any]:
    return {
        "finite_positive_count": int(np.sum(scores > 0)),
        "finite_positive_fraction": float(np.mean(scores > 0)),
        "maximum": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "minimum": float(np.min(scores)),
        "p01": float(np.quantile(scores, 0.01)),
        "p05": float(np.quantile(scores, 0.05)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "unique_score_count": int(len(np.unique(scores))),
    }


def top_overlap(
    molecule_ids: list[str],
    unidock_scores: np.ndarray,
    autodock_gpu_scores: np.ndarray,
    fraction: float,
) -> dict[str, Any]:
    total = len(molecule_ids)
    top_count = max(1, math.ceil(total * fraction))

    # Exact top-k sets use molecule_id only to resolve score ties
    # deterministically. Tie-inclusive statistics are recorded separately.
    unidock_order = sorted(
        range(total), key=lambda index: (unidock_scores[index], molecule_ids[index])
    )
    autodock_gpu_order = sorted(
        range(total),
        key=lambda index: (autodock_gpu_scores[index], molecule_ids[index]),
    )
    unidock_top = {molecule_ids[index] for index in unidock_order[:top_count]}
    autodock_gpu_top = {
        molecule_ids[index] for index in autodock_gpu_order[:top_count]
    }
    intersection = len(unidock_top & autodock_gpu_top)

    unidock_cutoff = float(
        np.partition(unidock_scores, top_count - 1)[top_count - 1]
    )
    autodock_gpu_cutoff = float(
        np.partition(autodock_gpu_scores, top_count - 1)[top_count - 1]
    )
    unidock_tie_inclusive = {
        molecule_ids[index]
        for index in range(total)
        if unidock_scores[index] <= unidock_cutoff
    }
    autodock_gpu_tie_inclusive = {
        molecule_ids[index]
        for index in range(total)
        if autodock_gpu_scores[index] <= autodock_gpu_cutoff
    }
    tie_intersection = len(unidock_tie_inclusive & autodock_gpu_tie_inclusive)

    return {
        "fraction": fraction,
        "random_expected_intersection": float(top_count * top_count / total),
        "random_expected_overlap_fraction": float(top_count / total),
        "top_count": top_count,
        "fixed_count": {
            "intersection": intersection,
            "overlap_fraction": float(intersection / top_count),
        },
        "tie_inclusive": {
            "autodock_gpu_count": len(autodock_gpu_tie_inclusive),
            "autodock_gpu_score_cutoff": autodock_gpu_cutoff,
            "intersection": tie_intersection,
            "jaccard": float(
                tie_intersection
                / len(unidock_tie_inclusive | autodock_gpu_tie_inclusive)
            ),
            "overlap_coefficient": float(
                tie_intersection
                / min(
                    len(unidock_tie_inclusive),
                    len(autodock_gpu_tie_inclusive),
                )
            ),
            "unidock_count": len(unidock_tie_inclusive),
            "unidock_score_cutoff": unidock_cutoff,
        },
    }


def analyze_run(
    benchmark_dir: Path,
    label: str,
    unidock_patterns: list[str],
    autodock_gpu_patterns: list[str],
) -> dict[str, Any]:
    unidock_paths = resolve_patterns(benchmark_dir, unidock_patterns)
    autodock_gpu_paths = resolve_patterns(benchmark_dir, autodock_gpu_patterns)
    unidock_rows, unidock_statuses = load_results(unidock_paths)
    autodock_gpu_rows, autodock_gpu_statuses = load_results(autodock_gpu_paths)

    paired_ids = sorted(
        molecule_id
        for molecule_id in unidock_rows.keys() & autodock_gpu_rows.keys()
        if unidock_rows[molecule_id]["status"] == "success"
        and autodock_gpu_rows[molecule_id]["status"] == "success"
        and math.isfinite(unidock_rows[molecule_id]["score"])
        and math.isfinite(autodock_gpu_rows[molecule_id]["score"])
    )
    mismatches = [
        molecule_id
        for molecule_id in paired_ids
        if unidock_rows[molecule_id]["canonical_smiles"]
        != autodock_gpu_rows[molecule_id]["canonical_smiles"]
    ]
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} paired molecule IDs have different canonical SMILES"
        )

    unidock_scores = np.array(
        [unidock_rows[molecule_id]["score"] for molecule_id in paired_ids],
        dtype=float,
    )
    autodock_gpu_scores = np.array(
        [autodock_gpu_rows[molecule_id]["score"] for molecule_id in paired_ids],
        dtype=float,
    )
    covariates = np.array(
        [
            [
                unidock_rows[molecule_id]["heavy_atoms"],
                unidock_rows[molecule_id]["rotatable_bonds"],
            ]
            for molecule_id in paired_ids
        ],
        dtype=float,
    )

    unidock_ranks = stats.rankdata(unidock_scores, method="average")
    autodock_gpu_ranks = stats.rankdata(autodock_gpu_scores, method="average")
    unidock_percentiles = (unidock_ranks - 1.0) / (len(paired_ids) - 1.0)
    autodock_gpu_percentiles = (autodock_gpu_ranks - 1.0) / (
        len(paired_ids) - 1.0
    )
    percentile_delta = np.abs(unidock_percentiles - autodock_gpu_percentiles)

    pearson_r = statistic(stats.pearsonr(unidock_scores, autodock_gpu_scores))
    unidock_low, unidock_high = np.quantile(unidock_scores, [0.01, 0.99])
    autodock_gpu_low, autodock_gpu_high = np.quantile(
        autodock_gpu_scores, [0.01, 0.99]
    )
    winsorized_pearson_r = statistic(
        stats.pearsonr(
            np.clip(unidock_scores, unidock_low, unidock_high),
            np.clip(
                autodock_gpu_scores,
                autodock_gpu_low,
                autodock_gpu_high,
            ),
        )
    )

    def source_metadata(paths: list[Path]) -> list[dict[str, Any]]:
        return [
            {
                "bytes": path.stat().st_size,
                "path": display_path(path),
                "sha256": sha256(path),
            }
            for path in paths
        ]

    return {
        "label": label,
        "input_files": {
            "autodock_gpu": source_metadata(autodock_gpu_paths),
            "unidock": source_metadata(unidock_paths),
        },
        "pairing": {
            "autodock_gpu_rows": len(autodock_gpu_rows),
            "autodock_gpu_status_counts": dict(
                sorted(autodock_gpu_statuses.items())
            ),
            "canonical_smiles_mismatch_count": len(mismatches),
            "paired_finite_successes": len(paired_ids),
            "unidock_rows": len(unidock_rows),
            "unidock_status_counts": dict(sorted(unidock_statuses.items())),
        },
        "score_correlation": {
            "pearson_r": pearson_r,
            "pearson_r_squared": pearson_r * pearson_r,
            "pearson_r_winsorized_1pct_each_tail": winsorized_pearson_r,
        },
        "rank_correlation": {
            "kendall_tau_b": statistic(
                stats.kendalltau(unidock_scores, autodock_gpu_scores)
            ),
            "spearman_rho": statistic(
                stats.spearmanr(unidock_scores, autodock_gpu_scores)
            ),
        },
        "partial_correlation": {
            "controls": ["heavy_atoms", "rotatable_bonds"],
            "pearson_r": partial_correlation(
                unidock_scores,
                autodock_gpu_scores,
                covariates,
                ranked=False,
            ),
            "spearman_rho": partial_correlation(
                unidock_scores,
                autodock_gpu_scores,
                covariates,
                ranked=True,
            ),
        },
        "ranking_distance": {
            "mean_absolute_percentile_points": float(
                np.mean(percentile_delta) * 100
            ),
            "median_absolute_percentile_points": float(
                np.median(percentile_delta) * 100
            ),
            "within_10_percentile_points_fraction": float(
                np.mean(percentile_delta <= 0.10)
            ),
            "within_25_percentile_points_fraction": float(
                np.mean(percentile_delta <= 0.25)
            ),
        },
        "score_property_spearman": {
            "autodock_gpu_vs_heavy_atoms": statistic(
                stats.spearmanr(autodock_gpu_scores, covariates[:, 0])
            ),
            "autodock_gpu_vs_rotatable_bonds": statistic(
                stats.spearmanr(autodock_gpu_scores, covariates[:, 1])
            ),
            "unidock_vs_heavy_atoms": statistic(
                stats.spearmanr(unidock_scores, covariates[:, 0])
            ),
            "unidock_vs_rotatable_bonds": statistic(
                stats.spearmanr(unidock_scores, covariates[:, 1])
            ),
        },
        "score_summary": {
            "autodock_gpu": score_summary(autodock_gpu_scores),
            "unidock": score_summary(unidock_scores),
        },
        "top_rank_overlap": [
            top_overlap(
                paired_ids,
                unidock_scores,
                autodock_gpu_scores,
                fraction,
            )
            for fraction in (0.001, 0.01, 0.05, 0.10)
        ],
    }


def analyze(benchmark_dir: Path) -> dict[str, Any]:
    benchmark_dir = benchmark_dir.resolve()
    runs = {
        run_name: analyze_run(
            benchmark_dir,
            definition["label"],
            definition["unidock"],
            definition["autodock_gpu"],
        )
        for run_name, definition in RUNS.items()
    }
    return {
        "analysis": "Cross-engine docking-score and ranking correlation",
        "analysis_date": "2026-08-30",
        "engines": {
            "autodock_gpu": {
                "commit": "e63e6f6280ebfad18caa3e8f48afdc269e79e063",
                "scoring_function": "ad4",
                "version": "1.6",
            },
            "unidock": {
                "commit": "95e409172b15dec0989aea70b0f2328e8ca52025",
                "scoring_function": "vina",
                "version": "1.2.0",
            },
        },
        "method": {
            "better_score_direction": "lower",
            "pair_key": "molecule_id",
            "pair_requirements": [
                "same canonical_smiles",
                "status == success for both engines",
                "finite docking_score for both engines",
            ],
            "rank_ties": "average ranks; exact top-k uses molecule_id tie-break",
            "scientific_scope": (
                "Agreement analysis only; no experimental labels or reference "
                "poses were used, so this is not an accuracy comparison."
            ),
        },
        "runs": runs,
        "schema_version": 1,
    }


def main() -> None:
    args = parse_args()
    result = analyze(args.benchmark_dir)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent raw 10k fast/balance validation for one RL target."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr

from igen3.generation import canonicalize_valid_smiles
from igen3.metrics import calculate_molecule_metrics
from igenvs_rl.oracle import DockingOutcome, IGenVSOracle, stable_molecule_id
from igenvs_rl.policy import load_policy_bundle, sample_policy


PROPERTY_COLUMNS = (
    "qed",
    "sa_score",
    "mol_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "heavy_atoms",
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
    "formal_charge",
)
SCORE_STATISTICS = (
    "score_min",
    "score_best_10_mean",
    "score_p01",
    "score_p10",
    "score_median",
    "score_p90",
    "score_p95",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--final-stage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=917003)
    parser.add_argument("--shards", type=int, default=4)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint(final_stage: Path) -> tuple[Path, str, int, dict[str, torch.Tensor]]:
    for name in ("best", "latest"):
        path = final_stage / "checkpoints" / f"{name}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            return path, name, int(payload["update"]), payload["policy_state"]
    raise FileNotFoundError(f"no trained checkpoint found below {final_stage}")


def _annotate_structure(frame: pd.DataFrame) -> pd.DataFrame:
    annotations: list[dict[str, object]] = []
    for canonical in frame["canonical_smiles"]:
        if not isinstance(canonical, str) or not canonical:
            annotations.append({})
            continue
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            annotations.append({})
            continue
        annotations.append(
            {
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "ring_count": Lipinski.RingCount(molecule),
                "aromatic_ring_count": Lipinski.NumAromaticRings(molecule),
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(molecule),
                "formal_charge": Chem.GetFormalCharge(molecule),
                "murcko_scaffold": (
                    MurckoScaffold.MurckoScaffoldSmiles(
                        mol=molecule,
                        includeChirality=True,
                    )
                    or "<acyclic>"
                ),
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(annotations)], axis=1)


def _sample_arm(
    *,
    arm: str,
    output_path: Path,
    model_root: Path,
    model_id: str,
    policy_state: dict[str, torch.Tensor] | None,
    count: int,
    seed: int,
    temperature: float,
    top_k: int,
) -> tuple[pd.DataFrame, float]:
    if output_path.is_file():
        frame = pd.read_csv(output_path)
        if len(frame) != count:
            raise RuntimeError(f"{output_path} has {len(frame)} rows, expected {count}")
        return frame, 0.0

    started = time.perf_counter()
    _seed_everything(seed)
    bundle = load_policy_bundle(
        model_root=model_root,
        model_id=model_id,
        policy_state=policy_state,
    )
    tokens, raw_smiles = sample_policy(
        bundle,
        count,
        temperature=temperature,
        top_k=top_k,
    )
    if len(raw_smiles) != count:
        raise RuntimeError(f"{arm}: sampled {len(raw_smiles)} raw strings, expected {count}")
    molecule_metrics, _ = calculate_molecule_metrics(
        raw_smiles,
        model_id=arm,
        isomeric_smiles=True,
    )
    molecule_metrics = molecule_metrics.rename(columns={"smiles": "generated_smiles"})
    molecule_metrics.insert(0, "sample_index", np.arange(count, dtype=int))
    molecule_metrics["canonical_smiles"] = [
        canonicalize_valid_smiles(smiles, isomeric_smiles=True) or ""
        for smiles in raw_smiles
    ]
    molecule_metrics = _annotate_structure(molecule_metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    molecule_metrics.to_csv(output_path, index=False)
    elapsed = time.perf_counter() - started
    del bundle, tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return molecule_metrics, elapsed


def _read_outcomes(path: Path, shards: int) -> dict[str, DockingOutcome]:
    result_paths = [path / f"shard-{index}" / "results.csv" for index in range(shards)]
    if not all(result.is_file() for result in result_paths):
        raise RuntimeError(f"incomplete existing docking directory: {path}")
    outcomes: dict[str, DockingOutcome] = {}
    for shard, result_path in enumerate(result_paths):
        with result_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_score = (row.get("docking_score") or "").strip()
                outcomes[row["molecule_id"]] = DockingOutcome(
                    molecule_id=row["molecule_id"],
                    canonical_smiles=row.get("canonical_smiles", ""),
                    status=row.get("status", "unknown"),
                    score=float(raw_score) if raw_score else None,
                    error=row.get("error", ""),
                )
        rejected = path / f"shard-{shard}" / "validation" / "rejected.csv"
        if rejected.is_file():
            with rejected.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    outcomes[row["molecule_id"]] = DockingOutcome(
                        molecule_id=row["molecule_id"],
                        canonical_smiles="",
                        status=row.get("status", "validation_rejected"),
                        score=None,
                        error=row.get("error", ""),
                    )
    return outcomes


def _dock_arm(
    *,
    frame: pd.DataFrame,
    target: Path,
    output_dir: Path,
    mode: str,
    shards: int,
    docking_seed: int,
) -> tuple[pd.DataFrame, float]:
    canonicals = sorted(
        {
            value
            for value in frame["canonical_smiles"]
            if isinstance(value, str) and value
        }
    )
    molecules = {stable_molecule_id(smiles): smiles for smiles in canonicals}
    started = time.perf_counter()
    if output_dir.is_dir():
        outcomes = _read_outcomes(output_dir, shards)
        elapsed = 0.0
    else:
        oracle = IGenVSOracle(
            target=target,
            engine="unidock",
            scoring="vina",
            search_mode=mode,
            shards=shards,
            prep_workers=16,
            validation_workers=8,
            seed=docking_seed,
        )
        outcomes = oracle.dock(molecules, output_dir)
        elapsed = time.perf_counter() - started
    if set(outcomes) != set(molecules):
        missing = len(set(molecules) - set(outcomes))
        extra = len(set(outcomes) - set(molecules))
        raise RuntimeError(f"{mode}: outcome identity mismatch (missing={missing}, extra={extra})")

    statuses: list[str] = []
    scores: list[float | None] = []
    errors: list[str] = []
    for canonical in frame["canonical_smiles"]:
        if not isinstance(canonical, str) or not canonical:
            statuses.append("invalid_smiles")
            scores.append(None)
            errors.append("")
            continue
        outcome = outcomes[stable_molecule_id(canonical)]
        statuses.append(outcome.status)
        scores.append(outcome.score)
        errors.append(outcome.error)
    result = frame.copy()
    result[f"{mode}_status"] = statuses
    result[f"{mode}_docking_score"] = scores
    result[f"{mode}_error"] = errors
    return result, elapsed


def _finite_float(value: object) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _property_summary(frame: pd.DataFrame) -> dict[str, object]:
    valid = frame.loc[frame["valid"] == True].copy()  # noqa: E712
    canonical_counts = Counter(valid["canonical_smiles"])
    scaffold_counts = Counter(valid["murcko_scaffold"].dropna().astype(str))
    summary: dict[str, object] = {
        "raw_count": len(frame),
        "valid_count": len(valid),
        "valid_fraction_raw": len(valid) / len(frame),
        "unique_valid_count": len(canonical_counts),
        "unique_valid_fraction": len(canonical_counts) / len(valid) if len(valid) else 0.0,
        "top_molecule_fraction_raw": (
            canonical_counts.most_common(1)[0][1] / len(frame) if canonical_counts else 0.0
        ),
        "lipinski_count": int(valid["lipinski_ro5_pass"].eq(True).sum()),  # noqa: E712
        "lipinski_fraction_raw": float(
            valid["lipinski_ro5_pass"].eq(True).sum() / len(frame)  # noqa: E712
        ),
        "lipinski_fraction_valid": float(
            valid["lipinski_ro5_pass"].eq(True).mean()  # noqa: E712
        ) if len(valid) else 0.0,
        "unique_scaffold_count": len(scaffold_counts),
        "top_scaffold_fraction_raw": (
            scaffold_counts.most_common(1)[0][1] / len(frame) if scaffold_counts else 0.0
        ),
    }
    for column in PROPERTY_COLUMNS:
        values = pd.to_numeric(valid[column], errors="coerce").dropna().to_numpy(dtype=float)
        summary[f"{column}_mean"] = float(values.mean()) if len(values) else None
        summary[f"{column}_median"] = float(np.median(values)) if len(values) else None

    if len(valid) >= 2:
        fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        unique_fingerprints = {
            smiles: fingerprint_generator.GetFingerprint(Chem.MolFromSmiles(smiles))
            for smiles in canonical_counts
        }
        fingerprints = [unique_fingerprints[smiles] for smiles in valid["canonical_smiles"]]
        rng = np.random.default_rng(20260907)
        pair_count = 100_000
        left = rng.integers(0, len(fingerprints), pair_count)
        right = rng.integers(0, len(fingerprints) - 1, pair_count)
        right += right >= left
        similarities = np.fromiter(
            (
                DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                for i, j in zip(left, right)
            ),
            dtype=float,
            count=pair_count,
        )
        summary["internal_diversity"] = float(1.0 - similarities.mean())
        summary["internal_diversity_sampled_pairs"] = pair_count
    else:
        summary["internal_diversity"] = None
        summary["internal_diversity_sampled_pairs"] = 0
    return summary


def _docking_summary(
    frame: pd.DataFrame,
    *,
    mode: str,
    elite_threshold: float,
    minimum_qed: float,
    maximum_absolute_formal_charge: int,
    minimum_fraction_csp3: float,
    maximum_aromatic_rings: int,
) -> dict[str, object]:
    score_column = f"{mode}_docking_score"
    status_column = f"{mode}_status"
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    successful = frame[status_column].eq("success") & np.isfinite(scores)
    successful_scores = scores.loc[successful].to_numpy(dtype=float)
    ordered = np.sort(successful_scores)
    lipinski = frame["lipinski_ro5_pass"].eq(True)  # noqa: E712
    qed = pd.to_numeric(frame["qed"], errors="coerce")
    formal_charge = pd.to_numeric(frame["formal_charge"], errors="coerce")
    fraction_csp3 = pd.to_numeric(frame["fraction_csp3"], errors="coerce")
    aromatic_rings = pd.to_numeric(frame["aromatic_ring_count"], errors="coerce")
    chemistry = (
        lipinski
        & qed.ge(minimum_qed)
        & formal_charge.abs().le(maximum_absolute_formal_charge)
        & fraction_csp3.ge(minimum_fraction_csp3)
        & aromatic_rings.le(maximum_aromatic_rings)
    )
    qualified = successful & chemistry & scores.le(elite_threshold)
    successful_canonicals = frame.loc[successful, "canonical_smiles"].astype(str)
    qualified_canonicals = frame.loc[qualified, "canonical_smiles"].astype(str)
    if not len(ordered):
        raise RuntimeError(f"{mode}: no successful docking scores")
    return {
        "raw_count": len(frame),
        "successful_count_raw": int(successful.sum()),
        "successful_fraction_raw": float(successful.mean()),
        "successful_unique_count": int(successful_canonicals.nunique()),
        "positive_score_count_raw": int((successful & scores.gt(0.0)).sum()),
        "positive_score_fraction_raw": float((successful & scores.gt(0.0)).mean()),
        "score_min": float(ordered[0]),
        "score_best_10_mean": float(ordered[: min(10, len(ordered))].mean()),
        "score_p01": float(np.quantile(ordered, 0.01)),
        "score_p10": float(np.quantile(ordered, 0.10)),
        "score_median": float(np.median(ordered)),
        "score_p90": float(np.quantile(ordered, 0.90)),
        "score_p95": float(np.quantile(ordered, 0.95)),
        "score_max": float(ordered[-1]),
        "elite_threshold": elite_threshold,
        "elite_count_raw": int((successful & scores.le(elite_threshold)).sum()),
        "elite_fraction_raw": float((successful & scores.le(elite_threshold)).mean()),
        "chemistry_pass_count_raw": int(chemistry.sum()),
        "chemistry_pass_fraction_raw": float(chemistry.mean()),
        "qualified_elite_count_raw": int(qualified.sum()),
        "qualified_elite_fraction_raw": float(qualified.mean()),
        "qualified_elite_unique_count": int(qualified_canonicals.nunique()),
    }


def _comparison(base: dict[str, object], rl: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {
        f"{key}_improvement": float(base[key]) - float(rl[key])
        for key in SCORE_STATISTICS
    }
    base_rate = float(base["qualified_elite_fraction_raw"])
    rl_rate = float(rl["qualified_elite_fraction_raw"])
    output.update(
        {
            "qualified_elite_absolute_gain": rl_rate - base_rate,
            "qualified_elite_enrichment": rl_rate / base_rate if base_rate else None,
            "positive_score_fraction_reduction": (
                float(base["positive_score_fraction_raw"])
                - float(rl["positive_score_fraction_raw"])
            ),
        }
    )
    return output


def _fast_balance_spearman(frame: pd.DataFrame) -> float | None:
    scores = frame[["canonical_smiles", "fast_status", "fast_docking_score", "balance_status", "balance_docking_score"]]
    scores = scores.drop_duplicates("canonical_smiles")
    scores = scores.loc[
        scores["fast_status"].eq("success") & scores["balance_status"].eq("success")
    ].copy()
    if len(scores) < 2:
        return None
    correlation = spearmanr(scores["fast_docking_score"], scores["balance_docking_score"]).statistic
    return _finite_float(correlation)


def _write_paired_csv(base: pd.DataFrame, rl: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "sample_index",
        "generated_smiles",
        "canonical_smiles",
        "valid",
        "lipinski_ro5_pass",
        "qed",
        "formal_charge",
        "fraction_csp3",
        "aromatic_ring_count",
        "fast_status",
        "fast_docking_score",
        "balance_status",
        "balance_docking_score",
    ]
    left = base[columns].rename(
        columns={
            "generated_smiles": "base_smiles",
            "canonical_smiles": "base_canonical_smiles",
            "valid": "base_valid",
            "lipinski_ro5_pass": "base_lipinski_ro5_pass",
            "qed": "base_qed",
            "formal_charge": "base_formal_charge",
            "fraction_csp3": "base_fraction_csp3",
            "aromatic_ring_count": "base_aromatic_ring_count",
            "fast_status": "base_fast_status",
            "fast_docking_score": "base_fast_docking_score",
            "balance_status": "base_balance_status",
            "balance_docking_score": "base_balance_docking_score",
        }
    )
    right = rl[columns].rename(
        columns={
            "generated_smiles": "rl_smiles",
            "canonical_smiles": "rl_canonical_smiles",
            "valid": "rl_valid",
            "lipinski_ro5_pass": "rl_lipinski_ro5_pass",
            "qed": "rl_qed",
            "formal_charge": "rl_formal_charge",
            "fraction_csp3": "rl_fraction_csp3",
            "aromatic_ring_count": "rl_aromatic_ring_count",
            "fast_status": "rl_fast_status",
            "fast_docking_score": "rl_fast_docking_score",
            "balance_status": "rl_balance_status",
            "balance_docking_score": "rl_balance_docking_score",
        }
    )
    left.merge(right, on="sample_index", validate="one_to_one").to_csv(output_path, index=False)


def main() -> None:
    args = _arguments()
    if args.count < 1 or args.shards < 1:
        raise ValueError("--count and --shards must be positive")
    final_stage = args.final_stage.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((final_stage / "config.json").read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    criteria = protocol["acceptance"]
    reference_payload = json.loads(
        (final_stage / "reference" / "scores.json").read_text(encoding="utf-8")
    )
    reference_scores = np.asarray(reference_payload["scores"], dtype=float)
    minimum_qed = float(config["minimum_qed"])
    maximum_absolute_formal_charge = int(config["maximum_absolute_formal_charge"])
    minimum_fraction_csp3 = float(config["minimum_fraction_csp3"])
    maximum_aromatic_rings = int(config["maximum_aromatic_rings"])
    checkpoint_path, checkpoint_name, checkpoint_update, policy_state = _checkpoint(final_stage)

    timing: dict[str, Any] = {
        "target": args.target_id,
        "started_at": _utc_now(),
        "gpu_count": args.shards,
        "sampling": {},
        "docking": {},
    }
    samples_dir = output_dir / "samples"
    frames: dict[str, pd.DataFrame] = {}
    for arm, state in (("base", None), ("rl", policy_state)):
        frame, elapsed = _sample_arm(
            arm=arm,
            output_path=samples_dir / f"{arm}.csv",
            model_root=Path(config["model_root"]),
            model_id=config["model_id"],
            policy_state=state,
            count=args.count,
            seed=args.seed,
            temperature=float(config["temperature"]),
            top_k=int(config["top_k"]),
        )
        timing["sampling"][arm] = elapsed
        frames[arm] = frame

    for arm in ("base", "rl"):
        for mode in ("fast", "balance"):
            frames[arm], elapsed = _dock_arm(
                frame=frames[arm],
                target=args.target.resolve(),
                output_dir=output_dir / "docking" / arm / mode,
                mode=mode,
                shards=args.shards,
                docking_seed=int(config["docking_seed"]),
            )
            timing["docking"][f"{arm}_{mode}"] = elapsed
        frames[arm].to_csv(output_dir / f"{arm}-raw10k.csv", index=False)

    thresholds: dict[str, float] = {}
    for mode in ("fast", "balance"):
        base_scores = pd.to_numeric(
            frames["base"].loc[
                frames["base"][f"{mode}_status"].eq("success"),
                f"{mode}_docking_score",
            ],
            errors="coerce",
        ).dropna()
        if base_scores.empty:
            raise RuntimeError(f"base {mode} docking produced no successful scores")
        thresholds[mode] = float(np.quantile(base_scores, 0.01))
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    comparisons: dict[str, dict[str, object]] = {}
    for mode in ("fast", "balance"):
        summaries[mode] = {
            arm: _docking_summary(
                frames[arm],
                mode=mode,
                elite_threshold=thresholds[mode],
                minimum_qed=minimum_qed,
                maximum_absolute_formal_charge=maximum_absolute_formal_charge,
                minimum_fraction_csp3=minimum_fraction_csp3,
                maximum_aromatic_rings=maximum_aromatic_rings,
            )
            for arm in ("base", "rl")
        }
        comparisons[mode] = _comparison(summaries[mode]["base"], summaries[mode]["rl"])

    properties = {arm: _property_summary(frames[arm]) for arm in ("base", "rl")}
    base_set = set(frames["base"].loc[frames["base"]["valid"] == True, "canonical_smiles"])  # noqa: E712
    rl_set = set(frames["rl"].loc[frames["rl"]["valid"] == True, "canonical_smiles"])  # noqa: E712
    acceptance = {
        "selected_qualified_checkpoint": checkpoint_name == "best",
        "raw_chemistry_pass": (
            float(summaries["fast"]["rl"]["chemistry_pass_fraction_raw"])
            >= float(criteria["minimum_raw_chemistry_pass_fraction"])
        ),
        "minimum_unique_valid_molecules": (
            int(properties["rl"]["unique_valid_count"])
            >= int(criteria["minimum_unique_valid_molecules"])
        ),
        "maximum_top_molecule_fraction": (
            float(properties["rl"]["top_molecule_fraction_raw"])
            <= float(criteria["maximum_top_molecule_fraction"])
        ),
    }
    for mode in ("fast", "balance"):
        acceptance[f"{mode}_raw_qualified_elite_fraction"] = (
            float(summaries[mode]["rl"]["qualified_elite_fraction_raw"])
            >= float(criteria["minimum_raw_qualified_elite_fraction"])
        )
        acceptance[f"{mode}_distinct_qualified_elites"] = (
            int(summaries[mode]["rl"]["qualified_elite_unique_count"])
            >= int(criteria["minimum_distinct_qualified_elites"])
        )
        acceptance[f"{mode}_positive_score_fraction"] = (
            float(summaries[mode]["rl"]["positive_score_fraction_raw"])
            <= float(criteria["maximum_positive_score_fraction"])
        )
        for field, minimum_gain in criteria["minimum_score_improvement"].items():
            acceptance[f"{mode}_{field}_improvement"] = (
                float(comparisons[mode][f"{field}_improvement"])
                >= float(minimum_gain)
            )
    acceptance["passed"] = all(acceptance.values())
    summary = {
        "target": args.target_id,
        "protocol": {
            "raw_draws_per_arm": args.count,
            "sampling_seed": args.seed,
            "temperature": float(config["temperature"]),
            "top_k": int(config["top_k"]),
            "engine": "unidock",
            "scoring": "vina",
            "docking_seed": int(config["docking_seed"]),
            "fast_elite_threshold": thresholds["fast"],
            "fast_elite_threshold_source": "matched validation base raw draws",
            "minimum_qed": minimum_qed,
            "maximum_absolute_formal_charge": maximum_absolute_formal_charge,
            "minimum_fraction_csp3": minimum_fraction_csp3,
            "maximum_aromatic_rings": maximum_aromatic_rings,
            "training_reference_elite_threshold": float(
                np.quantile(reference_scores, float(config["elite_fraction"]))
            ),
            "balance_elite_threshold": thresholds["balance"],
            "balance_elite_threshold_source": "matched validation base raw draws",
            "protocol_path": str(args.protocol.resolve()),
            "protocol_sha256": _sha256(args.protocol.resolve()),
        },
        "rl_checkpoint": {
            "selection": checkpoint_name,
            "update": checkpoint_update,
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "docking": summaries,
        "comparison": comparisons,
        "molecular_and_structural": properties,
        "chemical_space": {
            "base_rl_unique_molecule_overlap": len(base_set & rl_set),
            "base_only_unique_molecules": len(base_set - rl_set),
            "rl_only_unique_molecules": len(rl_set - base_set),
        },
        "fast_balance_spearman_unique": {
            arm: _fast_balance_spearman(frames[arm]) for arm in ("base", "rl")
        },
        "acceptance": acceptance,
    }
    _write_paired_csv(
        frames["base"],
        frames["rl"],
        output_dir / f"{args.target_id}_base-vs-rl_10k.csv",
    )
    _atomic_json(output_dir / "summary.json", summary)
    timing["ended_at"] = _utc_now()
    timing["validation_wall_seconds"] = sum(timing["sampling"].values()) + sum(
        timing["docking"].values()
    )
    timing["validation_gpu_hours"] = timing["validation_wall_seconds"] * args.shards / 3600.0
    _atomic_json(output_dir / "timing.json", timing)
    print(json.dumps(summary["acceptance"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

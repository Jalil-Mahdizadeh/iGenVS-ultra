"""On-policy iGen3 training driven by an iGenVS docking oracle."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean, median
from time import perf_counter, time_ns
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from igen3.generation import canonicalize_valid_smiles, write_de_novo_file
from igen3.metrics import _lipinski_pass as igen3_lipinski_pass
from igen3.registry import resolve_model

from .oracle import DockingOutcome, FakeOracle, IGenVSOracle, stable_molecule_id
from .policy import (
    PolicyBundle,
    load_policy_bundle,
    sample_policy,
    sequence_statistics,
    synchronize_sampler,
)
from .reward import (
    binary_elite_desirability,
    elite_desirability,
    empirical_quantile,
    hybrid_desirability,
    normalized_advantages,
    percentile_desirability,
)
from .state import (
    JobConfig,
    append_csv,
    append_score_cache,
    load_score_cache,
    load_seen_smiles,
)


SAMPLE_FIELDS = [
    "sample_index",
    "raw_smiles",
    "canonical_smiles",
    "molecule_id",
    "sample_status",
    "docking_status",
    "docking_score",
    "score_source",
    "lipinski_pass",
    "qed",
    "formal_charge",
    "fraction_csp3",
    "aromatic_ring_count",
    "chemistry_pass",
    "reward",
    "used_for_gradient",
    "error",
]

SEQUENCE_MICROBATCH_MAX = 256


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _oracle(config: JobConfig):
    if config.oracle == "fake":
        return FakeOracle()
    return IGenVSOracle(
        target=Path(config.target),
        engine=config.engine,
        scoring=config.scoring,
        search_mode=config.search_mode,
        shards=config.shards,
        prep_workers=config.prep_workers,
        validation_workers=config.validation_workers,
        seed=config.docking_seed,
    )


def _read_reference(job_dir: Path) -> list[float] | None:
    path = job_dir / "reference" / "scores.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = [float(value) for value in payload["scores"]]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return scores if scores and all(math.isfinite(value) for value in scores) else None


def _ensure_reference(
    job_dir: Path,
    config: JobConfig,
    bundle: PolicyBundle,
    oracle,
) -> list[float]:
    existing = _read_reference(job_dir)
    if existing is not None:
        return existing

    reference_dir = job_dir / "reference"
    if reference_dir.exists():
        reference_dir.rename(
            reference_dir.with_name(f"reference.incomplete-{time_ns()}")
        )
    reference_dir.mkdir(parents=True, exist_ok=True)
    smiles_path = reference_dir / "base-isomeric.smi"
    stats = write_de_novo_file(
        bundle.sampler,
        output_path=smiles_path,
        count=config.reference_count,
        batch_size=min(config.batch_size, config.reference_count),
        temperature=config.temperature,
        do_sample=True,
        top_k=config.top_k or None,
        progress=False,
    )
    smiles = [line.strip() for line in smiles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    molecules = {stable_molecule_id(value): value for value in smiles}
    outcomes = oracle.dock(molecules, reference_dir / "docking")
    scores = sorted(
        outcome.score
        for outcome in outcomes.values()
        if outcome.status == "success" and outcome.score is not None
    )
    minimum = min(config.reference_count, max(10, config.reference_count // 2))
    if len(scores) < minimum:
        raise RuntimeError(
            f"only {len(scores)} successful reference dockings; need at least {minimum}"
        )
    payload = {
        "model_id": config.model_id,
        "requested": config.reference_count,
        "generated": stats.generated,
        "candidates_generated": stats.candidates_generated,
        "successful_dockings": len(scores),
        "scores": scores,
    }
    _atomic_json(reference_dir / "scores.json", payload)
    print(
        f"[igenvs-rl] frozen reference: {len(scores)} scores, "
        f"median={median(scores):.3f}",
        flush=True,
    )
    return scores


def _scaffold_summary(smiles: list[str]) -> tuple[int, float]:
    scaffolds: list[str] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
        scaffolds.append(scaffold or "<acyclic>")
    if not scaffolds:
        return 0, 1.0
    counts = Counter(scaffolds)
    return len(counts), max(counts.values()) / len(scaffolds)


def _passes_igen3_lipinski(smiles: str) -> bool:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False
    return igen3_lipinski_pass(
        Descriptors.MolWt(molecule),
        Crippen.MolLogP(molecule),
        Lipinski.NumHDonors(molecule),
        Lipinski.NumHAcceptors(molecule),
    )


def _candidate_properties(
    smiles: str,
    *,
    minimum_qed: float,
    maximum_absolute_formal_charge: int | None,
    minimum_fraction_csp3: float = 0.0,
    maximum_aromatic_rings: int | None = None,
) -> tuple[bool, float, int, float, int, bool]:
    """Return molecular properties and the configured chemistry-gate result."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False, math.nan, 0, math.nan, 0, False
    lipinski_pass = igen3_lipinski_pass(
        Descriptors.MolWt(molecule),
        Crippen.MolLogP(molecule),
        Lipinski.NumHDonors(molecule),
        Lipinski.NumHAcceptors(molecule),
    )
    qed = float(QED.qed(molecule))
    formal_charge = int(Chem.GetFormalCharge(molecule))
    fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(molecule))
    aromatic_ring_count = int(Lipinski.NumAromaticRings(molecule))
    chemistry_pass = qed >= minimum_qed and (
        maximum_absolute_formal_charge is None
        or abs(formal_charge) <= maximum_absolute_formal_charge
    ) and fraction_csp3 >= minimum_fraction_csp3 and (
        maximum_aromatic_rings is None
        or aromatic_ring_count <= maximum_aromatic_rings
    )
    return (
        lipinski_pass,
        qed,
        formal_charge,
        fraction_csp3,
        aromatic_ring_count,
        chemistry_pass,
    )


def _write_sample_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _docking_desirability(
    reference_scores: list[float],
    score: float,
    *,
    reward_mode: str,
    tail_fraction: float,
    tail_weight: float,
    elite_fraction: float,
) -> float:
    if reward_mode == "percentile":
        return percentile_desirability(reference_scores, score)
    if reward_mode == "hybrid":
        return hybrid_desirability(
            reference_scores,
            score,
            tail_fraction=tail_fraction,
            tail_weight=tail_weight,
        )
    if reward_mode == "elite":
        return elite_desirability(
            reference_scores,
            score,
            elite_fraction=elite_fraction,
        )
    if reward_mode == "binary-elite":
        return binary_elite_desirability(
            reference_scores,
            score,
            elite_fraction=elite_fraction,
        )
    raise ValueError(f"unsupported reward mode: {reward_mode}")


def _sample_and_score(
    *,
    bundle: PolicyBundle,
    oracle,
    reference_scores: list[float],
    count: int,
    output_dir: Path,
    job_dir: Path,
    temperature: float,
    top_k: int | None,
    previously_seen: set[str],
    score_cache: dict[str, tuple[str, float]],
    reward_mode: str,
    tail_fraction: float,
    tail_weight: float,
    reward_seen_molecules: bool,
    elite_fraction: float,
    require_lipinski: bool,
    minimum_qed: float = 0.0,
    maximum_absolute_formal_charge: int | None = None,
    minimum_fraction_csp3: float = 0.0,
    maximum_aromatic_rings: int | None = None,
    reward_occurrence_cap: int | None = None,
    persist_score_cache: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, float]]:
    if output_dir.exists():
        output_dir.rename(
            output_dir.with_name(f"{output_dir.name}.incomplete-{time_ns()}")
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    tokens, raw_smiles = sample_policy(
        bundle,
        count,
        temperature=temperature,
        top_k=top_k,
    )
    rows: list[dict[str, Any]] = []
    molecules: dict[str, str] = {}
    molecule_rows: dict[str, list[int]] = {}
    batch_seen: set[str] = set()
    property_cache: dict[str, tuple[bool, float, int, float, int, bool]] = {}

    for index, raw in enumerate(raw_smiles):
        canonical = canonicalize_valid_smiles(raw, isomeric_smiles=True)
        row: dict[str, Any] = {
            "sample_index": index,
            "raw_smiles": raw,
            "canonical_smiles": canonical or "",
            "molecule_id": "",
            "sample_status": "candidate",
            "docking_status": "",
            "docking_score": "",
            "score_source": "",
            "lipinski_pass": "",
            "qed": "",
            "formal_charge": "",
            "chemistry_pass": "",
            "reward": 0.0,
            "used_for_gradient": True,
            "error": "",
        }
        if canonical is None:
            row["sample_status"] = "invalid_smiles"
        else:
            molecule_id = stable_molecule_id(canonical)
            row["molecule_id"] = molecule_id
            if canonical not in property_cache:
                property_cache[canonical] = _candidate_properties(
                    canonical,
                    minimum_qed=minimum_qed,
                    maximum_absolute_formal_charge=maximum_absolute_formal_charge,
                    minimum_fraction_csp3=minimum_fraction_csp3,
                    maximum_aromatic_rings=maximum_aromatic_rings,
                )
            (
                lipinski_pass,
                qed,
                formal_charge,
                fraction_csp3,
                aromatic_ring_count,
                chemistry_pass,
            ) = property_cache[canonical]
            row["lipinski_pass"] = lipinski_pass
            row["qed"] = qed
            row["formal_charge"] = formal_charge
            row["fraction_csp3"] = fraction_csp3
            row["aromatic_ring_count"] = aromatic_ring_count
            row["chemistry_pass"] = chemistry_pass
            if not reward_seen_molecules and canonical in batch_seen:
                row["sample_status"] = "batch_duplicate"
            elif not reward_seen_molecules and canonical in previously_seen:
                row["sample_status"] = "seen_duplicate"
            else:
                if canonical in batch_seen:
                    row["sample_status"] = "batch_repeat"
                elif canonical in previously_seen:
                    row["sample_status"] = "seen_repeat"
                molecule_rows.setdefault(molecule_id, []).append(index)
                if molecule_id not in score_cache:
                    molecules[molecule_id] = canonical
            batch_seen.add(canonical)
        rows.append(row)

    oracle_outcomes = oracle.dock(molecules, output_dir / "docking") if molecules else {}
    new_cache_entries: list[tuple[str, str, float]] = []
    successful_scores: list[float] = []
    successful_unique: set[str] = set()
    elite_unique: set[str] = set()
    qualified_elite_unique: set[str] = set()
    elite_threshold = empirical_quantile(reference_scores, elite_fraction)
    for molecule_id, row_indices in molecule_rows.items():
        canonical = str(rows[row_indices[0]]["canonical_smiles"])
        cached = score_cache.get(molecule_id)
        if cached is not None:
            cached_smiles, cached_score = cached
            if cached_smiles != canonical:
                raise RuntimeError(f"score-cache molecule ID collision for {molecule_id}")
            outcome: DockingOutcome | None = DockingOutcome(
                molecule_id=molecule_id,
                canonical_smiles=canonical,
                status="success",
                score=cached_score,
                error="",
            )
            score_source = "cache"
        else:
            outcome = oracle_outcomes.get(molecule_id)
            score_source = "oracle"
        if outcome is None:
            for row_index in row_indices:
                rows[row_index]["sample_status"] = "oracle_missing"
                rows[row_index]["used_for_gradient"] = False
            continue
        if outcome.status == "success" and outcome.score is not None:
            reward = _docking_desirability(
                reference_scores,
                outcome.score,
                reward_mode=reward_mode,
                tail_fraction=tail_fraction,
                tail_weight=tail_weight,
                elite_fraction=elite_fraction,
            )
            if require_lipinski and not bool(rows[row_indices[0]]["lipinski_pass"]):
                reward = 0.0
            if not bool(rows[row_indices[0]]["chemistry_pass"]):
                reward = 0.0
            if cached is None:
                cached_value = float(outcome.score)
                score_cache[molecule_id] = (canonical, cached_value)
                new_cache_entries.append((molecule_id, canonical, cached_value))
            for row_index in row_indices:
                row = rows[row_index]
                row["docking_status"] = "success"
                row["docking_score"] = outcome.score
                row["score_source"] = score_source
                row["reward"] = reward
                if row["sample_status"] == "candidate":
                    row["sample_status"] = "cache_hit" if cached is not None else "scored"
                successful_scores.append(float(outcome.score))
            successful_unique.add(molecule_id)
            if float(outcome.score) <= elite_threshold:
                elite_unique.add(molecule_id)
                if (
                    bool(rows[row_indices[0]]["lipinski_pass"])
                    and bool(rows[row_indices[0]]["chemistry_pass"])
                ):
                    qualified_elite_unique.add(molecule_id)
        else:
            molecule_caused = outcome.status != "docking_failed"
            for row_index in row_indices:
                row = rows[row_index]
                row["sample_status"] = outcome.status or "oracle_failure"
                row["docking_status"] = outcome.status
                row["score_source"] = score_source
                row["error"] = outcome.error
                row["reward"] = 0.0
                row["used_for_gradient"] = molecule_caused

    gradient_capped_count = 0
    if reward_occurrence_cap is not None:
        if reward_occurrence_cap <= 0:
            raise ValueError("reward_occurrence_cap must be positive")
        for row_indices in molecule_rows.values():
            for row_index in row_indices[reward_occurrence_cap:]:
                if rows[row_index]["used_for_gradient"]:
                    rows[row_index]["used_for_gradient"] = False
                    gradient_capped_count += 1

    if persist_score_cache:
        append_score_cache(job_dir, new_cache_entries)

    rewards = torch.tensor(
        [float(row["reward"]) for row in rows],
        dtype=torch.float32,
        device=bundle.device,
    )
    gradient_mask = torch.tensor(
        [bool(row["used_for_gradient"]) for row in rows],
        dtype=torch.bool,
        device=bundle.device,
    )
    canonical_valid = [str(row["canonical_smiles"]) for row in rows if row["canonical_smiles"]]
    unique_valid = set(canonical_valid)
    scaffold_count, top_scaffold_fraction = _scaffold_summary(list(unique_valid))
    molecule_counts = Counter(canonical_valid)
    elite_count = sum(
        1
        for row in rows
        if row["docking_score"] != "" and float(row["docking_score"]) <= elite_threshold
    )
    lipinski_count = sum(row["lipinski_pass"] is True for row in rows)
    chemistry_count = sum(row["chemistry_pass"] is True for row in rows)
    qualified_elite_count = sum(
        1
        for row in rows
        if row["lipinski_pass"] is True
        and row["chemistry_pass"] is True
        and row["docking_score"] != ""
        and float(row["docking_score"]) <= elite_threshold
    )
    positive_score_count = sum(
        1
        for row in rows
        if row["docking_score"] != "" and float(row["docking_score"]) > 0.0
    )
    successful_count = len(successful_scores)
    metrics = {
        "raw_count": float(count),
        "valid_count": float(len(canonical_valid)),
        "valid_fraction": len(canonical_valid) / max(1, count),
        "lipinski_count": float(lipinski_count),
        "lipinski_fraction": lipinski_count / max(1, count),
        "chemistry_count": float(chemistry_count),
        "chemistry_fraction": chemistry_count / max(1, count),
        "unique_valid_count": float(len(unique_valid)),
        "unique_fraction": len(unique_valid) / max(1, len(canonical_valid)),
        "top_molecule_fraction": (
            max(molecule_counts.values()) / max(1, count) if molecule_counts else 0.0
        ),
        "scaffold_count": float(scaffold_count),
        "top_scaffold_fraction": top_scaffold_fraction,
        "docked": float(successful_count),
        "docked_unique": float(len(successful_unique)),
        "oracle_unique_count": float(len(molecules)),
        "cache_hit_count": float(
            sum(1 for row in rows if row["score_source"] == "cache")
        ),
        "gradient_count": float(sum(bool(row["used_for_gradient"]) for row in rows)),
        "gradient_capped_count": float(gradient_capped_count),
        "score_success_fraction": successful_count / max(1, count),
        "positive_score_fraction": positive_score_count / max(1, count),
        "score_mean": mean(successful_scores) if successful_scores else math.nan,
        "score_median": median(successful_scores) if successful_scores else math.nan,
        "elite_threshold": elite_threshold,
        "elite_count": float(elite_count),
        "elite_fraction": elite_count / max(1, count),
        "elite_unique_count": float(len(elite_unique)),
        "qualified_elite_count": float(qualified_elite_count),
        "qualified_elite_fraction": qualified_elite_count / max(1, count),
        "qualified_elite_unique_count": float(len(qualified_elite_unique)),
        # Every raw draw is represented. Operational failures remain masked
        # from the gradient but cannot make this evaluation metric look better.
        "reward_mean": sum(float(row["reward"]) for row in rows) / max(1, count),
    }
    if successful_scores:
        ordered_scores = sorted(successful_scores)
        metrics.update(
            {
                "score_min": ordered_scores[0],
                "score_best_10_mean": mean(ordered_scores[:10]),
                "score_p01": empirical_quantile(ordered_scores, 0.01),
                "score_p10": empirical_quantile(ordered_scores, 0.10),
                "score_p90": empirical_quantile(ordered_scores, 0.90),
                "score_p95": empirical_quantile(ordered_scores, 0.95),
                "score_max": ordered_scores[-1],
            }
        )
    _write_sample_rows(output_dir / "samples.csv", rows)
    return tokens, rewards, gradient_mask, rows, metrics


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _backward_sequence_objective(
    bundle: PolicyBundle,
    tokens: torch.Tensor,
    advantages: torch.Tensor,
    gradient_mask: torch.Tensor,
    *,
    kl_beta: float,
    temperature: float,
    top_k: int | None,
    microbatch_size: int,
) -> dict[str, float | int]:
    """Backpropagate the unchanged full-batch mean objective in slices."""

    batch_size = len(tokens)
    gradient_count = int(gradient_mask.sum().item())
    if batch_size <= 0 or gradient_count <= 0 or microbatch_size <= 0:
        raise ValueError("sequence objective requires non-empty positive dimensions")
    policy_loss = 0.0
    kl = 0.0
    entropy = 0.0
    for begin in range(0, batch_size, microbatch_size):
        stop = min(batch_size, begin + microbatch_size)
        local_mask = gradient_mask[begin:stop]
        statistics = sequence_statistics(
            bundle.policy,
            bundle.prior,
            tokens[begin:stop],
            eos_idx=bundle.vocab.eos_idx,
            temperature=temperature,
            top_k=top_k,
        )
        if bool(local_mask.any()):
            policy_piece = -(
                advantages[begin:stop][local_mask]
                * statistics.sequence_log_probability[local_mask]
            ).sum() / gradient_count
        else:
            policy_piece = statistics.sequence_log_probability.sum() * 0.0
        kl_piece = statistics.kl_per_token.sum() / batch_size
        loss_piece = policy_piece + kl_beta * kl_piece
        if not bool(torch.isfinite(loss_piece)):
            raise RuntimeError("non-finite microbatched sequence objective")
        loss_piece.backward()
        policy_loss += float(policy_piece.detach().cpu())
        kl += float(kl_piece.detach().cpu())
        entropy += float(statistics.entropy_per_token.detach().sum().cpu()) / batch_size
    return {
        "loss": policy_loss + kl_beta * kl,
        "policy_loss": policy_loss,
        "kl_per_token": kl,
        "entropy_per_token": entropy,
        "microbatch_size": microbatch_size,
    }


def _backward_sequence_objective_adaptive(
    bundle: PolicyBundle,
    tokens: torch.Tensor,
    advantages: torch.Tensor,
    gradient_mask: torch.Tensor,
    *,
    kl_beta: float,
    temperature: float,
    top_k: int | None,
) -> dict[str, float | int]:
    microbatch_size = min(SEQUENCE_MICROBATCH_MAX, len(tokens))
    while True:
        bundle.policy.zero_grad(set_to_none=True)
        try:
            return _backward_sequence_objective(
                bundle,
                tokens,
                advantages,
                gradient_mask,
                kl_beta=kl_beta,
                temperature=temperature,
                top_k=top_k,
                microbatch_size=microbatch_size,
            )
        except RuntimeError as exc:
            recoverable = any(
                marker in str(exc).lower()
                for marker in (
                    "out of memory",
                    "failed to allocate",
                    "cublas_status_alloc_failed",
                    "launch out of resources",
                )
            )
            if not recoverable or microbatch_size <= 1:
                raise
            bundle.policy.zero_grad(set_to_none=True)
            if bundle.device.type == "cuda":
                torch.cuda.empty_cache()
            microbatch_size = max(1, microbatch_size // 2)
            print(
                f"[igenvs-rl] retrying sequence objective with microbatch "
                f"{microbatch_size}",
                flush=True,
            )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _atomic_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _update_number(path: Path) -> int | None:
    value = path.name.removeprefix("update-")
    return int(value) if value.isdigit() else None


def _reconcile_completed_state(job_dir: Path, checkpoint_update: int) -> None:
    """Derive lightweight state only from checkpoint-authorized updates."""

    history_path = job_dir / "history.csv"
    history_fields: list[str] = []
    histories: dict[int, dict[str, Any]] = {}
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            history_fields = list(reader.fieldnames or [])
            for row in reader:
                try:
                    update = int(row["update"])
                except (KeyError, TypeError, ValueError):
                    # A legacy append may have been interrupted mid-row.
                    # Durable completions below still have to cover every
                    # checkpoint-authorized update; no history is invented.
                    continue
                if 0 < update <= checkpoint_update and None not in row and None not in row.values():
                    histories[update] = dict(row)
    updates_root = job_dir / "updates"
    successful: set[str] = set()
    sampled_updates: set[int] = set()
    if updates_root.is_dir():
        for update_dir in sorted(updates_root.glob("update-*")):
            update = _update_number(update_dir)
            if update is None or update > checkpoint_update:
                continue
            completion_path = update_dir / "completion.json"
            if completion_path.is_file():
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                histories[update] = dict(completion["history"])
            samples_path = update_dir / "samples.csv"
            if samples_path.is_file():
                sampled_updates.add(update)
                with samples_path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if row.get("docking_status") == "success" and row.get(
                            "canonical_smiles"
                        ):
                            successful.add(row["canonical_smiles"])
    if checkpoint_update and set(histories) != set(range(1, checkpoint_update + 1)):
        missing = sorted(set(range(1, checkpoint_update + 1)).difference(histories))
        raise RuntimeError(
            f"checkpoint update {checkpoint_update} lacks durable history for updates {missing}"
        )
    if checkpoint_update and sampled_updates != set(range(1, checkpoint_update + 1)):
        missing = sorted(set(range(1, checkpoint_update + 1)).difference(sampled_updates))
        raise RuntimeError(
            f"checkpoint update {checkpoint_update} lacks sample records for updates {missing}"
        )
    ordered_histories = [histories[index] for index in sorted(histories)]
    for row in ordered_histories:
        for field in row:
            if field not in history_fields:
                history_fields.append(field)
    if ordered_histories:
        _atomic_csv_rows(history_path, ordered_histories, history_fields)
    elif history_path.exists():
        history_path.unlink()

    seen_path = job_dir / "seen.smi"
    seen_tmp = seen_path.with_suffix(".smi.tmp")
    with seen_tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for value in sorted(successful):
            handle.write(value + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    seen_tmp.replace(seen_path)

    evaluations_path = job_dir / "evaluations.csv"
    if evaluations_path.is_file():
        with evaluations_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            evaluation_fields = list(reader.fieldnames or [])
            evaluations = []
            for row in reader:
                if None in row or None in row.values():
                    continue
                label = row.get("label", "")
                suffix = label.removeprefix("update-")
                if not suffix.isdigit() or int(suffix) <= checkpoint_update:
                    evaluations.append(dict(row))
        _atomic_csv_rows(evaluations_path, evaluations, evaluation_fields)


def _publish_progress(job_dir: Path, update: int, *, model_latest_exported: bool) -> None:
    _atomic_json(
        job_dir / "progress.json",
        {
            "schema_version": 1,
            "status": "complete",
            "completed_updates": update,
            "model_latest_exported": model_latest_exported,
        },
    )


def _save_checkpoint(
    path: Path,
    *,
    bundle: PolicyBundle,
    optimizer: torch.optim.Optimizer,
    update: int,
    kl_beta: float,
    best_reward: float,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "update": update,
        "policy_state": _cpu_state_dict(bundle.policy),
        "optimizer_state": optimizer.state_dict(),
        "kl_beta": kl_beta,
        "best_reward": best_reward,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_initial_policy_state(config: JobConfig) -> dict[str, torch.Tensor] | None:
    if config.initial_model_root is None:
        return None
    spec = resolve_model(config.model_id)
    payload = torch.load(
        spec.weights_path(Path(config.initial_model_root)),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("the initial iGen3 checkpoint is not a state dictionary")
    return payload


def _restore_rng(payload: dict[str, Any]) -> None:
    torch.set_rng_state(payload["torch_rng_state"])
    random.setstate(payload["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])


def _export_i_gen3_model(
    job_dir: Path,
    config: JobConfig,
    bundle: PolicyBundle,
    *,
    directory: str = "model",
) -> None:
    _export_policy_state(job_dir, config, bundle.spec, _cpu_state_dict(bundle.policy), directory)


def _export_policy_state(job_dir, config, spec, policy_state, directory: str) -> None:
    """Publish a checkpoint's weights without constructing models or using RNG."""
    destination = job_dir / directory / config.model_id.replace("-", "_")
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / spec.weights_name
    weights_temporary = weights.with_suffix(weights.suffix + ".tmp")
    with weights_temporary.open("wb") as handle:
        torch.save(policy_state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    weights_temporary.replace(weights)
    _atomic_copy(
        spec.vocab_path(Path(config.model_root)),
        destination / spec.vocab_name,
    )


def recover_job(job_dir: Path, config: JobConfig) -> int:
    """Reconcile checkpoint-authorized state without sampling or training."""
    job_dir = job_dir.resolve()
    latest_path = job_dir / "checkpoints/latest.pt"
    checkpoint = _load_checkpoint(latest_path)
    update = int(checkpoint["update"]) if checkpoint else 0
    _reconcile_completed_state(job_dir, update)
    # Invalidate any previous export claim before the multi-file repair.
    _publish_progress(job_dir, update, model_latest_exported=False)
    if checkpoint is not None:
        completion_path = job_dir / "updates" / f"update-{update:04d}" / "completion.json"
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("promote_best"):
                _atomic_copy(latest_path, job_dir / "checkpoints/best.pt")
        spec = resolve_model(config.model_id)
        _export_policy_state(job_dir, config, spec, checkpoint["policy_state"], "model-latest")
        del checkpoint
        best = _load_checkpoint(job_dir / "checkpoints/best.pt")
        if best is not None:
            if int(best["update"]) > update:
                raise RuntimeError("best checkpoint is ahead of the authoritative latest checkpoint")
            _export_policy_state(job_dir, config, spec, best["policy_state"], "model")
        _publish_progress(job_dir, update, model_latest_exported=True)
    return update


def _evaluate(
    *,
    job_dir: Path,
    config: JobConfig,
    bundle: PolicyBundle,
    oracle,
    reference_scores: list[float],
    label: str,
    count: int,
    score_cache: dict[str, tuple[str, float]],
) -> dict[str, float]:
    evaluation_cache = {} if config.fresh_evaluation_docking else score_cache
    _, _, _, _, metrics = _sample_and_score(
        bundle=bundle,
        oracle=oracle,
        reference_scores=reference_scores,
        count=count,
        output_dir=job_dir / "evaluations" / label,
        job_dir=job_dir,
        temperature=config.temperature,
        top_k=config.top_k or None,
        previously_seen=set(),
        score_cache=evaluation_cache,
        reward_mode=config.reward_mode,
        tail_fraction=config.tail_fraction,
        tail_weight=config.tail_weight,
        reward_seen_molecules=config.reward_seen_molecules,
        elite_fraction=config.elite_fraction,
        require_lipinski=config.require_lipinski,
        minimum_qed=config.minimum_qed,
        maximum_absolute_formal_charge=config.maximum_absolute_formal_charge,
        minimum_fraction_csp3=config.minimum_fraction_csp3,
        maximum_aromatic_rings=config.maximum_aromatic_rings,
        reward_occurrence_cap=None,
        persist_score_cache=not config.fresh_evaluation_docking,
    )
    append_csv(job_dir / "evaluations.csv", {"label": label, **metrics})
    return metrics


def train_job(
    job_dir: Path,
    config: JobConfig,
    *,
    updates: int | None = None,
    target_update: int | None = None,
) -> None:
    if (updates is None) == (target_update is None):
        raise ValueError("provide exactly one of updates or target_update")
    if updates is not None and updates <= 0:
        raise ValueError("updates must be positive")
    if target_update is not None and target_update <= 0:
        raise ValueError("target_update must be positive")
    job_dir = job_dir.resolve()
    latest_path = job_dir / "checkpoints" / "latest.pt"
    checkpoint = _load_checkpoint(latest_path)
    policy_state = checkpoint["policy_state"] if checkpoint else _load_initial_policy_state(config)

    _seed_everything(config.generator_seed)
    docking_oracle = _oracle(config)
    reference_scores = _read_reference(job_dir)
    bundle: PolicyBundle | None = None
    if reference_scores is None:
        base_bundle = load_policy_bundle(
            model_root=Path(config.model_root),
            model_id=config.model_id,
        )
        reference_scores = _ensure_reference(job_dir, config, base_bundle, docking_oracle)
        if policy_state is None:
            bundle = base_bundle
    if bundle is None:
        bundle = load_policy_bundle(
            model_root=Path(config.model_root),
            model_id=config.model_id,
            policy_state=policy_state,
        )
    optimizer = torch.optim.Adam(bundle.policy.parameters(), lr=config.learning_rate)
    start_update = 0
    kl_beta = config.kl_beta
    best_reward = -math.inf
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _optimizer_to_device(optimizer, bundle.device)
        start_update = int(checkpoint["update"])
        kl_beta = float(checkpoint["kl_beta"])
        best_reward = float(checkpoint.get("best_reward", -math.inf))
        _restore_rng(checkpoint)
        print(f"[igenvs-rl] resumed after update {start_update}", flush=True)
    elif config.initial_model_root is not None:
        print(f"[igenvs-rl] warm-started policy from {config.initial_model_root}", flush=True)

    stop_update = (
        int(target_update)
        if target_update is not None
        else start_update + int(updates)
    )
    if stop_update < start_update:
        raise ValueError(
            f"checkpoint is already at update {start_update}, beyond target {stop_update}"
        )
    prior_progress = {}
    progress_path = job_dir / "progress.json"
    if progress_path.is_file():
        prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    _reconcile_completed_state(job_dir, start_update)
    prior_exported = bool(
        prior_progress.get("completed_updates") == start_update
        and prior_progress.get("model_latest_exported")
    )
    _publish_progress(
        job_dir, start_update, model_latest_exported=prior_exported
    )

    if checkpoint and start_update > 0:
        completion_path = job_dir / "updates" / f"update-{start_update:04d}" / "completion.json"
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("promote_best"):
                best_checkpoint = _load_checkpoint(job_dir / "checkpoints" / "best.pt")
                if best_checkpoint is None or int(best_checkpoint["update"]) != start_update:
                    _atomic_copy(latest_path, job_dir / "checkpoints" / "best.pt")
                # The checkpoint and exported iGen3 files are separate atomic
                # publications. Re-export even when best.pt already landed so
                # a crash between those publications is fully recoverable.
                _export_i_gen3_model(job_dir, config, bundle)

    seen = load_seen_smiles(job_dir)
    score_cache = load_score_cache(job_dir)

    for update in range(start_update + 1, stop_update + 1):
        started = perf_counter()
        update_dir = job_dir / "updates" / f"update-{update:04d}"
        tokens, rewards, gradient_mask, rows, metrics = _sample_and_score(
            bundle=bundle,
            oracle=docking_oracle,
            reference_scores=reference_scores,
            count=config.batch_size,
            output_dir=update_dir,
            job_dir=job_dir,
            temperature=config.temperature,
            top_k=config.top_k or None,
            previously_seen=seen,
            score_cache=score_cache,
            reward_mode=config.reward_mode,
            tail_fraction=config.tail_fraction,
            tail_weight=config.tail_weight,
            reward_seen_molecules=config.reward_seen_molecules,
            elite_fraction=config.elite_fraction,
            require_lipinski=config.require_lipinski,
            minimum_qed=config.minimum_qed,
            maximum_absolute_formal_charge=config.maximum_absolute_formal_charge,
            minimum_fraction_csp3=config.minimum_fraction_csp3,
            maximum_aromatic_rings=config.maximum_aromatic_rings,
            reward_occurrence_cap=config.reward_occurrence_cap,
        )
        if not bool(gradient_mask.any()):
            raise RuntimeError("no authoritative rewards were available for this update")

        advantages = normalized_advantages(rewards, gradient_mask)
        optimizer.zero_grad(set_to_none=True)
        objective = _backward_sequence_objective_adaptive(
            bundle,
            tokens,
            advantages,
            gradient_mask,
            kl_beta=kl_beta,
            temperature=config.temperature,
            top_k=config.top_k or None,
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            bundle.policy.parameters(),
            config.max_grad_norm,
        )
        if not bool(torch.isfinite(gradient_norm)):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"non-finite gradient at update {update}")
        optimizer.step()
        synchronize_sampler(bundle)

        successful = {
            str(row["canonical_smiles"])
            for row in rows
            if row["docking_status"] == "success"
        }
        measured_kl = float(objective["kl_per_token"])
        if measured_kl > config.target_kl:
            kl_beta = min(1.0, kl_beta * 1.5)

        evaluation_reward = math.nan
        evaluation_elite_fraction = math.nan
        evaluation_qualified_elite_fraction = math.nan
        if config.evaluation_every > 0 and update % config.evaluation_every == 0:
            evaluation = _evaluate(
                job_dir=job_dir,
                config=config,
                bundle=bundle,
                oracle=docking_oracle,
                reference_scores=reference_scores,
                label=f"update-{update:04d}",
                count=config.evaluation_count,
                score_cache=score_cache,
            )
            evaluation_reward = evaluation["reward_mean"]
            evaluation_elite_fraction = evaluation["elite_fraction"]
            evaluation_qualified_elite_fraction = evaluation["qualified_elite_fraction"]

        if config.evaluation_every > 0:
            selection_metrics = evaluation if math.isfinite(evaluation_reward) else None
        else:
            selection_metrics = metrics
        if selection_metrics is None:
            selection_reward = math.nan
        elif config.reward_mode in {"elite", "binary-elite"}:
            fraction_key = (
                "qualified_elite_fraction" if config.require_lipinski else "elite_fraction"
            )
            unique_key = (
                "qualified_elite_unique_count"
                if config.require_lipinski
                else "elite_unique_count"
            )
            checkpoint_eligible = (
                selection_metrics[unique_key] >= config.minimum_elite_unique
                and selection_metrics["top_molecule_fraction"]
                <= config.maximum_top_molecule_fraction
            )
            selection_reward = (
                selection_metrics[fraction_key] if checkpoint_eligible else -math.inf
            )
        else:
            selection_reward = selection_metrics["reward_mean"]
        promote_best = math.isfinite(selection_reward) and selection_reward > best_reward
        history_row = {
            "update": update,
            **metrics,
            "loss": float(objective["loss"]),
            "policy_loss": float(objective["policy_loss"]),
            "kl_per_token": measured_kl,
            "entropy_per_token": float(objective["entropy_per_token"]),
            "sequence_microbatch_size": int(objective["microbatch_size"]),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "kl_beta": kl_beta,
            "evaluation_reward_mean": evaluation_reward,
            "evaluation_elite_fraction": evaluation_elite_fraction,
            "evaluation_qualified_elite_fraction": evaluation_qualified_elite_fraction,
            "seconds": perf_counter() - started,
        }
        _atomic_json(
            update_dir / "completion.json",
            {
                "schema_version": 1,
                "status": "ready_for_checkpoint",
                "update": update,
                "history": history_row,
                "successful_smiles": sorted(successful),
                "promote_best": promote_best,
            },
        )
        _save_checkpoint(
            latest_path,
            bundle=bundle,
            optimizer=optimizer,
            update=update,
            kl_beta=kl_beta,
            best_reward=max(best_reward, selection_reward),
        )
        _reconcile_completed_state(job_dir, update)
        _publish_progress(job_dir, update, model_latest_exported=False)
        seen.update(successful)
        if promote_best:
            best_reward = selection_reward
            _atomic_copy(latest_path, job_dir / "checkpoints" / "best.pt")
            _export_i_gen3_model(job_dir, config, bundle)

        tail_summary = ""
        if "score_min" in metrics:
            tail_summary = (
                f" min={metrics['score_min']:.3f} p10={metrics['score_p10']:.3f}"
                f" p90={metrics['score_p90']:.3f}"
            )
        print(
            f"[igenvs-rl] update={update} docked={int(metrics['docked'])} "
            f"reward={metrics['reward_mean']:.3f} score={metrics['score_median']:.3f} "
            f"elite={metrics['elite_fraction']:.3f}/{int(metrics['elite_unique_count'])} "
            f"qualified={metrics['qualified_elite_fraction']:.3f}/"
            f"{int(metrics['qualified_elite_unique_count'])} "
            f"valid={metrics['valid_fraction']:.3f} unique={metrics['unique_fraction']:.3f} "
            f"kl={measured_kl:.5f}{tail_summary}",
            flush=True,
        )

    # Multi-stage protocols advance from the policy after a fixed number of
    # updates. Keep that hand-off separate from ``model/``, which remains the
    # best checkpoint selected by fresh evaluation.
    _export_i_gen3_model(job_dir, config, bundle, directory="model-latest")
    _publish_progress(job_dir, stop_update, model_latest_exported=True)


def evaluate_job(
    job_dir: Path,
    config: JobConfig,
    *,
    count: int,
    checkpoint_name: str = "best",
    seed: int | None = None,
) -> dict[str, float]:
    job_dir = job_dir.resolve()
    if checkpoint_name not in {"base", "best", "latest"}:
        raise ValueError("checkpoint_name must be base, best, or latest")
    update = 0
    policy_state = None
    if checkpoint_name != "base":
        checkpoint = _load_checkpoint(job_dir / "checkpoints" / f"{checkpoint_name}.pt")
        if checkpoint is None:
            raise FileNotFoundError(f"no {checkpoint_name} checkpoint exists")
        update = int(checkpoint["update"])
        policy_state = checkpoint["policy_state"]
    seed_offset = {"base": 300_000, "best": 400_000, "latest": 500_000}[checkpoint_name]
    _seed_everything(seed if seed is not None else config.generator_seed + update + seed_offset)
    bundle = load_policy_bundle(
        model_root=Path(config.model_root),
        model_id=config.model_id,
        policy_state=policy_state,
    )
    reference_scores = _read_reference(job_dir)
    if reference_scores is None:
        raise FileNotFoundError("the frozen docking reference is missing")
    label = f"manual-{checkpoint_name}-{time_ns()}"
    return _evaluate(
        job_dir=job_dir,
        config=config,
        bundle=bundle,
        oracle=_oracle(config),
        reference_scores=reference_scores,
        label=label,
        count=count,
        score_cache=load_score_cache(job_dir),
    )


def generate_job(
    job_dir: Path,
    config: JobConfig,
    *,
    count: int,
    output: Path,
    allow_repeats: bool = False,
) -> None:
    checkpoint = _load_checkpoint(job_dir / "checkpoints" / "best.pt")
    if checkpoint is None:
        checkpoint = _load_checkpoint(job_dir / "checkpoints" / "latest.pt")
    if checkpoint is None:
        raise FileNotFoundError("no trained checkpoint exists")
    _seed_everything(config.generator_seed + int(checkpoint["update"]) + 200_000)
    bundle = load_policy_bundle(
        model_root=Path(config.model_root),
        model_id=config.model_id,
        policy_state=checkpoint["policy_state"],
    )
    if allow_repeats:
        generated: list[str] = []
        candidates = 0
        candidate_limit = max(count * 10, count + config.batch_size)
        while len(generated) < count and candidates < candidate_limit:
            batch_count = min(
                config.batch_size,
                count - len(generated),
                candidate_limit - candidates,
            )
            _, raw_smiles = sample_policy(
                bundle,
                batch_count,
                temperature=config.temperature,
                top_k=config.top_k or None,
            )
            candidates += batch_count
            for raw in raw_smiles:
                canonical = canonicalize_valid_smiles(raw, isomeric_smiles=True)
                if canonical is not None:
                    generated.append(canonical)
                    if len(generated) == count:
                        break
        if len(generated) != count:
            raise RuntimeError(
                f"generated only {len(generated)} valid occurrences from {candidates} candidates"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(generated) + "\n", encoding="utf-8")
        print(
            f"[igenvs-rl] wrote {len(generated)} valid molecule occurrences "
            f"({len(set(generated))} unique) to {output}",
            flush=True,
        )
    else:
        stats = write_de_novo_file(
            bundle.sampler,
            output_path=output,
            count=count,
            batch_size=min(config.batch_size, count),
            temperature=config.temperature,
            do_sample=True,
            top_k=config.top_k or None,
            progress=True,
        )
        print(
            f"[igenvs-rl] wrote {stats.generated} valid unique molecules to {output}",
            flush=True,
        )

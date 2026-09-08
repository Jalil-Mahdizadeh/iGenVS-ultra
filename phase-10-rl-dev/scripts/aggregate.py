#!/usr/bin/env python3
"""Audit and summarize the complete universal-protocol development panel."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCORE_FIELDS = (
    "score_min",
    "score_best_10_mean",
    "score_p01",
    "score_p10",
    "score_median",
    "score_p90",
    "score_p95",
)
PROPERTY_FIELDS = (
    "valid_fraction_raw",
    "unique_valid_count",
    "unique_valid_fraction",
    "top_molecule_fraction_raw",
    "lipinski_fraction_raw",
    "unique_scaffold_count",
    "top_scaffold_fraction_raw",
    "internal_diversity",
    "qed_mean",
    "sa_score_mean",
    "mol_weight_mean",
    "logp_mean",
    "tpsa_mean",
    "hbd_mean",
    "hba_mean",
    "rotatable_bonds_mean",
    "heavy_atoms_mean",
    "ring_count_mean",
    "aromatic_ring_count_mean",
    "fraction_csp3_mean",
    "formal_charge_mean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired_ok(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        indices = [int(row["sample_index"]) for row in csv.DictReader(handle)]
    return indices == list(range(expected))


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def _failed_gates(summary: dict[str, Any]) -> str:
    return ", ".join(
        key for key, passed in summary["acceptance"].items()
        if key != "passed" and not passed
    ) or "none"


def _history_updates(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["update"]) if rows else 0


def main() -> None:
    phase = Path(__file__).resolve().parents[1]
    repo = phase.parent
    protocol_path = phase / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = _sha256(protocol_path)
    targets = (phase / "targets.txt").read_text(encoding="utf-8").split()
    expected_targets = int(protocol["acceptance"]["required_development_passes"])
    if len(targets) != expected_targets or len(set(targets)) != len(targets):
        raise RuntimeError("development target list does not match the acceptance contract")
    count = int(protocol["validation"]["raw_draws_per_arm"])
    stages = protocol["stages"]
    final_stage = stages[-1]["name"]

    summaries: dict[str, dict[str, Any]] = {}
    timings: dict[str, dict[str, Any]] = {}
    stoppings: dict[str, dict[str, Any]] = {}
    audit_targets: dict[str, Any] = {}
    base_hashes: dict[str, str] = {}

    for target in targets:
        root = phase / target
        required = {
            "run": root / "run.json",
            "summary": root / "validation/summary.json",
            "timing": root / "training/timing.json",
            "stopping": root / f"training/{final_stage}/stopping.json",
            "base_samples": root / "validation/samples/base.csv",
            "paired": root / f"validation/{target}_base-vs-rl_10k.csv",
        }
        absent = [name for name, path in required.items() if not path.is_file()]
        if absent:
            raise RuntimeError(f"{target}: incomplete outputs: {', '.join(absent)}")

        run = json.loads(required["run"].read_text(encoding="utf-8"))
        summary = json.loads(required["summary"].read_text(encoding="utf-8"))
        timing = json.loads(required["timing"].read_text(encoding="utf-8"))
        stopping = json.loads(required["stopping"].read_text(encoding="utf-8"))
        prepared_manifest = (
            repo / protocol["prepared_target_root"] / target / "manifest.json"
        )
        manifests = sorted(
            (root / "validation/docking").glob("*/*/shard-*/manifest.json")
        )
        model = (
            root
            / f"training/{final_stage}/model/base_isomeric/"
            "iGen3_base_isomeric_256d.pth"
        )
        stage_updates = {
            stage["name"]: _history_updates(
                root / "training" / stage["name"] / "history.csv"
            )
            for stage in stages
        }
        fixed_updates_ok = all(
            stage_updates[stage["name"]] == int(stage["updates"])
            for stage in stages
            if "adaptive_stopping" not in stage
        )
        adaptive_stage_checks: dict[str, bool] = {}
        adaptive_gate_checks: dict[str, bool] = {}
        for stage in stages:
            if "adaptive_stopping" not in stage:
                continue
            stage_stopping_path = root / f'training/{stage["name"]}/stopping.json'
            if not stage_stopping_path.is_file():
                adaptive_stage_checks[stage["name"]] = False
                adaptive_gate_checks[stage["name"]] = False
                continue
            stage_stopping = json.loads(stage_stopping_path.read_text(encoding="utf-8"))
            adaptive = stage["adaptive_stopping"]
            adaptive_stage_checks[stage["name"]] = (
                int(adaptive["minimum_updates"])
                <= stage_updates[stage["name"]]
                <= int(adaptive["maximum_updates"])
                and stage_updates[stage["name"]]
                == int(stage_stopping["stopped_at_update"])
            )
            adaptive_gate_checks[stage["name"]] = bool(stage_stopping["gate_met"])
        checks = {
            "protocol_hash_matches_run": run["protocol_sha256"] == protocol_hash,
            "protocol_hash_matches_validation": (
                summary["protocol"]["protocol_sha256"] == protocol_hash
            ),
            "prepared_target_hash_matches": (
                run["prepared_target_manifest_sha256"] == _sha256(prepared_manifest)
            ),
            "target_identity_matches": (
                run["target"] == target and summary["target"] == target
            ),
            "fixed_stage_updates_match": fixed_updates_ok,
            "adaptive_stage_updates_valid": all(adaptive_stage_checks.values()),
            "all_online_gates_met": all(adaptive_gate_checks.values()),
            "selected_checkpoint_is_qualified_best": (
                summary["rl_checkpoint"]["selection"] == "best"
            ),
            "selected_checkpoint_hash_matches": (
                _sha256(Path(summary["rl_checkpoint"]["path"]))
                == summary["rl_checkpoint"]["sha256"]
            ),
            "exported_model_exists": model.is_file(),
            "paired_csv_has_contiguous_rows": _paired_ok(required["paired"], count),
            "all_16_validation_manifests_complete": (
                len(manifests) == 16
                and all(
                    json.loads(path.read_text(encoding="utf-8")).get("status")
                    == "complete"
                    for path in manifests
                )
            ),
            "independent_acceptance_passed": bool(summary["acceptance"]["passed"]),
        }
        audit_targets[target] = {
            "passed": all(checks.values()),
            "checks": checks,
            "stage_updates": stage_updates,
            "failed_acceptance_gates": _failed_gates(summary),
        }
        base_hashes[target] = _sha256(required["base_samples"])
        summaries[target] = summary
        timings[target] = timing
        stoppings[target] = stopping

    audit = {
        "protocol_sha256": protocol_hash,
        "expected_target_count": expected_targets,
        "base_raw_sample_sha256": base_hashes,
        "base_raw_samples_identical": len(set(base_hashes.values())) == 1,
        "targets": audit_targets,
    }
    audit["passed"] = audit["base_raw_samples_identical"] and all(
        entry["passed"] for entry in audit_targets.values()
    )
    (phase / "integrity-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    docking_rows: list[dict[str, object]] = []
    property_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for target in targets:
        summary = summaries[target]
        stopping = stoppings[target]
        for mode in protocol["acceptance"]["required_search_modes"]:
            base = summary["docking"][mode]["base"]
            rl = summary["docking"][mode]["rl"]
            comparison = summary["comparison"][mode]
            row: dict[str, object] = {
                "target": target,
                "mode": mode,
                "accepted": summary["acceptance"]["passed"],
                "base_success_fraction_raw": base["successful_fraction_raw"],
                "rl_success_fraction_raw": rl["successful_fraction_raw"],
                "base_positive_fraction_raw": base["positive_score_fraction_raw"],
                "rl_positive_fraction_raw": rl["positive_score_fraction_raw"],
                "elite_threshold": rl["elite_threshold"],
                "base_qualified_elite_fraction_raw": base[
                    "qualified_elite_fraction_raw"
                ],
                "rl_qualified_elite_fraction_raw": rl[
                    "qualified_elite_fraction_raw"
                ],
                "rl_qualified_elite_unique": rl["qualified_elite_unique_count"],
                "rl_chemistry_pass_fraction_raw": rl[
                    "chemistry_pass_fraction_raw"
                ],
            }
            for field in SCORE_FIELDS:
                row[f"base_{field}"] = base[field]
                row[f"rl_{field}"] = rl[field]
                row[f"{field}_improvement"] = comparison[f"{field}_improvement"]
            docking_rows.append(row)

        for arm in ("base", "rl"):
            source = summary["molecular_and_structural"][arm]
            property_rows.append(
                {
                    "target": target,
                    "arm": arm,
                    **{field: source[field] for field in PROPERTY_FIELDS},
                }
            )

        timing = timings[target]
        timing_rows.append(
            {
                "target": target,
                "gpu_count": timing["gpu_count"],
                "final_stage_updates": stopping["stopped_at_update"],
                "selected_checkpoint_update": summary["rl_checkpoint"]["update"],
                "online_gate_met": stopping["gate_met"],
                "training_wall_seconds": timing["training_wall_seconds"],
                "training_wall_minutes": timing["training_wall_seconds"] / 60.0,
                "training_gpu_hours": timing["training_gpu_hours"],
                **{
                    f"{stage['stage']}_update_loop_seconds": stage[
                        "update_loop_seconds"
                    ]
                    for stage in timing["stages"]
                },
            }
        )

        model = (
            phase
            / target
            / f"training/{final_stage}/model/base_isomeric/"
            "iGen3_base_isomeric_256d.pth"
        )
        model_rows.append(
            {
                "target": target,
                "checkpoint_update": summary["rl_checkpoint"]["update"],
                "final_stage_stop_update": stopping["stopped_at_update"],
                "model_path": str(model.relative_to(phase)),
                "model_sha256": _sha256(model),
                "model_bytes": model.stat().st_size,
            }
        )

    _write_csv(phase / "docking-summary.csv", docking_rows)
    _write_csv(phase / "molecular-structural-summary.csv", property_rows)
    _write_csv(phase / "training-times.csv", timing_rows)
    _write_csv(phase / "model-manifest.csv", model_rows)

    accepted = [target for target in targets if summaries[target]["acceptance"]["passed"]]
    universal_pass = len(accepted) == expected_targets and bool(audit["passed"])
    total_seconds = sum(
        float(timing["training_wall_seconds"]) for timing in timings.values()
    )
    total_gpu_hours = sum(
        float(timing["training_gpu_hours"]) for timing in timings.values()
    )
    lines = [
        "# Eight-target universal RL protocol development",
        "",
        "## Decision",
        "",
        (
            "**Accepted: all 8/8 development targets passed every predeclared "
            "online, independent docking, diversity, and chemistry-safety gate.**"
            if universal_pass
            else f"**Not accepted: {len(accepted)}/{expected_targets} targets passed "
            "the independent endpoint; the benchmark remains locked.**"
        ),
        "",
        "The same receptor-agnostic recipe was used independently for every target. "
        "Each run began from `base-isomeric`; only the receptor and its target-local "
        "base docking references differed. A bounded percentile warm-up was followed "
        "by binary top-1% Uni-Dock/Vina `balance` concentration and binary top-0.5% "
        "`fast` refinement. Every stage retained an immutable base KL prior; the binary "
        "stages used chemistry qualification, repeat-capped reward gradients, and "
        "uncached checkpoint docking.",
        "",
        "Final evaluation used matched independent 10,000-draw base and RL samples. "
        "Invalid strings, repeats, docking failures, non-elites, and positive scores "
        "remain in raw denominators. Every distinct molecule was freshly docked in "
        "both Uni-Dock `fast` and `balance`.",
        "",
        "## Headline results",
        "",
        "| Target | Pass | Online stop | Stop / selected | Training min | Fast q-elite | Balance q-elite | Fast distinct | Top molecule | Chemistry | Fast gains best10 / median / p95 | Balance gains best10 / median / p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in targets:
        summary = summaries[target]
        stopping = stoppings[target]
        fast = summary["docking"]["fast"]["rl"]
        balance = summary["docking"]["balance"]["rl"]
        fast_gain = summary["comparison"]["fast"]
        balance_gain = summary["comparison"]["balance"]
        props = summary["molecular_and_structural"]["rl"]
        timing = timings[target]
        lines.append(
            "| "
            + " | ".join(
                [
                    target,
                    _fmt(summary["acceptance"]["passed"]),
                    _fmt(stopping["gate_met"]),
                    f'{stopping["stopped_at_update"]} / {summary["rl_checkpoint"]["update"]}',
                    _fmt(float(timing["training_wall_seconds"]) / 60.0, 1),
                    f'{100 * fast["qualified_elite_fraction_raw"]:.2f}%',
                    f'{100 * balance["qualified_elite_fraction_raw"]:.2f}%',
                    _fmt(fast["qualified_elite_unique_count"]),
                    f'{100 * props["top_molecule_fraction_raw"]:.2f}%',
                    f'{100 * fast["chemistry_pass_fraction_raw"]:.2f}%',
                    "/".join(
                        _fmt(fast_gain[f"{field}_improvement"])
                        for field in ("score_best_10_mean", "score_median", "score_p95")
                    ),
                    "/".join(
                        _fmt(balance_gain[f"{field}_improvement"])
                        for field in ("score_best_10_mean", "score_median", "score_p95")
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Acceptance details",
            "",
            "| Target | Failed gates | Fast positive | Balance positive | Unique valid | Unique scaffolds | Internal diversity | QED | fraction-Csp3 | Aromatic rings | Formal charge |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target in targets:
        summary = summaries[target]
        props = summary["molecular_and_structural"]["rl"]
        lines.append(
            "| "
            + " | ".join(
                [
                    target,
                    _failed_gates(summary),
                    f'{100 * summary["docking"]["fast"]["rl"]["positive_score_fraction_raw"]:.3f}%',
                    f'{100 * summary["docking"]["balance"]["rl"]["positive_score_fraction_raw"]:.3f}%',
                    _fmt(props["unique_valid_count"]),
                    _fmt(props["unique_scaffold_count"]),
                    _fmt(props["internal_diversity"]),
                    _fmt(props["qed_mean"]),
                    _fmt(props["fraction_csp3_mean"]),
                    _fmt(props["aromatic_ring_count_mean"]),
                    _fmt(props["formal_charge_mean"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Runtime and integrity",
            "",
            f"Recorded training time totals {_fmt(total_seconds / 3600.0, 2)} node-hours "
            f"and {_fmt(total_gpu_hours, 2)} allocated GPU-hours. Integrity audit: "
            f"**{'pass' if audit['passed'] else 'fail'}**.",
            "",
            "Docking scores are computational prioritization estimates, not measured "
            "binding affinities. The chemistry guards directly block the low-QED, "
            "high-charge, and excessively planar/aromatic failure modes seen previously, "
            "but they do not establish synthesis or activity. The single best score is "
            "reported in `docking-summary.csv` as a diagnostic; acceptance uses the "
            "best-10 mean because a one-sample minimum is an unstable extreme statistic.",
            "",
            "## Output index",
            "",
            "- `docking-summary.csv`: full fast/balance distribution comparisons.",
            "- `molecular-structural-summary.csv`: molecular properties and diversity.",
            "- `training-times.csv`: wall time and allocated GPU-hours.",
            "- `model-manifest.csv`: accepted model paths and hashes.",
            "- `integrity-audit.json`: protocol, target, row, checkpoint, and docking checks.",
            "- `<target>/validation/<target>_base-vs-rl_10k.csv`: paired raw libraries.",
            "",
        ]
    )
    (phase / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"wrote development report; accepted={len(accepted)}/{expected_targets}; "
        f"integrity={audit['passed']}"
    )


if __name__ == "__main__":
    main()

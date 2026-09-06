#!/usr/bin/env python3
"""Validate all cold cases and render the concise benchmark report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common import (
    PROTOCOL,
    SPEED,
    atomic_json,
    load_and_verify_protocol,
    load_json,
    sha256,
    utc_now,
)
from submit import cases


def number(value: float) -> str:
    return f"{value:,.0f}"


def seconds(value: float) -> str:
    return f"{value:,.3f}"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def load_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docking = []
    screening = []
    missing = []
    protocol_hash = sha256(PROTOCOL)
    for specification in cases():
        summary_path = (
            SPEED / "results" / specification["kind"] / specification["case"] / "summary.json"
        )
        if not summary_path.is_file():
            missing.append(specification["case"])
            continue
        summary = load_json(summary_path)
        if summary.get("status") != "complete":
            raise RuntimeError(f"case is not complete: {specification['case']}")
        if summary.get("sample") != "single cold run":
            raise RuntimeError(f"case is not a cold singleton: {specification['case']}")
        if summary.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"protocol mismatch: {specification['case']}")
        if int(summary.get("gpus", 0)) != int(specification["gpus"]):
            raise RuntimeError(f"GPU-count mismatch: {specification['case']}")
        (docking if specification["kind"] == "docking" else screening).append(summary)
    if missing:
        raise RuntimeError("benchmark cases are incomplete: " + ", ".join(missing))
    return docking, screening


def main() -> None:
    protocol = load_and_verify_protocol()
    docking, screening = load_cases()
    docking_rows = []
    docking_stage_rows = []
    for item in docking:
        docking_rows.append(
            [
                item["engine"],
                item["mode"],
                str(item["gpus"]),
                f"{item['input_rows']:,}",
                f"{item['successful_rows']:,}",
                f"{100 * item['yield']:.3f}%",
                seconds(item["complete_wall_seconds"]),
                number(item["input_per_hour_complete_wall"]),
                number(item["successful_per_hour_complete_wall"]),
                number(item["successful_per_hour_engine_wall"]),
            ]
        )
        stage = item["stage_timings"]
        docking_stage_rows.append(
            [
                item["case"],
                seconds(stage["target_setup_seconds"]),
                seconds(stage["validation_seconds"]),
                seconds(stage["preparation_wait_seconds_critical"]),
                seconds(stage["preparation_cpu_seconds_sum"]),
                seconds(stage["engine_wall_seconds_critical"]),
                seconds(stage["engine_worker_seconds_sum"]),
                seconds(stage["result_processing_seconds_critical"]),
                seconds(stage["cleanup_seconds_critical"]),
                seconds(stage["wrapper_docking_stage_seconds"]),
            ]
        )
    screening_rows = []
    screening_stage_rows = []
    for item in screening:
        stage = item["stage_timings"]
        screening_rows.append(
            [
                str(item["gpus"]),
                f"{item['committed_finite_scores']:,}",
                f"{100 * item['end_to_end_candidate_yield']:.3f}%",
                f"{100 * item['encoding_yield']:.3f}%",
                seconds(item["complete_wall_seconds"]),
                number(item["screening_per_hour_complete_wall"]),
                number(item["generation_candidate_slots_per_hour"]),
                number(item["encoding_per_hour"]),
                number(item["head_inference_per_hour"]),
            ]
        )
        screening_stage_rows.append(
            [
                str(item["gpus"]),
                str(stage["stream_batches"]),
                seconds(stage["worker_startup_seconds"]),
                seconds(stage["generation_seconds_critical_sum"]),
                seconds(stage["validation_seconds_sum"]),
                seconds(stage["admission_seconds_sum"]),
                seconds(stage["policy_seconds_critical_sum"]),
                seconds(stage["encoding_seconds_critical_sum"]),
                seconds(stage["head_inference_seconds_critical_sum"]),
                seconds(stage["score_stage_seconds_critical_sum"]),
                seconds(stage["batch_processing_wall_seconds"]),
                seconds(stage["worker_shutdown_seconds"]),
                seconds(stage["finalization_seconds"]),
            ]
        )
    lines = [
        "# iGenVS-ultra cold speed benchmark",
        "",
        f"Completed: {utc_now()}",
        "",
        (
            "Each row is one cold timing sample on Arrhenius GH200 120GB GPUs. "
            "No warm-up run and no repeated timing sample were used. All public-wrapper "
            "performance controls remained `auto`; docking used score-only output and "
            "screening used only the count argument with cross-batch overlap."
        ),
        "",
        "## Docking",
        "",
        *table(
            [
                "Engine", "Mode", "GPUs", "Input", "Finite success", "Yield",
                "Complete wall (s)", "Input/h", "Successful/h", "Engine-only successful/h",
            ],
            docking_rows,
        ),
        "",
        "Docking timings are seconds. Preparation CPU is summed across workers; engine "
        "worker time is summed across GPU shards. Critical values are the slowest concurrent shard.",
        "",
        *table(
            [
                "Case", "Target", "Validate", "Prep critical", "Prep CPU sum",
                "Engine critical", "Engine worker sum", "Results critical", "Cleanup critical",
                "Wrapper docking stage",
            ],
            docking_stage_rows,
        ),
        "",
        "## Ultra screening",
        "",
        *table(
            [
                "GPUs", "Committed finite scores", "Candidate yield", "Encoding yield",
                "Complete wall (s)", "Screening/h", "Candidate generation/h", "Encoding/h",
                "Head inference/h",
            ],
            screening_rows,
        ),
        "",
        "Screening timings are seconds and stage sums may exceed complete wall time because "
        "generation and scoring overlap across batches. GPU-sharded stage values use each "
        "batch's slowest shard before summing batches.",
        "",
        *table(
            [
                "GPUs", "Batches", "Startup", "Generate", "Validate", "Admit", "Policy",
                "Encode", "Heads", "Score stage", "Batch wall", "Shutdown", "Finalize",
            ],
            screening_stage_rows,
        ),
        "",
        "## Audit",
        "",
        f"- Protocol SHA-256: `{sha256(PROTOCOL)}`",
        f"- Locked at: `{protocol['locked_at']}`",
        "- Full machine snapshots, resolved automatic plans, commands, hashes, counts, and "
        "  unrounded measurements are retained in each case's `summary.json`.",
        "",
    ]
    report = SPEED / "REPORT.md"
    temporary = report.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, report)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol_sha256": sha256(PROTOCOL),
        "report_sha256": sha256(report),
        "docking_cases": len(docking),
        "screening_cases": len(screening),
    }
    atomic_json(SPEED / "audit/completion.json", completion)
    print(report)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

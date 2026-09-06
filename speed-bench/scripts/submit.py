#!/usr/bin/env python3
"""Submit all 15 immutable, single-sample cold benchmark cases."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from common import (
    PROTOCOL,
    ROOT,
    SPEED,
    atomic_json,
    load_and_verify_protocol,
    sha256,
    utc_now,
)


ACCOUNT = "naiss2025-3-10-gpu"
PARTITION = "gpu"
GPU_COUNTS = (1, 2, 4)
EXCLUDED_NODES = ("n481",)


def cases() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for engine, modes in (
        ("unidock", ("fast", "balance", "detail")),
        ("autodock-gpu", ("fast",)),
    ):
        for mode in modes:
            for gpus in GPU_COUNTS:
                records.append(
                    {
                        "kind": "docking",
                        "engine": engine,
                        "mode": mode,
                        "gpus": gpus,
                        "case": f"{engine}-{mode}-{gpus}gpu-cold",
                        "time_limit": "08:00:00",
                    }
                )
    for gpus in GPU_COUNTS:
        records.append(
            {
                "kind": "screening",
                "gpus": gpus,
                "case": f"screening-{gpus}gpu-cold",
                "time_limit": "12:00:00",
            }
        )
    return records


def sbatch_command(case: dict[str, Any]) -> list[str]:
    log_dir = SPEED / "logs/slurm"
    export = [
        "ALL",
        f"BENCH_KIND={case['kind']}",
        f"BENCH_GPUS={case['gpus']}",
    ]
    if case["kind"] == "docking":
        export.extend(
            [
                f"BENCH_ENGINE={case['engine']}",
                f"BENCH_MODE={case['mode']}",
            ]
        )
    return [
        "sbatch",
        "--parsable",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        f"--exclude={','.join(EXCLUDED_NODES)}",
        "--nodes=1",
        "--ntasks=1",
        f"--gpus-per-node={case['gpus']}",
        f"--cpus-per-task={72 * int(case['gpus'])}",
        f"--time={case['time_limit']}",
        f"--job-name=sb-{case['case']}",
        f"--output={log_dir}/%j-{case['case']}.out",
        f"--error={log_dir}/%j-{case['case']}.out",
        "--open-mode=truncate",
        f"--chdir={ROOT}",
        f"--export={','.join(export)}",
        str(SPEED / "slurm/benchmark.sbatch"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Print all scheduler commands only."
    )
    args = parser.parse_args()
    load_and_verify_protocol()
    submission_path = SPEED / "audit/submission.json"
    if submission_path.exists() and not args.dry_run:
        raise FileExistsError(f"refusing duplicate submission: {submission_path}")
    expected = cases()
    for case in expected:
        if (SPEED / "results" / case["kind"] / case["case"]).exists():
            raise FileExistsError(f"result path already exists for {case['case']}")
    commands = [sbatch_command(case) for case in expected]
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    (SPEED / "logs/slurm").mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "submitting",
        "submitted_at": utc_now(),
        "account": ACCOUNT,
        "partition": PARTITION,
        "excluded_nodes": list(EXCLUDED_NODES),
        "protocol_sha256": sha256(PROTOCOL),
        "cases": [],
    }
    atomic_json(submission_path, record)
    try:
        for case, command in zip(expected, commands):
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            job_id = completed.stdout.strip().split(";")[0]
            submitted = {**case, "job_id": job_id, "command": command}
            record["cases"].append(submitted)
            atomic_json(submission_path, record)
            print(f"{job_id}\t{case['case']}", flush=True)
    except BaseException:
        record["status"] = "partially_submitted"
        atomic_json(submission_path, record)
        raise
    record["status"] = "submitted"
    record["completed_submission_at"] = utc_now()
    atomic_json(submission_path, record)


if __name__ == "__main__":
    main()

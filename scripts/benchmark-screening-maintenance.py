#!/usr/bin/env python3
"""Measure maintained screening without overwriting the frozen benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "speed-bench/scripts"))
from common import GMOLAI_IMAGE, IGENVS_IMAGE, atomic_json, hardware_snapshot, sha256, stage_model_job
from run_case import collect_screen_stages, validate_screen_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory outside the repository.")
    parser.add_argument("--count", type=int, default=10_000_000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT) or args.count <= 0:
        parser.error("use a new directory outside the repository and a positive count")
    output.mkdir(parents=True, exist_ok=False)
    job = output / "job"
    job.mkdir()
    stage_model_job(job)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = os.environ.copy()
    environment.update({
        "IGENVS_ULTRA_ASSETS": str(ROOT), "IGENVS_ULTRA_JOB": str(job),
        "IGENVS_IMAGE": str(IGENVS_IMAGE), "GMOLAI_IMAGE": str(GMOLAI_IMAGE),
        "IGENVS_ULTRA_PROFILE_CACHE": str(output / "profile"),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TMPDIR": str(scratch),
    })
    command = [str(ROOT / "igenvs-ultra"), "screen-fast", str(args.count)]
    record = {"classification": "operational_maintenance_not_frozen_benchmark", "command": command,
              "hardware": hardware_snapshot(), "count": args.count, "status": "running"}
    atomic_json(output / "summary.json", record)
    started = time.perf_counter()
    with (output / "pipeline.log").open("w") as log:
        result = subprocess.run(command, cwd=job, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record.update(wall_seconds=time.perf_counter() - started, returncode=result.returncode)
    if result.returncode:
        record["status"] = "failed"
        atomic_json(output / "summary.json", record)
        raise RuntimeError(f"screen failed; see {output / 'pipeline.log'}")
    screen = job / f"screens/fast-{args.count}"
    manifest = json.loads((screen / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["exact_requested_score_count"]
    record.update(
        status="complete", stages=collect_screen_stages(screen),
        validation=validate_screen_results(Path(manifest["results"]), args.count),
        scientific_protocol_sha256=sha256(ROOT / "phase-10-rl-dev/protocol.json"),
        source_sha256={relative: sha256(ROOT / relative) for relative in (
            "user-pipeline/src/igenvs_ultra/workflow.py", "user-pipeline/src/igenvs_ultra/generation_worker.py",
            "user-pipeline/src/igenvs_ultra/model_ops.py",
        )},
    )
    atomic_json(output / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

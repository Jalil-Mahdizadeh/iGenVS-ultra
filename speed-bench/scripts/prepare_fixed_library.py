#!/usr/bin/env python3
"""Generate and freeze nested 20k/40k/80k base-isomeric fixtures."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from common import (
    IGENVS_IMAGE,
    ROOT,
    SPEED,
    atomic_json,
    hardware_snapshot,
    sha256,
    source_pythonpath,
    utc_now,
)


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    with log_path.open("w", encoding="utf-8", newline="") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
    if completed.returncode:
        raise RuntimeError(f"fixture command failed ({completed.returncode}): {log_path}")


def main() -> None:
    inputs = SPEED / "inputs"
    manifest_path = inputs / "fixed-libraries.manifest.json"
    if manifest_path.exists() or inputs.exists() and any(inputs.iterdir()):
        raise FileExistsError(f"refusing to replace benchmark inputs: {inputs}")
    inputs.mkdir(parents=True, exist_ok=True)
    scratch_parent = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    with tempfile.TemporaryDirectory(prefix="igenvs-fixed-library-", dir=scratch_parent) as raw_tmp:
        scratch = Path(raw_tmp)
        raw = scratch / "generated.smi"
        validation = scratch / "validation"
        environment = os.environ.copy()
        environment.update(
            {
                "APPTAINERENV_PYTHONPATH": source_pythonpath(),
                "PYTHONUNBUFFERED": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        binds = ["--bind", f"{ROOT}:{ROOT}", "--bind", f"{scratch}:{scratch}"]
        generation = [
            "apptainer", "exec", "--nv", *binds, str(IGENVS_IMAGE),
            "igen3", "generate", "--model", "base-isomeric", "--mode", "de-novo",
            "--output", str(raw), "--count", "80000", "--batch-size", "auto",
            "--temperature", "1.0", "--top-k", "64", "--seed", "2026090601",
            "--no-progress",
        ]
        started = time.perf_counter()
        run_logged(generation, inputs / "generation.log", environment)
        generation_seconds = time.perf_counter() - started
        validate = [
            "apptainer", "exec", *binds, str(IGENVS_IMAGE), "igenvs", "validate",
            "--input", str(raw), "--input-format", "smi", "--fragment-policy", "reject",
            "--workers", "auto", "--output-dir", str(validation),
        ]
        started = time.perf_counter()
        run_logged(validate, inputs / "validation.log", environment)
        validation_seconds = time.perf_counter() - started
        validated = validation / "validated.csv"
        with validated.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 80_000:
            raise RuntimeError(f"fixture produced {len(rows):,} validated rows, expected 80,000")
        canonical = [row["canonical_smiles"] for row in rows]
        if len(set(canonical)) != len(canonical):
            raise RuntimeError("fixture is not exactly canonical-SMILES deduplicated")
        libraries = {}
        for size in (20_000, 40_000, 80_000):
            path = inputs / f"fixed-{size // 1000}k.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["molecule_id", "smiles"])
                writer.writeheader()
                for index, smiles in enumerate(canonical[:size], start=1):
                    writer.writerow(
                        {"molecule_id": f"SPEEDBENCH-{index:06d}", "smiles": smiles}
                    )
            libraries[str(size)] = {
                "path": str(path.relative_to(ROOT)),
                "rows": size,
                "sha256": sha256(path),
            }
        atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "complete",
                "created_at": utc_now(),
                "relationship": "nested prefixes of one generated canonical exact-deduplicated fixture",
                "generation": {
                    "model": "base-isomeric",
                    "mode": "de-novo",
                    "temperature": 1.0,
                    "top_k": 64,
                    "seed": 2026090601,
                    "requested": 80_000,
                    "batch_size": "auto",
                    "seconds": generation_seconds,
                },
                "validation": {
                    "fragment_policy": "reject",
                    "deduplicate": True,
                    "workers": "auto",
                    "seconds": validation_seconds,
                },
                "libraries": libraries,
                "hardware": hardware_snapshot(),
                "raw_generation_sha256": sha256(raw),
            },
        )
    print(json.dumps(json.loads(manifest_path.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

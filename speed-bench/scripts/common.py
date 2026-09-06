#!/usr/bin/env python3
"""Shared helpers for the cold full-speed benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPEED = Path(__file__).resolve().parents[1]
ROOT = SPEED.parent
PROTOCOL = SPEED / "protocol.json"
IGENVS_IMAGE = Path(
    "/nobackup/proj/disk/theo-storage/personal/jalil/iGenVS/containers/iGenVS.SIF"
)
GMOLAI_IMAGE = Path(
    "/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/"
    "gmolai-pyg-25.09-arm64.sif"
)
TARGET = ROOT / "phase-7-benchmark-docking/targets/4ag8"
RELEASED_MODELS = (
    ROOT / "phase-9-benchmark-active-learning/rounds/round-5"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path, patterns: Iterable[str] = ("*.py",)) -> str:
    files = sorted(
        {
            path
            for pattern in patterns
            for path in root.rglob(pattern)
            if path.is_file() and "__pycache__" not in path.parts
        },
        key=lambda path: str(path.relative_to(root)),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(root)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_pythonpath() -> str:
    return os.pathsep.join(
        [str(ROOT / "iGenVS/src"), str(ROOT / "iGenVS/iGen3/src")]
    )


def benchmark_environment(profile_cache: Path, scratch: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "IGENVS_ULTRA_ASSETS": str(ROOT),
            "IGENVS_IMAGE": str(IGENVS_IMAGE),
            "GMOLAI_IMAGE": str(GMOLAI_IMAGE),
            "IGENVS_ULTRA_PROFILE_CACHE": str(profile_cache),
            "APPTAINERENV_PYTHONPATH": source_pythonpath(),
            "PYTHONUNBUFFERED": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TMPDIR": str(scratch),
        }
    )
    return environment


def visible_gpu_tokens() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if value and value not in {"-1", "NoDevFiles"}:
        return [item.strip() for item in value.split(",") if item.strip()]
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def hardware_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,pstate,"
            "clocks.max.sm,power.limit",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    cpu = subprocess.run(
        ["lscpu", "--json"], check=False, capture_output=True, text=True
    )
    return {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_CPUS_PER_TASK",
                "SLURM_GPUS_ON_NODE",
                "SLURM_JOB_GPUS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "gpus": [line.strip() for line in gpu.stdout.splitlines() if line.strip()],
        "lscpu": json.loads(cpu.stdout) if cpu.returncode == 0 and cpu.stdout else None,
    }


def require_gpu_allocation(expected: int) -> list[str]:
    tokens = visible_gpu_tokens()
    if len(tokens) != expected:
        raise RuntimeError(
            f"benchmark expected exactly {expected} visible GPUs, observed {tokens}"
        )
    names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not names or any("GH200 120GB" not in name for name in names):
        raise RuntimeError(f"benchmark requires GH200 120GB GPUs, observed {names}")
    return tokens


def load_and_verify_protocol() -> dict[str, Any]:
    protocol = load_json(PROTOCOL)
    if protocol.get("status") != "locked_before_execution":
        raise RuntimeError("benchmark protocol is not locked for execution")
    for relative, expected in protocol["locked_files"].items():
        path = ROOT / relative
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"locked file changed: {path}; expected {expected}, observed {observed}"
            )
    for relative, expected in protocol["locked_source_trees"].items():
        path = ROOT / relative
        patterns = ("*.sbatch",) if relative == "speed-bench/slurm" else ("*.py",)
        observed = tree_sha256(path, patterns)
        if observed != expected:
            raise RuntimeError(
                f"locked source tree changed: {path}; expected {expected}, observed {observed}"
            )
    for relative, expected in protocol.get("locked_asset_trees", {}).items():
        path = ROOT / relative
        observed = tree_sha256(path, ("*",))
        if observed != expected:
            raise RuntimeError(
                f"locked asset tree changed: {path}; expected {expected}, observed {observed}"
            )
    for label, record in protocol["containers"].items():
        path = Path(record["path"])
        stat = path.stat()
        if stat.st_size != int(record["size_bytes"]):
            raise RuntimeError(f"{label} container size changed: {path}")
        if stat.st_mtime_ns != int(record["mtime_ns"]):
            raise RuntimeError(f"{label} container modification time changed: {path}")
    return protocol


def stage_model_job(job: Path) -> Path:
    released_path = RELEASED_MODELS / "models/ensemble-manifest.json"
    released = load_json(released_path)
    target = released["targets"]["4ag8"]
    members = []
    for member in target["members"]:
        checkpoint = RELEASED_MODELS / member["checkpoint"]
        if sha256(checkpoint) != member["checkpoint_sha256"]:
            raise RuntimeError(f"released checkpoint changed: {checkpoint}")
        members.append({**member, "checkpoint": str(checkpoint)})
    model_dir = job / "models/round-5"
    model_dir.mkdir(parents=True)
    manifest = model_dir / "ensemble-manifest.json"
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "target": "4ag8",
            "stage": "round-5",
            "architecture": released["architecture"],
            "round": 5,
            "standardizer_sha256": released["standardizer_sha256"],
            "members": members,
        },
    )
    atomic_json(job / "fit-config.json", {"target_name": "4ag8", "benchmark_stub": True})
    atomic_json(
        job / "models/final.json",
        {
            "schema_version": 1,
            "status": "complete",
            "stage": "round-5",
            "ensemble_manifest": str(manifest.relative_to(job)),
            "ensemble_manifest_sha256": sha256(manifest),
        },
    )
    return manifest


def copy_debug_logs(job: Path, destination: Path) -> None:
    source = job / "logs"
    if source.is_dir():
        target = destination / "worker-logs"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

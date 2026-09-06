from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .util import ensure_directory, runtime_versions


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def gather_rank_objects(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    source, destination = Path(source), Path(destination)
    ensure_directory(destination.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Any,
    global_step: int,
    best_validation: float,
    data_state: dict[str, int],
    rank: int,
    world_size: int,
    identity: dict[str, str],
) -> dict[str, Any] | None:
    data_states = gather_rank_objects(data_state, world_size)
    rng_states = gather_rank_objects(capture_rng_state(), world_size)
    if rank != 0:
        return None
    return {
        "checkpoint_version": 1,
        **identity,
        "world_size": world_size,
        "global_step": int(global_step),
        "best_validation": float(best_validation),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp_scaler": amp_scaler.state_dict() if amp_scaler is not None else None,
        "data_states": data_states,
        "rng_states": rng_states,
        "runtime": runtime_versions(),
    }


def validate_checkpoint(
    checkpoint: dict[str, Any], identity: dict[str, str], world_size: int | None = None
) -> None:
    if checkpoint.get("checkpoint_version") != 1:
        raise RuntimeError("Unsupported checkpoint format")
    for key, expected in identity.items():
        actual = checkpoint.get(key)
        if actual != expected:
            raise RuntimeError(f"Checkpoint {key} mismatch: expected {expected}, got {actual}")
    if world_size is not None and int(checkpoint["world_size"]) != world_size:
        raise RuntimeError(
            f"Exact resume requires the original world size {checkpoint['world_size']}; requested {world_size}"
        )

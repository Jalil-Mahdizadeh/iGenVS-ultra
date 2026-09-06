from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable


def ensure_directory(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV artifact")
    destination = Path(path)
    ensure_directory(destination.parent)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_jsonl_row(path: str | Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise IndexError(index)
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Task {index} does not exist in {path}")


def sha256_file(path: str | Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(text: str, seed: int = 0) -> int:
    payload = seed.to_bytes(8, "little", signed=False) + text.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big", signed=False)


def stable_fraction(text: str, seed: int = 0) -> float:
    return stable_u64(text, seed) / float(1 << 64)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def runtime_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for module_name in (
        "gmolai_retrain",
        "numpy",
        "pyarrow",
        "duckdb",
        "rdkit",
        "torch",
        "torch_geometric",
    ):
        try:
            module = __import__(module_name)
            result[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            result[module_name] = "not-installed"
    return result

"""Stable subprocess adapter around the existing iGen3 CLI."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .errors import ExternalToolError


@dataclass(frozen=True)
class GenerationConfig:
    count: int
    model: str = "rl-nonisomeric"
    mode: str = "de-novo"
    batch_size: str = "auto"
    max_batch_size: int = 32_768
    model_dir: Path | None = None
    seed_file: Path | None = None
    samples_per_seed: int = 1
    temperature: float | None = None
    top_k: int | None = None
    compile_model: bool = False
    seed: int = 13
    executable: str = "igen3"


def run_igen3(config: GenerationConfig, *, output_path: Path, log_dir: Path) -> dict[str, object]:
    if config.count <= 0:
        raise ValueError("generation count must be positive")
    if shutil.which(config.executable) is None:
        raise ExternalToolError(f"iGen3 executable not found: {config.executable}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [config.executable]
    if config.model_dir is not None:
        command.extend(["--model-dir", str(config.model_dir)])
    command.extend(
        [
            "generate",
            "--model",
            config.model,
            "--mode",
            config.mode,
            "--count",
            str(config.count),
            "--output",
            str(output_path),
            "--batch-size",
            str(config.batch_size),
            "--max-batch-size",
            str(config.max_batch_size),
            "--seed",
            str(config.seed),
            "--compile",
            "on" if config.compile_model else "off",
            "--no-progress",
        ]
    )
    if config.temperature is not None:
        command.extend(["--temperature", str(config.temperature)])
    if config.top_k is not None:
        command.extend(["--top-k", str(config.top_k)])
    if config.mode == "derivative":
        if config.seed_file is None:
            raise ValueError("derivative generation requires a seed file")
        command.extend(["--seed-file", str(config.seed_file), "--samples-per-seed", str(config.samples_per_seed)])

    stdout_path = log_dir / "igen3.stdout.log"
    stderr_path = log_dir / "igen3.stderr.log"
    started = perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    seconds = perf_counter() - started
    if process.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4_000:]
        raise ExternalToolError(f"iGen3 failed with exit code {process.returncode}: {tail}")
    if not output_path.is_file():
        raise ExternalToolError("iGen3 reported success but did not create its SMILES output")
    return {
        "command": command,
        "seconds": seconds,
        "output": str(output_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }

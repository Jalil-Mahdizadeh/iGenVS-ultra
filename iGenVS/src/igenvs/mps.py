"""Run-private CUDA MPS lifecycle for automatic AutoDock-GPU concurrency."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ManagedCudaMPS:
    workers: int
    source: str
    executable: str | None = None
    environment: dict[str, str] | None = None
    root: Path | None = None
    previous_pipe: str | None = None
    previous_log: str | None = None

    def close(self) -> None:
        if self.executable is None or self.environment is None:
            return
        try:
            subprocess.run(
                [self.executable],
                input="quit\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self.environment,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        if self.previous_pipe is None:
            os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        else:
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = self.previous_pipe
        if self.previous_log is None:
            os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
        else:
            os.environ["CUDA_MPS_LOG_DIRECTORY"] = self.previous_log
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
        self.executable = None
        self.environment = None


def start_cuda_mps(
    workers: int,
    *,
    scratch_root: Path,
    automatic: bool,
) -> ManagedCudaMPS:
    """Start a private MPS daemon, falling back only for automatic choices."""

    if workers <= 1:
        return ManagedCudaMPS(workers, "not-required")
    existing = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if existing:
        return ManagedCudaMPS(workers, "existing-environment")
    executable = shutil.which("nvidia-cuda-mps-control")
    if executable is None:
        if automatic:
            return ManagedCudaMPS(1, "fallback-no-mps-control")
        raise RuntimeError(
            "multiple AutoDock-GPU workers require nvidia-cuda-mps-control"
        )

    root = Path(tempfile.mkdtemp(prefix="igenvs-mps-", dir=scratch_root))
    pipe = root / "pipe"
    log = root / "log"
    pipe.mkdir()
    log.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe),
            "CUDA_MPS_LOG_DIRECTORY": str(log),
        }
    )
    try:
        started = subprocess.run(
            [executable, "-d"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(root, ignore_errors=True)
        if automatic:
            return ManagedCudaMPS(1, f"fallback-mps-start:{type(exc).__name__}")
        raise RuntimeError(f"could not start CUDA MPS: {exc}") from exc
    if started.returncode != 0:
        shutil.rmtree(root, ignore_errors=True)
        detail = (started.stderr or started.stdout).strip()[-1_000:]
        if automatic:
            return ManagedCudaMPS(1, f"fallback-mps-start-exit-{started.returncode}")
        raise RuntimeError(
            f"could not start CUDA MPS (exit {started.returncode}): {detail}"
        )

    previous_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    previous_log = os.environ.get("CUDA_MPS_LOG_DIRECTORY")
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(log)
    return ManagedCudaMPS(
        workers,
        "managed-run-private",
        executable,
        environment,
        root,
        previous_pipe,
        previous_log,
    )

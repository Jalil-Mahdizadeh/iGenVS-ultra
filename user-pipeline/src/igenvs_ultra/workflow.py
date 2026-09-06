"""Lightweight, resumable orchestration for iGenVS-ultra."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import heapq
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence


SEEDS = (260904, 260905, 260906)
SCORE_FIELDS = [
    "molecule_id",
    "smiles",
    "original_smiles",
    "source_kind",
    "source_batch",
    "source_row",
    "target",
    "model_stage",
    *[f"member_probability_{seed}" for seed in SEEDS],
    "ensemble_probability",
    "ensemble_mutual_information",
]
WATER_AND_SMALL_ADDITIVES = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "ACT", "ACE", "EDO", "GOL", "PEG",
    "NA", "CL", "K", "CA", "MG", "MN", "ZN", "FE", "CO", "CU", "NI", "CD",
}
DEFAULT_IGENVS_DOCKER_IMAGE = "igenvs-ultra/igenvs:latest"
DEFAULT_GMOLAI_DOCKER_IMAGE = "igenvs-ultra/gmolai:latest"
DOCKER_FORWARDED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONUNBUFFERED",
    "IGENVS_ULTRA_PROFILE_CACHE",
)


class PipelineError(RuntimeError):
    """A user-actionable pipeline failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def atomic_concatenate_csv(paths: Sequence[Path], output: Path) -> int:
    """Concatenate identical-schema CSV partitions without reparsing rows."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    header: Optional[bytes] = None
    rows = 0
    try:
        with temporary.open("wb") as target:
            for path in paths:
                with path.open("rb") as source:
                    observed = source.readline()
                    if not observed:
                        raise PipelineError(f"CSV partition is empty: {path}")
                    if header is None:
                        header = observed
                        target.write(observed)
                    elif observed != header:
                        raise PipelineError("screen score batch schemas differ")
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        target.write(block)
                rows += file_rows(path)
            if header is None:
                target.write((",".join(SCORE_FIELDS) + "\n").encode("utf-8"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return rows


def tail(path: Path, characters: int = 4000) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-characters:]


def discover_project_root() -> Path:
    override = os.environ.get("IGENVS_ULTRA_ASSETS")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (
            (parent / "user-pipeline").is_dir()
            and (parent / "iGenVS").is_dir()
            and (parent / "gMolAI-v2.0").is_dir()
        ):
            return parent
    raise PipelineError("cannot discover project assets; pass --assets-dir or set IGENVS_ULTRA_ASSETS")


def default_igenvs_image(project: Path) -> Path:
    candidates = [
        project.parent / "iGenVS/containers/iGenVS.SIF",
        project / "iGenVS/containers/iGenVS.SIF",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def default_gmolai_image(project: Path) -> Path:
    candidates = [
        project.parent / "gMolAI/containers/gmolai-pyg-25.09-arm64.sif",
        project / "gMolAI-v2.0/containers/gmolai-pyg-25.09-arm64.sif",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def resolve_assets(args: Any) -> Path:
    value = getattr(args, "assets_dir", None)
    return Path(value).expanduser().resolve() if value else discover_project_root()


def resolved_runtime_paths(args: Any, assets: Path) -> tuple[Path, Path]:
    igenvs = getattr(args, "igenvs_image", None) or os.environ.get("IGENVS_IMAGE")
    gmolai = getattr(args, "gmolai_image", None) or os.environ.get("GMOLAI_IMAGE")
    return (
        Path(igenvs).expanduser().resolve() if igenvs else default_igenvs_image(assets),
        Path(gmolai).expanduser().resolve() if gmolai else default_gmolai_image(assets),
    )


def resolved_docker_images(args: Any) -> tuple[str, str]:
    igenvs = (
        getattr(args, "igenvs_docker_image", None)
        or os.environ.get("IGENVS_DOCKER_IMAGE")
        or DEFAULT_IGENVS_DOCKER_IMAGE
    )
    gmolai = (
        getattr(args, "gmolai_docker_image", None)
        or os.environ.get("GMOLAI_DOCKER_IMAGE")
        or DEFAULT_GMOLAI_DOCKER_IMAGE
    )
    return str(igenvs), str(gmolai)


def docker_image_exists(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def visible_gpu_tokens(explicit: Optional[str] = None) -> list[str]:
    if explicit:
        tokens = [item.strip() for item in explicit.split(",") if item.strip()]
        if not tokens:
            raise PipelineError("--gpu-ids did not contain a GPU identifier")
        return tokens
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"-1", "NoDevFiles"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []


def gpu_memory_mib() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    values = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                values.append(int(line.strip()))
            except ValueError:
                pass
    return values


def available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def profile_cache_directory(job: Path) -> Path:
    """Return a portable, reusable cache for tiny hardware calibration records."""
    explicit = os.environ.get("IGENVS_ULTRA_PROFILE_CACHE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    candidate = (base / "igenvs-ultra/profiles").resolve()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        fallback = job / ".igenvs-ultra/profiles"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def choose_stream_batch_size(
    output_parent: Path,
    requested: Any,
    *,
    molecule_count: Optional[int] = None,
    gpu_count: int = 1,
) -> tuple[int, dict[str, Any]]:
    memories = gpu_memory_mib()
    gpu_mib = min(memories) if memories else 0
    host_available = available_memory_bytes()
    disk_available = shutil.disk_usage(output_parent).free
    constraints: dict[str, Any] = {}
    if requested != "auto":
        chosen = int(requested)
        reason = "user override"
    else:
        if gpu_mib >= 80 * 1024:
            per_gpu = 1_000_000
        elif gpu_mib >= 40 * 1024:
            per_gpu = 750_000
        elif gpu_mib >= 20 * 1024:
            per_gpu = 500_000
        elif gpu_mib >= 10 * 1024:
            per_gpu = 250_000
        else:
            per_gpu = 100_000
        # A score row temporarily owns Python/SMILES metadata and a 1,536-byte
        # FP32 embedding.  Reserve most RAM and scratch for the OS, worker
        # pools, output, resume state, and unusually large molecules.
        host_bound = max(10_000, int(host_available * 0.20 / 4096)) if host_available else 100_000
        disk_bound = max(10_000, int(disk_available * 0.10 / 2048))
        candidates = [per_gpu * max(1, gpu_count), host_bound, disk_bound]
        if molecule_count is not None:
            candidates.append(int(molecule_count))
        chosen = min(candidates)
        if chosen >= 100_000:
            chosen = max(100_000, (chosen // 100_000) * 100_000)
        reason = "resource-model bound for persistent streaming"
        constraints = {
            "per_gpu_rows": per_gpu,
            "gpu_scaled_rows": per_gpu * max(1, gpu_count),
            "host_memory_bound_rows": host_bound,
            "scratch_bound_rows": disk_bound,
            "requested_molecule_count": molecule_count,
        }
    if chosen <= 0:
        raise PipelineError("stream batch size must be positive")
    return chosen, {
        "requested": requested,
        "selected": chosen,
        "reason": reason,
        "gpu_count": gpu_count,
        "constraints": constraints,
        "visible_gpu_memory_mib": memories,
        "host_memory_available_gib": round(host_available / 2**30, 3) if host_available else None,
        "disk_available_gib": round(disk_available / 2**30, 3),
    }


class Runtime:
    """Run iGenVS and gMolAI natively, in Docker, or in released SIFs."""

    def __init__(self, args: Any, assets: Path, job: Path, *, require_gmolai: bool = True):
        self.assets = assets.resolve()
        self.job = job.resolve()
        explicit_gpus = getattr(args, "gpu_ids", None)
        self.gpu_ids = ",".join(visible_gpu_tokens(explicit_gpus)) if explicit_gpus else None
        self.igenvs_image, self.gmolai_image = resolved_runtime_paths(args, assets)
        self.igenvs_docker_image, self.gmolai_docker_image = resolved_docker_images(args)
        requested = getattr(args, "execution", "auto")
        if requested == "auto":
            sif_images_available = self.igenvs_image.is_file() and (
                self.gmolai_image.is_file() or not require_gmolai
            )
            docker_images_available = docker_image_exists(self.igenvs_docker_image) and (
                docker_image_exists(self.gmolai_docker_image) or not require_gmolai
            )
            if shutil.which("apptainer") and sif_images_available:
                requested = "apptainer"
            elif shutil.which("docker") and docker_images_available:
                requested = "docker"
            else:
                requested = "native"
        self.execution = requested
        if self.execution == "apptainer" and shutil.which("apptainer") is None:
            raise PipelineError("Apptainer execution was requested but 'apptainer' is not on PATH")
        if self.execution == "docker" and shutil.which("docker") is None:
            raise PipelineError("Docker execution was requested but 'docker' is not on PATH")

    def _wrap(
        self,
        tool: str,
        command: Sequence[str],
        *,
        gpu: bool,
        extra_paths: Sequence[Path] = (),
        cpu_affinity: Sequence[int] = (),
    ) -> list[str]:
        if not command:
            raise PipelineError("cannot execute an empty command")
        binds = {self.assets}
        try:
            self.job.relative_to(self.assets)
        except ValueError:
            binds.add(self.job)
        for path in extra_paths:
            resolved = path.expanduser().resolve()
            candidate = resolved if resolved.is_dir() else resolved.parent
            contained = False
            for base in binds:
                try:
                    candidate.relative_to(base)
                    contained = True
                    break
                except ValueError:
                    pass
            if not contained:
                binds.add(candidate)
        if self.execution == "native":
            wrapped = list(command)
        elif self.execution == "docker":
            image = (
                self.igenvs_docker_image
                if tool == "igenvs"
                else self.gmolai_docker_image
            )
            wrapped = ["docker", "run", "--rm", "--init", "-i"]
            if gpu:
                wrapped.extend(["--gpus", "all"])
            if hasattr(os, "getuid") and hasattr(os, "getgid"):
                wrapped.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
            if cpu_affinity:
                wrapped.extend(["--cpuset-cpus", taskset_cpu_list(cpu_affinity)])
            wrapped.extend(["--ipc", "host", "--env", "HOME=/tmp"])
            for name in DOCKER_FORWARDED_ENV:
                wrapped.extend(["--env", name])
            for path in sorted(binds, key=str):
                wrapped.extend(["--volume", f"{path}:{path}:rw"])
            wrapped.extend(
                [
                    "--workdir",
                    str(self.assets),
                    "--entrypoint",
                    str(command[0]),
                    image,
                    *command[1:],
                ]
            )
            return wrapped
        else:
            image = self.igenvs_image if tool == "igenvs" else self.gmolai_image
            if not image.is_file():
                raise PipelineError(f"{tool} container does not exist: {image}")
            wrapped = ["apptainer", "exec"]
            if gpu:
                wrapped.append("--nv")
            for path in sorted(binds, key=str):
                wrapped.extend(["--bind", f"{path}:{path}"])
            wrapped.extend([str(image), *command])
        if cpu_affinity:
            taskset = shutil.which("taskset")
            if taskset is None:
                raise PipelineError("CPU isolation requires the standard 'taskset' utility")
            wrapped = [taskset, "--cpu-list", taskset_cpu_list(cpu_affinity), *wrapped]
        return wrapped

    def run_logged(
        self,
        tool: str,
        command: Sequence[str],
        log_path: Path,
        *,
        gpu: bool = True,
        extra_paths: Sequence[Path] = (),
        env: Optional[dict[str, str]] = None,
    ) -> None:
        full = self._wrap(tool, command, gpu=gpu, extra_paths=extra_paths)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[iGenVS-ultra] running: {' '.join(full)}", flush=True)
        process_env = os.environ.copy()
        if gpu and self.gpu_ids:
            process_env.update(
                {
                    "CUDA_VISIBLE_DEVICES": self.gpu_ids,
                    "APPTAINERENV_CUDA_VISIBLE_DEVICES": self.gpu_ids,
                }
            )
        if env:
            process_env.update(env)
        with log_path.open("w", encoding="utf-8", newline="") as log:
            process = subprocess.Popen(
                full,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=process_env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            process.stdout.close()
            returncode = process.wait()
        if returncode != 0:
            raise PipelineError(f"command failed with exit code {returncode}; see {log_path}\n{tail(log_path)}")

    def capture(
        self,
        tool: str,
        command: Sequence[str],
        *,
        gpu: bool = True,
        extra_paths: Sequence[Path] = (),
    ) -> subprocess.CompletedProcess[str]:
        process_env = os.environ.copy()
        if gpu and self.gpu_ids:
            process_env.update(
                {
                    "CUDA_VISIBLE_DEVICES": self.gpu_ids,
                    "APPTAINERENV_CUDA_VISIBLE_DEVICES": self.gpu_ids,
                }
            )
        try:
            return subprocess.run(
                self._wrap(tool, command, gpu=gpu, extra_paths=extra_paths),
                check=False,
                capture_output=True,
                text=True,
                env=process_env,
            )
        except OSError as exc:
            raise PipelineError(f"cannot execute {command[0]!r}: {exc}") from exc

    def model_command(self, operation: Sequence[str]) -> list[str]:
        script = self.assets / "user-pipeline/src/igenvs_ultra/model_ops.py"
        return ["python", str(script), *operation]


class PersistentJsonWorker:
    """One long-lived container process with a strict JSON-lines protocol."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        environment: dict[str, str],
        log_path: Path,
        label: str,
    ) -> None:
        self.label = label
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8", newline="")
        self.process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._request_number = 0
        self.ready: Optional[dict[str, Any]] = None

    def _receive(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            returncode = self.process.poll()
            raise PipelineError(
                f"persistent worker {self.label} exited unexpectedly "
                f"(return code {returncode}); see {self.log_path}\n{tail(self.log_path)}"
            )
        try:
            message = json.loads(line)
        except ValueError as exc:
            raise PipelineError(
                f"persistent worker {self.label} emitted invalid protocol data: {line!r}; "
                f"see {self.log_path}"
            ) from exc
        if message.get("event") == "error":
            raise PipelineError(
                f"persistent worker {self.label} failed: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        return message

    def wait_ready(self) -> dict[str, Any]:
        message = self._receive()
        if message.get("event") != "ready":
            raise PipelineError(
                f"persistent worker {self.label} did not send a ready event: {message}"
            )
        self.ready = message
        return message

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.ready is None:
            raise PipelineError(f"persistent worker {self.label} is not ready")
        if self.process.poll() is not None:
            raise PipelineError(
                f"persistent worker {self.label} has exited; see {self.log_path}\n"
                f"{tail(self.log_path)}"
            )
        self._request_number += 1
        request_id = f"{self.label}-{self._request_number}"
        message = {**payload, "request_id": request_id}
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message, sort_keys=True) + "\n")
        self.process.stdin.flush()
        response = self._receive()
        if response.get("event") != "result" or response.get("request_id") != request_id:
            raise PipelineError(
                f"persistent worker {self.label} returned an unexpected response: {response}"
            )
        return dict(response["result"])

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request_number += 1
                request_id = f"{self.label}-shutdown-{self._request_number}"
                assert self.process.stdin is not None
                self.process.stdin.write(
                    json.dumps({"command": "shutdown", "request_id": request_id}) + "\n"
                )
                self.process.stdin.flush()
                response = self._receive()
                if response.get("event") != "stopped":
                    self.process.terminate()
            except (BrokenPipeError, OSError, PipelineError):
                self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._log.close()


def _worker_affinity_groups(gpu_ids: Sequence[str]) -> list[list[int]]:
    inherited = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    return partition_cpu_affinity(inherited, len(gpu_ids))


def _start_workers(workers: Sequence[PersistentJsonWorker]) -> list[dict[str, Any]]:
    try:
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            return list(executor.map(lambda worker: worker.wait_ready(), workers))
    except BaseException:
        for worker in workers:
            worker.close()
        raise


class PersistentGenerationPool:
    """One resident iGen3 worker per selected GPU."""

    def __init__(
        self,
        runtime: Runtime,
        args: Any,
        job: Path,
        screen: Path,
        *,
        expected_count: int,
    ) -> None:
        self.gpu_ids = screening_gpu_ids(args)
        affinities = _worker_affinity_groups(self.gpu_ids)
        taskset = shutil.which("taskset")
        if runtime.execution != "docker" and taskset is None:
            raise PipelineError("persistent screening requires the standard 'taskset' utility")
        script = runtime.assets / "user-pipeline/src/igenvs_ultra/generation_worker.py"
        if not script.is_file():
            raise PipelineError(f"persistent generation worker is missing: {script}")
        extras = [Path(args.seed_file).expanduser().resolve()] if args.seed_file else []
        if args.model_dir:
            extras.append(Path(args.model_dir).expanduser().resolve())
        profile_cache = profile_cache_directory(job)
        extras.append(profile_cache)
        self.workers = []
        for lane, (gpu, affinity) in enumerate(zip(self.gpu_ids, affinities)):
            operation = [
                "python3", str(script), "--model", args.model,
                "--mode", args.generation_mode,
                "--batch-size", str(args.generator_batch_size),
                "--max-batch-size", str(args.generator_max_batch_size),
                "--expected-count", str(max(1, math.ceil(expected_count / len(self.gpu_ids)))),
                "--profile-cache", str(profile_cache / "generation.json"),
                "--samples-per-seed", str(args.samples_per_seed),
                "--max-candidate-multiplier", str(
                    args.max_candidate_multiplier if args.max_candidate_multiplier is not None else 50.0
                ),
                "--device", args.generator_device,
                "--dtype", args.generator_dtype,
                "--compile-mode", args.compile_mode,
            ]
            _append_option(operation, "--model-dir", args.model_dir)
            _append_option(operation, "--temperature", args.temperature)
            _append_option(operation, "--top-k", args.top_k)
            _append_option(operation, "--seed-file", args.seed_file)
            _append_option(operation, "--max-candidates", args.max_candidates)
            _append_option(operation, "--stagnation-limit", args.stagnation_limit)
            for value in args.seed_smiles or []:
                operation.extend(["--seed-smiles", value])
            if args.greedy:
                operation.append("--greedy")
            if args.compile_generator:
                operation.append("--compile")
            if args.include_seed_molecules:
                operation.append("--include-seed-molecules")
            full = runtime._wrap(
                "igenvs",
                operation,
                gpu=True,
                extra_paths=tuple(extras),
                cpu_affinity=affinity,
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "APPTAINERENV_CUDA_VISIBLE_DEVICES": gpu,
                    "OMP_NUM_THREADS": str(len(affinity)),
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            self.workers.append(
                PersistentJsonWorker(
                    full,
                    environment=environment,
                    log_path=job / f"logs/screen-{screen.name}-generation-worker-{lane}.log",
                    label=f"generation-{lane}",
                )
            )
        self.ready = _start_workers(self.workers)

    def run_wave(self, requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(requests) > len(self.workers):
            raise PipelineError("generation wave exceeds the persistent GPU worker count")
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = [
                executor.submit(worker.request, request)
                for worker, request in zip(self.workers, requests)
            ]
            return [future.result() for future in futures]

    def close(self) -> None:
        for worker in self.workers:
            worker.close()


class PersistentScorePool:
    """One resident gMolAI + target-ensemble worker per selected GPU."""

    def __init__(
        self,
        runtime: Runtime,
        args: Any,
        job: Path,
        assets: Path,
        screen: Path,
        model_manifest: Path,
    ) -> None:
        self.gpu_ids = screening_gpu_ids(args)
        affinities = _worker_affinity_groups(self.gpu_ids)
        taskset = shutil.which("taskset")
        if runtime.execution != "docker" and taskset is None:
            raise PipelineError("persistent screening requires the standard 'taskset' utility")
        self.workers = []
        profile_cache = profile_cache_directory(job)
        for lane, (gpu, affinity) in enumerate(zip(self.gpu_ids, affinities)):
            operation = build_score_worker_operation(
                args,
                job,
                assets,
                model_manifest,
                profile_cache,
            )
            full = runtime._wrap(
                "gmolai",
                runtime.model_command(operation),
                gpu=True,
                extra_paths=(runtime.assets / "gMolAI-v2.0", profile_cache),
                cpu_affinity=affinity,
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "APPTAINERENV_CUDA_VISIBLE_DEVICES": gpu,
                    "OMP_NUM_THREADS": str(len(affinity)),
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            self.workers.append(
                PersistentJsonWorker(
                    full,
                    environment=environment,
                    log_path=job / f"logs/screen-{screen.name}-score-worker-{lane}.log",
                    label=f"score-{lane}",
                )
            )
        self.ready = _start_workers(self.workers)

    def run(self, requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(requests) > len(self.workers):
            raise PipelineError("score request exceeds the persistent GPU worker count")
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            futures = [
                executor.submit(worker.request, request)
                for worker, request in zip(self.workers, requests)
            ]
            return [future.result() for future in futures]

    def close(self) -> None:
        for worker in self.workers:
            worker.close()


def sanitize_target_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not cleaned:
        raise PipelineError("target name is empty after filename-safe normalization")
    return cleaned


def detect_ligand_id(complex_pdb: Path) -> str:
    residues: dict[tuple[str, str, str], int] = {}
    with complex_pdb.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("HETATM") or len(line) < 27:
                continue
            resname = line[17:20].strip().upper()
            if resname in WATER_AND_SMALL_ADDITIVES:
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if element == "H":
                continue
            chain = line[21].strip()
            residue = (chain, resname, line[22:27].strip())
            residues[residue] = residues.get(residue, 0) + 1
    candidates = [item for item, heavy_atoms in residues.items() if heavy_atoms >= 5]
    if len(candidates) != 1:
        rendered = ", ".join(
            f"{chain or '_'}:{name}:{number} ({residues[(chain, name, number)]} heavy atoms)"
            for chain, name, number in sorted(candidates)
        ) or "none"
        raise PipelineError(
            "--ligand-id was omitted and a unique ligand could not be inferred; "
            f"candidate residues: {rendered}"
        )
    chain, name, residue = candidates[0]
    return f"{chain}:{name}:{residue}" if chain else name


def copy_verified(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise PipelineError(f"input file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256(destination) != sha256(source):
            raise PipelineError(f"job input already exists with different contents: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def target_input_descriptor(args: Any) -> tuple[str, dict[str, Any]]:
    if getattr(args, "prepared_target", None):
        source = Path(args.prepared_target).expanduser().resolve()
        manifest = source / "manifest.json"
        if not manifest.is_file():
            raise PipelineError(f"prepared target lacks manifest.json: {source}")
        return "prepared-target", {"path": str(source), "manifest_sha256": sha256(manifest)}
    if getattr(args, "complex", None):
        source = Path(args.complex).expanduser().resolve()
        if not source.is_file():
            raise PipelineError(f"complex PDB does not exist: {source}")
        ligand = getattr(args, "ligand_id", None) or detect_ligand_id(source)
        return "complex-pdb", {"path": str(source), "sha256": sha256(source), "ligand_id": ligand}
    receptor = Path(args.receptor).expanduser().resolve()
    ligand = Path(args.reference_ligand).expanduser().resolve()
    if not receptor.is_file() or not ligand.is_file():
        raise PipelineError("both receptor PDB and reference-ligand SDF must exist")
    return "separate-files", {
        "receptor": str(receptor),
        "receptor_sha256": sha256(receptor),
        "reference_ligand": str(ligand),
        "reference_ligand_sha256": sha256(ligand),
    }


def docking_config(args: Any) -> dict[str, Any]:
    names = (
        "engine", "search_mode", "scoring", "num_modes", "energy_range", "refine_step",
        "no_refine", "unidock_verbosity", "seed", "max_gpu_memory", "adgpu_runs",
        "adgpu_evaluations", "adgpu_no_heuristics", "adgpu_no_autostop",
        "adgpu_local_search", "adgpu_cpu_threads", "adgpu_workers", "adgpu_executable",
        "batch_size", "batch_profile", "prep_workers", "prep_mode", "embed_max_attempts",
        "embed_timeout", "validation_workers",
        "fragment_policy", "no_deduplicate", "keep_work", "pose_output", "scratch_dir",
        "device_id", "num_shards", "shard_index", "docking_gpus", "docking_logical_shards",
    )
    result = {}
    for name in names:
        value = getattr(args, name, None)
        result[name] = str(value) if isinstance(value, Path) else value
    result["padding"] = float(args.padding)
    return result


def release_equivalent_docking(config: dict[str, Any]) -> bool:
    return (
        config["engine"] == "unidock"
        and config["search_mode"] == "fast"
        and config["scoring"] in {"auto", "vina"}
        and config["num_modes"] == 1
        and config["energy_range"] == 3.0
        and config["refine_step"] == 3
        and not config["no_refine"]
        and config["fragment_policy"] == "reject"
        and not config["no_deduplicate"]
        and config["seed"] == 181129
        and config["padding"] == 5.0
        and config["docking_logical_shards"] == 4
    )


def make_target_config(args: Any, assets: Path) -> dict[str, Any]:
    mode, descriptor = target_input_descriptor(args)
    default_name = Path(descriptor.get("path", descriptor.get("receptor", "target"))).stem
    target_name = sanitize_target_name(getattr(args, "target_name", None) or default_name)
    return {
        "schema_version": 1,
        "target_name": target_name,
        "target_input_mode": mode,
        "target_input": descriptor,
        "assets_dir": str(assets),
    }


def make_fit_config(args: Any, assets: Path) -> dict[str, Any]:
    config = make_target_config(args, assets)
    dock = docking_config(args)
    config.update({
        "docking": dock,
        "release_equivalent_docking_protocol": release_equivalent_docking(dock),
        "head": {
            "architecture": "wide_mlp_rank_aux",
            "embedding_space": "released_hybrid_w3",
            "ensemble_seeds": list(SEEDS),
            "epochs": 7,
        },
    })
    return config


def ensure_fit_config(args: Any, job: Path, assets: Path) -> dict[str, Any]:
    config = make_fit_config(args, assets)
    path = job / "fit-config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise PipelineError(
                f"existing job was created with different target/protocol settings: {path}; "
                "use a new --output-dir"
            )
    else:
        if job.exists() and any(job.iterdir()):
            raise PipelineError(f"output directory is non-empty but is not an iGenVS-ultra job: {job}")
        job.mkdir(parents=True, exist_ok=True)
        atomic_json(path, config)
    if not config["release_equivalent_docking_protocol"]:
        print(
            "[iGenVS-ultra] WARNING: custom docking settings are internally consistent but are not "
            "release-equivalent to Uni-Dock/Vina fast with 5 A padding.",
            file=sys.stderr,
            flush=True,
        )
    return config


def prepare_target(args: Any, job: Path, runtime: Runtime, config: dict[str, Any]) -> Path:
    target = job / "target"
    manifest = target / "manifest.json"
    if manifest.is_file():
        print("[iGenVS-ultra] target already prepared", flush=True)
        return target
    mode = config["target_input_mode"]
    inputs = job / "inputs"
    if mode == "prepared-target":
        source = Path(config["target_input"]["path"])
        if target.exists():
            failed = target.with_name(f"target.incomplete-{int(time.time())}")
            target.rename(failed)
        shutil.copytree(source, target)
        if sha256(target / "manifest.json") != config["target_input"]["manifest_sha256"]:
            raise PipelineError("prepared target changed while it was copied")
        return target
    if target.exists():
        target.rename(target.with_name(f"target.incomplete-{int(time.time())}"))
    command = ["igenvs", "prepare-target"]
    if mode == "complex-pdb":
        source = Path(config["target_input"]["path"])
        copied = inputs / "complex.pdb"
        copy_verified(source, copied)
        command.extend(["--complex", str(copied), "--ligand-id", config["target_input"]["ligand_id"]])
    else:
        receptor = inputs / "receptor.pdb"
        ligand = inputs / "reference-ligand.sdf"
        copy_verified(Path(config["target_input"]["receptor"]), receptor)
        copy_verified(Path(config["target_input"]["reference_ligand"]), ligand)
        command.extend(["--receptor", str(receptor), "--reference-ligand", str(ligand)])
    command.extend(["--padding", str(args.padding), "--output-dir", str(target)])
    runtime.run_logged("igenvs", command, job / "logs/prepare-target.log", gpu=False)
    if not manifest.is_file():
        raise PipelineError("iGenVS did not produce a prepared-target manifest")
    return target


def regular_source_config(args: Any) -> dict[str, Any]:
    """Describe exactly one source accepted by the original iGenVS screen command."""
    if getattr(args, "input", None):
        path = Path(args.input).expanduser().resolve()
        if not path.is_file():
            raise PipelineError(f"docking library does not exist: {path}")
        return {
            "kind": "external",
            "path": str(path),
            "sha256": sha256(path),
            "format": args.input_format,
            "smiles_column": args.smiles_column,
            "id_column": args.id_column,
            "delimiter": args.delimiter,
        }
    seed_file = Path(args.seed_file).expanduser().resolve() if args.seed_file else None
    if seed_file is not None and not seed_file.is_file():
        raise PipelineError(f"generation seed file does not exist: {seed_file}")
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else None
    if model_dir is not None and not model_dir.is_dir():
        raise PipelineError(f"iGen3 model directory does not exist: {model_dir}")
    return {
        "kind": "iGen3",
        "generate_count": int(args.generate_count),
        "model": args.model,
        "generation_mode": args.generation_mode,
        "seed_file": str(seed_file) if seed_file else None,
        "seed_file_sha256": sha256(seed_file) if seed_file else None,
        "samples_per_seed": args.samples_per_seed,
        "generator_batch_size": args.generator_batch_size,
        "generator_max_batch_size": args.generator_max_batch_size,
        "model_dir": str(model_dir) if model_dir else None,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "compile_generator": args.compile_generator,
        "generator_seed": args.generator_seed,
    }


def make_regular_config(args: Any, assets: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": "regular-iGenVS-docking",
        "target": make_target_config(args, assets),
        "source": regular_source_config(args),
        "docking": docking_config(args),
    }


def ensure_regular_config(job: Path, config: dict[str, Any]) -> None:
    path = job / "regular-config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise PipelineError(
                f"existing regular-docking job has different settings: {path}; "
                "use a new --output-dir"
            )
        return
    if job.exists() and any(job.iterdir()):
        raise PipelineError(f"output directory is non-empty but is not a regular iGenVS job: {job}")
    job.mkdir(parents=True, exist_ok=True)
    atomic_json(path, config)


def build_regular_docking_command(
    args: Any,
    *,
    source: dict[str, Any],
    target: Path,
    output: Path,
    prevalidated_input: Optional[Path] = None,
    num_shards: Optional[int] = None,
    shard_index: Optional[int] = None,
    device_id: Optional[int] = None,
    scratch_dir: Optional[Path] = None,
) -> list[str]:
    """Build a transparent pass-through to the released ``igenvs screen`` CLI."""
    command = ["igenvs", "screen"]
    if prevalidated_input is not None:
        command.extend(["--prevalidated-input", str(prevalidated_input)])
    elif source["kind"] == "external":
        command.extend(
            [
                "--input", source["path"],
                "--input-format", source["format"],
                "--smiles-column", source["smiles_column"],
                "--delimiter", source["delimiter"],
            ]
        )
        _append_option(command, "--id-column", source["id_column"])
    else:
        command.extend(
            [
                "--generate-count", str(source["generate_count"]),
                "--model", source["model"],
                "--generation-mode", source["generation_mode"],
                "--samples-per-seed", str(source["samples_per_seed"]),
                "--generator-batch-size", str(source["generator_batch_size"]),
                "--generator-max-batch-size", str(source["generator_max_batch_size"]),
                "--generator-seed", str(source["generator_seed"]),
            ]
        )
        _append_option(command, "--seed-file", source["seed_file"])
        _append_option(command, "--model-dir", source["model_dir"])
        _append_option(command, "--temperature", source["temperature"])
        _append_option(command, "--top-k", source["top_k"])
        if source["compile_generator"]:
            command.append("--compile-generator")
    command.extend(
        [
            "--target", str(target),
            "--engine", args.engine,
            "--search-mode", args.search_mode,
            "--scoring", args.scoring,
            "--num-modes", str(args.num_modes),
            "--energy-range", str(args.energy_range),
            "--refine-step", str(args.refine_step),
            "--unidock-verbosity", str(args.unidock_verbosity),
            "--seed", str(args.seed),
            "--device-id", str(args.device_id if device_id is None else device_id),
            "--max-gpu-memory", str(args.max_gpu_memory),
            "--batch-size", str(args.batch_size),
            "--prep-workers", str(args.prep_workers),
            "--prep-mode", args.prep_mode,
            "--embed-max-attempts", str(args.embed_max_attempts),
            "--embed-timeout", str(args.embed_timeout),
            "--validation-workers", str(args.validation_workers),
            "--fragment-policy", args.fragment_policy,
            "--num-shards", str(args.num_shards if num_shards is None else num_shards),
            "--shard-index", str(args.shard_index if shard_index is None else shard_index),
            "--pose-output", args.pose_output,
            "--adgpu-local-search", args.adgpu_local_search,
            "--adgpu-cpu-threads", str(args.adgpu_cpu_threads),
            "--adgpu-workers", str(args.adgpu_workers),
            "--adgpu-executable", args.adgpu_executable,
            "--output-dir", str(output),
        ]
    )
    if args.no_refine:
        command.append("--no-refine")
    if args.no_deduplicate:
        command.append("--no-deduplicate")
    if args.keep_work:
        command.append("--keep-work")
    _append_option(command, "--batch-profile", args.batch_profile)
    _append_option(command, "--scratch-dir", scratch_dir if scratch_dir is not None else args.scratch_dir)
    _append_option(command, "--adgpu-runs", args.adgpu_runs)
    _append_option(command, "--adgpu-evaluations", args.adgpu_evaluations)
    if args.adgpu_no_heuristics:
        command.append("--adgpu-no-heuristics")
    if args.adgpu_no_autostop:
        command.append("--adgpu-no-autostop")
    return command


def _regular_extra_paths(args: Any, source: dict[str, Any]) -> tuple[Path, ...]:
    extras: list[Path] = []
    for value in (
        source.get("path"),
        source.get("seed_file"),
        source.get("model_dir"),
        getattr(args, "batch_profile", None),
        getattr(args, "scratch_dir", None),
    ):
        if value:
            extras.append(Path(value).expanduser().resolve())
    executable = Path(args.adgpu_executable).expanduser()
    if executable.is_file():
        extras.append(executable.resolve())
    return tuple(extras)


def _generate_regular_library(
    runtime: Runtime,
    job: Path,
    source: dict[str, Any],
) -> Path:
    """Generate once before multi-GPU docking instead of once per shard."""

    generated = job / "library/generated.smi"
    manifest_path = job / "library/generation-manifest.json"
    if generated.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "complete"
            and manifest.get("output_sha256") == sha256(generated)
        ):
            return generated
    generated.parent.mkdir(parents=True, exist_ok=True)
    temporary = generated.with_suffix(".smi.partial")
    temporary.unlink(missing_ok=True)
    command = ["igen3"]
    if source.get("model_dir"):
        command.extend(["--model-dir", source["model_dir"]])
    command.extend(
        [
            "generate",
            "--model", source["model"],
            "--mode", source["generation_mode"],
            "--count", str(source["generate_count"]),
            "--output", str(temporary),
            "--batch-size", str(source["generator_batch_size"]),
            "--max-batch-size", str(source["generator_max_batch_size"]),
            "--seed", str(source["generator_seed"]),
            "--compile", "on" if source["compile_generator"] else "off",
            "--no-progress",
        ]
    )
    _append_option(command, "--temperature", source.get("temperature"))
    _append_option(command, "--top-k", source.get("top_k"))
    if source["generation_mode"] == "derivative":
        command.extend(
            [
                "--seed-file", source["seed_file"],
                "--samples-per-seed", str(source["samples_per_seed"]),
            ]
        )
    runtime.run_logged(
        "igenvs",
        command,
        job / "logs/regular-generation.log",
        gpu=True,
        extra_paths=tuple(
            Path(value).expanduser().resolve()
            for value in (source.get("seed_file"), source.get("model_dir"))
            if value
        ),
    )
    if not temporary.is_file():
        raise PipelineError("iGen3 did not produce the shared regular-docking library")
    os.replace(temporary, generated)
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "source": source,
            "output": str(generated),
            "output_sha256": sha256(generated),
            "rows": sum(1 for line in generated.open(encoding="utf-8") if line.strip()),
        },
    )
    return generated


def prepare_shared_docking_library(
    args: Any,
    runtime: Runtime,
    job: Path,
    source: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Validate and globally deduplicate once before launching GPU shards."""

    started = time.perf_counter()
    source_path = (
        Path(source["path"])
        if source["kind"] == "external"
        else _generate_regular_library(runtime, job, source)
    )
    validation_dir = job / "library/validation"
    validated = validation_dir / "validated.csv"
    manifest_path = job / "library/validation-manifest.json"
    desired = {
        "schema_version": 1,
        "source": str(source_path.resolve()),
        "source_sha256": sha256(source_path),
        "fragment_policy": args.fragment_policy,
        "deduplicate": not args.no_deduplicate,
    }
    if validated.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            all(existing.get(key) == value for key, value in desired.items())
            and existing.get("status") == "complete"
            and existing.get("validated_sha256") == sha256(validated)
        ):
            return validated, existing
    if validation_dir.exists():
        validation_dir.rename(
            validation_dir.with_name(f"validation.incomplete-{int(time.time())}")
        )
    command = ["igenvs", "validate", "--input", str(source_path)]
    if source["kind"] == "external":
        command.extend(
            [
                "--input-format", source["format"],
                "--smiles-column", source["smiles_column"],
                "--delimiter", source["delimiter"],
            ]
        )
        _append_option(command, "--id-column", source.get("id_column"))
    else:
        command.extend(["--input-format", "smi"])
    command.extend(
        [
            "--fragment-policy", args.fragment_policy,
            "--workers", str(args.validation_workers),
            "--output-dir", str(validation_dir),
        ]
    )
    if args.no_deduplicate:
        command.append("--no-deduplicate")
    runtime.run_logged(
        "igenvs",
        command,
        job / "logs/regular-validation.log",
        gpu=False,
        extra_paths=(source_path,),
    )
    if not validated.is_file():
        raise PipelineError("iGenVS did not produce the shared validated library")
    manifest = {
        **desired,
        "status": "complete",
        "validated": str(validated),
        "validated_sha256": sha256(validated),
        "valid_rows": file_rows(validated),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(manifest_path, manifest)
    return validated, manifest


def _launch_regular_docking_shards(
    args: Any,
    runtime: Runtime,
    job: Path,
    source: dict[str, Any],
    validated: Path,
    gpus: Sequence[str],
) -> tuple[list[Path], float]:
    output = job / "docking"
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    groups = partition_cpu_affinity(affinity, len(gpus))
    taskset = shutil.which("taskset")
    if len(gpus) > 1 and runtime.execution != "docker" and taskset is None:
        raise PipelineError("multi-GPU docking requires the standard 'taskset' utility")
    pending: list[tuple[int, subprocess.Popen[Any], Any, Path]] = []
    started = time.perf_counter()
    for shard, (gpu, cores) in enumerate(zip(gpus, groups)):
        shard_dir = runs / f"shard-{shard}"
        manifest_path = shard_dir / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                continue
        if shard_dir.exists():
            shard_dir.rename(
                shard_dir.with_name(f"{shard_dir.name}.incomplete-{int(time.time())}")
            )
        shard_scratch = (
            Path(args.scratch_dir).expanduser().resolve() / f"regular-shard-{shard}"
            if args.scratch_dir
            else None
        )
        if shard_scratch is not None:
            shard_scratch.mkdir(parents=True, exist_ok=True)
        command = build_regular_docking_command(
            args,
            source=source,
            target=job / "target",
            output=shard_dir,
            prevalidated_input=validated,
            num_shards=len(gpus),
            shard_index=shard,
            device_id=0,
            scratch_dir=shard_scratch,
        )
        full = runtime._wrap(
            "igenvs",
            command,
            gpu=True,
            extra_paths=(*_regular_extra_paths(args, source), validated),
            cpu_affinity=cores if len(gpus) > 1 else (),
        )
        log_path = job / f"logs/regular-docking-shard-{shard}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8", newline="")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "APPTAINERENV_CUDA_VISIBLE_DEVICES": gpu,
                "OMP_NUM_THREADS": str(len(cores)),
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        print(
            f"[iGenVS-ultra] starting regular shard {shard}/{len(gpus)} on GPU {gpu}; "
            f"CPU affinity={taskset_cpu_list(cores)}",
            flush=True,
        )
        process = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, env=environment)
        pending.append((shard, process, log, log_path))
    try:
        while pending:
            remaining = []
            for shard, process, log, log_path in pending:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((shard, process, log, log_path))
                    continue
                log.close()
                if returncode:
                    for _, other, other_log, _ in remaining:
                        other.terminate()
                        other_log.close()
                    raise PipelineError(
                        f"regular docking shard {shard} failed with exit code {returncode}; "
                        f"see {log_path}\n{tail(log_path)}"
                    )
                print(f"[iGenVS-ultra] completed regular shard {shard}", flush=True)
            pending = remaining
            if pending:
                time.sleep(1)
    except BaseException:
        for _, process, log, _ in pending:
            if process.poll() is None:
                process.terminate()
            log.close()
        raise
    return [runs / f"shard-{index}" for index in range(len(gpus))], time.perf_counter() - started


def _merge_regular_docking_shards(
    args: Any,
    job: Path,
    shard_dirs: Sequence[Path],
    gpus: Sequence[str],
    launcher_seconds: float,
    validation: dict[str, Any],
) -> dict[str, Any]:
    output = job / "docking"
    manifests = [
        json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for directory in shard_dirs
    ]
    if any(manifest.get("status") != "complete" for manifest in manifests):
        raise PipelineError("cannot merge incomplete regular docking shards")

    readers = []
    handles = []
    fields: Optional[list[str]] = None
    heap: list[tuple[int, int, dict[str, str], Any]] = []
    temporary = output / "results.csv.partial"
    try:
        for shard, directory in enumerate(shard_dirs):
            handle = (directory / "results.csv").open("r", encoding="utf-8", newline="")
            handles.append(handle)
            reader = csv.DictReader(handle)
            current_fields = list(reader.fieldnames or [])
            if fields is None:
                fields = current_fields
            elif current_fields != fields:
                raise PipelineError("regular docking shard result schemas differ")
            readers.append(reader)
            row = next(reader, None)
            if row is not None:
                heapq.heappush(heap, (int(row["source_row"]), shard, row, reader))
        if fields is None:
            raise PipelineError("regular docking shards produced no result schema")
        last_source_row = 0
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while heap:
                source_row, shard, row, reader = heapq.heappop(heap)
                if source_row <= last_source_row:
                    raise PipelineError("regular docking shards contain duplicate/out-of-order source rows")
                writer.writerow(row)
                last_source_row = source_row
                following = next(reader, None)
                if following is not None:
                    heapq.heappush(
                        heap,
                        (int(following["source_row"]), shard, following, reader),
                    )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / "results.csv")
    finally:
        temporary.unlink(missing_ok=True)
        for handle in handles:
            handle.close()

    poses: Optional[Path] = None
    if args.pose_output == "merged":
        poses = output / "poses.pdbqt"
        pose_tmp = poses.with_suffix(".pdbqt.partial")
        with pose_tmp.open("wb") as target:
            for directory in shard_dirs:
                source = directory / "poses.pdbqt"
                if source.is_file():
                    with source.open("rb") as handle:
                        shutil.copyfileobj(handle, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(pose_tmp, poses)
    elif args.pose_output == "individual":
        poses = output / "poses"
        poses.mkdir(exist_ok=True)
        for directory in shard_dirs:
            for source in (directory / "poses").glob("*.pdbqt"):
                destination = poses / source.name
                if destination.exists():
                    raise PipelineError(f"duplicate individual pose filename: {source.name}")
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)

    count_keys = {
        key
        for manifest in manifests
        for key, value in manifest.get("counts", {}).items()
        if isinstance(value, int)
    }
    counts = {
        key: sum(int(manifest.get("counts", {}).get(key, 0)) for manifest in manifests)
        for key in sorted(count_keys)
    }
    timings = {
        "launcher_wall_seconds": launcher_seconds,
        "parallel_docking_wall_seconds": max(
            float(manifest.get("timings", {}).get("docking_wall_seconds", 0.0))
            for manifest in manifests
        ),
        "parallel_preparation_wait_seconds": max(
            float(manifest.get("timings", {}).get("preparation_wait_seconds", 0.0))
            for manifest in manifests
        ),
    }
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "completed_at": utc_now(),
        "workflow": "regular-iGenVS-docking-multi-gpu",
        "engine": args.engine,
        "search_mode": args.search_mode,
        "gpu_ids": list(gpus),
        "logical_shards": len(shard_dirs),
        "validation": validation,
        "counts": counts,
        "timings": timings,
        "shards": [
            {
                "index": index,
                "directory": str(directory),
                "manifest_sha256": sha256(directory / "manifest.json"),
                "results_sha256": sha256(directory / "results.csv"),
            }
            for index, directory in enumerate(shard_dirs)
        ],
        "outputs": {
            "results": str(output / "results.csv"),
            "poses": str(poses) if poses is not None else None,
            "manifest": str(output / "manifest.json"),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def regular_dock(args: Any) -> dict[str, Any]:
    """Run ordinary iGenVS generation/ingress, preparation, and docking only."""
    workflow_started = time.perf_counter()
    assets = resolve_assets(args)
    job = Path(args.output_dir).expanduser().resolve()
    config = make_regular_config(args, assets)
    runtime = Runtime(args, assets, job, require_gmolai=False)
    output = job / "docking"
    command = build_regular_docking_command(
        args,
        source=config["source"],
        target=job / "target",
        output=output,
    )
    if getattr(args, "dry_run", False):
        plan = {
            "command": "dock",
            "workflow": "regular-iGenVS-docking",
            "job": str(job),
            "runtime": runtime.execution,
            "target": config["target"],
            "source": config["source"],
            "docking": config["docking"],
            "gpu_plan": {
                "requested": args.docking_gpus,
                "automatic_multi_gpu": args.num_shards == 1 and args.shard_index == 0,
            },
            "igenvs_command": command,
            "outputs": {
                "results": str(output / "results.csv"),
                "poses": None if args.pose_output == "none" else str(output / ("poses" if args.pose_output == "individual" else "poses.pdbqt")),
            },
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan

    ensure_regular_config(job, config)
    target_started = time.perf_counter()
    prepare_target(args, job, runtime, config["target"])
    target_setup_seconds = time.perf_counter() - target_started
    selected_gpus = docking_gpu_ids(args)
    automatic_parallel = (
        len(selected_gpus) > 1
        and args.num_shards == 1
        and args.shard_index == 0
    )
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_outputs = manifest.get("outputs", {})
        results_path = Path(manifest_outputs.get("results", output / "results.csv"))
        poses_value = manifest_outputs.get("poses")
        artifacts_complete = results_path.is_file() and (
            not poses_value or Path(poses_value).exists()
        )
        if manifest.get("status") == "complete" and artifacts_complete:
            print("[iGenVS-ultra] regular iGenVS docking already complete", flush=True)
        elif not automatic_parallel:
            output.rename(output.with_name(f"docking.incomplete-{int(time.time())}"))
            manifest = {}
        else:
            manifest = {}
    else:
        manifest = {}
        if output.exists() and not automatic_parallel:
            output.rename(output.with_name(f"docking.incomplete-{int(time.time())}"))
    docking_stage_started = time.perf_counter()
    if manifest.get("status") != "complete":
        if automatic_parallel:
            output.mkdir(parents=True, exist_ok=True)
            validated, validation = prepare_shared_docking_library(
                args,
                runtime,
                job,
                config["source"],
            )
            shard_dirs, launcher_seconds = _launch_regular_docking_shards(
                args,
                runtime,
                job,
                config["source"],
                validated,
                selected_gpus,
            )
            manifest = _merge_regular_docking_shards(
                args,
                job,
                shard_dirs,
                selected_gpus,
                launcher_seconds,
                validation,
            )
        else:
            runtime.run_logged(
                "igenvs",
                command,
                job / "logs/regular-docking.log",
                gpu=True,
                extra_paths=_regular_extra_paths(args, config["source"]),
            )
        if not manifest_path.is_file():
            raise PipelineError("iGenVS did not produce a regular docking manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results_path = Path(manifest.get("outputs", {}).get("results", output / "results.csv"))
        if manifest.get("status") != "complete" or not results_path.is_file():
            raise PipelineError(f"regular iGenVS docking did not complete: {manifest_path}")
    docking_stage_seconds = time.perf_counter() - docking_stage_started
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": manifest.get("completed_at", utc_now()),
        "workflow": "regular-iGenVS-docking",
        "target": config["target"]["target_name"],
        "engine": args.engine,
        "gpu_ids": manifest.get("gpu_ids", selected_gpus[:1]),
        "counts": manifest.get("counts", {}),
        "results": manifest.get("outputs", {}).get("results", str(output / "results.csv")),
        "poses": manifest.get("outputs", {}).get("poses"),
        "manifest": str(manifest_path),
        "timings": {
            "target_setup_seconds": target_setup_seconds,
            "docking_stage_seconds": docking_stage_seconds,
            "wrapper_wall_seconds": time.perf_counter() - workflow_started,
        },
    }
    atomic_json(job / "regular-summary.json", summary)
    print(
        f"[iGenVS-ultra] regular docking complete; results: {summary['results']}",
        flush=True,
    )
    return summary


def docking_gpu_ids(args: Any) -> list[str]:
    available = visible_gpu_tokens(getattr(args, "gpu_ids", None))
    requested = getattr(args, "docking_gpus", "auto")
    if requested == "auto":
        count = len(available)
        slurm_per_task = os.environ.get("SLURM_GPUS_PER_TASK", "").strip()
        if slurm_per_task:
            match = re.search(r"(\d+)$", slurm_per_task)
            if match and int(match.group(1)) > 0:
                count = min(count, int(match.group(1)))
    else:
        count = int(requested)
    if count < 1:
        raise PipelineError("docking requires at least one visible GPU")
    if len(available) < count:
        raise PipelineError(f"requested {count} docking GPUs but only {len(available)} are visible")
    return available[:count]


def partition_cpu_affinity(cpu_ids: Sequence[int], partitions: int) -> list[list[int]]:
    """Split the inherited allocation into stable, disjoint shard affinities."""
    values = sorted({int(value) for value in cpu_ids})
    if partitions <= 0:
        raise PipelineError("CPU-affinity partition count must be positive")
    if len(values) < partitions:
        raise PipelineError(
            f"only {len(values)} CPU cores are available for {partitions} docking shards"
        )
    quotient, remainder = divmod(len(values), partitions)
    groups = []
    start = 0
    for index in range(partitions):
        width = quotient + int(index < remainder)
        groups.append(values[start : start + width])
        start += width
    return groups


def taskset_cpu_list(cpu_ids: Sequence[int]) -> str:
    """Render an explicit taskset list without assumptions about CPU numbering."""
    return ",".join(str(value) for value in cpu_ids)


def _append_option(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([name, str(value)])


def build_docking_command(
    args: Any,
    *,
    input_path: Path,
    target: Path,
    output: Path,
    shards: int,
    shard: int,
    cpu_workers: str | int,
    prevalidated: bool = False,
) -> list[str]:
    command = ["igenvs", "screen"]
    if prevalidated:
        command.extend(["--prevalidated-input", str(input_path)])
    else:
        command.extend(
            [
                "--input", str(input_path), "--input-format", "csv",
                "--smiles-column", "smiles", "--id-column", "molecule_id",
            ]
        )
    command.extend([
        "--target", str(target), "--engine", args.engine,
        "--search-mode", args.search_mode, "--scoring", args.scoring,
        "--num-modes", str(args.num_modes), "--energy-range", str(args.energy_range),
        "--refine-step", str(args.refine_step), "--unidock-verbosity", str(args.unidock_verbosity),
        "--seed", str(args.seed), "--device-id", "0", "--max-gpu-memory", str(args.max_gpu_memory),
        "--batch-size", str(args.batch_size), "--prep-workers",
        str(cpu_workers if args.prep_workers == "auto" else args.prep_workers),
        "--prep-mode", args.prep_mode,
        "--embed-max-attempts", str(args.embed_max_attempts),
        "--embed-timeout", str(args.embed_timeout), "--validation-workers",
        str(cpu_workers if args.validation_workers == "auto" else args.validation_workers),
        "--fragment-policy", args.fragment_policy, "--pose-output", args.pose_output,
        "--num-shards", str(shards), "--shard-index", str(shard),
        "--output-dir", str(output),
    ])
    if args.no_refine:
        command.append("--no-refine")
    if args.no_deduplicate:
        command.append("--no-deduplicate")
    if args.keep_work:
        command.append("--keep-work")
    _append_option(command, "--batch-profile", args.batch_profile)
    _append_option(command, "--adgpu-runs", args.adgpu_runs)
    _append_option(command, "--adgpu-evaluations", args.adgpu_evaluations)
    if args.adgpu_no_heuristics:
        command.append("--adgpu-no-heuristics")
    if args.adgpu_no_autostop:
        command.append("--adgpu-no-autostop")
    command.extend(
        [
            "--adgpu-local-search", args.adgpu_local_search,
            "--adgpu-cpu-threads", str(args.adgpu_cpu_threads),
            "--adgpu-workers", str(args.adgpu_workers),
            "--adgpu-executable", args.adgpu_executable,
        ]
    )
    if args.scratch_dir:
        scratch = Path(args.scratch_dir).expanduser().resolve() / f"shard-{shard}"
        scratch.mkdir(parents=True, exist_ok=True)
        command.extend(["--scratch-dir", str(scratch)])
    return command


def prepare_fit_docking_library(
    args: Any,
    runtime: Runtime,
    job: Path,
    label: str,
    input_path: Path,
    base: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create one globally validated artifact shared by all logical shards."""

    validation = base / "validation"
    validated = validation / "validated.csv"
    manifest_path = base / "validation-manifest.json"
    desired = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "fragment_policy": args.fragment_policy,
        "deduplicate": not args.no_deduplicate,
    }
    if validated.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            all(existing.get(key) == value for key, value in desired.items())
            and existing.get("status") == "complete"
            and existing.get("validated_sha256") == sha256(validated)
        ):
            return validated, existing
    if validation.exists():
        validation.rename(
            validation.with_name(f"validation.incomplete-{int(time.time())}")
        )
    command = [
        "igenvs", "validate",
        "--input", str(input_path),
        "--input-format", "csv",
        "--smiles-column", "smiles",
        "--id-column", "molecule_id",
        "--fragment-policy", args.fragment_policy,
        "--workers", str(args.validation_workers),
        "--output-dir", str(validation),
    ]
    if args.no_deduplicate:
        command.append("--no-deduplicate")
    runtime.run_logged(
        "igenvs",
        command,
        job / f"logs/{label}-validation.log",
        gpu=False,
        extra_paths=(input_path,),
    )
    if not validated.is_file():
        raise PipelineError(f"shared validation did not produce {validated}")
    manifest = {
        **desired,
        "status": "complete",
        "validated": str(validated),
        "validated_sha256": sha256(validated),
        "valid_rows": file_rows(validated),
    }
    atomic_json(manifest_path, manifest)
    return validated, manifest


def run_docking(
    args: Any,
    runtime: Runtime,
    job: Path,
    label: str,
    input_path: Path,
) -> tuple[list[Path], list[str]]:
    base = job / (
        f"docking/{label}" if label.startswith("UDRL-") else f"al/{label}/docking"
    )
    if (base / "merge-manifest.json").is_file() and (base / "scores.csv").is_file():
        print(f"[iGenVS-ultra] {label} docking already merged", flush=True)
        return [], []
    validated_input, validation_manifest = prepare_fit_docking_library(
        args,
        runtime,
        job,
        label,
        input_path,
        base,
    )
    shard_plan_path = base / "shard-plan.json"
    existing_plan = (
        json.loads(shard_plan_path.read_text(encoding="utf-8"))
        if shard_plan_path.is_file()
        else None
    )
    logical_shards = int(getattr(args, "docking_logical_shards", 4))
    if logical_shards < 1:
        raise PipelineError("docking logical-shard count must be positive")
    gpus = docking_gpu_ids(args)[:logical_shards]
    desired_plan = {
        "schema_version": 2,
        "label": label,
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "num_shards": logical_shards,
        "logical_shards": logical_shards,
        "validated_input": str(validated_input),
        "validated_sha256": validation_manifest["validated_sha256"],
        "sharding_strategy": "globally_validated_fixed_modulo_logical_shards_v2",
    }
    if existing_plan is not None and existing_plan != desired_plan:
        raise PipelineError(
            f"incomplete docking stage has an incompatible shard plan: {shard_plan_path}; "
            "hardware-count-dependent legacy shards cannot be mixed with fixed logical shards"
        )
    if existing_plan is None:
        atomic_json(shard_plan_path, desired_plan)
    runs = base / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    inherited_affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    affinity_groups = partition_cpu_affinity(inherited_affinity, len(gpus))
    taskset = shutil.which("taskset")
    if len(gpus) > 1 and runtime.execution != "docker" and taskset is None:
        raise PipelineError("multi-GPU docking requires the standard 'taskset' utility for CPU isolation")
    cpu_workers: str | int = "auto" if args.prep_workers == "auto" else args.prep_workers
    queues: list[list[int]] = [[] for _ in gpus]
    shard_dirs = []
    for shard in range(logical_shards):
        output = runs / f"shard-{shard}"
        shard_dirs.append(output)
        manifest = output / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete":
            print(f"[iGenVS-ultra] {label} shard {shard} already complete", flush=True)
            continue
        if output.exists():
            output.rename(output.with_name(f"{output.name}.incomplete-{int(time.time())}"))
        queues[shard % len(gpus)].append(shard)

    pending: list[tuple[int, int, subprocess.Popen[Any], Any, Path]] = []

    def launch(gpu_slot: int, shard: int) -> None:
        gpu = gpus[gpu_slot]
        output = runs / f"shard-{shard}"
        command = build_docking_command(
            args,
            input_path=validated_input,
            target=job / "target",
            output=output,
            shards=logical_shards,
            shard=shard,
            cpu_workers=cpu_workers,
            prevalidated=True,
        )
        full = runtime._wrap(
            "igenvs",
            command,
            gpu=True,
            extra_paths=(validated_input,),
            cpu_affinity=(affinity_groups[gpu_slot] if len(gpus) > 1 else ()),
        )
        log_path = job / f"logs/{label}-shard-{shard}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8", newline="")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "APPTAINERENV_CUDA_VISIBLE_DEVICES": gpu,
                "OMP_NUM_THREADS": str(len(affinity_groups[gpu_slot])),
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        print(
            f"[iGenVS-ultra] starting {label} logical shard {shard}/{logical_shards} "
            f"on GPU {gpu}; CPU affinity={taskset_cpu_list(affinity_groups[gpu_slot])}",
            flush=True,
        )
        try:
            process = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, env=environment)
        except BaseException:
            log.close()
            for _, _, running, running_log, _ in pending:
                if running.poll() is None:
                    running.terminate()
                running_log.close()
            raise
        pending.append((gpu_slot, shard, process, log, log_path))

    for gpu_slot, queue in enumerate(queues):
        if queue:
            launch(gpu_slot, queue.pop(0))
    last_update = time.monotonic()
    try:
        while pending:
            remaining = []
            completed_slots = []
            for gpu_slot, shard, process, log, log_path in pending:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((gpu_slot, shard, process, log, log_path))
                    continue
                log.close()
                if returncode != 0:
                    for _, _, other, other_log, _ in remaining:
                        other.terminate()
                        other_log.close()
                    raise PipelineError(
                        f"{label} shard {shard} failed with exit code {returncode}; see {log_path}\n{tail(log_path)}"
                    )
                print(f"[iGenVS-ultra] completed {label} shard {shard}", flush=True)
                completed_slots.append(gpu_slot)
            pending = remaining
            for gpu_slot in completed_slots:
                if queues[gpu_slot]:
                    launch(gpu_slot, queues[gpu_slot].pop(0))
            if pending:
                now = time.monotonic()
                if now - last_update >= 60:
                    print(f"[iGenVS-ultra] {label}: {len(pending)} docking shard(s) still running", flush=True)
                    last_update = now
                time.sleep(2)
    except BaseException:
        for _, _, process, log, _ in pending:
            if process.poll() is None:
                process.terminate()
            log.close()
        raise
    return shard_dirs, gpus


def load_source_identity(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "molecule_id" not in fields or "smiles" not in fields:
            raise PipelineError(f"docking source must have molecule_id and smiles columns: {path}")
        return list(reader), fields


def merge_docking(
    job: Path,
    label: str,
    input_path: Path,
    shard_dirs: Sequence[Path],
    target_name: str,
) -> Path:
    base = job / (
        f"docking/{label}" if label.startswith("UDRL-") else f"al/{label}/docking"
    )
    output = base / "scores.csv"
    merge_manifest = base / "merge-manifest.json"
    if output.is_file() and merge_manifest.is_file():
        print(f"[iGenVS-ultra] {label} docking merge already complete", flush=True)
        return output
    sources, source_fields = load_source_identity(input_path)
    rows: list[Optional[dict[str, str]]] = [None] * len(sources)
    statuses: dict[str, int] = {}
    raw_fields: Optional[list[str]] = None
    shards = []
    for shard_index, directory in enumerate(shard_dirs):
        manifest_path = directory / "manifest.json"
        results_path = directory / "results.csv"
        if not manifest_path.is_file() or not results_path.is_file():
            raise PipelineError(f"incomplete docking shard: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise PipelineError(f"docking shard did not complete: {directory}")
        shard_count = 0
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            current_fields = list(reader.fieldnames or [])
            if raw_fields is None:
                raw_fields = current_fields
            elif current_fields != raw_fields:
                raise PipelineError("docking shard result schemas differ")
            for row in reader:
                index = int(row["source_row"]) - 1
                if index < 0 or index >= len(rows) or rows[index] is not None:
                    raise PipelineError(f"invalid/duplicate source row in docking shards: {index + 1}")
                if (index % len(shard_dirs)) != shard_index:
                    raise PipelineError(f"source row {index + 1} belongs to the wrong shard")
                source = sources[index]
                if row["molecule_id"] != source["molecule_id"]:
                    raise PipelineError(f"docking identity mismatch at source row {index + 1}")
                rows[index] = row
                statuses[row["status"]] = statuses.get(row["status"], 0) + 1
                shard_count += 1
        shards.append(
            {
                "shard_index": shard_index,
                "rows": shard_count,
                "manifest_sha256": sha256(manifest_path),
                "results_sha256": sha256(results_path),
            }
        )
    if any(row is None for row in rows):
        raise PipelineError(f"docking merge lacks {sum(row is None for row in rows)} terminal rows")
    assert raw_fields is not None
    preserved_fields = {
        "selection_order", "al_source_row", "acquisition_category", "category_rank",
        "ensemble_probability", "ensemble_mutual_information",
        "diversity_min_cosine_distance", "diversity_candidate_pool_size",
        "target", "model_stage", "original_smiles", "source_kind", "source_batch",
        *[f"member_probability_{seed}" for seed in SEEDS],
    }
    acquisition_fields = [
        field for field in source_fields
        if field not in {"molecule_id", "smiles"}
        and field not in raw_fields
        and field in preserved_fields
    ]
    fields = ["target_id", "library", "shard_index", *acquisition_fields, *raw_fields]
    temporary = output.with_suffix(".csv.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, raw in enumerate(rows):
            assert raw is not None
            prefix = {"target_id": target_name, "library": label, "shard_index": index % len(shard_dirs)}
            prefix.update({field: sources[index][field] for field in acquisition_fields})
            writer.writerow({**prefix, **raw})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    atomic_json(
        merge_manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "target": target_name,
            "library": label,
            "input": str(input_path),
            "input_sha256": sha256(input_path),
            "rows": len(rows),
            "status_counts": dict(sorted(statuses.items())),
            "successful": statuses.get("success", 0),
            "failed": len(rows) - statuses.get("success", 0),
            "output": str(output),
            "output_sha256": sha256(output),
            "shards": shards,
        },
    )
    return output


def model_operation(
    runtime: Runtime,
    job: Path,
    name: str,
    operation: Sequence[str],
) -> None:
    runtime.run_logged(
        "gmolai",
        runtime.model_command(operation),
        job / f"logs/{name}.log",
        gpu=True,
        extra_paths=(runtime.assets / "gMolAI-v2.0",),
    )


def fit(args: Any) -> dict[str, Any]:
    assets = resolve_assets(args)
    job = Path(args.output_dir).expanduser().resolve()
    config = make_fit_config(args, assets)
    runtime = Runtime(args, assets, job)
    if getattr(args, "dry_run", False):
        plan = {
            "command": "fit",
            "job": str(job),
            "target": config,
            "al_rounds": args.al_rounds,
            "docking_gpu_ids": visible_gpu_tokens(getattr(args, "gpu_ids", None)),
            "docking_logical_shards": args.docking_logical_shards,
            "reference_docking_rows": 300_000,
            "al_docking_rows": 30_000 * args.al_rounds,
            "runtime": runtime.execution,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan
    config = ensure_fit_config(args, job, assets)
    required = [
        assets / "phase-1-udrl/library/UDRL-train.csv",
        assets / "phase-1-udrl/library/UDRL-valid.csv",
        assets / "phase-1-udrl/embeddings/UDRL-train-embeddings.npz",
        assets / "phase-1-udrl/embeddings/UDRL-valid-embeddings.npz",
        assets / "phase-5-head-selection/artifacts/input-standardizer.npz",
    ]
    for round_number in range(1, args.al_rounds + 1):
        required.extend(
            [
                assets / f"phase-2-al-sets/library/AL-set-{round_number}.csv",
                assets / f"phase-2-al-sets/embeddings/AL-set-{round_number}-embeddings.npz",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PipelineError(f"required release assets are missing: {missing}")
    prepare_target(args, job, runtime, config)
    for library in ("UDRL-train", "UDRL-valid"):
        source = assets / f"phase-1-udrl/library/{library}.csv"
        shard_dirs, _ = run_docking(args, runtime, job, library, source)
        merge_docking(job, library, source, shard_dirs, config["target_name"])
    model_operation(
        runtime,
        job,
        "train-initial",
        ["train", "--job-dir", str(job), "--assets-dir", str(assets), "--stage", "initial"],
    )
    for round_number in range(1, args.al_rounds + 1):
        label = f"round-{round_number}"
        model_operation(
            runtime,
            job,
            f"acquire-{label}",
            ["acquire", "--job-dir", str(job), "--assets-dir", str(assets), "--round", str(round_number)],
        )
        source = job / f"al/{label}/acquisition/selected.csv"
        shard_dirs, _ = run_docking(args, runtime, job, label, source)
        merge_docking(job, label, source, shard_dirs, config["target_name"])
        model_operation(
            runtime,
            job,
            f"train-{label}",
            ["train", "--job-dir", str(job), "--assets-dir", str(assets), "--stage", label],
        )
    final = json.loads((job / "models/final.json").read_text(encoding="utf-8"))
    completed_rounds = 0 if final["stage"] == "initial" else int(final["stage"].split("-", 1)[1])
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "target": config["target_name"],
        "al_rounds_requested": args.al_rounds,
        "al_rounds_completed": completed_rounds,
        "final_model_stage": final["stage"],
        "final_model_manifest": final["ensemble_manifest"],
        "release_equivalent_docking_protocol": config["release_equivalent_docking_protocol"],
    }
    atomic_json(job / "fit-summary.json", summary)
    print(f"[iGenVS-ultra] fit complete; final head: {final['stage']}", flush=True)
    return summary


def screening_source_config(args: Any) -> dict[str, Any]:
    if getattr(args, "input", None):
        path = Path(args.input).expanduser().resolve()
        if not path.is_file():
            raise PipelineError(f"screening library does not exist: {path}")
        return {
            "kind": "external",
            "path": str(path),
            "sha256": sha256(path),
            "format": args.input_format,
            "smiles_column": args.smiles_column,
            "id_column": args.id_column,
            "delimiter": args.delimiter,
        }
    seed_file = Path(args.seed_file).expanduser().resolve() if args.seed_file else None
    if seed_file is not None and not seed_file.is_file():
        raise PipelineError(f"generation seed file does not exist: {seed_file}")
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else None
    if model_dir is not None and not model_dir.is_dir():
        raise PipelineError(f"iGen3 model directory does not exist: {model_dir}")
    return {
        "kind": "iGen3",
        "generate_count": int(args.generate_count),
        "model": args.model,
        "generation_mode": args.generation_mode,
        "seed_file": str(seed_file) if seed_file else None,
        "seed_file_sha256": sha256(seed_file) if seed_file else None,
        "seed_smiles": list(args.seed_smiles or []),
        "samples_per_seed": args.samples_per_seed,
        "generator_batch_size": args.generator_batch_size,
        "generator_max_batch_size": args.generator_max_batch_size,
        "model_dir": str(model_dir) if model_dir else None,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "greedy": args.greedy,
        "compile_generator": args.compile_generator,
        "compile_mode": args.compile_mode,
        "generator_seed": args.generator_seed,
        "include_seed_molecules": args.include_seed_molecules,
        "max_candidates": args.max_candidates,
        "max_candidate_multiplier": args.max_candidate_multiplier,
        "stagnation_limit": args.stagnation_limit,
        "generator_device": args.generator_device,
        "generator_dtype": args.generator_dtype,
        "generator_metrics": args.generator_metrics,
    }


def ensure_screen_config(
    args: Any,
    job: Path,
    assets: Path,
    screen: Path,
    batch_size: int,
    batch_decision: dict[str, Any],
    model_manifest: Path,
) -> dict[str, Any]:
    score_gpus = screening_gpu_ids(args)
    source = screening_source_config(args)
    generation_shards = (
        generation_logical_shard_count(args, score_gpus)
        if source["kind"] == "iGen3"
        else None
    )
    config = {
        "schema_version": 1,
        "source": source,
        "stream_batch_size": batch_size,
        "stream_batch_decision": batch_decision,
        "fragment_policy": args.fragment_policy,
        "validation_workers": args.validation_workers,
        "exclude_reference_libraries": args.exclude_reference_libraries,
        "save_policy": args.save_policy,
        "score_threshold": args.score_threshold if args.save_policy == "threshold" else None,
        "save_embeddings": args.keep_embeddings,
        "encoder": {
            "backend": args.encoder_backend,
            "batch_size": args.encoder_batch_size,
            "node_budget": args.encoder_node_budget,
            "workers": args.encoder_workers,
            "verify_rows": args.encoder_verify_rows,
            "threads": args.encoder_threads,
            "device": args.encoder_device,
        },
        "execution_engine": (
            "persistent_v2"
            if not args.keep_embeddings
            and os.environ.get("IGENVS_ULTRA_EPHEMERAL_WORKERS", "0") != "1"
            else "ephemeral_compatibility"
        ),
        "profile_cache": str(profile_cache_directory(job)),
        "cross_batch_overlap": (
            source["kind"] == "iGen3"
            and not args.keep_embeddings
            and os.environ.get("IGENVS_ULTRA_EPHEMERAL_WORKERS", "0") != "1"
        ),
        "screen_gpu_count": len(score_gpus),
        "generation_logical_shards": generation_shards,
        "model_manifest": str(model_manifest),
        "model_manifest_sha256": sha256(model_manifest),
        "assets_dir": str(assets),
    }
    path = screen / "config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        # Screens created before local score sharding were necessarily
        # single-GPU. Treat the absent field as that locked historical value.
        comparable.setdefault("screen_gpu_count", 1)
        # Generated screens created before multi-GPU iGen3 fan-out used one
        # generator process per stream batch. Preserve that resume contract.
        comparable.setdefault(
            "generation_logical_shards",
            1 if comparable.get("source", {}).get("kind") == "iGen3" else None,
        )
        if comparable != config:
            raise PipelineError(
                f"screen name already has different source/scoring settings: {path}; "
                "use another --screen-name"
            )
    else:
        screen.mkdir(parents=True, exist_ok=True)
        atomic_json(path, config)
    return config


def reference_libraries(assets: Path) -> list[Path]:
    return [
        assets / "phase-1-udrl/library/UDRL-train.csv",
        assets / "phase-1-udrl/library/UDRL-valid.csv",
        *[assets / f"phase-2-al-sets/library/AL-set-{number}.csv" for number in range(1, 6)],
        assets / "phase-3-test-set/library/test-set.csv",
    ]


def open_dedup_database(screen: Path) -> sqlite3.Connection:
    path = screen / "dedup.sqlite3"
    connection = sqlite3.connect(str(path), timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS smiles (value TEXT PRIMARY KEY, owner INTEGER NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.commit()
    return connection


def seed_reference_identities(
    connection: sqlite3.Connection,
    screen: Path,
    assets: Path,
    enabled: bool,
) -> dict[str, Any]:
    desired = "all-fixed-reference-libraries" if enabled else "disabled"
    stored = connection.execute("SELECT value FROM metadata WHERE key='reference_seed'").fetchone()
    if stored and stored[0] == desired:
        count = connection.execute("SELECT COUNT(*) FROM smiles WHERE owner=0").fetchone()[0]
        return {"status": "complete", "enabled": enabled, "unique_rows": int(count)}
    if stored and stored[0] != desired:
        raise PipelineError("dedup database reference policy differs from the screen configuration")
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("DELETE FROM smiles WHERE owner=0")
    inserted = 0
    inputs = []
    if enabled:
        for path in reference_libraries(assets):
            if not path.is_file():
                connection.rollback()
                raise PipelineError(f"reference library is missing: {path}")
            file_inserted = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if "smiles" not in (reader.fieldnames or []):
                    connection.rollback()
                    raise PipelineError(f"reference library lacks a smiles column: {path}")
                batch = []
                for row in reader:
                    batch.append((row["smiles"], 0))
                    if len(batch) == 10_000:
                        before = connection.total_changes
                        connection.executemany("INSERT OR IGNORE INTO smiles(value, owner) VALUES (?, ?)", batch)
                        file_inserted += connection.total_changes - before
                        batch = []
                if batch:
                    before = connection.total_changes
                    connection.executemany("INSERT OR IGNORE INTO smiles(value, owner) VALUES (?, ?)", batch)
                    file_inserted += connection.total_changes - before
            inserted += file_inserted
            inputs.append({"path": str(path), "sha256": sha256(path), "new_unique_rows": file_inserted})
            print(f"[iGenVS-ultra] seeded known identities from {path.name}: {file_inserted:,}", flush=True)
    connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('reference_seed', ?)", (desired,))
    connection.commit()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "enabled": enabled,
        "unique_rows": inserted,
        "inputs": inputs,
    }
    atomic_json(screen / "reference-dedup-manifest.json", manifest)
    return manifest


def replay_completed_admissions(connection: sqlite3.Connection, screen: Path) -> None:
    for marker in sorted((screen / "batches").glob("batch-*/admission.json")) if (screen / "batches").is_dir() else []:
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        owner = int(record["batch"])
        prepared = Path(record["prepared"])
        if not prepared.is_file():
            raise PipelineError(f"completed admission lacks prepared CSV: {prepared}")
        with prepared.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            values = [(row["smiles"], owner) for row in reader]
        connection.executemany("INSERT OR IGNORE INTO smiles(value, owner) VALUES (?, ?)", values)
    connection.commit()


def load_seen_smiles(connection: sqlite3.Connection) -> set[str]:
    """Materialize the exact dedup authority once for O(1) hot-path lookups."""
    started = time.perf_counter()
    values = {str(row[0]) for row in connection.execute("SELECT value FROM smiles")}
    print(
        f"[iGenVS-ultra] loaded {len(values):,} exact identities into memory "
        f"in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return values


def validate_library(
    runtime: Runtime,
    args: Any,
    input_path: Path,
    output: Path,
    log_path: Path,
    *,
    input_format: str,
    smiles_column: str,
    id_column: Optional[str],
    delimiter: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    marker = output / "validation.json"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    if output.exists():
        output.rename(output.with_name(f"{output.name}.incomplete-{int(time.time())}"))
    command = [
        "igenvs", "validate", "--input", str(input_path), "--input-format", input_format,
        "--smiles-column", smiles_column, "--delimiter", delimiter,
        "--fragment-policy", args.fragment_policy, "--workers", str(args.validation_workers),
        "--output-dir", str(output),
    ]
    if id_column:
        command.extend(["--id-column", id_column])
    runtime.run_logged("igenvs", command, log_path, gpu=False, extra_paths=(input_path,))
    valid = output / "validated.csv"
    rejected = output / "rejected.csv"
    if not valid.is_file() or not rejected.is_file():
        raise PipelineError("iGenVS validation did not produce its expected CSV outputs")
    record = {
        "schema_version": 1,
        "status": "complete",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "validated": str(valid),
        "validated_rows": file_rows(valid),
        "rejected": str(rejected),
        "rejected_rows": file_rows(rejected),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(marker, record)
    return record


def generated_rows_from_iGen3_contract(
    input_path: Path,
    output: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Use iGen3's valid-canonical output contract without a third RDKit pass.

    gMolAI's stricter inference policy remains authoritative immediately before
    encoding.  With the default reject-fragments policy, any structure that
    iGenVS validation would reject is therefore still rejected before scoring.
    """
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    rejected = output / "rejected.csv"
    if not rejected.is_file():
        temporary = rejected.with_suffix(".csv.partial")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=[
                    "molecule_id", "original_smiles", "source_row", "reason", "error"
                ],
                lineterminator="\n",
            ).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, rejected)
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle, start=1):
            smiles = line.strip()
            if not smiles or smiles.startswith("#"):
                continue
            rows.append(
                {
                    "molecule_id": f"generated-{source_row}",
                    "original_smiles": smiles,
                    "canonical_smiles": smiles,
                    "source_row": str(source_row),
                }
            )
    record = {
        "schema_version": 1,
        "status": "complete",
        "mode": "iGen3_valid_canonical_contract_then_gMol_policy",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "validated_rows": len(rows),
        "rejected": str(rejected),
        "rejected_rows": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "validation.json", record)
    return rows, record


def generation_logical_shard_count(args: Any, gpu_ids: Sequence[str]) -> int:
    """Resolve a hardware-independent iGen3 work decomposition."""
    if not gpu_ids:
        raise PipelineError("iGen3 generation requires at least one selected GPU")
    requested = getattr(args, "generation_logical_shards", "auto")
    count = len(gpu_ids) if requested == "auto" else int(requested)
    if count < len(gpu_ids):
        raise PipelineError(
            "--generation-logical-shards cannot be smaller than --screen-gpus; "
            "otherwise some selected GPUs would be idle during generation"
        )
    return count


def generation_shard_plan(
    requested: int,
    *,
    batch_number: int,
    base_seed: int,
    logical_shards: int,
) -> list[dict[str, int]]:
    """Split a batch deterministically without making it hardware-count dependent."""
    if requested <= 0 or batch_number <= 0 or logical_shards <= 0:
        raise PipelineError("invalid generated-batch shard dimensions")
    active_shards = min(requested, logical_shards)
    quotient, remainder = divmod(requested, active_shards)
    seed_offset = (batch_number - 1) * logical_shards
    return [
        {
            "logical_shard": index,
            "requested": quotient + int(index < remainder),
            "seed": base_seed + seed_offset + index,
        }
        for index in range(active_shards)
    ]


def build_generation_command(
    args: Any,
    *,
    output: Path,
    requested: int,
    seed: int,
    metrics_dir: Path,
) -> list[str]:
    command = ["igen3"]
    if args.model_dir:
        command.extend(["--model-dir", str(Path(args.model_dir).expanduser().resolve())])
    command.extend(
        [
            "generate", "--model", args.model, "--mode", args.generation_mode,
            "--output", str(output), "--count", str(requested),
            "--batch-size", str(args.generator_batch_size),
            "--max-batch-size", str(args.generator_max_batch_size),
            "--samples-per-seed", str(args.samples_per_seed),
            "--seed", str(seed), "--device", args.generator_device,
            "--dtype", args.generator_dtype,
            "--compile", "on" if args.compile_generator else "off",
            "--compile-mode", args.compile_mode, "--no-progress",
        ]
    )
    if args.seed_file:
        command.extend(["--seed-file", str(Path(args.seed_file).expanduser().resolve())])
    for item in args.seed_smiles or []:
        command.extend(["--seed-smiles", item])
    if args.include_seed_molecules:
        command.append("--include-seed-molecules")
    if args.temperature is not None:
        command.extend(["--temperature", str(args.temperature)])
    if args.top_k is not None:
        command.extend(["--top-k", str(args.top_k)])
    if args.greedy:
        command.append("--greedy")
    _append_option(command, "--max-candidates", args.max_candidates)
    _append_option(command, "--max-candidate-multiplier", args.max_candidate_multiplier)
    _append_option(command, "--stagnation-limit", args.stagnation_limit)
    if args.generator_metrics:
        command.extend(["--metrics", "--metrics-dir", str(metrics_dir)])
    return command


def generate_batch(
    runtime: Runtime,
    args: Any,
    screen: Path,
    batch_number: int,
    requested: int,
    generation_pool: Optional[PersistentGenerationPool] = None,
) -> tuple[Path, dict[str, Any]]:
    batch_dir = screen / f"batches/batch-{batch_number:06d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    raw = batch_dir / "generated.smi"
    marker = batch_dir / "generation.json"
    gpu_ids = screening_gpu_ids(args)
    logical_shards = generation_logical_shard_count(args, gpu_ids)
    plan = generation_shard_plan(
        requested,
        batch_number=batch_number,
        base_seed=int(args.generator_seed),
        logical_shards=logical_shards,
    )
    expected = {
        "requested": requested,
        "logical_shards": logical_shards,
        "shard_plan": plan,
    }
    if marker.is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
        comparable = {key: record.get(key) for key in expected}
        legacy_single_gpu = (
            logical_shards == 1
            and record.get("requested") == requested
            and record.get("seed") == plan[0]["seed"]
            and "logical_shards" not in record
        )
        if comparable != expected and not legacy_single_gpu:
            raise PipelineError(f"existing generated batch has incompatible settings: {marker}")
        if raw.is_file() and sha256(raw) == record.get("sha256"):
            return raw, record
    if raw.exists():
        raw.rename(raw.with_name(f"generated.incomplete-{int(time.time())}.smi"))

    shard_root = batch_dir / "generation-shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    extras = [Path(args.seed_file).expanduser().resolve()] if args.seed_file else []
    if args.model_dir:
        extras.append(Path(args.model_dir).expanduser().resolve())
    inherited_affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    affinity_groups = partition_cpu_affinity(inherited_affinity, len(gpu_ids))
    taskset = shutil.which("taskset")
    if len(gpu_ids) > 1 and runtime.execution != "docker" and taskset is None:
        raise PipelineError("multi-GPU generation requires the standard 'taskset' utility")

    shard_records: list[Optional[dict[str, Any]]] = [None] * len(plan)
    pending_plan = []
    for item in plan:
        index = item["logical_shard"]
        directory = shard_root / f"shard-{index:04d}"
        output = directory / "generated.smi"
        shard_marker = directory / "manifest.json"
        shard_expected = {
            "logical_shard": index,
            "requested": item["requested"],
            "seed": item["seed"],
        }
        if shard_marker.is_file() and output.is_file():
            record = json.loads(shard_marker.read_text(encoding="utf-8"))
            if (
                all(record.get(key) == value for key, value in shard_expected.items())
                and record.get("status") == "complete"
                and record.get("sha256") == sha256(output)
            ):
                shard_records[index] = record
                continue
        if directory.exists():
            directory.rename(
                directory.with_name(f"{directory.name}.incomplete-{time.time_ns()}")
            )
        directory.mkdir(parents=True)
        pending_plan.append((item, directory, output, shard_marker))

    generation_started = time.perf_counter()
    if generation_pool is not None and pending_plan:
        for wave_start in range(0, len(pending_plan), len(gpu_ids)):
            wave = pending_plan[wave_start : wave_start + len(gpu_ids)]
            requests = [
                {
                    "command": "generate",
                    "output": str(output),
                    "count": int(item["requested"]),
                    "seed": int(item["seed"]),
                    "metrics_dir": str(directory / "metrics") if args.generator_metrics else None,
                }
                for item, directory, output, _ in wave
            ]
            results = generation_pool.run_wave(requests)
            for lane, ((item, directory, output, shard_marker), result) in enumerate(
                zip(wave, results)
            ):
                if not output.is_file():
                    raise PipelineError(
                        f"persistent iGen3 shard {item['logical_shard']} produced no output"
                    )
                with output.open("r", encoding="utf-8") as handle:
                    produced = sum(
                        1 for line in handle if line.strip() and not line.startswith("#")
                    )
                if produced != item["requested"] or int(result["generated"]) != produced:
                    raise PipelineError(
                        f"persistent iGen3 shard {item['logical_shard']} produced "
                        f"{produced:,} rows; expected {item['requested']:,}"
                    )
                record = {
                    "schema_version": 1,
                    "status": "complete",
                    **item,
                    "produced": produced,
                    "gpu": gpu_ids[lane],
                    "elapsed_seconds": float(result["seconds"]),
                    "output": str(output),
                    "sha256": sha256(output),
                    "persistent_worker": True,
                    "generator": result,
                }
                atomic_json(shard_marker, record)
                shard_records[item["logical_shard"]] = record
                print(
                    f"[iGenVS-ultra] completed generated batch {batch_number} "
                    f"logical shard {item['logical_shard']} with resident iGen3",
                    flush=True,
                )
        pending_plan = []
    for wave_start in range(0, len(pending_plan), len(gpu_ids)):
        wave = pending_plan[wave_start : wave_start + len(gpu_ids)]
        running = []
        try:
            for lane, (item, directory, output, shard_marker) in enumerate(wave):
                gpu = gpu_ids[lane]
                command = build_generation_command(
                    args,
                    output=output,
                    requested=item["requested"],
                    seed=item["seed"],
                    metrics_dir=directory / "metrics",
                )
                full = runtime._wrap(
                    "igenvs",
                    command,
                    gpu=True,
                    extra_paths=tuple(extras),
                    cpu_affinity=(
                        affinity_groups[lane]
                        if runtime.execution == "docker" or taskset is not None
                        else ()
                    ),
                )
                log_path = directory / "generation.log"
                log = log_path.open("w", encoding="utf-8", newline="")
                environment = os.environ.copy()
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "APPTAINERENV_CUDA_VISIBLE_DEVICES": gpu,
                        "OMP_NUM_THREADS": str(len(affinity_groups[lane])),
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "PYTHONUNBUFFERED": "1",
                    }
                )
                print(
                    f"[iGenVS-ultra] starting generated batch {batch_number} "
                    f"logical shard {item['logical_shard']}/{len(plan)} on GPU {gpu}; "
                    f"count={item['requested']:,}, seed={item['seed']}",
                    flush=True,
                )
                process = subprocess.Popen(
                    full, stdout=log, stderr=subprocess.STDOUT, env=environment
                )
                running.append(
                    (item, gpu, process, log, log_path, output, shard_marker, time.perf_counter())
                )
            while running:
                remaining = []
                for item, gpu, process, log, log_path, output, shard_marker, started in running:
                    returncode = process.poll()
                    if returncode is None:
                        remaining.append(
                            (item, gpu, process, log, log_path, output, shard_marker, started)
                        )
                        continue
                    elapsed = time.perf_counter() - started
                    log.close()
                    if returncode != 0:
                        for _, _, other, other_log, _, _, _, _ in remaining:
                            other.terminate()
                            other_log.close()
                        raise PipelineError(
                            f"iGen3 generation shard {item['logical_shard']} failed with "
                            f"exit code {returncode}; see {log_path}\n{tail(log_path)}"
                        )
                    if not output.is_file():
                        raise PipelineError(
                            f"iGen3 generation shard {item['logical_shard']} produced no output"
                        )
                    with output.open("r", encoding="utf-8") as handle:
                        produced = sum(
                            1 for line in handle if line.strip() and not line.startswith("#")
                        )
                    if produced != item["requested"]:
                        raise PipelineError(
                            f"iGen3 generation shard {item['logical_shard']} produced "
                            f"{produced:,} rows; expected {item['requested']:,}"
                        )
                    record = {
                        "schema_version": 1,
                        "status": "complete",
                        **item,
                        "produced": produced,
                        "gpu": gpu,
                        "elapsed_seconds": elapsed,
                        "output": str(output),
                        "sha256": sha256(output),
                    }
                    atomic_json(shard_marker, record)
                    shard_records[item["logical_shard"]] = record
                    print(
                        f"[iGenVS-ultra] completed generated batch {batch_number} "
                        f"logical shard {item['logical_shard']}",
                        flush=True,
                    )
                running = remaining
                if running:
                    time.sleep(0.25)
        except BaseException:
            for _, _, process, log, _, _, _, _ in running:
                if process.poll() is None:
                    process.terminate()
                log.close()
            raise

    if any(record is None for record in shard_records):
        raise PipelineError("generated batch lacks a terminal logical-shard record")
    completed = [record for record in shard_records if record is not None]
    temporary = raw.with_suffix(".smi.partial")
    with temporary.open("wb") as destination:
        for record in completed:
            with Path(record["output"]).open("rb") as source:
                shutil.copyfileobj(source, destination)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, raw)
    produced = sum(int(record["produced"]) for record in completed)
    if produced != requested:
        raise PipelineError(
            f"merged iGen3 batch contains {produced:,} rows; expected {requested:,}"
        )
    record = {
        "schema_version": 1,
        "status": "complete",
        **expected,
        "produced": produced,
        "generation_gpu_count": len(gpu_ids),
        "parallel_elapsed_seconds": time.perf_counter() - generation_started,
        "worker_elapsed_seconds_sum": sum(
            float(item.get("elapsed_seconds", 0.0)) for item in completed
        ),
        "persistent_workers": generation_pool is not None,
        "shards": completed,
        "output": str(raw),
        "sha256": sha256(raw),
    }
    atomic_json(marker, record)
    return raw, record


def admit_batch(
    connection: sqlite3.Connection,
    screen: Path,
    batch_number: int,
    rows: Sequence[dict[str, str]],
    source_kind: str,
    accepted_offset: int,
    seen_smiles: Optional[set[str]] = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    batch_dir = screen / f"batches/batch-{batch_number:06d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    marker = batch_dir / "admission.json"
    if marker.is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if Path(record["prepared"]).is_file():
            return record
    prepared = batch_dir / "prepared.csv"
    rejected = batch_dir / "dedup-rejections.csv"
    prepared_tmp = prepared.with_suffix(".csv.partial")
    rejected_tmp = rejected.with_suffix(".csv.partial")
    accepted = 0
    duplicates = 0
    hot_seen = seen_smiles if seen_smiles is not None else load_seen_smiles(connection)
    newly_seen: list[str] = []
    connection.execute("BEGIN IMMEDIATE")
    connection.execute("DELETE FROM smiles WHERE owner=?", (batch_number,))
    try:
        with prepared_tmp.open("w", encoding="utf-8", newline="") as valid_handle, rejected_tmp.open(
            "w", encoding="utf-8", newline=""
        ) as rejected_handle:
            fields = ["molecule_id", "smiles", "original_smiles", "source_kind", "source_batch", "source_row"]
            valid_writer = csv.DictWriter(valid_handle, fieldnames=fields, lineterminator="\n")
            reject_writer = csv.DictWriter(
                rejected_handle,
                fieldnames=["molecule_id", "smiles", "source_batch", "source_row", "reason"],
                lineterminator="\n",
            )
            valid_writer.writeheader()
            reject_writer.writeheader()
            for row in rows:
                canonical = row["canonical_smiles"]
                if canonical in hot_seen:
                    duplicates += 1
                    reject_writer.writerow(
                        {
                            "molecule_id": row["molecule_id"],
                            "smiles": canonical,
                            "source_batch": batch_number,
                            "source_row": row["source_row"],
                            "reason": "duplicate_canonical_smiles",
                        }
                    )
                    continue
                hot_seen.add(canonical)
                newly_seen.append(canonical)
                accepted += 1
                molecule_id = (
                    f"IGEN3-{accepted_offset + accepted:012d}"
                    if source_kind == "iGen3"
                    else row["molecule_id"]
                )
                valid_writer.writerow(
                    {
                        "molecule_id": molecule_id,
                        "smiles": canonical,
                        "original_smiles": row["original_smiles"],
                        "source_kind": source_kind,
                        "source_batch": batch_number,
                        "source_row": row["source_row"],
                    }
                )
            valid_handle.flush()
            rejected_handle.flush()
            os.fsync(valid_handle.fileno())
            os.fsync(rejected_handle.fileno())
        connection.executemany(
            "INSERT INTO smiles(value, owner) VALUES (?, ?)",
            ((value, batch_number) for value in newly_seen),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        for value in newly_seen:
            hot_seen.discard(value)
        prepared_tmp.unlink(missing_ok=True)
        rejected_tmp.unlink(missing_ok=True)
        raise
    os.replace(prepared_tmp, prepared)
    os.replace(rejected_tmp, rejected)
    record = {
        "schema_version": 1,
        "status": "complete",
        "batch": batch_number,
        "source_kind": source_kind,
        "input_rows": len(rows),
        "accepted_rows": accepted,
        "duplicate_rows": duplicates,
        "prepared": str(prepared),
        "prepared_sha256": sha256(prepared),
        "dedup_rejections": str(rejected),
        "dedup_rejections_sha256": sha256(rejected),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(marker, record)
    return record


def write_empty_score_batch(output: Path, input_path: Path, model_manifest: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=SCORE_FIELDS, lineterminator="\n").writeheader()
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    atomic_json(
        output.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "status": "complete",
            "input": str(input_path),
            "input_rows": 0,
            "encoded_rows": 0,
            "encoder_rejected_rows": 0,
            "retained_rows": 0,
            "output": str(output),
            "output_sha256": sha256(output),
            "model_manifest": str(model_manifest),
        },
    )


def screening_gpu_ids(args: Any) -> list[str]:
    """Resolve the local GPU fan-out for streamed model scoring."""
    available = visible_gpu_tokens(getattr(args, "gpu_ids", None))
    requested = getattr(args, "screen_gpus", "auto")
    if requested == "auto":
        count = len(available)
        # A Slurm step may leave every node GPU visible even when its task was
        # granted fewer devices. Never exceed the task-local GPU request.
        slurm_per_task = os.environ.get("SLURM_GPUS_PER_TASK", "").strip()
        if slurm_per_task:
            match = re.search(r"(\d+)$", slurm_per_task)
            if match and int(match.group(1)) > 0:
                count = min(count, int(match.group(1)))
    else:
        count = int(requested)
    if count < 1:
        raise PipelineError("screening requires at least one visible GPU")
    if len(available) < count:
        raise PipelineError(
            f"requested {count} screening GPUs but only {len(available)} are visible"
        )
    if count > 1 and getattr(args, "encoder_device", "auto") != "auto":
        raise PipelineError(
            "multi-GPU screening requires --encoder-device auto; use --gpu-ids to choose devices"
        )
    return available[:count]


def aligned_score_shard_lengths(rows: int, shards: int, alignment: int) -> list[int]:
    """Balance contiguous shards without changing full encoder batch boundaries."""
    if rows < 0 or shards <= 0 or alignment <= 0:
        raise PipelineError("invalid score-shard dimensions")
    if rows == 0:
        return []
    full_batches, remainder = divmod(rows, alignment)
    # A split is allowed only where the one-GPU encoder would already end a
    # batch. Small inputs therefore use fewer devices instead of changing the
    # numerical operation grouping.
    maximum_aligned_shards = full_batches + int(remainder > 0)
    shards = min(shards, maximum_aligned_shards)
    if full_batches >= shards:
        quotient, extra = divmod(full_batches, shards)
        lengths = [(quotient + int(index < extra)) * alignment for index in range(shards)]
        lengths[-1] += remainder
    else:
        lengths = [alignment] * full_batches + [remainder]
    if sum(lengths) != rows or any(length <= 0 for length in lengths):
        raise PipelineError("could not construct a valid score-shard plan")
    return lengths


def prepare_score_shards(
    input_path: Path,
    output: Path,
    *,
    rows: int,
    shards: int,
    alignment: int,
) -> list[dict[str, Any]]:
    """Create stable contiguous CSV shards and an auditable resume marker."""
    root = output.parent / "score-shards"
    plan_path = root / "plan.json"
    input_hash = sha256(input_path)
    lengths = aligned_score_shard_lengths(rows, shards, alignment)
    expected = {
        "schema_version": 1,
        "status": "complete",
        "input": str(input_path),
        "input_sha256": input_hash,
        "input_rows": rows,
        "alignment": alignment,
        "num_shards": len(lengths),
        "lengths": lengths,
    }
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise PipelineError(f"existing score-shard plan is incompatible: {plan_path}")
        records = list(existing.get("shards", []))
        if len(records) != len(lengths):
            raise PipelineError(f"score-shard plan is incomplete: {plan_path}")
        for record in records:
            path = Path(record["input"])
            if not path.is_file() or sha256(path) != record.get("input_sha256"):
                raise PipelineError(f"score-shard input changed: {path}")
        return records
    if root.exists():
        root.rename(root.with_name(f"{root.name}.incomplete-{int(time.time())}"))
    root.mkdir(parents=True)
    temporary_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            if not fields:
                raise PipelineError(f"score input has no CSV header: {input_path}")
            offset = 0
            for index, length in enumerate(lengths):
                directory = root / f"shard-{index:04d}"
                directory.mkdir()
                path = directory / "prepared.csv"
                temporary = path.with_suffix(".csv.partial")
                temporary_paths.append(temporary)
                with temporary.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    for _ in range(length):
                        try:
                            writer.writerow(next(reader))
                        except StopIteration as exc:
                            raise PipelineError(
                                f"score input ended before its declared {rows:,} rows"
                            ) from exc
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                records.append(
                    {
                        "shard_index": index,
                        "input": str(path),
                        "input_sha256": sha256(path),
                        "input_rows": length,
                        "input_row_offset": offset,
                    }
                )
                offset += length
            try:
                next(reader)
            except StopIteration:
                pass
            else:
                raise PipelineError(f"score input exceeds its declared {rows:,} rows")
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    atomic_json(plan_path, {**expected, "shards": records})
    return records


def build_score_operation(
    args: Any,
    job: Path,
    assets: Path,
    model_manifest: Path,
    input_path: Path,
    output: Path,
) -> list[str]:
    command = [
        "score", "--job-dir", str(job), "--assets-dir", str(assets),
        "--gmolai-dir", str(assets / "gMolAI-v2.0"),
        "--gmolai-models-dir", str(assets / "gMolAI-v2.0/inference/models"),
        "--model-manifest", str(model_manifest), "--input", str(input_path),
        "--output", str(output), "--save-policy", args.save_policy,
        "--device", args.encoder_device, "--encoder-backend", args.encoder_backend,
        "--encoder-batch-size", str(args.encoder_batch_size),
        "--encoder-node-budget", str(args.encoder_node_budget),
        "--encoder-workers", str(args.encoder_workers),
        "--encoder-verify-rows", str(args.encoder_verify_rows),
        "--encoder-threads", str(args.encoder_threads),
        "--profile-cache", str(profile_cache_directory(job)),
    ]
    if args.save_policy == "threshold":
        command.extend(["--score-threshold", str(args.score_threshold)])
    if args.keep_embeddings:
        command.append("--keep-embeddings")
    return command


def build_score_worker_operation(
    args: Any,
    job: Path,
    assets: Path,
    model_manifest: Path,
    profile_cache: Path,
) -> list[str]:
    command = [
        "score-worker", "--job-dir", str(job), "--assets-dir", str(assets),
        "--gmolai-dir", str(assets / "gMolAI-v2.0"),
        "--gmolai-models-dir", str(assets / "gMolAI-v2.0/inference/models"),
        "--model-manifest", str(model_manifest), "--save-policy", args.save_policy,
        "--device", args.encoder_device, "--encoder-backend", args.encoder_backend,
        "--encoder-batch-size", str(args.encoder_batch_size),
        "--encoder-node-budget", str(args.encoder_node_budget),
        "--encoder-workers", str(args.encoder_workers),
        "--encoder-verify-rows", str(args.encoder_verify_rows),
        "--encoder-threads", str(args.encoder_threads),
        "--profile-cache", str(profile_cache),
    ]
    if args.save_policy == "threshold":
        command.extend(["--score-threshold", str(args.score_threshold)])
    return command


def merge_score_shards(
    input_path: Path,
    output: Path,
    model_manifest: Path,
    shard_plan: Sequence[dict[str, Any]],
    shard_manifests: Sequence[dict[str, Any]],
    *,
    parallel_wall_seconds: float,
    keep_embeddings: bool,
) -> dict[str, Any]:
    """Restore source order and expose one ordinary score-batch manifest."""
    retained = atomic_concatenate_csv(
        [Path(manifest["output"]) for manifest in shard_manifests], output
    )

    rejection_path = output.with_suffix(".embeddings.rejections.csv")
    rejection_tmp = rejection_path.with_suffix(".csv.partial")
    rejection_header: Optional[list[str]] = None
    rejected = 0
    with rejection_tmp.open("w", encoding="utf-8", newline="") as target:
        writer = None
        for plan, manifest in zip(shard_plan, shard_manifests):
            source_path = Path(manifest["encoder_rejections"])
            with source_path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                fields = list(reader.fieldnames or [])
                if rejection_header is None:
                    rejection_header = fields
                    writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                elif fields != rejection_header:
                    raise PipelineError("parallel encoder rejection schemas differ")
                assert writer is not None
                for row in reader:
                    if row.get("input_row", "").strip():
                        row["input_row"] = str(
                            int(row["input_row"]) + int(plan["input_row_offset"])
                        )
                    writer.writerow(row)
                    rejected += 1
        if rejection_header is None:
            rejection_header = ["input_row", "input_id", "input_smiles", "reason", "error"]
            csv.DictWriter(target, fieldnames=rejection_header, lineterminator="\n").writeheader()
        target.flush()
        os.fsync(target.fileno())
    os.replace(rejection_tmp, rejection_path)

    input_rows = sum(int(item["input_rows"]) for item in shard_manifests)
    encoded_rows = sum(int(item["encoded_rows"]) for item in shard_manifests)
    if rejected != input_rows - encoded_rows:
        raise PipelineError("parallel score rejection accounting differs from encoded rows")
    first = shard_manifests[0]
    member_timings: dict[int, float] = {}
    member_templates: dict[int, dict[str, Any]] = {}
    for manifest in shard_manifests:
        if manifest["model_manifest_sha256"] != first["model_manifest_sha256"]:
            raise PipelineError("parallel score shards used different model manifests")
        for member in manifest.get("members", []):
            seed = int(member["seed"])
            member_templates[seed] = member
            member_timings[seed] = member_timings.get(seed, 0.0) + float(
                member.get("inference_seconds", 0.0)
            )
    members = []
    for seed in sorted(member_templates):
        member = dict(member_templates[seed])
        member["inference_seconds"] = member_timings[seed]
        member["inference_seconds_shard_sum"] = member_timings[seed]
        members.append(member)
    inference_sum = sum(
        float(item.get("inference_seconds_member_sum", 0.0)) for item in shard_manifests
    )
    encoder_elapsed_max = max(float(item["encoder"]["elapsed_seconds"]) for item in shard_manifests)
    metadata_path = output.with_suffix(".embeddings.metadata.json")
    metadata_shards = [Path(item["encoder_metadata"]) for item in shard_manifests]
    atomic_json(
        metadata_path,
        {
            "schema_version": 1,
            "status": "complete",
            "artifact_type": "igenvs_ultra_parallel_gmol_encoding",
            "input": {"path": str(input_path), "sha256": sha256(input_path)},
            "rows": {"total": input_rows, "accepted": encoded_rows, "rejected": rejected},
            "execution": {
                "score_shards": len(shard_manifests),
                "parallel_elapsed_seconds": parallel_wall_seconds,
                "maximum_shard_encoder_elapsed_seconds": encoder_elapsed_max,
                "rows_per_second_parallel": encoded_rows / parallel_wall_seconds,
            },
            "shards": [
                {
                    "metadata": str(path),
                    "metadata_sha256": sha256(path),
                    "input_rows": int(manifest["input_rows"]),
                    "encoded_rows": int(manifest["encoded_rows"]),
                }
                for path, manifest in zip(metadata_shards, shard_manifests)
            ],
        },
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "input_rows": input_rows,
        "encoded_rows": encoded_rows,
        "encoder_rejected_rows": rejected,
        "retained_rows": retained,
        "save_policy": first["save_policy"],
        "score_threshold": first["score_threshold"],
        "output": str(output),
        "output_sha256": sha256(output),
        "model_manifest": str(model_manifest),
        "model_manifest_sha256": sha256(model_manifest),
        "members": members,
        "inference_seconds_member_sum": inference_sum,
        "inference_seconds_member_shard_sum": inference_sum,
        "parallel_score_wall_seconds": parallel_wall_seconds,
        "encoder": {
            "accepted": encoded_rows,
            "rejected": rejected,
            "dimensions": first["encoder"]["dimensions"],
            "embedding_space": first["encoder"]["embedding_space"],
            "backend": first["encoder"]["backend"],
            "device": "multi-gpu",
            "devices": [item["encoder"].get("device") for item in shard_manifests],
            "workers_per_shard": [item["encoder"].get("workers") for item in shard_manifests],
            "parallel_elapsed_seconds": parallel_wall_seconds,
            "elapsed_seconds": encoder_elapsed_max,
            "maximum_shard_elapsed_seconds": encoder_elapsed_max,
            "rows_per_second_parallel": encoded_rows / parallel_wall_seconds,
            "rows_per_second": encoded_rows / encoder_elapsed_max,
            "shards": len(shard_manifests),
            "metadata": str(metadata_path),
            "rejections": str(rejection_path),
        },
        "encoder_rejections": str(rejection_path),
        "encoder_rejections_sha256": sha256(rejection_path),
        "encoder_metadata": str(metadata_path),
        "encoder_metadata_shards": [str(path) for path in metadata_shards],
        "embeddings_retained": keep_embeddings,
        "embedding_shards": [
            str(Path(item["output"]).with_suffix(".embeddings.npz"))
            for item in shard_manifests
        ] if keep_embeddings else [],
        "score_shards": [
            {
                **dict(plan),
                "output": manifest["output"],
                "output_sha256": manifest["output_sha256"],
                "encoded_rows": manifest["encoded_rows"],
                "retained_rows": manifest["retained_rows"],
            }
            for plan, manifest in zip(shard_plan, shard_manifests)
        ],
    }
    atomic_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def run_parallel_score_shards(
    runtime: Runtime,
    args: Any,
    job: Path,
    assets: Path,
    screen: Path,
    batch_number: int,
    input_path: Path,
    output: Path,
    model_manifest: Path,
    shard_plan: Sequence[dict[str, Any]],
    gpu_ids: Sequence[str],
) -> dict[str, Any]:
    inherited_affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    affinity_groups = partition_cpu_affinity(inherited_affinity, len(shard_plan))
    taskset = shutil.which("taskset")
    if runtime.execution != "docker" and taskset is None:
        raise PipelineError("multi-GPU screening requires the standard 'taskset' utility")
    pending = []
    manifests: list[Optional[dict[str, Any]]] = [None] * len(shard_plan)
    started = time.perf_counter()
    try:
        for plan, gpu, affinity in zip(shard_plan, gpu_ids, affinity_groups):
            index = int(plan["shard_index"])
            shard_input = Path(plan["input"])
            shard_output = shard_input.parent / "scores.csv"
            shard_manifest = shard_output.with_suffix(".manifest.json")
            if shard_manifest.is_file() and shard_output.is_file():
                record = json.loads(shard_manifest.read_text(encoding="utf-8"))
                if record.get("status") == "complete" and record.get("output_sha256") == sha256(shard_output):
                    manifests[index] = record
                    print(
                        f"[iGenVS-ultra] screen {screen.name} batch {batch_number} "
                        f"score shard {index} already complete",
                        flush=True,
                    )
                    continue
            operation = build_score_operation(
                args, job, assets, model_manifest, shard_input, shard_output
            )
            full = runtime._wrap(
                "gmolai",
                runtime.model_command(operation),
                gpu=True,
                extra_paths=(runtime.assets / "gMolAI-v2.0",),
                cpu_affinity=affinity,
            )
            log_path = job / (
                f"logs/screen-{screen.name}-batch-{batch_number:06d}-shard-{index:04d}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8", newline="")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "APPTAINERENV_CUDA_VISIBLE_DEVICES": gpu,
                    "OMP_NUM_THREADS": str(len(affinity)),
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            print(
                f"[iGenVS-ultra] starting screen {screen.name} batch {batch_number} "
                f"score shard {index}/{len(shard_plan)} on GPU {gpu}; "
                f"CPU affinity={taskset_cpu_list(affinity)}",
                flush=True,
            )
            process = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, env=environment)
            pending.append((index, process, log, log_path, shard_manifest, shard_output))
        while pending:
            remaining = []
            for index, process, log, log_path, shard_manifest, shard_output in pending:
                returncode = process.poll()
                if returncode is None:
                    remaining.append((index, process, log, log_path, shard_manifest, shard_output))
                    continue
                log.close()
                if returncode != 0:
                    for _, other, other_log, _, _, _ in remaining:
                        other.terminate()
                        other_log.close()
                    raise PipelineError(
                        f"screen score shard {index} failed with exit code {returncode}; "
                        f"see {log_path}\n{tail(log_path)}"
                    )
                manifests[index] = json.loads(shard_manifest.read_text(encoding="utf-8"))
                print(
                    f"[iGenVS-ultra] completed screen {screen.name} batch {batch_number} "
                    f"score shard {index}",
                    flush=True,
                )
            pending = remaining
            if pending:
                time.sleep(1)
    except BaseException:
        for _, process, log, _, _, _ in pending:
            if process.poll() is None:
                process.terminate()
            log.close()
        raise
    if any(item is None for item in manifests):
        raise PipelineError("parallel score shard did not produce a terminal manifest")
    completed = [item for item in manifests if item is not None]
    return merge_score_shards(
        input_path,
        output,
        model_manifest,
        shard_plan,
        completed,
        parallel_wall_seconds=time.perf_counter() - started,
        keep_embeddings=bool(args.keep_embeddings),
    )


def score_batch(
    runtime: Runtime,
    args: Any,
    job: Path,
    assets: Path,
    screen: Path,
    batch_number: int,
    admission: dict[str, Any],
    model_manifest: Path,
    score_pool: Optional[PersistentScorePool] = None,
) -> dict[str, Any]:
    input_path = Path(admission["prepared"])
    output = screen / f"batches/batch-{batch_number:06d}/scores.csv"
    manifest = output.with_suffix(".manifest.json")
    if manifest.is_file() and output.is_file():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("status") == "complete" and record.get("output_sha256") == sha256(output):
            return record
    if int(admission["accepted_rows"]) == 0:
        write_empty_score_batch(output, input_path, model_manifest)
        return json.loads(manifest.read_text(encoding="utf-8"))
    gpu_ids = screening_gpu_ids(args)
    encoder_alignment = (
        512 if str(args.encoder_batch_size).lower() == "auto" else int(args.encoder_batch_size)
    )
    aligned_lengths = aligned_score_shard_lengths(
        int(admission["accepted_rows"]), len(gpu_ids), encoder_alignment
    )
    if len(aligned_lengths) > 1:
        shard_plan = prepare_score_shards(
            input_path,
            output,
            rows=int(admission["accepted_rows"]),
            shards=len(gpu_ids),
            alignment=encoder_alignment,
        )
        if score_pool is not None:
            started = time.perf_counter()
            responses = score_pool.run(
                [
                    {
                        "command": "score",
                        "input": plan["input"],
                        "output": str(Path(plan["input"]).parent / "scores.csv"),
                    }
                    for plan in shard_plan
                ]
            )
            manifests = [
                json.loads(Path(response["manifest"]).read_text(encoding="utf-8"))
                for response in responses
            ]
            return merge_score_shards(
                input_path,
                output,
                model_manifest,
                shard_plan,
                manifests,
                parallel_wall_seconds=time.perf_counter() - started,
                keep_embeddings=False,
            )
        return run_parallel_score_shards(
            runtime,
            args,
            job,
            assets,
            screen,
            batch_number,
            input_path,
            output,
            model_manifest,
            shard_plan,
            gpu_ids[: len(shard_plan)],
        )
    if score_pool is not None:
        response = score_pool.run(
            [{"command": "score", "input": str(input_path), "output": str(output)}]
        )[0]
        return json.loads(Path(response["manifest"]).read_text(encoding="utf-8"))
    command = build_score_operation(args, job, assets, model_manifest, input_path, output)
    model_operation(runtime, job, f"screen-{screen.name}-batch-{batch_number:06d}", command)
    return json.loads(manifest.read_text(encoding="utf-8"))


def iter_csv_chunks(path: Path, size: int) -> Iterator[list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


def process_external_screen(
    runtime: Runtime,
    args: Any,
    job: Path,
    assets: Path,
    screen: Path,
    connection: sqlite3.Connection,
    batch_size: int,
    model_manifest: Path,
    seen_smiles: set[str],
    score_pool: Optional[PersistentScorePool] = None,
) -> list[dict[str, Any]]:
    source = Path(args.input).expanduser().resolve()
    validation = validate_library(
        runtime,
        args,
        source,
        screen / "source-validation",
        screen / "logs/source-validation.log",
        input_format=args.input_format,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        delimiter=args.delimiter,
    )
    records = []
    accepted_total = 0
    for batch_number, rows in enumerate(iter_csv_chunks(Path(validation["validated"]), batch_size), start=1):
        admission = admit_batch(
            connection,
            screen,
            batch_number,
            rows,
            "external",
            accepted_total,
            seen_smiles,
        )
        accepted_total += int(admission["accepted_rows"])
        score = score_batch(
            runtime,
            args,
            job,
            assets,
            screen,
            batch_number,
            admission,
            model_manifest,
            score_pool,
        )
        records.append({"batch": batch_number, "admission": admission, "score": score})
    return records


def process_generated_screen(
    runtime: Runtime,
    args: Any,
    job: Path,
    assets: Path,
    screen: Path,
    connection: sqlite3.Connection,
    batch_size: int,
    model_manifest: Path,
    seen_smiles: set[str],
    generation_pool: Optional[PersistentGenerationPool] = None,
    score_pool: Optional[PersistentScorePool] = None,
) -> list[dict[str, Any]]:
    target_count = int(args.generate_count)
    records = []
    admitted_total = 0
    committed_total = 0
    batch_number = 1
    stagnant = 0
    default_max = int(math.ceil(target_count / batch_size)) * 3 + 10
    maximum_batches = args.max_stream_batches or default_max
    # The resident generator and scorer own separate processes on every selected
    # GPU. Keep at most one future generation batch in flight while the current
    # admitted batch is scored. The next request excludes rows already admitted
    # in the current batch, so an exactly-completing batch never launches a
    # wasteful full-size tail. Encoding rejections are replenished afterwards.
    overlap = generation_pool is not None and score_pool is not None
    generation_executor = ThreadPoolExecutor(max_workers=1) if overlap else None
    pending_generation = None
    try:
        while committed_total < target_count:
            if batch_number > maximum_batches:
                raise PipelineError(
                    f"generation did not reach {target_count:,} globally unique molecules within "
                    f"{maximum_batches} stream batches; increase --max-stream-batches"
                )
            requested = min(batch_size, target_count - committed_total)
            if pending_generation is None:
                raw, generation = generate_batch(
                    runtime,
                    args,
                    screen,
                    batch_number,
                    requested,
                    generation_pool,
                )
            else:
                raw, generation = pending_generation.result()
                pending_generation = None
            batch_dir = screen / f"batches/batch-{batch_number:06d}"
            if args.fragment_policy == "reject":
                rows, validation = generated_rows_from_iGen3_contract(
                    raw, batch_dir / "validation"
                )
            else:
                validation = validate_library(
                    runtime,
                    args,
                    raw,
                    batch_dir / "validation",
                    batch_dir / "validation.log",
                    input_format="smi",
                    smiles_column="smiles",
                    id_column=None,
                    delimiter="auto",
                )
                with Path(validation["validated"]).open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    rows = list(csv.DictReader(handle))
            admission = admit_batch(
                connection,
                screen,
                batch_number,
                rows,
                "iGen3",
                admitted_total,
                seen_smiles,
            )
            accepted = int(admission["accepted_rows"])
            minimum_next = target_count - committed_total - accepted
            prefetched_next = overlap and minimum_next > 0
            if prefetched_next:
                assert generation_executor is not None
                next_requested = min(batch_size, minimum_next)
                pending_generation = generation_executor.submit(
                    generate_batch,
                    runtime,
                    args,
                    screen,
                    batch_number + 1,
                    next_requested,
                    generation_pool,
                )
            score = score_batch(
                runtime,
                args,
                job,
                assets,
                screen,
                batch_number,
                admission,
                model_manifest,
                score_pool,
            )
            committed = int(score.get("encoded_rows", 0))
            records.append(
                {
                    "batch": batch_number,
                    "generation": generation,
                    "validation": validation,
                    "admission": admission,
                    "score": score,
                    "prefetched_next_generation": prefetched_next,
                }
            )
            admitted_total += accepted
            committed_total += committed
            stagnant = stagnant + 1 if committed == 0 else 0
            print(
                f"[iGenVS-ultra] generated stream batch {batch_number}: "
                f"admitted={accepted:,}, scored={committed:,}, "
                f"committed={committed_total:,}/{target_count:,}, "
                f"next_generation_prefetched={prefetched_next}",
                flush=True,
            )
            if stagnant >= 3:
                raise PipelineError("three consecutive generated batches committed no new score")
            batch_number += 1
    finally:
        if generation_executor is not None:
            generation_executor.shutdown(wait=True, cancel_futures=True)
    if committed_total != target_count:
        raise PipelineError(
            f"generated screen committed {committed_total:,} scores; expected exactly {target_count:,}"
        )
    return records


def normalize_rejections(screen: Path, batch_records: Sequence[dict[str, Any]]) -> tuple[Path, int]:
    output = screen / "rejections.csv"
    temporary = output.with_suffix(".csv.partial")
    fields = ["stage", "batch", "input_row", "molecule_id", "smiles", "reason", "detail"]
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()

        def append(path: Path, stage: str, batch: Any) -> None:
            nonlocal count
            if not path.is_file():
                return
            with path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    writer.writerow(
                        {
                            "stage": stage,
                            "batch": batch,
                            "input_row": row.get("input_row", row.get("source_row", "")),
                            "molecule_id": row.get("input_id", row.get("molecule_id", "")),
                            "smiles": row.get("input_smiles", row.get("original_smiles", row.get("smiles", ""))),
                            "reason": row.get("reason", row.get("status", "")),
                            "detail": row.get("error", ""),
                        }
                    )
                    count += 1

        source_rejections = screen / "source-validation/rejected.csv"
        append(source_rejections, "validation", "")
        for item in batch_records:
            number = item["batch"]
            batch_dir = screen / f"batches/batch-{number:06d}"
            if "generation" in item:
                append(batch_dir / "validation/rejected.csv", "validation", number)
            append(Path(item["admission"]["dedup_rejections"]), "deduplication", number)
            score = item["score"]
            if score.get("encoder_rejections"):
                append(Path(score["encoder_rejections"]), "encoding", number)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output)
    return output, count


def finalize_screen(
    screen: Path,
    config: dict[str, Any],
    batch_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    final_manifest = screen / "manifest.json"
    results = screen / "results.csv"
    if final_manifest.is_file() and results.is_file():
        record = json.loads(final_manifest.read_text(encoding="utf-8"))
        if record.get("results_sha256") == sha256(results):
            return record
    retained = atomic_concatenate_csv(
        [Path(item["score"]["output"]) for item in batch_records], results
    )
    rejection_path, rejection_count = normalize_rejections(screen, batch_records)
    admitted = sum(int(item["admission"]["accepted_rows"]) for item in batch_records)
    encoded = sum(int(item["score"].get("encoded_rows", 0)) for item in batch_records)
    record = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "source_kind": config["source"]["kind"],
        "execution_engine": config.get("execution_engine", "ephemeral_compatibility"),
        "cross_batch_overlap": any(
            bool(item.get("prefetched_next_generation")) for item in batch_records
        ),
        "stream_batch_size": config["stream_batch_size"],
        "stream_batches": len(batch_records),
        "admitted_unique_rows": admitted,
        "encoded_rows": encoded,
        "encoding_rejections": admitted - encoded,
        "saved_rows": retained,
        "save_policy": config["save_policy"],
        "score_threshold": config["score_threshold"],
        "results": str(results),
        "results_sha256": sha256(results),
        "rejections": str(rejection_path),
        "rejection_rows": rejection_count,
        "model_manifest": config["model_manifest"],
        "model_manifest_sha256": config["model_manifest_sha256"],
        "requested_successful_scores": (
            int(config["source"]["generate_count"])
            if config["source"]["kind"] == "iGen3"
            else None
        ),
        "exact_requested_score_count": (
            encoded == int(config["source"]["generate_count"])
            if config["source"]["kind"] == "iGen3"
            else None
        ),
        "batches": [
            {
                "batch": item["batch"],
                "admitted": int(item["admission"]["accepted_rows"]),
                "encoded": int(item["score"].get("encoded_rows", 0)),
                "saved": int(item["score"].get("retained_rows", 0)),
            }
            for item in batch_records
        ],
    }
    atomic_json(final_manifest, record)
    return record


def screen(args: Any) -> dict[str, Any]:
    screen_started = time.perf_counter()
    assets = resolve_assets(args)
    job_value = getattr(args, "job_dir", None) or getattr(args, "output_dir", None)
    job = Path(job_value).expanduser().resolve()
    fit_config_path = job / "fit-config.json"
    final_path = job / "models/final.json"
    if not fit_config_path.is_file() or not final_path.is_file():
        raise PipelineError(f"job has no completed target head; run 'fit' first: {job}")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    model_manifest = job / final["ensemble_manifest"]
    if not model_manifest.is_file():
        raise PipelineError(f"final model manifest is missing: {model_manifest}")
    screen_name = sanitize_target_name(args.screen_name)
    screen_dir = job / "screens" / screen_name
    source_config = screening_source_config(args)
    existing_config_path = screen_dir / "config.json"
    if existing_config_path.is_file():
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        batch_size = int(existing_config["stream_batch_size"])
        decision = existing_config["stream_batch_decision"]
        if args.stream_batch_size != "auto" and int(args.stream_batch_size) != batch_size:
            raise PipelineError(
                f"screen was started with stream batch size {batch_size}; "
                "use a new --screen-name to change it"
            )
    else:
        output_parent = screen_dir.parent if screen_dir.parent.exists() else job
        batch_size, decision = choose_stream_batch_size(
            output_parent,
            args.stream_batch_size,
            molecule_count=(
                int(source_config["generate_count"])
                if source_config["kind"] == "iGen3"
                else None
            ),
            gpu_count=len(screening_gpu_ids(args)),
        )
    expected_batches = (
        math.ceil(int(source_config["generate_count"]) / batch_size)
        if source_config["kind"] == "iGen3"
        else None
    )
    if getattr(args, "dry_run", False):
        plan = {
            "command": "screen",
            "job": str(job),
            "screen": screen_name,
            "source": source_config,
            "model_stage": final["stage"],
            "stream_batch": decision,
            "minimum_expected_batches": expected_batches,
            "save_policy": args.save_policy,
            "score_threshold": args.score_threshold,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan
    runtime = Runtime(args, assets, job)
    config = ensure_screen_config(
        args, job, assets, screen_dir, batch_size, decision, model_manifest
    )
    completed_manifest = screen_dir / "manifest.json"
    completed_results = screen_dir / "results.csv"
    if completed_manifest.is_file() and completed_results.is_file():
        completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
        if completed.get("results_sha256") == sha256(completed_results):
            print(
                f"[iGenVS-ultra] screen already complete: {completed_results}",
                flush=True,
            )
            return completed
    connection = open_dedup_database(screen_dir)
    generation_pool: Optional[PersistentGenerationPool] = None
    score_pool: Optional[PersistentScorePool] = None
    try:
        seed_reference_identities(
            connection, screen_dir, assets, enabled=args.exclude_reference_libraries
        )
        replay_completed_admissions(connection, screen_dir)
        seen_smiles = load_seen_smiles(connection)
        persistent = config["execution_engine"] in {"persistent_v1", "persistent_v2"}
        worker_started = time.perf_counter()
        worker_startup_seconds = 0.0
        processing_seconds = 0.0
        worker_shutdown_seconds = 0.0
        if persistent:
            score_pool = PersistentScorePool(
                runtime, args, job, assets, screen_dir, model_manifest
            )
        # Keep the scorer resident before sizing iGen3 so generation tuning
        # observes the memory pressure of the real steady-state process set.
        if persistent and source_config["kind"] == "iGen3":
            generation_pool = PersistentGenerationPool(
                runtime,
                args,
                job,
                screen_dir,
                expected_count=int(source_config["generate_count"]),
            )
        if persistent:
            worker_startup_seconds = time.perf_counter() - worker_started
            atomic_json(
                screen_dir / "workers.json",
                {
                    "schema_version": 1,
                    "status": "ready",
                    "engine": config["execution_engine"],
                    "startup_seconds": worker_startup_seconds,
                    "generation": generation_pool.ready if generation_pool else [],
                    "scoring": score_pool.ready if score_pool else [],
                },
            )
        processing_started = time.perf_counter()
        if source_config["kind"] == "external":
            batches = process_external_screen(
                runtime,
                args,
                job,
                assets,
                screen_dir,
                connection,
                batch_size,
                model_manifest,
                seen_smiles,
                score_pool,
            )
        else:
            batches = process_generated_screen(
                runtime,
                args,
                job,
                assets,
                screen_dir,
                connection,
                batch_size,
                model_manifest,
                seen_smiles,
                generation_pool,
                score_pool,
            )
        processing_seconds = time.perf_counter() - processing_started
    finally:
        shutdown_started = time.perf_counter()
        if score_pool is not None:
            score_pool.close()
        if generation_pool is not None:
            generation_pool.close()
        worker_shutdown_seconds = time.perf_counter() - shutdown_started
        connection.close()
    finalization_started = time.perf_counter()
    result = finalize_screen(screen_dir, config, batches)
    finalization_seconds = time.perf_counter() - finalization_started
    result["screen_wall_seconds"] = time.perf_counter() - screen_started
    result["timings"] = {
        "worker_startup_seconds": worker_startup_seconds,
        "batch_processing_wall_seconds": processing_seconds,
        "worker_shutdown_seconds": worker_shutdown_seconds,
        "finalization_seconds": finalization_seconds,
    }
    workers_path = screen_dir / "workers.json"
    if workers_path.is_file():
        result["workers"] = json.loads(workers_path.read_text(encoding="utf-8"))
    atomic_json(screen_dir / "manifest.json", result)
    print(
        f"[iGenVS-ultra] screen complete: scored={result['encoded_rows']:,}, "
        f"saved={result['saved_rows']:,}; {result['results']}",
        flush=True,
    )
    return result


def plan_run(args: Any) -> dict[str, Any]:
    """Print a mutation-free combined plan for the one-command workflow."""
    assets = resolve_assets(args)
    job = Path(args.output_dir).expanduser().resolve()
    target = make_fit_config(args, assets)
    runtime = Runtime(args, assets, job)
    source = screening_source_config(args)
    parent = job.parent if job.parent.exists() else Path.cwd()
    batch_size, decision = choose_stream_batch_size(parent, args.stream_batch_size)
    plan = {
        "command": "run",
        "job": str(job),
        "runtime": runtime.execution,
        "target": target,
        "fit": {
            "al_rounds": args.al_rounds,
            "reference_docking_rows": 300_000,
            "al_docking_rows": 30_000 * args.al_rounds,
            "docking_gpu_ids": visible_gpu_tokens(getattr(args, "gpu_ids", None)),
        },
        "screen": {
            "name": sanitize_target_name(args.screen_name),
            "source": source,
            "stream_batch": decision,
            "minimum_expected_batches": (
                math.ceil(int(source["generate_count"]) / batch_size)
                if source["kind"] == "iGen3"
                else None
            ),
            "save_policy": args.save_policy,
            "score_threshold": args.score_threshold,
        },
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def status(args: Any) -> dict[str, Any]:
    job = Path(args.job_dir).expanduser().resolve()
    result: dict[str, Any] = {"job": str(job), "exists": job.is_dir()}
    if not job.is_dir():
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    regular_config = job / "regular-config.json"
    if regular_config.is_file():
        result["workflow"] = "regular-iGenVS-docking"
        result["regular-config.json"] = json.loads(regular_config.read_text(encoding="utf-8"))
        summary = job / "regular-summary.json"
        manifest = job / "docking/manifest.json"
        result["summary"] = (
            json.loads(summary.read_text(encoding="utf-8"))
            if summary.is_file()
            else {"status": "incomplete"}
        )
        result["docking"] = (
            json.loads(manifest.read_text(encoding="utf-8"))
            if manifest.is_file()
            else {"status": "not-started"}
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    result["workflow"] = "iGenVS-ultra"
    for name in ("fit-config.json", "fit-summary.json", "models/final.json"):
        path = job / name
        if path.is_file():
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    result["reference_docking"] = {
        library: (job / f"docking/{library}/merge-manifest.json").is_file()
        for library in ("UDRL-train", "UDRL-valid")
    }
    result["al_rounds"] = {
        str(number): {
            "acquisition": (job / f"al/round-{number}/acquisition/manifest.json").is_file(),
            "docking": (job / f"al/round-{number}/docking/merge-manifest.json").is_file(),
            "model": (job / f"models/round-{number}/ensemble-manifest.json").is_file(),
        }
        for number in range(1, 6)
    }
    result["screens"] = {}
    screens = job / "screens"
    if screens.is_dir():
        for directory in sorted(path for path in screens.iterdir() if path.is_dir()):
            manifest = directory / "manifest.json"
            result["screens"][directory.name] = (
                json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {"status": "incomplete"}
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def doctor(args: Any) -> dict[str, Any]:
    assets = resolve_assets(args)
    igenvs_image, gmolai_image = resolved_runtime_paths(args, assets)
    igenvs_docker_image, gmolai_docker_image = resolved_docker_images(args)
    scratch_job = assets / "user-pipeline"
    runtime = Runtime(args, assets, scratch_job)
    core_required = [
        assets / "phase-5-head-selection/artifacts/input-standardizer.npz",
        assets / "gMolAI-v2.0/inference/gmolai.py",
        assets / "gMolAI-v2.0/inference/models/SHA256SUMS",
        assets / "user-pipeline/src/igenvs_ultra/model_ops.py",
        assets / "user-pipeline/src/igenvs_ultra/generation_worker.py",
    ]
    fit_required = [
        assets / "phase-1-udrl/library/UDRL-train.csv",
        assets / "phase-1-udrl/library/UDRL-valid.csv",
        assets / "phase-1-udrl/embeddings/UDRL-train-embeddings.npz",
        assets / "phase-1-udrl/embeddings/UDRL-valid-embeddings.npz",
    ]
    for number in range(1, 6):
        fit_required.extend(
            [
                assets / f"phase-2-al-sets/library/AL-set-{number}.csv",
                assets / f"phase-2-al-sets/embeddings/AL-set-{number}-embeddings.npz",
            ]
        )
    core_files = {str(path): path.is_file() for path in core_required}
    fit_files = {str(path): path.is_file() for path in fit_required}
    fit_ready = all(fit_files.values())
    igenvs = runtime.capture("igenvs", ["igenvs", "doctor", "--json", *( ["--no-gpu"] if args.no_gpu else [] )], gpu=not args.no_gpu)
    model = runtime.capture(
        "gmolai",
        runtime.model_command(["self-test"]),
        gpu=False,
        extra_paths=(assets / "gMolAI-v2.0",),
    )
    try:
        igenvs_payload = json.loads(igenvs.stdout) if igenvs.returncode == 0 else {"error": igenvs.stderr or igenvs.stdout}
    except json.JSONDecodeError:
        igenvs_payload = {"error": igenvs.stderr or igenvs.stdout}
    require_fit_assets = bool(getattr(args, "require_fit_assets", False))
    report = {
        "ok": (
            all(core_files.values())
            and (fit_ready or not require_fit_assets)
            and igenvs.returncode == 0
            and model.returncode == 0
        ),
        "execution": runtime.execution,
        "assets_dir": str(assets),
        "apptainer_images": {
            "igenvs": {"path": str(igenvs_image), "exists": igenvs_image.is_file()},
            "gmolai": {"path": str(gmolai_image), "exists": gmolai_image.is_file()},
        },
        "docker_images": {
            "igenvs": {
                "name": igenvs_docker_image,
                "exists": docker_image_exists(igenvs_docker_image),
            },
            "gmolai": {
                "name": gmolai_docker_image,
                "exists": docker_image_exists(gmolai_docker_image),
            },
        },
        "required_assets": core_files,
        "fit_assets": {
            "required": require_fit_assets,
            "ready": fit_ready,
            "files": fit_files,
        },
        "visible_gpu_ids": visible_gpu_tokens(getattr(args, "gpu_ids", None)),
        "gpu_memory_mib": gpu_memory_mib(),
        "igenvs": igenvs_payload,
        "model_ops_self_test": {"returncode": model.returncode, "stdout": model.stdout.strip(), "stderr": model.stderr.strip()},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report

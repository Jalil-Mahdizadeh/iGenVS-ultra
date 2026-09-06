"""Hardware discovery and conservative automatic defaults."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    uuid: str
    memory_total_mib: int
    memory_free_mib: int
    compute_capability: str
    driver_version: str


def available_cpu_count() -> int:
    """Return CPUs available to this process, respecting a Slurm/cgroup cpuset."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def available_physical_cpu_count() -> int:
    """Count physical cores inside the current affinity without trusting SMT."""

    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return available_cpu_count()
    identities: set[tuple[str, str]] = set()
    try:
        for cpu in affinity:
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            package = (topology / "physical_package_id").read_text(
                encoding="utf-8"
            ).strip()
            core = (topology / "core_id").read_text(encoding="utf-8").strip()
            identities.add((package, core))
    except OSError:
        return available_cpu_count()
    return max(1, len(identities))


def available_memory_bytes() -> int:
    """Best-effort memory available to this process or its cgroup."""

    proc_available = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                proc_available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass

    cgroup_available = 0
    try:
        relative = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
            if "::" in line
        )
        root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        raw_limit = (root / "memory.max").read_text(encoding="utf-8").strip()
        if raw_limit != "max":
            limit = int(raw_limit)
            current = int((root / "memory.current").read_text(encoding="utf-8").strip())
            cgroup_available = max(0, limit - current)
    except (OSError, StopIteration, ValueError, IndexError):
        pass
    candidates = [value for value in (proc_available, cgroup_available) if value > 0]
    return min(candidates) if candidates else 0


def auto_worker_count(requested: str | int, *, cap: int = 32) -> int:
    if isinstance(requested, int) or str(requested).lower() != "auto":
        value = int(requested)
        if value < 1:
            raise ValueError("worker count must be positive or 'auto'")
        return value
    return max(1, min(cap, available_cpu_count()))


def auto_preparation_worker_count(
    requested: str | int,
    *,
    cap: int = 64,
    memory_per_worker_mib: int = 768,
) -> int:
    """Choose RDKit workers from physical cores and conservative live RAM."""

    if isinstance(requested, int) or str(requested).lower() != "auto":
        value = int(requested)
        if value < 1:
            raise ValueError("worker count must be positive or 'auto'")
        return value
    physical = available_physical_cpu_count()
    # Leave one physical core for the supervisor, driver and result writer
    # when the allocation is large enough to do so.
    cpu_bound = max(1, physical - int(physical >= 8))
    memory = available_memory_bytes()
    memory_bound = cap
    if memory > 0:
        usable = int(memory * 0.70)
        memory_bound = max(1, usable // (memory_per_worker_mib * 1024 * 1024))
    return max(1, min(cap, cpu_bound, memory_bound))


def query_gpus() -> list[GPUInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    fields = "index,name,uuid,memory.total,memory.free,compute_cap,driver_version"
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    result: list[GPUInfo] = []
    for line in proc.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            result.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    uuid=parts[2],
                    memory_total_mib=int(float(parts[3])),
                    memory_free_mib=int(float(parts[4])),
                    compute_capability=parts[5],
                    driver_version=parts[6],
                )
            )
        except ValueError:
            continue
    return result


def selected_gpu(device_id: int = 0) -> GPUInfo | None:
    gpus = query_gpus()
    if not gpus:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"-1", "NoDevFiles"}:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if 0 <= device_id < len(tokens):
            token = tokens[device_id]
            if token.isdigit():
                physical_index = int(token)
                match = next((gpu for gpu in gpus if gpu.index == physical_index), None)
                if match is not None:
                    return match
            else:
                normalized = token.removeprefix("GPU-")
                match = next(
                    (
                        gpu
                        for gpu in gpus
                        if gpu.uuid == token
                        or gpu.uuid.removeprefix("GPU-").startswith(normalized)
                    ),
                    None,
                )
                if match is not None:
                    return match
    if 0 <= device_id < len(gpus):
        return gpus[device_id]
    return gpus[0]


def heuristic_docking_batch_size(
    gpu: GPUInfo | None,
    *,
    engine: str = "unidock",
) -> int:
    """Choose an engine-appropriate outer process batch."""

    if engine == "autodock-gpu":
        # AD-GPU handles one ligand's LGA populations on-device while its
        # OpenMP file-list pipeline overlaps setup and result processing.
        # This is an amortization/resilience choice, not a GPU-memory limit.
        if gpu is None:
            return 512
        if gpu.memory_total_mib >= 40 * 1024:
            return 4_096
        if gpu.memory_total_mib >= 16 * 1024:
            return 2_048
        if gpu.memory_total_mib >= 8 * 1024:
            return 1_024
        return 512
    if engine != "unidock":
        raise ValueError(f"unsupported docking engine: {engine}")
    if gpu is None:
        return 512
    target = max(512, min(32_768, (gpu.memory_free_mib // 1024) * 384))
    power = 1 << (int(target).bit_length() - 1)
    return max(512, min(32_768, power))


def default_tuning_sizes(
    gpu: GPUInfo | None,
    *,
    engine: str = "unidock",
) -> list[int]:
    if engine == "autodock-gpu":
        # Large file lists amortize process startup. Keep the geometric sweep
        # through the measured GH200 optimum while allowing the profile to
        # select an earlier point on smaller hardware.
        return [512, 1_024, 2_048, 4_096]
    ceiling = heuristic_docking_batch_size(gpu, engine=engine) * 2
    values = [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768]
    return [value for value in values if value <= ceiling]


def hardware_snapshot(device_id: int = 0) -> dict[str, Any]:
    gpu = selected_gpu(device_id)
    return {
        "cpu_count": available_cpu_count(),
        "physical_cpu_count": available_physical_cpu_count(),
        "memory_available_bytes": available_memory_bytes(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "cuda_mps_log_directory": os.environ.get("CUDA_MPS_LOG_DIRECTORY"),
        "cuda_mps_active_thread_percentage": os.environ.get(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ),
        "gpu": asdict(gpu) if gpu else None,
    }


def load_batch_profile(
    path: Path,
    *,
    engine: str | None = None,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    value = data.get("selected_batch_size")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} does not contain a positive selected_batch_size")
    if engine is not None:
        config = data.get("config")
        profile_engine = data.get("engine")
        if profile_engine is None and isinstance(config, dict):
            profile_engine = config.get("engine")
        if profile_engine is None:
            profile_engine = "unidock"
        if profile_engine != engine:
            raise ValueError(
                f"{path} is a {profile_engine} profile, not a {engine} profile"
            )
    return data


def resolve_docking_batch_size(
    value: str | int,
    *,
    profile_path: Path | None = None,
    device_id: int = 0,
    engine: str = "unidock",
) -> tuple[int, str]:
    if str(value).lower() != "auto":
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("batch size must be positive or 'auto'")
        return parsed, "explicit"
    if profile_path is not None:
        profile = load_batch_profile(profile_path, engine=engine)
        return int(profile["selected_batch_size"]), f"profile:{profile_path}"
    return (
        heuristic_docking_batch_size(selected_gpu(device_id), engine=engine),
        "hardware-heuristic",
    )


def resolve_autodock_gpu_workers(
    value: str | int,
    *,
    profile_path: Path | None = None,
    engine: str = "unidock",
    device_id: int = 0,
    cpu_threads_per_worker: int = 4,
) -> tuple[int, str]:
    """Resolve same-GPU AD-GPU processes, with measured profiles preferred."""

    is_auto = str(value).lower() == "auto"
    if engine != "autodock-gpu":
        if not is_auto and int(value) != 1:
            raise ValueError("AutoDock-GPU workers cannot be used with Uni-Dock")
        return 1, "not-applicable"
    if not is_auto:
        parsed = int(value)
        if not 1 <= parsed <= 64:
            raise ValueError("AutoDock-GPU workers must be between 1 and 64")
        return parsed, "explicit"
    if profile_path is not None:
        profile = load_batch_profile(profile_path, engine=engine)
        candidate = profile.get("selected_autodock_gpu_workers")
        if candidate is None:
            profile_config = profile.get("config")
            if isinstance(profile_config, dict):
                candidate = profile_config.get("autodock_gpu_workers")
        if candidate is None:
            candidate = 1
        if not isinstance(candidate, int) or not 1 <= candidate <= 64:
            raise ValueError(
                f"{profile_path} does not contain a valid AutoDock-GPU worker count"
            )
        return candidate, f"profile:{profile_path}"
    if cpu_threads_per_worker < 1:
        raise ValueError("AutoDock-GPU CPU threads per worker must be positive")
    gpu = selected_gpu(device_id)
    if gpu is None or shutil.which("nvidia-cuda-mps-control") is None:
        return 1, "safe-default-no-mps"
    if gpu.memory_total_mib >= 40 * 1024:
        gpu_bound = 6
    elif gpu.memory_total_mib >= 16 * 1024:
        gpu_bound = 4
    elif gpu.memory_total_mib >= 8 * 1024:
        gpu_bound = 2
    else:
        return 1, "safe-default-small-gpu"
    cpu_bound = max(1, available_physical_cpu_count() // cpu_threads_per_worker)
    return min(gpu_bound, cpu_bound), "hardware-cpu-heuristic-mps"

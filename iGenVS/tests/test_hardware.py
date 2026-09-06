from __future__ import annotations

import json

import igenvs.hardware as hardware

from igenvs.hardware import (
    GPUInfo,
    auto_preparation_worker_count,
    default_tuning_sizes,
    heuristic_docking_batch_size,
    resolve_autodock_gpu_workers,
    resolve_docking_batch_size,
)


def _gpu(free_mib: int, total_mib: int = 97871, index: int = 0) -> GPUInfo:
    return GPUInfo(index, "NVIDIA test GPU", f"GPU-test-{index}", total_mib, free_mib, "9.0", "580.159.04")


def test_gh200_heuristic_is_bounded_power_of_two() -> None:
    assert heuristic_docking_batch_size(_gpu(97280)) == 32768
    assert heuristic_docking_batch_size(_gpu(512)) == 512
    assert heuristic_docking_batch_size(None) == 512
    assert default_tuning_sizes(_gpu(97280))[-1] == 32768
    assert (
        heuristic_docking_batch_size(
            _gpu(97280),
            engine="autodock-gpu",
        )
        == 4096
    )
    assert default_tuning_sizes(
        _gpu(97280),
        engine="autodock-gpu",
    ) == [512, 1_024, 2_048, 4_096]


def test_autodock_gpu_defaults_scale_with_device_capacity() -> None:
    assert heuristic_docking_batch_size(_gpu(40_000, 48 * 1024), engine="autodock-gpu") == 4096
    assert heuristic_docking_batch_size(_gpu(12_000, 16 * 1024), engine="autodock-gpu") == 2048
    assert heuristic_docking_batch_size(_gpu(7_000, 8 * 1024), engine="autodock-gpu") == 1024
    assert heuristic_docking_batch_size(_gpu(3_000, 4 * 1024), engine="autodock-gpu") == 512


def test_visible_gpu_token_resolves_physical_inventory(monkeypatch) -> None:
    inventory = [_gpu(10_000, index=0), _gpu(20_000, index=1)]
    monkeypatch.setattr(hardware, "query_gpus", lambda: inventory)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert hardware.selected_gpu(0) == inventory[1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inventory[0].uuid)
    assert hardware.selected_gpu(0) == inventory[0]


def test_preparation_workers_are_physical_core_and_memory_bounded(monkeypatch) -> None:
    monkeypatch.setattr(hardware, "available_physical_cpu_count", lambda: 32)
    monkeypatch.setattr(hardware, "available_memory_bytes", lambda: 12 * 1024**3)
    assert auto_preparation_worker_count("auto") == 11
    assert auto_preparation_worker_count(7) == 7


def test_autodock_gpu_auto_workers_are_gpu_and_cpu_bounded(monkeypatch) -> None:
    monkeypatch.setattr(
        hardware,
        "selected_gpu",
        lambda device_id=0: _gpu(40_000, 48 * 1024),
    )
    monkeypatch.setattr(
        hardware.shutil,
        "which",
        lambda _: "/bin/nvidia-cuda-mps-control",
    )
    monkeypatch.setattr(hardware, "available_physical_cpu_count", lambda: 10)
    workers, source = resolve_autodock_gpu_workers(
        "auto",
        engine="autodock-gpu",
        cpu_threads_per_worker=4,
    )
    assert workers == 2
    assert source == "hardware-cpu-heuristic-mps"


def test_saved_profile_overrides_heuristic(tmp_path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"selected_batch_size": 2048}), encoding="utf-8")
    value, source = resolve_docking_batch_size("auto", profile_path=profile)
    assert value == 2048
    assert source.startswith("profile:")


def test_batch_profile_cannot_cross_engines(tmp_path) -> None:
    profile = tmp_path / "adgpu-profile.json"
    profile.write_text(
        json.dumps(
            {
                "engine": "autodock-gpu",
                "selected_batch_size": 256,
                "selected_autodock_gpu_workers": 4,
            }
        ),
        encoding="utf-8",
    )
    value, _ = resolve_docking_batch_size(
        "auto",
        profile_path=profile,
        engine="autodock-gpu",
    )
    assert value == 256
    workers, worker_source = resolve_autodock_gpu_workers(
        "auto",
        profile_path=profile,
        engine="autodock-gpu",
    )
    assert workers == 4
    assert worker_source.startswith("profile:")

    try:
        resolve_docking_batch_size(
            "auto",
            profile_path=profile,
            engine="unidock",
        )
    except ValueError as exc:
        assert "not a unidock profile" in str(exc)
    else:
        raise AssertionError("cross-engine batch profile should be rejected")


def test_hardware_snapshot_records_cuda_mps(monkeypatch) -> None:
    monkeypatch.setattr(hardware, "selected_gpu", lambda device_id=0: None)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/mps-pipe")
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", "/tmp/mps-log")
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "25")
    snapshot = hardware.hardware_snapshot()
    assert snapshot["cuda_mps_pipe_directory"] == "/tmp/mps-pipe"
    assert snapshot["cuda_mps_log_directory"] == "/tmp/mps-log"
    assert snapshot["cuda_mps_active_thread_percentage"] == "25"

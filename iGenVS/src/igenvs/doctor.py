"""Installation and GPU diagnostics."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
from typing import Any

from .hardware import hardware_snapshot


def _executable_status(name: str, version_args: list[str]) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"ok": False, "path": None, "version": None, "error": "not found on PATH"}
    try:
        process = subprocess.run([name, *version_args], capture_output=True, text=True, timeout=30, check=False)
        output = (process.stdout or process.stderr).strip()
        return {
            "ok": process.returncode == 0,
            "path": path,
            "version": output.splitlines()[0].strip() if output else None,
            "returncode": process.returncode,
        }
    except Exception as exc:
        return {"ok": False, "path": path, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def _package_status(distribution: str) -> dict[str, Any]:
    try:
        return {"ok": True, "version": importlib.metadata.version(distribution)}
    except importlib.metadata.PackageNotFoundError:
        return {"ok": False, "version": None, "error": "not installed"}


def run_doctor(*, require_gpu: bool = True) -> dict[str, Any]:
    packages = {
        name: _package_status(name)
        for name in ("igenvs", "igen3", "torch", "rdkit", "meeko", "numpy", "gemmi")
    }
    executables = {
        "unidock": _executable_status("unidock", ["--version"]),
        "autodock_gpu": _executable_status("autodock_gpu", ["--help"]),
        "autogrid4": _executable_status("autogrid4", ["--version"]),
        "igen3": _executable_status("igen3", ["list-models"]),
        "mk_prepare_receptor.py": _executable_status("mk_prepare_receptor.py", ["--help"]),
        "obabel": _executable_status("obabel", ["-V"]),
    }
    torch_cuda: dict[str, Any]
    try:
        import torch

        available = bool(torch.cuda.is_available())
        torch_cuda = {
            "ok": available or not require_gpu,
            "available": available,
            "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "device_capability": list(torch.cuda.get_device_capability(0)) if available else None,
        }
    except Exception as exc:
        torch_cuda = {"ok": False, "available": False, "error": f"{type(exc).__name__}: {exc}"}

    required_packages = ("igenvs", "igen3", "torch", "rdkit", "meeko")
    required_executables = (
        "unidock",
        "autodock_gpu",
        "autogrid4",
        "igen3",
        "mk_prepare_receptor.py",
    )
    ok = all(packages[name]["ok"] for name in required_packages)
    ok = ok and all(executables[name]["ok"] for name in required_executables)
    ok = ok and bool(torch_cuda["ok"])
    return {
        "ok": ok,
        "platform": {"machine": platform.machine(), "python": platform.python_version()},
        "hardware": hardware_snapshot(),
        "torch_cuda": torch_cuda,
        "packages": packages,
        "executables": executables,
    }


def format_doctor(report: dict[str, Any]) -> str:
    lines = [f"iGenVS environment: {'OK' if report['ok'] else 'NOT READY'}"]
    gpu = report["hardware"].get("gpu")
    if gpu:
        lines.append(
            f"GPU: {gpu['name']} (CC {gpu['compute_capability']}, "
            f"{gpu['memory_free_mib']}/{gpu['memory_total_mib']} MiB free)"
        )
    else:
        lines.append("GPU: not visible")
    for name, status in report["packages"].items():
        lines.append(f"package {name}: {status.get('version') or status.get('error')}" )
    for name, status in report["executables"].items():
        lines.append(f"tool {name}: {status.get('version') or status.get('error')}")
    return "\n".join(lines)

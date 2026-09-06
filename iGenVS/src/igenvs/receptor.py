"""Explicit wrappers around Meeko receptor preparation and AutoGrid."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .errors import ExternalToolError, InputError


def prepare_receptor(
    input_path: Path,
    output_path: Path,
    *,
    gpf_path: Path | None = None,
    center: tuple[float, float, float] | None = None,
    size: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    """Prepare a rigid PDBQT and, when requested, an AutoGrid GPF."""

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise InputError(f"receptor input does not exist: {input_path}")
    grid_values = (gpf_path is not None, center is not None, size is not None)
    if any(grid_values) and not all(grid_values):
        raise InputError("GPF preparation requires gpf_path, center, and size together")
    executable = shutil.which("mk_prepare_receptor.py")
    if executable is None:
        raise ExternalToolError(
            "mk_prepare_receptor.py is not available; install Meeko with receptor dependencies"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "--read_pdb",
        str(input_path),
        "--write_pdbqt",
        str(output_path),
    ]
    resolved_gpf = None
    if gpf_path is not None and center is not None and size is not None:
        resolved_gpf = gpf_path.expanduser().resolve()
        if resolved_gpf.parent != output_path.parent:
            raise InputError(
                "receptor PDBQT and AutoGrid GPF must share one directory"
            )
        command.extend(
            [
                "--write_gpf",
                str(resolved_gpf),
                "--box_center",
                *(str(value) for value in center),
                "--box_size",
                *(str(value) for value in size),
            ]
        )
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if (
        process.returncode != 0
        or not output_path.is_file()
        or (resolved_gpf is not None and not resolved_gpf.is_file())
    ):
        raise ExternalToolError(
            f"Meeko receptor preparation failed with exit {process.returncode}: "
            f"{process.stderr[-4000:]}"
        )
    return {
        "input": str(input_path),
        "output": str(output_path),
        "command": command,
        "gpf": str(resolved_gpf) if resolved_gpf is not None else None,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def run_autogrid(
    gpf_path: Path,
    *,
    log_path: Path,
    executable_name: str = "autogrid4",
) -> dict[str, object]:
    """Calculate AD4 maps from a generated GPF and return the FLD descriptor."""

    gpf_path = gpf_path.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    if not gpf_path.is_file():
        raise InputError(f"AutoGrid parameter file does not exist: {gpf_path}")
    executable = shutil.which(executable_name)
    if executable is None:
        raise ExternalToolError(f"{executable_name} is not available")

    gridfld = None
    for line in gpf_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        fields = line.split()
        if fields and fields[0].lower() == "gridfld" and len(fields) >= 2:
            gridfld = fields[1]
            break
    if gridfld is None or Path(gridfld).name != gridfld:
        raise InputError("AutoGrid GPF has no safe gridfld output filename")

    version_process = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    version_output = (version_process.stdout or version_process.stderr).strip()
    version = version_output.splitlines()[0].strip() if version_output else None

    fld_path = gpf_path.parent / gridfld
    command = [executable, "-p", gpf_path.name, "-l", log_path.name]
    process = subprocess.run(
        command,
        cwd=gpf_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0 or not fld_path.is_file():
        diagnostic = process.stderr or process.stdout
        raise ExternalToolError(
            f"AutoGrid failed with exit {process.returncode}: {diagnostic[-4000:]}"
        )
    return {
        "gpf": str(gpf_path),
        "fld": str(fld_path.resolve()),
        "log": str(log_path),
        "command": command,
        "version": version,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }

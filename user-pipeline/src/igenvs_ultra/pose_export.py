"""Stream saved docking poses to SDF using the image's Meeko/RDKit tools."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Iterator


MOLECULE_MARKER = "REMARK IGENVS MOLECULE_ID "


def merged_molecules(path: Path) -> Iterator[tuple[str, str]]:
    """Yield one ligand at a time, retaining all its docked conformers."""
    name = None
    lines: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(MOLECULE_MARKER):
                if name is not None:
                    yield name, "".join(lines)
                name = line[len(MOLECULE_MARKER):].rstrip("\r\n")
                lines = []
            elif name is not None:
                lines.append(line)
            elif line.strip():
                raise ValueError(f"missing iGenVS molecule boundary in {path}")
    if name is not None:
        yield name, "".join(lines)


def sdf_string(name: str, pdbqt: str) -> tuple[str, int]:
    from meeko import PDBQTMolecule, RDKitMolCreate

    try:
        molecule = PDBQTMolecule(pdbqt, name=name, skip_typing=True)
        text, failures = RDKitMolCreate.write_sd_string(molecule)
        poses = text.count("$$$$\n")
        if failures or not poses:
            raise ValueError("Meeko could not reconstruct every ligand")
    except Exception as exc:
        raise ValueError(f"cannot export SDF poses for {name!r}: {exc}") from exc
    return text, poses


def write_sdf(destination: Path, molecules: Iterator[tuple[str, str]]) -> dict[str, int]:
    """Publish only a complete SDF; preserve any previous export on failure."""
    temporary = destination.with_suffix(".sdf.partial")
    counts = {"molecules": 0, "poses": 0}
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for name, pdbqt in molecules:
                text, poses = sdf_string(name, pdbqt)
                handle.write(text)
                counts["molecules"] += 1
                counts["poses"] += poses
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return counts


def export_poses(docking_dir: Path, pose_output: str) -> dict:
    if pose_output == "merged":
        destination = docking_dir / "poses.sdf"
        counts = write_sdf(destination, merged_molecules(docking_dir / "poses.pdbqt"))
    elif pose_output == "individual":
        destination = docking_dir / "poses"
        destination.mkdir(exist_ok=True)
        counts = {"molecules": 0, "poses": 0}
        # Results keep IDs even when filenames have been normalized for safety.
        with (docking_dir / "results.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "success" or not row["pose_ref"]:
                    continue
                source = (docking_dir / row["pose_ref"]).resolve()
                if source.parent != destination.resolve() or source.suffix != ".pdbqt":
                    raise ValueError(f"individual pose is outside the poses folder: {source}")
                result = write_sdf(source.with_suffix(".sdf"), iter([(row["molecule_id"], source.read_text(encoding="utf-8"))]))
                for key in counts:
                    counts[key] += result[key]
    else:
        raise ValueError("SDF export requires merged or individual poses")
    return {"output": str(destination), **counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docking-dir", required=True, type=Path)
    parser.add_argument("--pose-output", required=True, choices=("merged", "individual"))
    args = parser.parse_args()
    try:
        report = export_poses(args.docking_dir.resolve(), args.pose_output)
    except (OSError, ValueError) as exc:
        print(f"pose export: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

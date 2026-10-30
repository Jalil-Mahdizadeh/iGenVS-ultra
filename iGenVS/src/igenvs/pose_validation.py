"""Validate ring closure in saved poses without dropping or inventing poses."""

from __future__ import annotations

import math
from pathlib import Path


def _atoms(text: str) -> dict[int, tuple[str, tuple[float, float, float]]]:
    atoms = {}
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError("pose contains non-finite coordinates")
            atoms[int(line[6:11])] = (line.split()[-1], xyz)
    return atoms


def validate_ring_closures(prepared: Path, output: Path) -> None:
    """Check Meeko's opened bonds and dummy/closure-atom overlap in every pose.

    Restraints are checked against the prepared geometry, so this also catches
    stretched rings from an old, unpatched Uni-Dock binary. A failed pose makes
    the entire result invalid; the requested pose count is never silently cut.
    """
    reference = _atoms(prepared.read_text(encoding="utf-8"))
    groups: dict[str, list[int]] = {}
    for serial, (atom_type, _) in reference.items():
        if atom_type.startswith("CG") and atom_type[2:].isdigit():
            groups.setdefault(atom_type[2:], []).append(serial)
    if not groups:
        return

    bonds = []
    overlaps = []
    for group, carbons in groups.items():
        dummies = [serial for serial, (kind, _) in reference.items() if kind == f"G{group}"]
        if len(carbons) != 2 or len(dummies) != 2:
            raise ValueError(f"ambiguous ring-closure atom group {group}")
        a, b = carbons
        bonds.append((a, b, math.dist(reference[a][1], reference[b][1])))
        for dummy in dummies:
            partner = min(carbons, key=lambda serial: math.dist(reference[dummy][1], reference[serial][1]))
            overlaps.append((dummy, partner))

    text = output.read_text(encoding="utf-8")
    models = [block for block in text.split("ENDMDL") if any(
        line.startswith(("ATOM  ", "HETATM")) for line in block.splitlines()
    )]
    if not models:
        raise ValueError("ring-containing output has no atom coordinates")
    for pose, model in enumerate(models, 1):
        atoms = _atoms(model)
        for a, b, expected in bonds:
            if a not in atoms or b not in atoms:
                raise ValueError(f"pose {pose} is missing ring-closure atoms")
            distance = math.dist(atoms[a][1], atoms[b][1])
            if abs(distance - expected) > 0.25:
                raise ValueError(
                    f"pose {pose}: unclosed ring bond {a}-{b} is {distance:.3f} A "
                    f"(prepared {expected:.3f} A)"
                )
        for dummy, partner in overlaps:
            if dummy not in atoms or partner not in atoms:
                raise ValueError(f"pose {pose} is missing ring-closure dummy atoms")
            if math.dist(atoms[dummy][1], atoms[partner][1]) > 0.30:
                raise ValueError(f"pose {pose}: ring-closure dummy {dummy} misses atom {partner}")

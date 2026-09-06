"""Prepare and load reusable receptor/pocket target bundles."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem

from .errors import InputError
from .receptor import prepare_receptor, run_autogrid


DEFAULT_BOX_SIZE = (22.5, 22.5, 22.5)
DEFAULT_POCKET_PADDING = 5.0
POCKET_FILENAME = "pocket.json"
TARGET_MANIFEST_FILENAME = "manifest.json"
TARGET_RECEPTOR_FILENAME = "receptor.pdbqt"
TARGET_SCHEMA_VERSION = 2
AUTODOCK_GPU_GPF_FILENAME = "receptor.gpf"
AUTODOCK_GPU_FLD_FILENAME = "receptor.maps.fld"
AUTOGRID_LOG_FILENAME = "autogrid.log"
AUTOGRID_SPACING = 0.375
AUTODOCK_GPU_MAX_GRID_POINTS = 256


@dataclass(frozen=True)
class PreparedTarget:
    """Resolved files and docking box from a prepared target directory."""

    directory: Path
    receptor: Path
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    pocket: Path
    manifest: Path
    autodock_gpu_fld: Path | None = None


@dataclass(frozen=True)
class _PDBAtom:
    line_index: int
    line: str
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    occupancy: float
    element: str
    coordinates: tuple[float, float, float]

    @property
    def residue_key(self) -> tuple[str, str, int, str]:
        return self.chain, self.resname, self.resseq, self.icode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise InputError(f"{label} does not exist: {resolved}")
    return resolved


def _prepare_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise InputError(f"target output exists and is not a directory: {resolved}")
    if resolved.is_dir() and any(resolved.iterdir()):
        raise InputError(f"target output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _parse_pdb_atom(line: str, line_index: int) -> _PDBAtom:
    padded = line.ljust(80)
    try:
        serial = int(padded[6:11])
        resseq = int(padded[22:26])
        coordinates = tuple(float(padded[start : start + 8]) for start in (30, 38, 46))
        occupancy_text = padded[54:60].strip()
        occupancy = float(occupancy_text) if occupancy_text else 0.0
    except ValueError as exc:
        raise InputError(f"invalid PDB atom record at line {line_index + 1}") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise InputError(f"non-finite PDB coordinate at line {line_index + 1}")
    return _PDBAtom(
        line_index=line_index,
        line=line,
        serial=serial,
        atom_name=padded[12:16].strip(),
        altloc=padded[16].strip(),
        resname=padded[17:20].strip().upper(),
        chain=padded[21].strip(),
        resseq=resseq,
        icode=padded[26].strip().upper(),
        occupancy=occupancy,
        element=padded[76:78].strip().upper(),
        coordinates=coordinates,  # type: ignore[arg-type]
    )


def _read_pdb(path: Path) -> tuple[list[str], list[_PDBAtom]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    model_count = sum(line[:6].strip().upper() == "MODEL" for line in lines)
    if model_count > 1:
        raise InputError("multi-model PDB inputs are ambiguous; provide a single selected model")
    atoms = [
        _parse_pdb_atom(line, index)
        for index, line in enumerate(lines)
        if line[:6].strip().upper() in {"ATOM", "HETATM"}
    ]
    if not atoms:
        raise InputError(f"PDB contains no ATOM or HETATM records: {path}")
    return lines, atoms


def _residue_label(key: tuple[str, str, int, str]) -> str:
    chain, resname, resseq, icode = key
    return f"{chain or '_'}:{resname}:{resseq}{icode}"


def _format_residue_list(keys: Iterable[tuple[str, str, int, str]]) -> str:
    ordered = sorted(keys)
    labels = [_residue_label(key) for key in ordered[:20]]
    if len(ordered) > 20:
        labels.append(f"... (+{len(ordered) - 20} more)")
    return ", ".join(labels)


def _parse_residue_number(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", value.strip())
    if match is None:
        raise InputError("ligand residue number must be an integer with an optional insertion code")
    return int(match.group(1)), match.group(2).upper()


def _select_ligand_residue(
    atoms: Iterable[_PDBAtom], ligand_id: str
) -> tuple[tuple[str, str, int, str], list[_PDBAtom]]:
    residues: dict[tuple[str, str, int, str], list[_PDBAtom]] = {}
    for atom in atoms:
        if atom.line[:6].strip().upper() == "HETATM":
            residues.setdefault(atom.residue_key, []).append(atom)
    if not residues:
        raise InputError("complex PDB contains no HETATM ligand candidates")

    parts = [part.strip() for part in ligand_id.split(":")]
    if len(parts) == 1 and parts[0]:
        resname = parts[0].upper()
        matches = [key for key in residues if key[1] == resname]
    elif len(parts) in {3, 4}:
        chain = "" if parts[0] in {"", "_"} else parts[0]
        resname = parts[1].upper()
        resseq, embedded_icode = _parse_residue_number(parts[2])
        explicit_icode = parts[3].upper() if len(parts) == 4 else ""
        if embedded_icode and explicit_icode:
            raise InputError("specify the ligand insertion code only once")
        key = (chain, resname, resseq, explicit_icode or embedded_icode)
        matches = [key] if key in residues else []
    else:
        raise InputError(
            "ligand ID must be RESNAME or CHAIN:RESNAME:RESSEQ, for example LIG or A:LIG:501"
        )

    if len(matches) != 1:
        available = _format_residue_list(residues)
        if not matches:
            raise InputError(f"ligand ID {ligand_id!r} was not found; available HETATM residues: {available}")
        selected = _format_residue_list(matches)
        raise InputError(
            f"ligand ID {ligand_id!r} is ambiguous ({selected}); use CHAIN:RESNAME:RESSEQ"
        )
    key = matches[0]
    return key, residues[key]


def _inferred_element(atom: _PDBAtom) -> str:
    if atom.element:
        return atom.element
    name = atom.atom_name.lstrip("0123456789").upper()
    if name.startswith(("CL", "BR")):
        return name[:2]
    return name[:1]


def _reference_coordinates(atoms: Iterable[_PDBAtom]) -> list[tuple[float, float, float]]:
    alternatives: dict[str, list[_PDBAtom]] = {}
    for atom in atoms:
        alternatives.setdefault(atom.atom_name, []).append(atom)
    selected: list[_PDBAtom] = []
    for variants in alternatives.values():
        selected.append(
            max(
                variants,
                key=lambda atom: (
                    atom.altloc == "",
                    atom.occupancy,
                    atom.altloc == "A",
                ),
            )
        )
    coordinates = [
        atom.coordinates
        for atom in selected
        if _inferred_element(atom) not in {"H", "D", "T"}
    ]
    if not coordinates:
        raise InputError("selected reference ligand contains no heavy atoms")
    return coordinates


def _validate_reference_alignment(
    coordinates: Iterable[tuple[float, float, float]], receptor_atoms: Iterable[_PDBAtom]
) -> float:
    receptor_coordinates = [
        atom.coordinates
        for atom in receptor_atoms
        if _inferred_element(atom) not in {"H", "D", "T"}
    ]
    if not receptor_coordinates:
        raise InputError("receptor PDB contains no heavy atoms")
    nearest_squared = min(
        sum((reference[axis] - receptor[axis]) ** 2 for axis in range(3))
        for reference in coordinates
        for receptor in receptor_coordinates
    )
    nearest = math.sqrt(nearest_squared)
    if nearest < 0.25:
        raise InputError(
            "reference SDF overlaps receptor atoms; provide a ligand-free receptor PDB"
        )
    if nearest > 8.0:
        raise InputError(
            "reference SDF is spatially separated from the receptor; verify that both files share a coordinate frame"
        )
    return round(nearest, 6)


def _sdf_coordinates(path: Path) -> tuple[list[tuple[float, float, float]], int]:
    supplier = Chem.SDMolSupplier(
        str(path), removeHs=False, sanitize=False, strictParsing=False
    )
    if len(supplier) != 1:
        raise InputError("reference SDF must contain exactly one molecule")
    molecule = supplier[0]
    if molecule is None or molecule.GetNumConformers() != 1:
        raise InputError("reference SDF could not be parsed with one coordinate conformer")
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise InputError("reference SDF must contain aligned 3D coordinates, not a 2D depiction")
    coordinates = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        point = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in point):
            raise InputError("reference SDF contains a non-finite coordinate")
        coordinates.append(point)
    if not coordinates:
        raise InputError("reference SDF contains no heavy atoms")
    return coordinates, len(coordinates)


def _calculate_box(
    coordinates: Iterable[tuple[float, float, float]], padding: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not math.isfinite(padding) or padding < 0:
        raise InputError("pocket padding must be a finite non-negative number")
    points = list(coordinates)
    if not points:
        raise InputError("cannot calculate a pocket without reference coordinates")
    minima = tuple(min(point[axis] for point in points) for axis in range(3))
    maxima = tuple(max(point[axis] for point in points) for axis in range(3))
    center = tuple(round((low + high) / 2.0, 6) for low, high in zip(minima, maxima))
    size = tuple(round(high - low + 2.0 * padding, 6) for low, high in zip(minima, maxima))
    if any(value <= 0 for value in size):
        raise InputError("calculated docking box has a non-positive dimension; increase padding")
    return center, size


def _autodock_grid_geometry(
    requested_size: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    """Round outward to the even grid intervals required by AutoGrid."""

    quantum = 2.0 * AUTOGRID_SPACING
    npts = tuple(
        2 * math.ceil(value / quantum)
        for value in requested_size
    )
    if any(points + 1 > AUTODOCK_GPU_MAX_GRID_POINTS for points in npts):
        maximum = (AUTODOCK_GPU_MAX_GRID_POINTS - 2) * AUTOGRID_SPACING
        raise InputError(
            "docking box exceeds AutoDock-GPU's per-axis grid limit "
            f"of {maximum:.3f} Angstrom at {AUTOGRID_SPACING:.3f} Angstrom spacing"
        )
    actual_size = tuple(round(points * AUTOGRID_SPACING, 6) for points in npts)
    return actual_size, npts  # type: ignore[return-value]


def _read_gpf_geometry(
    gpf_path: Path,
) -> tuple[tuple[int, int, int], float, tuple[float, float, float]]:
    values: dict[str, list[str]] = {}
    for line in gpf_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if fields and fields[0].lower() in {"npts", "spacing", "gridcenter"}:
            values[fields[0].lower()] = fields[1:]
    try:
        npts = tuple(int(value) for value in values["npts"])
        spacing_values = values["spacing"]
        center = tuple(float(value) for value in values["gridcenter"])
        if len(npts) != 3 or len(spacing_values) != 1 or len(center) != 3:
            raise ValueError
        spacing = float(spacing_values[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError("AutoGrid GPF has invalid npts, spacing, or gridcenter") from exc
    if (
        not math.isfinite(spacing)
        or spacing <= 0
        or not all(math.isfinite(value) for value in center)
        or any(points <= 0 or points % 2 for points in npts)
    ):
        raise InputError("AutoGrid GPF has invalid grid geometry")
    return npts, spacing, center  # type: ignore[return-value]


def _validate_grid_coverage(
    *,
    npts: tuple[int, int, int],
    spacing: float,
    grid_center: tuple[float, float, float],
    requested_center: tuple[float, float, float],
    requested_size: tuple[float, float, float],
) -> tuple[float, float, float]:
    if abs(spacing - AUTOGRID_SPACING) > 1e-9:
        raise InputError(
            f"AutoDock-GPU target requires {AUTOGRID_SPACING:.3f} Angstrom grid spacing"
        )
    if any(points + 1 > AUTODOCK_GPU_MAX_GRID_POINTS for points in npts):
        raise InputError("AutoDock-GPU target exceeds the 256-point grid limit")
    if any(
        abs(actual - requested) > 0.001
        for actual, requested in zip(grid_center, requested_center)
    ):
        raise InputError("AutoGrid center does not match the requested pocket center")
    actual_size = tuple(round(points * spacing, 6) for points in npts)
    if any(
        actual + 1e-6 < requested
        for actual, requested in zip(actual_size, requested_size)
    ):
        raise InputError(
            "AutoGrid dimensions do not fully cover the requested docking pocket"
        )
    return actual_size  # type: ignore[return-value]


def _conect_serials(line: str) -> set[int]:
    serials: set[int] = set()
    for field in line[6:].split():
        try:
            serials.add(int(field))
        except ValueError:
            continue
    return serials


def _write_complex_components(
    lines: list[str],
    atoms: list[_PDBAtom],
    selected_key: tuple[str, str, int, str],
    receptor_path: Path,
    ligand_path: Path,
) -> None:
    selected_atoms = [
        atom
        for atom in atoms
        if atom.line[:6].strip().upper() == "HETATM" and atom.residue_key == selected_key
    ]
    selected_lines = {atom.line_index for atom in selected_atoms}
    selected_serials = {atom.serial for atom in selected_atoms}
    external_bonds = []
    for line in lines:
        if line[:6].strip().upper() != "CONECT":
            continue
        serials = _conect_serials(line)
        if serials & selected_serials and serials - selected_serials:
            external_bonds.append(line)
    if external_bonds:
        raise InputError(
            "selected ligand has CONECT records to the receptor; covalent complexes are unsupported"
        )

    receptor_lines: list[str] = []
    ligand_lines = [atom.line for atom in selected_atoms]
    for index, line in enumerate(lines):
        record = line[:6].strip().upper()
        if index in selected_lines:
            continue
        if record == "ANISOU":
            try:
                if int(line.ljust(11)[6:11]) in selected_serials:
                    continue
            except ValueError:
                pass
        if record == "CONECT":
            serials = _conect_serials(line)
            if serials & selected_serials:
                if serials <= selected_serials:
                    ligand_lines.append(line)
                continue
        receptor_lines.append(line)

    remaining_protein = [
        atom for atom in atoms if atom.line[:6].strip().upper() == "ATOM" and atom.line_index not in selected_lines
    ]
    if not remaining_protein:
        raise InputError("complex PDB contains no protein ATOM records after ligand removal")
    receptor_path.write_text("\n".join(receptor_lines).rstrip() + "\n", encoding="utf-8")
    ligand_path.write_text("\n".join(ligand_lines) + "\nEND\n", encoding="utf-8")


def _complete_target(
    *,
    output_dir: Path,
    receptor_pdb: Path,
    reference_path: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    padding: float,
    mode: str,
    reference_atoms: int,
    inputs: dict[str, Any],
    ligand_selector: str | None = None,
) -> PreparedTarget:
    receptor_pdbqt = output_dir / TARGET_RECEPTOR_FILENAME
    gpf_path = output_dir / AUTODOCK_GPU_GPF_FILENAME
    autogrid_request_size, expected_npts = _autodock_grid_geometry(size)
    preparation = prepare_receptor(
        receptor_pdb,
        receptor_pdbqt,
        gpf_path=gpf_path,
        center=center,
        size=autogrid_request_size,
    )
    grid_npts, grid_spacing, grid_center = _read_gpf_geometry(gpf_path)
    if grid_npts != expected_npts:
        raise InputError(
            f"Meeko wrote AutoGrid npts {grid_npts}; expected {expected_npts}"
        )
    grid_size = _validate_grid_coverage(
        npts=grid_npts,
        spacing=grid_spacing,
        grid_center=grid_center,
        requested_center=center,
        requested_size=size,
    )
    grid = run_autogrid(
        gpf_path,
        log_path=output_dir / AUTOGRID_LOG_FILENAME,
    )
    fld_path = Path(str(grid["fld"])).resolve()
    expected_fld = output_dir / AUTODOCK_GPU_FLD_FILENAME
    if fld_path != expected_fld or not fld_path.is_file():
        raise InputError(
            f"AutoGrid produced an unexpected grid descriptor: {fld_path}"
        )

    pocket_path = output_dir / POCKET_FILENAME
    pocket = {
        "schema_version": 1,
        "method": "reference-ligand-axis-aligned-bounds",
        "center": list(center),
        "size": list(size),
        "padding_angstrom": padding,
        "reference_ligand": reference_path.name,
        "reference_heavy_atoms": reference_atoms,
    }
    _write_json(pocket_path, pocket)

    asset_candidates = [
        gpf_path,
        fld_path,
        output_dir / AUTOGRID_LOG_FILENAME,
        output_dir / "boron-silicon-atom_par.dat",
        output_dir / "receptor.box.pdb",
        output_dir / "receptor.maps.xyz",
        *sorted(output_dir.glob("receptor.*.map")),
    ]
    grid_assets: list[Path] = []
    seen_assets: set[Path] = set()
    for candidate in asset_candidates:
        resolved = candidate.resolve()
        if (
            resolved.parent == output_dir
            and resolved.is_file()
            and resolved not in seen_assets
        ):
            grid_assets.append(resolved)
            seen_assets.add(resolved)
    if gpf_path.resolve() not in seen_assets or fld_path not in seen_assets:
        raise InputError("AutoGrid did not produce the required GPF and FLD assets")

    manifest_path = output_dir / TARGET_MANIFEST_FILENAME
    manifest = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "inputs": inputs,
        "ligand_selector": ligand_selector,
        "outputs": {
            "receptor_pdb": receptor_pdb.name,
            "receptor_pdbqt": receptor_pdbqt.name,
            "reference_ligand": reference_path.name,
            "pocket": pocket_path.name,
        },
        "sha256": {
            "receptor_pdb": _sha256(receptor_pdb),
            "receptor_pdbqt": _sha256(receptor_pdbqt),
            "reference_ligand": _sha256(reference_path),
            "pocket": _sha256(pocket_path),
        },
        "autodock_gpu": {
            "engine_version": "1.6",
            "scoring_function": "ad4",
            "grid_generator": grid.get("version"),
            "grid": {
                "center": list(grid_center),
                "npts": list(grid_npts),
                "requested_center": list(center),
                "requested_size": list(size),
                "rounding": "outward-even-grid-intervals",
                "size": list(grid_size),
                "spacing": grid_spacing,
            },
            "gpf": gpf_path.name,
            "fld": fld_path.name,
            "files": [path.name for path in grid_assets],
            "sha256": {path.name: _sha256(path) for path in grid_assets},
            "receptor_preparation_command": [
                str(item) for item in preparation["command"]
            ],
            "autogrid_command": [str(item) for item in grid["command"]],
        },
        "receptor_preparation_command": [
            str(item) for item in preparation["command"]
        ],
    }
    _write_json(manifest_path, manifest)
    return load_target(output_dir)


def prepare_target(
    *,
    output_dir: Path,
    complex_pdb: Path | None = None,
    ligand_id: str | None = None,
    receptor_pdb: Path | None = None,
    reference_ligand_sdf: Path | None = None,
    padding: float = DEFAULT_POCKET_PADDING,
) -> PreparedTarget:
    """Prepare a target from either a complex PDB or aligned receptor/SDF pair."""

    complex_mode = complex_pdb is not None
    pair_mode = receptor_pdb is not None or reference_ligand_sdf is not None
    if complex_mode == pair_mode:
        raise InputError(
            "select exactly one target mode: --complex with --ligand-id, or --receptor with --reference-ligand"
        )
    if complex_mode and not ligand_id:
        raise InputError("--complex requires --ligand-id")
    if complex_mode and reference_ligand_sdf is not None:
        raise InputError("--reference-ligand cannot be combined with --complex")
    if pair_mode and (receptor_pdb is None or reference_ligand_sdf is None):
        raise InputError("--receptor and --reference-ligand must be provided together")
    if pair_mode and ligand_id is not None:
        raise InputError("--ligand-id is only valid with --complex")
    if not math.isfinite(padding) or padding < 0:
        raise InputError("pocket padding must be a finite non-negative number")

    if complex_pdb is not None:
        source = _resolve_existing_file(complex_pdb, "complex PDB")
        lines, atoms = _read_pdb(source)
        selected_key, selected_atoms = _select_ligand_residue(atoms, ligand_id or "")
        coordinates = _reference_coordinates(selected_atoms)
        center, size = _calculate_box(coordinates, padding)
        destination = _prepare_output_directory(output_dir)
        receptor_copy = destination / "receptor.pdb"
        reference_copy = destination / "reference_ligand.pdb"
        _write_complex_components(
            lines, atoms, selected_key, receptor_copy, reference_copy
        )
        return _complete_target(
            output_dir=destination,
            receptor_pdb=receptor_copy,
            reference_path=reference_copy,
            center=center,
            size=size,
            padding=padding,
            mode="complex-pdb",
            reference_atoms=len(coordinates),
            inputs={"complex_pdb": {"path": str(source), "sha256": _sha256(source)}},
            ligand_selector=_residue_label(selected_key),
        )

    assert receptor_pdb is not None and reference_ligand_sdf is not None
    receptor_source = _resolve_existing_file(receptor_pdb, "receptor PDB")
    ligand_source = _resolve_existing_file(reference_ligand_sdf, "reference ligand SDF")
    _, receptor_atoms = _read_pdb(receptor_source)
    coordinates, atom_count = _sdf_coordinates(ligand_source)
    nearest_distance = _validate_reference_alignment(coordinates, receptor_atoms)
    center, size = _calculate_box(coordinates, padding)
    destination = _prepare_output_directory(output_dir)
    receptor_copy = destination / "receptor.pdb"
    reference_copy = destination / "reference_ligand.sdf"
    shutil.copy2(receptor_source, receptor_copy)
    shutil.copy2(ligand_source, reference_copy)
    return _complete_target(
        output_dir=destination,
        receptor_pdb=receptor_copy,
        reference_path=reference_copy,
        center=center,
        size=size,
        padding=padding,
        mode="receptor-pdb-reference-sdf",
        reference_atoms=atom_count,
        inputs={
            "receptor_pdb": {"path": str(receptor_source), "sha256": _sha256(receptor_source)},
            "nearest_heavy_atom_distance_angstrom": nearest_distance,
            "reference_ligand_sdf": {"path": str(ligand_source), "sha256": _sha256(ligand_source)},
        },
    )


def _numeric_triple(value: Any, label: str, *, positive: bool) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise InputError(f"target {label} must contain exactly three numbers")
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"target {label} must contain exactly three numbers") from exc
    if not all(math.isfinite(item) for item in parsed):
        raise InputError(f"target {label} contains a non-finite value")
    if positive and any(item <= 0 for item in parsed):
        raise InputError(f"target {label} dimensions must be positive")
    return parsed  # type: ignore[return-value]


def _verify_autodock_gpu_assets(
    directory: Path,
    manifest: dict[str, Any],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> Path | None:
    section = manifest.get("autodock_gpu")
    if section is None and manifest.get("schema_version") == 1:
        return None
    if not isinstance(section, dict):
        raise InputError("prepared-target manifest is missing AutoDock-GPU assets")
    files = section.get("files")
    checksums = section.get("sha256")
    fld_filename = section.get("fld")
    gpf_filename = section.get("gpf")
    if (
        not isinstance(files, list)
        or not files
        or not isinstance(checksums, dict)
        or not isinstance(fld_filename, str)
        or not isinstance(gpf_filename, str)
    ):
        raise InputError("prepared-target AutoDock-GPU metadata is incomplete")
    if len(files) != len(set(files)):
        raise InputError("prepared-target AutoDock-GPU file list contains duplicates")

    verified: dict[str, Path] = {}
    for filename in files:
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise InputError(
                "prepared-target manifest has an invalid AutoDock-GPU filename"
            )
        expected = checksums.get(filename)
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise InputError(
                f"prepared-target manifest has an invalid checksum for {filename}"
            )
        candidate = (directory / filename).resolve()
        if candidate.parent != directory or not candidate.is_file():
            raise InputError(f"prepared target is missing {filename}: {directory}")
        if _sha256(candidate) != expected:
            raise InputError(f"prepared-target checksum mismatch for {filename}")
        verified[filename] = candidate

    fld = verified.get(fld_filename)
    if fld is None or fld.suffix.lower() != ".fld":
        raise InputError("prepared-target AutoDock-GPU FLD is not in its asset list")
    gpf = verified.get(gpf_filename)
    if gpf is None or gpf.suffix.lower() != ".gpf":
        raise InputError("prepared-target AutoDock-GPU GPF is not in its asset list")
    npts, spacing, grid_center = _read_gpf_geometry(gpf)
    grid_size = _validate_grid_coverage(
        npts=npts,
        spacing=spacing,
        grid_center=grid_center,
        requested_center=center,
        requested_size=size,
    )
    expected_grid = {
        "center": list(grid_center),
        "npts": list(npts),
        "requested_center": list(center),
        "requested_size": list(size),
        "rounding": "outward-even-grid-intervals",
        "size": list(grid_size),
        "spacing": spacing,
    }
    if section.get("grid") != expected_grid:
        raise InputError("prepared-target AutoDock-GPU grid metadata is inconsistent")
    return fld


def _verify_target_files(
    directory: Path,
    manifest: dict[str, Any],
    receptor: Path,
    pocket: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> Path | None:
    outputs = manifest.get("outputs")
    checksums = manifest.get("sha256")
    if not isinstance(outputs, dict) or not isinstance(checksums, dict):
        raise InputError("prepared-target manifest is missing outputs or sha256 metadata")
    paths: dict[str, Path] = {}
    for label in ("receptor_pdb", "receptor_pdbqt", "reference_ligand", "pocket"):
        filename = outputs.get(label)
        expected = checksums.get(label)
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise InputError(f"prepared-target manifest has an invalid {label} filename")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise InputError(f"prepared-target manifest has an invalid {label} checksum")
        candidate = (directory / filename).resolve()
        if candidate.parent != directory or not candidate.is_file():
            raise InputError(f"prepared target is missing {filename}: {directory}")
        if _sha256(candidate) != expected:
            raise InputError(f"prepared-target checksum mismatch for {filename}")
        paths[label] = candidate
    if paths["receptor_pdbqt"] != receptor.resolve() or paths["pocket"] != pocket.resolve():
        raise InputError("prepared-target manifest does not reference the standard receptor and pocket files")
    return _verify_autodock_gpu_assets(directory, manifest, center, size)


def load_target(path: Path) -> PreparedTarget:
    """Validate and resolve a target bundle directory."""

    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise InputError(f"prepared target directory does not exist: {directory}")
    receptor = directory / TARGET_RECEPTOR_FILENAME
    pocket_path = directory / POCKET_FILENAME
    manifest_path = directory / TARGET_MANIFEST_FILENAME
    for required in (receptor, pocket_path, manifest_path):
        if not required.is_file():
            raise InputError(f"prepared target is missing {required.name}: {directory}")
    try:
        pocket = json.loads(pocket_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"prepared target metadata is unreadable: {directory}") from exc
    if not isinstance(pocket, dict) or not isinstance(manifest, dict):
        raise InputError("prepared-target metadata must contain JSON objects")
    if pocket.get("schema_version") != 1 or manifest.get("schema_version") not in {
        1,
        TARGET_SCHEMA_VERSION,
    }:
        raise InputError("unsupported prepared-target schema version")
    center = _numeric_triple(pocket.get("center"), "center", positive=False)
    size = _numeric_triple(pocket.get("size"), "size", positive=True)
    autodock_gpu_fld = _verify_target_files(
        directory,
        manifest,
        receptor,
        pocket_path,
        center,
        size,
    )
    return PreparedTarget(
        directory=directory,
        receptor=receptor,
        center=center,
        size=size,
        pocket=pocket_path,
        manifest=manifest_path,
        autodock_gpu_fld=autodock_gpu_fld,
    )

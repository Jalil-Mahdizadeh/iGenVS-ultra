from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from igenvs.errors import InputError
from igenvs.target import _autodock_grid_geometry, load_target, prepare_target


def _pdb_atom(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    coordinates: tuple[float, float, float],
    *,
    record: str = "ATOM",
    element: str = "C",
) -> str:
    x, y, z = coordinates
    return (
        f"{record:<6}{serial:5d} {name:<4} {resname:>3} {chain:1}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2}"
    )


def _fake_receptor_preparation(
    input_path: Path,
    output_path: Path,
    *,
    gpf_path: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> dict[str, object]:
    assert input_path.is_file()
    output_path.write_text("REMARK fake prepared receptor\n", encoding="utf-8")
    npts = tuple(round(value / 0.375) for value in size)
    gpf_path.write_text(
        f"npts {npts[0]} {npts[1]} {npts[2]}\n"
        "gridfld receptor.maps.fld\n"
        "spacing 0.375\n"
        f"gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n"
        "map receptor.C.map\n",
        encoding="utf-8",
    )
    (output_path.parent / "boron-silicon-atom_par.dat").write_text(
        "fake parameters\n", encoding="utf-8"
    )
    (output_path.parent / "receptor.box.pdb").write_text(
        f"REMARK {center} {size}\n", encoding="utf-8"
    )
    return {
        "input": str(input_path),
        "output": str(output_path),
        "gpf": str(gpf_path),
        "command": ["fake-meeko", str(input_path), str(output_path)],
        "stdout": "",
        "stderr": "",
    }


def _fake_autogrid(gpf_path: Path, *, log_path: Path) -> dict[str, object]:
    assert gpf_path.is_file()
    directory = gpf_path.parent
    fld = directory / "receptor.maps.fld"
    fld.write_text("label=Fake AD4 maps\n", encoding="utf-8")
    (directory / "receptor.C.map").write_text("fake map\n", encoding="utf-8")
    (directory / "receptor.maps.xyz").write_text("0 0 0\n", encoding="utf-8")
    log_path.write_text("AutoGrid complete\n", encoding="utf-8")
    return {
        "gpf": str(gpf_path),
        "fld": str(fld),
        "log": str(log_path),
        "command": ["fake-autogrid", "-p", gpf_path.name],
        "version": "AutoGrid test-version",
        "stdout": "",
        "stderr": "",
    }


def test_autodock_grid_rounds_outward_and_enforces_engine_limit() -> None:
    size, npts = _autodock_grid_geometry((18.664, 26.739, 23.526))
    assert size == (18.75, 27.0, 24.0)
    assert npts == (50, 72, 64)
    with pytest.raises(InputError, match="grid limit"):
        _autodock_grid_geometry((96.0, 20.0, 20.0))


def test_prepare_target_from_complex_removes_selected_ligand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "\n".join(
            [
                "HEADER    TEST COMPLEX",
                _pdb_atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)),
                _pdb_atom(2, "C1", "LIG", "A", 501, (10.0, 20.0, 30.0), record="HETATM"),
                _pdb_atom(3, "O1", "LIG", "A", 501, (14.0, 26.0, 38.0), record="HETATM", element="O"),
                _pdb_atom(4, "ZN", "ZN", "B", 900, (2.0, 2.0, 2.0), record="HETATM", element="ZN"),
                "CONECT    2    3",
                "CONECT    3    2",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("igenvs.target.prepare_receptor", _fake_receptor_preparation)
    monkeypatch.setattr("igenvs.target.run_autogrid", _fake_autogrid)

    target = prepare_target(
        complex_pdb=complex_pdb,
        ligand_id="A:LIG:501",
        padding=5.0,
        output_dir=tmp_path / "target",
    )

    assert target.center == (12.0, 23.0, 34.0)
    assert target.size == (14.0, 16.0, 18.0)
    receptor_text = (target.directory / "receptor.pdb").read_text(encoding="utf-8")
    ligand_text = (target.directory / "reference_ligand.pdb").read_text(encoding="utf-8")
    assert " LIG " not in receptor_text
    assert " ZN " in receptor_text
    assert " LIG " in ligand_text
    assert "CONECT" in ligand_text
    assert load_target(target.directory) == target
    manifest = json.loads(target.manifest.read_text(encoding="utf-8"))
    assert manifest["mode"] == "complex-pdb"
    assert manifest["ligand_selector"] == "A:LIG:501"
    assert manifest["schema_version"] == 2
    assert manifest["autodock_gpu"]["grid_generator"] == "AutoGrid test-version"
    assert manifest["autodock_gpu"]["grid"]["npts"] == [38, 44, 48]
    assert manifest["autodock_gpu"]["grid"]["requested_size"] == [14.0, 16.0, 18.0]
    assert manifest["autodock_gpu"]["grid"]["size"] == [14.25, 16.5, 18.0]
    assert target.autodock_gpu_fld == target.directory / "receptor.maps.fld"
    target.receptor.write_text("REMARK tampered\n", encoding="utf-8")
    with pytest.raises(InputError, match="checksum mismatch"):
        load_target(target.directory)


def test_short_ligand_id_must_be_unique(tmp_path: Path) -> None:
    complex_pdb = tmp_path / "ambiguous.pdb"
    complex_pdb.write_text(
        "\n".join(
            [
                _pdb_atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)),
                _pdb_atom(2, "C1", "LIG", "A", 501, (1.0, 1.0, 1.0), record="HETATM"),
                _pdb_atom(3, "C1", "LIG", "B", 501, (2.0, 2.0, 2.0), record="HETATM"),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="ambiguous"):
        prepare_target(
            complex_pdb=complex_pdb,
            ligand_id="LIG",
            output_dir=tmp_path / "target",
        )


def test_prepare_target_from_aligned_sdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        _pdb_atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)) + "\nEND\n",
        encoding="utf-8",
    )
    molecule = Chem.MolFromSmiles("CO")
    assert molecule is not None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    conformer.SetAtomPosition(0, Point3D(1.0, 2.0, 3.0))
    conformer.SetAtomPosition(1, Point3D(5.0, 8.0, 9.0))
    molecule.AddConformer(conformer)
    ligand = tmp_path / "ligand.sdf"
    writer = Chem.SDWriter(str(ligand))
    writer.write(molecule)
    writer.close()
    monkeypatch.setattr("igenvs.target.prepare_receptor", _fake_receptor_preparation)
    monkeypatch.setattr("igenvs.target.run_autogrid", _fake_autogrid)

    target = prepare_target(
        receptor_pdb=receptor,
        reference_ligand_sdf=ligand,
        padding=5.0,
        output_dir=tmp_path / "target",
    )

    assert target.center == (3.0, 5.0, 6.0)
    assert target.size == (14.0, 16.0, 16.0)
    assert (target.directory / "reference_ligand.sdf").is_file()
    pocket = json.loads(target.pocket.read_text(encoding="utf-8"))
    assert pocket["reference_heavy_atoms"] == 2
    assert target.autodock_gpu_fld is not None
    target.autodock_gpu_fld.write_text("tampered map descriptor\n", encoding="utf-8")
    with pytest.raises(InputError, match="checksum mismatch"):
        load_target(target.directory)


def test_target_modes_require_complete_input_pairs(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="provided together"):
        prepare_target(receptor_pdb=tmp_path / "receptor.pdb", output_dir=tmp_path / "target")


def test_load_target_rejects_non_positive_box(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "receptor.pdbqt").write_text("REMARK receptor\n", encoding="utf-8")
    (target / "manifest.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    (target / "pocket.json").write_text(
        '{"schema_version": 1, "center": [0, 0, 0], "size": [20, 0, 20]}\n',
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="positive"):
        load_target(target)


def test_pair_rejects_spatially_unaligned_sdf(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        _pdb_atom(1, "CA", "ALA", "A", 1, (0.0, 0.0, 0.0)) + "\nEND\n",
        encoding="utf-8",
    )
    molecule = Chem.MolFromSmiles("CO")
    assert molecule is not None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    conformer.SetAtomPosition(0, Point3D(100.0, 100.0, 100.0))
    conformer.SetAtomPosition(1, Point3D(101.0, 100.0, 100.0))
    molecule.AddConformer(conformer)
    ligand = tmp_path / "unaligned.sdf"
    writer = Chem.SDWriter(str(ligand))
    writer.write(molecule)
    writer.close()

    with pytest.raises(InputError, match="coordinate frame"):
        prepare_target(
            receptor_pdb=receptor,
            reference_ligand_sdf=ligand,
            output_dir=tmp_path / "target",
        )

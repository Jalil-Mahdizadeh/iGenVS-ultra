from pathlib import Path

import pytest

from igenvs.pose_validation import validate_ring_closures


def ring_atoms(distance: float = 1.5, dummy_shift: float = 0) -> str:
    return "\n".join(
        f"ATOM  {serial:5d}  C   UNL     1    {x:8.3f}{y:8.3f}{0.:8.3f}  1.00  0.00     0.000 {kind}"
        for serial, x, y, kind in (
            (1, 0., 0., "CG0"), (2, distance, 0., "CG0"),
            (3, 1.5, dummy_shift, "G0"), (4, 0., 0., "G0"),
        )
    ) + "\n"


def test_all_pose_ring_closures_are_checked(tmp_path: Path) -> None:
    prepared, output = tmp_path / "in.pdbqt", tmp_path / "out.pdbqt"
    prepared.write_text(ring_atoms())
    output.write_text("MODEL 1\n" + ring_atoms() + "ENDMDL\n")
    validate_ring_closures(prepared, output)
    output.write_text(output.read_text() + "MODEL 2\n" + ring_atoms(6.98) + "ENDMDL\n")
    with pytest.raises(ValueError, match="pose 2: unclosed ring"):
        validate_ring_closures(prepared, output)


def test_correct_bond_length_alone_does_not_prove_ring_closure(tmp_path: Path) -> None:
    prepared, output = tmp_path / "in.pdbqt", tmp_path / "out.pdbqt"
    prepared.write_text(ring_atoms())
    output.write_text("MODEL 1\n" + ring_atoms(dummy_shift=1) + "ENDMDL\n")
    with pytest.raises(ValueError, match="ring-closure dummy"):
        validate_ring_closures(prepared, output)

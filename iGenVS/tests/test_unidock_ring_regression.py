"""Real GPU regression; opt in with IGENVS_RUN_GPU_TESTS=1 inside the image."""

import os
import shutil
from pathlib import Path

import pytest
from rdkit import Chem

from igenvs.docking import DockingConfig, dock_batch_resilient
from igenvs.preparation import _prepare_one
from igenvs.records import PreparedLigand, ValidatedRecord


@pytest.mark.skipif(os.environ.get("IGENVS_RUN_GPU_TESTS") != "1" or not shutil.which("unidock"),
                    reason="requires an explicitly enabled Uni-Dock GPU runtime")
@pytest.mark.parametrize("seed,search_mode", [(181129, "fast"), (104729, "balance"), (271828, "detail")])
def test_ring_sizes_and_multiple_closures_return_three_intact_poses(
    tmp_path: Path, seed: int, search_mode: str,
) -> None:
    # A separate synthetic receptor isolates ring closure from a particular
    # crystal structure, ligand selector, or reference-ligand box calculation.
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
    )
    prep = tmp_path / "prepared"
    prep.mkdir()
    inputs = ["C1" + "C" * (size - 1) + "1" for size in (6, 7, 8, 9, 12, 16, 24, 33)]
    inputs += ["C1CCCCCC1CC2CCCCCC2"]
    ligands = []
    for row, smiles in enumerate(inputs, 1):
        mol = Chem.MolFromSmiles(smiles)
        record = ValidatedRecord(f"ring-{row}", smiles, Chem.MolToSmiles(mol), row,
                                 mol.GetNumHeavyAtoms(), 0)
        ligand = _prepare_one(record, prep, seed, 300, 48)
        assert isinstance(ligand, PreparedLigand), ligand
        ligands.append(ligand)
    results, _ = dock_batch_resilient(ligands, config=DockingConfig(
        receptor=receptor, center=(0, 0, 0), size=(24, 24, 24), num_modes=3,
        energy_range=0, search_mode=search_mode, seed=seed,
    ), work_dir=tmp_path, label="rings")
    assert len(results) == len(inputs)
    assert not [(r.ligand.record.molecule_id, r.error) for r in results if r.status != "success"]
    for result in results:
        assert result.status == "success", (result.ligand.record.molecule_id, result.error)
        assert len(result.scores) == 3
        # The production collector checks closure in all PDBQT models. Verify
        # exported chemistry too, including the restored covalent ring bonds.
        from meeko import PDBQTMolecule, RDKitMolCreate

        text = result.pose_path.read_text()
        molecules = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(text, skip_typing=True))
        assert len(molecules) == 1 and molecules[0] is not None
        molecule = molecules[0]
        assert molecule.GetNumConformers() == 3
        for conformer in molecule.GetConformers():
            for bond in molecule.GetBonds():
                if bond.GetBeginAtom().GetAtomicNum() == 1 or bond.GetEndAtom().GetAtomicNum() == 1:
                    continue
                distance = (conformer.GetAtomPosition(bond.GetBeginAtomIdx())
                            - conformer.GetAtomPosition(bond.GetEndAtomIdx())).Length()
                assert 1.0 < distance < 2.1, (result.ligand.record.molecule_id, distance)

"""SDF chemistry/coordinates and exporting a completed job without redocking."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from igenvs_ultra import pose_export, workflow
from igenvs_ultra.cli import build_parser

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None


PROJECT = Path(__file__).resolve().parents[2]


class ExportRecoveryTests(unittest.TestCase):
    def test_merged_reader_keeps_multiple_models_inside_each_ligand(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "poses.pdbqt"
            source.write_text("REMARK IGENVS MOLECULE_ID one\nMODEL 1\nENDMDL\nMODEL 2\nENDMDL\n"
                              "REMARK IGENVS MOLECULE_ID two\nROOT\nENDROOT\n")
            blocks = list(pose_export.merged_molecules(source))
            self.assertEqual([name for name, _ in blocks], ["one", "two"])
            self.assertEqual(blocks[0][1].count("MODEL"), 2)
            source.write_text("ROOT\nENDROOT\n")
            with self.assertRaisesRegex(ValueError, "boundary"):
                list(pose_export.merged_molecules(source))

    def test_conversion_failure_preserves_old_sdf_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "poses.sdf"
            output.write_text("existing export")
            with mock.patch.object(pose_export, "sdf_string", side_effect=[("first\n$$$$\n", 1), ValueError("bad pose")]):
                with self.assertRaisesRegex(ValueError, "bad pose"):
                    pose_export.write_sdf(output, iter([("one", "pdbqt1"), ("two", "pdbqt2")]))
            self.assertEqual(output.read_text(), "existing export")
            self.assertFalse(output.with_suffix(".sdf.partial").exists())

    def test_completed_job_exports_and_repairs_missing_sdf_without_redocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            (target / "manifest.json").write_text("{}")
            library = root / "input.smi"
            library.write_text("CCO ethanol\n")
            job = root / "job"
            args = build_parser().parse_args([
                "dock", "--prepared-target", str(target), "--input", str(library),
                "--output-dir", str(job), "--assets-dir", str(PROJECT),
            ])
            workflow.ensure_regular_config(job, workflow.make_regular_config(args, PROJECT))
            output = job / "docking"
            output.mkdir()
            results = output / "results.csv"
            results.write_text("molecule_id,source_row,status,num_poses\nethanol,1,success,1\n")
            (output / "poses.pdbqt").write_text("saved poses")
            manifest = {"status": "complete", "counts": {"docked": 1}, "outputs": {
                "results": str(results), "poses": str(output / "poses.pdbqt"),
            }}
            workflow.atomic_json(output / "manifest.json", manifest)
            stamp = results.stat().st_mtime_ns
            runtime = mock.Mock()

            def export(_tool, command, _log, **kwargs):
                self.assertTrue(command[1].endswith("pose_export.py"))
                self.assertFalse(kwargs["gpu"])
                (output / "poses.sdf").write_text("exported\n$$$$\n")

            runtime.run_logged.side_effect = export
            with mock.patch.object(workflow, "Runtime", return_value=runtime), \
                    mock.patch.object(workflow, "prepare_target"), \
                    mock.patch.object(workflow, "docking_gpu_ids", return_value=["0"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                first = workflow.regular_dock(args)
                self.assertEqual(runtime.run_logged.call_count, 1)
                self.assertEqual(first["poses_sdf"], str(output / "poses.sdf"))
                workflow.regular_dock(args)
                self.assertEqual(runtime.run_logged.call_count, 1)
                (output / "poses.sdf").unlink()
                workflow.regular_dock(args)
                self.assertEqual(runtime.run_logged.call_count, 2)
            self.assertEqual(results.stat().st_mtime_ns, stamp)
            self.assertEqual((output / "poses.pdbqt").read_text(), "saved poses")

    def test_scores_only_never_launches_export(self):
        runtime = mock.Mock()
        workflow.ensure_regular_sdf(argparse.Namespace(pose_output="none"), runtime, Path("unused"), {})
        runtime.run_logged.assert_not_called()


@unittest.skipIf(Chem is None, "run inside the released iGenVS environment")
class SdfChemistryTests(unittest.TestCase):
    def prepare(self, smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=13), 0)
        setup = MoleculePreparation().prepare(mol)[0]
        pdbqt, success, error = PDBQTWriterLegacy.write_string(setup)
        self.assertTrue(success, error)
        return pdbqt

    def assert_coordinates(self, mol, pdbqt, shift=0):
        expected = sorted(tuple(float(line[a:b]) + shift for a, b in ((30, 38), (38, 46), (46, 54)))
                          for line in pdbqt.splitlines() if line.startswith("ATOM") and line.split()[-1] not in {"H", "HD"})
        actual = sorted(tuple(mol.GetConformer().GetAtomPosition(atom.GetIdx())) for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        self.assertEqual(len(actual), len(expected))
        for found, wanted in zip(actual, expected):
            for a, b in zip(found, wanted):
                self.assertAlmostEqual(a, b, places=4)

    def test_merged_sdf_preserves_identity_stereochemistry_and_every_pose(self):
        smiles = "C[C@H](O)C(=O)Nc1ccccc1"
        pdbqt = self.prepare(smiles)
        shifted = "\n".join(
            line[:30] + "".join(f"{float(line[a:b]) + 1:8.3f}" for a, b in ((30, 38), (38, 46), (46, 54))) + line[54:]
            if line.startswith("ATOM") else line for line in pdbqt.splitlines()
        ) + "\n"
        vina = f"MODEL 1\nREMARK VINA RESULT: -7.0 0 0\n{pdbqt}ENDMDL\nMODEL 2\nREMARK VINA RESULT: -6.0 0 0\n{shifted}ENDMDL\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "poses.pdbqt").write_text("REMARK IGENVS MOLECULE_ID chiral ligand\n" + vina
                                             + "REMARK IGENVS MOLECULE_ID best pose\n" + pdbqt)
            report = pose_export.export_poses(root, "merged")
            self.assertEqual((report["molecules"], report["poses"]), (2, 3))
            mols = list(Chem.SDMolSupplier(str(root / "poses.sdf")))
        self.assertEqual(len(mols), 3)
        for mol in mols:
            self.assertIsNotNone(mol)
            self.assertEqual(Chem.MolToSmiles(mol), Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
        self.assertEqual([mol.GetProp("_Name") for mol in mols], ["chiral ligand", "chiral ligand", "best pose"])
        self.assert_coordinates(mols[0], pdbqt)
        self.assert_coordinates(mols[1], pdbqt, shift=1)
        self.assert_coordinates(mols[2], pdbqt)

    def test_individual_sdf_uses_original_id_and_stays_beside_pdbqt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "poses").mkdir()
            source = root / "poses/safe_filename.pdbqt"
            source.write_text(self.prepare("CC(=O)Oc1ccccc1C(=O)O"))
            with (root / "results.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["molecule_id", "status", "pose_ref"])
                writer.writerow(["original ID", "success", "poses/safe_filename.pdbqt"])
                writer.writerow(["failed", "docking_failed", ""])
            report = pose_export.export_poses(root, "individual")
            self.assertEqual((report["molecules"], report["poses"]), (1, 1))
            self.assertTrue(source.is_file())
            mol = next(iter(Chem.SDMolSupplier(str(source.with_suffix(".sdf")))))
            self.assertEqual(mol.GetProp("_Name"), "original ID")


if __name__ == "__main__":
    unittest.main()

"""Host orchestration test whose actual chemistry runs in the dedicated SIF."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from igenvs_ultra.workflow import Runtime

PROJECT = Path(__file__).resolve().parents[2]
SIF = Path("/nobackup/proj/disk/theo-storage/personal/jalil/iGenVS/containers/iGenVS.SIF")


@unittest.skipUnless(os.environ.get("IGENVS_ULTRA_SIF_INTEGRATION") == "1", "set IGENVS_ULTRA_SIF_INTEGRATION=1 with host Apptainer available")
class SourceRuntimeIntegrationTests(unittest.TestCase):
    def test_installed_sif_cannot_bypass_source_zero_success_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.csv"
            source.write_text("molecule_id,smiles\nethanol,CCO\n")
            binary_dir = root / "bin"
            binary_dir.mkdir()
            (binary_dir / "unidock").symlink_to("/bin/false")
            args = argparse.Namespace(execution="apptainer", gpu_ids=None, igenvs_image=SIF)
            runtime = Runtime(args, PROJECT, root, require_gmolai=False)
            with mock.patch.dict(os.environ, {
                "PYTHONPATH": "", "APPTAINERENV_PYTHONPATH": "",
                "PYTHONDONTWRITEBYTECODE": "1", "APPTAINERENV_PYTHONDONTWRITEBYTECODE": "1",
                "APPTAINERENV_PREPEND_PATH": str(binary_dir),
            }):
                result = runtime.capture("igenvs", [
                    "igenvs", "screen", "--input", str(source), "--id-column", "molecule_id",
                    "--receptor", str(PROJECT / "phase-7-benchmark-docking/targets/1err/receptor.pdbqt"),
                    "--center", "0", "0", "0", "--size", "20", "20", "20",
                    "--output-dir", str(root / "docking"), "--batch-size", "1",
                    "--prep-workers", "1", "--validation-workers", "1", "--pose-output", "none",
                ], gpu=False)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((root / "docking/manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["counts"]["prepared"], 1)
            self.assertEqual(manifest["counts"]["docked"], 0)
            self.assertIn("zero successful ligands", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()

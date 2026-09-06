from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from igenvs_ultra.cli import build_parser, validate_args
from igenvs_ultra.workflow import regular_dock


PROJECT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("IGENVS_ULTRA_INTEGRATION") == "1", "set IGENVS_ULTRA_INTEGRATION=1")
class RegularDockIntegrationTests(unittest.TestCase):
    def test_native_igenvs_outputs_and_resume(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory(dir=str(PROJECT)) as temporary:
            root = Path(temporary)
            source = root / "tiny.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["id", "smiles"])
                writer.writerow(["aspirin", "CC(=O)Oc1ccccc1C(=O)O"])
                writer.writerow(["caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"])
            job = root / "regular-job"
            args = parser.parse_args(
                [
                    "dock",
                    "--assets-dir", str(PROJECT),
                    "--target", str(PROJECT / "phase-7-benchmark-docking/targets/1err"),
                    "--input", str(source),
                    "--smiles-column", "smiles",
                    "--id-column", "id",
                    "--output-dir", str(job),
                    "--search-mode", "fast",
                    "--pose-output", "none",
                    "--batch-size", "8",
                    "--prep-workers", "1",
                    "--validation-workers", "1",
                ]
            )
            validate_args(parser, args)
            first = regular_dock(args)
            first_mtime = (job / "docking/results.csv").stat().st_mtime_ns
            second = regular_dock(args)
            self.assertEqual(first["counts"]["docked"], 2)
            self.assertEqual(first, second)
            self.assertEqual((job / "docking/results.csv").stat().st_mtime_ns, first_mtime)
            self.assertTrue((job / "docking/manifest.json").is_file())
            self.assertFalse((job / "models").exists())
            self.assertFalse((job / "screens").exists())


if __name__ == "__main__":
    unittest.main()

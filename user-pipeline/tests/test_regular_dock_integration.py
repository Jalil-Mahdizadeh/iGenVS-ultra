from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from igenvs_ultra import workflow
from igenvs_ultra.cli import build_parser, validate_args
from igenvs_ultra.workflow import regular_dock


PROJECT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("IGENVS_ULTRA_INTEGRATION") == "1", "set IGENVS_ULTRA_INTEGRATION=1")
class RegularDockIntegrationTests(unittest.TestCase):
    def test_native_igenvs_outputs_and_resume(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temporary:
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
            # A resume measures fresh wrapper wall time; scientific results and
            # persisted engine artifacts must stay identical.
            self.assertEqual(
                {key: value for key, value in first.items() if key != "timings"},
                {key: value for key, value in second.items() if key != "timings"},
            )
            self.assertEqual((job / "docking/results.csv").stat().st_mtime_ns, first_mtime)
            self.assertTrue((job / "docking/manifest.json").is_file())
            self.assertFalse((job / "models").exists())
            self.assertFalse((job / "screens").exists())


@unittest.skipUnless(os.environ.get("IGENVS_ULTRA_DOCKER_INTEGRATION") == "1", "set IGENVS_ULTRA_DOCKER_INTEGRATION=1 with Docker/GPU")
class GeneratedDockDockerIntegrationTests(unittest.TestCase):
    def test_single_and_shared_generation_with_resume(self):
        available = workflow.visible_gpu_tokens(None)
        self.assertTrue(available, "at least one GPU is required")
        parser = build_parser()
        count = 8
        for shared in (False, True):
            with self.subTest(shared_generation=shared), tempfile.TemporaryDirectory(prefix="generated-dock-") as temporary:
                job = Path(temporary) / "job"
                gpus = available[:2] if shared else available[:1]
                args = parser.parse_args([
                    "dock", "--execution", "docker", "--assets-dir", str(PROJECT),
                    "--complex", str(PROJECT / "complexes/4ag8.pdb"), "--ligand-id", "A:AXI:2000",
                    "--output-dir", str(job), "--gpu-ids", ",".join(gpus),
                    "--generate-count", str(count), "--model", "base-isomeric",
                    "--generator-batch-size", "64", "--generator-max-batch-size", "64",
                    "--search-mode", "fast", "--pose-output", "merged", "--num-modes", "1",
                    "--batch-size", "8", "--prep-workers", "1", "--validation-workers", "1",
                ])
                validate_args(parser, args)
                if shared:
                    workflow.ensure_regular_config(job, workflow.make_regular_config(args, PROJECT))
                    partial = job / "library/generated.smi.partial"
                    partial.parent.mkdir()
                    partial.write_text("incomplete output from an interrupted generation\n")
                    if len(gpus) == 1:
                        # A parallel job resumed on one GPU still uses shared generation
                        # and the shard launcher. Exercise this real path on single-GPU hosts.
                        (job / "docking/runs").mkdir(parents=True)
                first = regular_dock(args)
                generated = job / ("library/generated.smi" if shared else "docking/input/generated.smi")
                self.assertEqual(len(generated.read_text().splitlines()), count)
                self.assertGreater(first["counts"]["docked"], 0)
                results = job / "docking/results.csv"
                with results.open(newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count)
                sdf = job / "docking/poses.sdf"
                self.assertEqual(sdf.read_text().count("$$$$\n"), first["counts"]["docked"])
                if shared:
                    self.assertFalse(partial.exists())
                    manifest = json.loads((job / "library/generation-manifest.json").read_text())
                    self.assertEqual(manifest["status"], "complete")
                    self.assertEqual(manifest["rows"], count)
                    self.assertEqual(manifest["output_sha256"], workflow.sha256(generated))
                    self.assertTrue((job / "docking/shard-plan.json").is_file())
                paths = [generated, results, sdf, job / "docking/poses.pdbqt", job / "target/manifest.json"]
                timestamps = {path: path.stat().st_mtime_ns for path in paths}
                second = regular_dock(args)
                self.assertEqual(second["counts"], first["counts"])
                self.assertEqual({path: path.stat().st_mtime_ns for path in paths}, timestamps)


if __name__ == "__main__":
    unittest.main()

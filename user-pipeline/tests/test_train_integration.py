from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    from igenvs_ultra import model_ops
except ModuleNotFoundError:
    model_ops = None


PROJECT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    os.environ.get("IGENVS_ULTRA_TRAIN_INTEGRATION") == "1" and model_ops is not None,
    "run in the released gMolAI environment with IGENVS_ULTRA_TRAIN_INTEGRATION=1",
)
class TargetHeadTrainingIntegrationTests(unittest.TestCase):
    def test_three_seed_initial_fit_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            (job / "fit-config.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "target_name": "1err",
                        "head": {
                            "architecture": "wide_mlp_rank_aux",
                            "ensemble_seeds": list(model_ops.SEEDS),
                            "epochs": model_ops.EPOCHS,
                        },
                    }
                )
            )
            for library in ("UDRL-train", "UDRL-valid"):
                destination = job / "docking" / library / "scores.csv"
                destination.parent.mkdir(parents=True)
                destination.symlink_to(PROJECT / "phase-7-benchmark-docking/scores/1err" / f"{library}.csv")
            args = argparse.Namespace(
                job_dir=job,
                assets_dir=PROJECT,
                stage="initial",
                device="auto",
            )
            model_ops.run_train(args)
            manifest_path = job / "models/initial/ensemble-manifest.json"
            validation_path = job / "models/initial/validation-metrics.json"
            manifest = json.loads(manifest_path.read_text())
            validation = json.loads(validation_path.read_text())
            self.assertEqual(manifest["seeds"], list(model_ops.SEEDS))
            self.assertEqual(manifest["epochs"], 7)
            self.assertEqual(len(manifest["members"]), 3)
            self.assertEqual(validation["status"], "complete")
            checkpoint_times = {
                member["checkpoint"]: (job / member["checkpoint"]).stat().st_mtime_ns
                for member in manifest["members"]
            }
            model_ops.run_train(args)
            self.assertEqual(
                checkpoint_times,
                {
                    member["checkpoint"]: (job / member["checkpoint"]).stat().st_mtime_ns
                    for member in manifest["members"]
                },
            )


if __name__ == "__main__":
    unittest.main()

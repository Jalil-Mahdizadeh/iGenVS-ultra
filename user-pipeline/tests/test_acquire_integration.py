from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

try:
    from igenvs_ultra import model_ops
except ModuleNotFoundError:
    model_ops = None


PROJECT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    os.environ.get("IGENVS_ULTRA_ACQUIRE_INTEGRATION") == "1" and model_ops is not None,
    "run in the released gMolAI environment with IGENVS_ULTRA_ACQUIRE_INTEGRATION=1",
)
class AcquisitionIntegrationTests(unittest.TestCase):
    def test_released_round_one_quota_and_resume(self) -> None:
        released = json.loads(
            (PROJECT / "phase-8-benchmark-models/models/ensemble-manifest.json").read_text()
        )
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
            model_dir = job / "models/initial"
            model_dir.mkdir(parents=True)
            members = []
            for member in released["targets"]["1err"]["members"]:
                checkpoint = PROJECT / "phase-8-benchmark-models" / member["checkpoint"]
                members.append({**member, "checkpoint": str(checkpoint)})
            (model_dir / "ensemble-manifest.json").write_text(
                json.dumps(
                    {
                        "target": "1err",
                        "stage": "initial",
                        "standardizer_sha256": released["standardizer_sha256"],
                        "members": members,
                    }
                )
            )
            args = argparse.Namespace(
                job_dir=job,
                assets_dir=PROJECT,
                round=1,
                device="auto",
            )
            model_ops.run_acquire(args)
            selection = job / "al/round-1/acquisition/selected.csv"
            manifest_path = job / "al/round-1/acquisition/manifest.json"
            with selection.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            counts = Counter(row["acquisition_category"] for row in rows)
            self.assertEqual(len(rows), 30_000)
            self.assertEqual(
                counts,
                {"exploitation": 15_000, "uncertainty": 7_500, "diversity": 7_500},
            )
            self.assertEqual(len({row["al_source_row"] for row in rows}), 30_000)
            first_hash = model_ops.sha256(selection)
            released_selection = (
                PROJECT / "phase-9-benchmark-active-learning/rounds/round-1/acquisition/1err.csv"
            )
            with released_selection.open(newline="") as handle:
                released_rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(released_rows))
            for index, (actual, expected) in enumerate(zip(rows, released_rows), start=1):
                if actual != expected:
                    self.fail(f"released acquisition differs at row {index}: {actual} != {expected}")
            model_ops.run_acquire(args)
            self.assertEqual(model_ops.sha256(selection), first_hash)
            self.assertEqual(json.loads(manifest_path.read_text())["status"], "complete")

            for library in ("UDRL-train", "UDRL-valid"):
                destination = job / "docking" / library / "scores.csv"
                destination.parent.mkdir(parents=True)
                destination.symlink_to(
                    PROJECT / "phase-7-benchmark-docking/scores/1err" / f"{library}.csv"
                )
            round_scores = job / "al/round-1/docking/scores.csv"
            round_scores.parent.mkdir(parents=True)
            round_scores.symlink_to(
                PROJECT
                / "phase-9-benchmark-active-learning/rounds/round-1/docking/scores/1err.csv"
            )
            model_ops.run_train(
                argparse.Namespace(
                    job_dir=job,
                    assets_dir=PROJECT,
                    stage="round-1",
                    device="auto",
                )
            )
            trained = json.loads((job / "models/round-1/ensemble-manifest.json").read_text())
            released_round = json.loads(
                (
                    PROJECT
                    / "phase-9-benchmark-active-learning/rounds/round-1/models/ensemble-manifest.json"
                ).read_text()
            )
            for actual_member, expected_member in zip(
                trained["members"], released_round["targets"]["1err"]["members"]
            ):
                actual_checkpoint = model_ops.torch.load(
                    job / actual_member["checkpoint"], map_location="cpu", weights_only=False
                )
                expected_checkpoint = model_ops.torch.load(
                    PROJECT
                    / "phase-9-benchmark-active-learning/rounds/round-1"
                    / expected_member["checkpoint"],
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertTrue(
                    all(
                        model_ops.torch.equal(actual_checkpoint["model_state"][name], value)
                        for name, value in expected_checkpoint["model_state"].items()
                    )
                )


if __name__ == "__main__":
    unittest.main()

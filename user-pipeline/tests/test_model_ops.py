from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from igenvs_ultra import model_ops
except ModuleNotFoundError:  # The lightweight host driver intentionally has no ML dependency.
    torch = None
    model_ops = None


PROJECT = Path(__file__).resolve().parents[2]


@unittest.skipIf(torch is None or model_ops is None, "run inside the released gMolAI environment")
class ModelOperationIntegrationTests(unittest.TestCase):
    def test_default_scoring_removes_embedding_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            model_dir = job / "models/initial"
            model_dir.mkdir(parents=True)
            _, _, _, standardizer_hash = model_ops.load_standardizer(PROJECT)
            members = []
            for seed in model_ops.SEEDS:
                checkpoint = model_dir / f"seed-{seed}.pt"
                torch.save(
                    {
                        "candidate": model_ops.ARCHITECTURE,
                        "seed": seed,
                        "target": "synthetic",
                        "model_state": model_ops.make_model().state_dict(),
                    },
                    checkpoint,
                )
                members.append(
                    {
                        "seed": seed,
                        "checkpoint": str(checkpoint.relative_to(job)),
                        "checkpoint_sha256": model_ops.sha256(checkpoint),
                    }
                )
            ensemble = model_dir / "ensemble-manifest.json"
            ensemble.write_text(
                json.dumps(
                    {
                        "target": "synthetic",
                        "stage": "initial",
                        "standardizer_sha256": standardizer_hash,
                        "members": members,
                    }
                )
            )
            source = job / "prepared.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["molecule_id", "smiles", "original_smiles", "source_kind", "source_batch", "source_row"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "molecule_id": "mol-1",
                        "smiles": "CCO",
                        "original_smiles": "CCO",
                        "source_kind": "external",
                        "source_batch": 1,
                        "source_row": 1,
                    }
                )
                writer.writerow(
                    {
                        "molecule_id": "mol-2",
                        "smiles": "c1ccccc1",
                        "original_smiles": "c1ccccc1",
                        "source_kind": "external",
                        "source_batch": 1,
                        "source_row": 2,
                    }
                )
            output = job / "scores.csv"
            args = argparse.Namespace(
                job_dir=job,
                assets_dir=PROJECT,
                gmolai_dir=PROJECT / "gMolAI-v2.0",
                gmolai_models_dir=PROJECT / "gMolAI-v2.0/inference/models",
                model_manifest=ensemble,
                input=source,
                output=output,
                save_policy="all",
                score_threshold=None,
                keep_embeddings=False,
                device="auto",
                encoder_backend="optimized",
                encoder_batch_size=16,
                encoder_node_budget=512,
                encoder_workers="1",
                encoder_verify_rows=16,
                encoder_threads=1,
            )
            model_ops.run_score(args)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["molecule_id"] for row in rows], ["mol-1", "mol-2"])
            self.assertIn("ensemble_probability", rows[0])
            self.assertFalse(output.with_suffix(".embeddings.npz").exists())
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertFalse(manifest["embeddings_retained"])
            self.assertEqual(manifest["encoded_rows"], 2)

            saved_output = job / "scores-saved.csv"
            args.output = saved_output
            args.keep_embeddings = True
            model_ops.run_score(args)
            self.assertTrue(saved_output.with_suffix(".embeddings.npz").is_file())
            self.assertEqual(saved_output.read_bytes(), output.read_bytes())
            saved_manifest = json.loads(saved_output.with_suffix(".manifest.json").read_text())
            self.assertTrue(saved_manifest["embeddings_retained"])

            probabilities = [float(row["ensemble_probability"]) for row in rows]
            threshold = sum(probabilities) / len(probabilities)
            threshold_output = job / "scores-threshold.csv"
            args.output = threshold_output
            args.keep_embeddings = False
            args.save_policy = "threshold"
            args.score_threshold = threshold
            model_ops.run_score(args)
            with threshold_output.open(newline="") as handle:
                retained = list(csv.DictReader(handle))
            self.assertEqual(
                len(retained),
                sum(probability >= threshold for probability in probabilities),
            )
            self.assertTrue(
                all(float(row["ensemble_probability"]) >= threshold for row in retained)
            )


if __name__ == "__main__":
    unittest.main()

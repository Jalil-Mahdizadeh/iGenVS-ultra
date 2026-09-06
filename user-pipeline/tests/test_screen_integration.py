from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from igenvs_ultra.workflow import screen, sha256


PROJECT = Path(__file__).resolve().parents[2]


def prepare_released_model_job(job: Path) -> None:
    released = json.loads(
        (PROJECT / "phase-8-benchmark-models/models/ensemble-manifest.json").read_text()
    )
    (job / "models/initial").mkdir(parents=True)
    (job / "fit-config.json").write_text(json.dumps({"target_name": "1err"}))
    members = []
    for member in released["targets"]["1err"]["members"]:
        checkpoint = PROJECT / "phase-8-benchmark-models" / member["checkpoint"]
        members.append({**member, "checkpoint": str(checkpoint)})
    manifest = job / "models/initial/ensemble-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target": "1err",
                "stage": "initial",
                "standardizer_sha256": released["standardizer_sha256"],
                "members": members,
            }
        )
    )
    (job / "models/final.json").write_text(
        json.dumps({"stage": "initial", "ensemble_manifest": str(manifest.relative_to(job))})
    )


@unittest.skipUnless(os.environ.get("IGENVS_ULTRA_INTEGRATION") == "1", "set IGENVS_ULTRA_INTEGRATION=1")
class EndToEndScreenTests(unittest.TestCase):
    def test_external_library_default_drops_embeddings(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(PROJECT)) as temporary:
            job = Path(temporary)
            prepare_released_model_job(job)
            source = job / "tiny.smi"
            source.write_text("CCO mol-1\nCCO duplicate\nc1ccccc1 mol-2\ninvalid mol-bad\n")
            args = argparse.Namespace(
                assets_dir=PROJECT,
                job_dir=job,
                execution="auto",
                igenvs_image=None,
                gmolai_image=None,
                gpu_ids=None,
                input=source,
                generate_count=None,
                input_format="smi",
                smiles_column="smiles",
                id_column=None,
                delimiter="auto",
                model="base-isomeric",
                generation_mode="de-novo",
                seed_file=None,
                seed_smiles=[],
                samples_per_seed=1,
                generator_batch_size="auto",
                generator_max_batch_size=32768,
                model_dir=None,
                temperature=None,
                top_k=None,
                greedy=False,
                compile_generator=False,
                compile_mode="reduce-overhead",
                generator_seed=13,
                include_seed_molecules=False,
                max_candidates=None,
                max_candidate_multiplier=None,
                stagnation_limit=None,
                generator_device="auto",
                generator_dtype="auto",
                generator_metrics=False,
                screen_name="smoke",
                stream_batch_size=10,
                max_stream_batches=None,
                exclude_reference_libraries=False,
                save_policy="all",
                score_threshold=None,
                keep_embeddings=False,
                encoder_backend="optimized",
                encoder_batch_size=16,
                encoder_node_budget=512,
                encoder_workers="1",
                encoder_verify_rows=16,
                encoder_threads=1,
                encoder_device="auto",
                validation_workers=1,
                fragment_policy="reject",
                dry_run=False,
            )
            result = screen(args)
            self.assertEqual(result["encoded_rows"], 2)
            self.assertEqual(result["saved_rows"], 2)
            self.assertFalse(list((job / "screens/smoke").glob("**/*.embeddings.npz")))
            self.assertEqual(sha256(Path(result["results"])), result["results_sha256"])

    def test_generated_library_streams_over_multiple_batches(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(PROJECT)) as temporary:
            job = Path(temporary)
            prepare_released_model_job(job)
            args = argparse.Namespace(
                assets_dir=PROJECT,
                job_dir=job,
                execution="auto",
                igenvs_image=None,
                gmolai_image=None,
                gpu_ids=None,
                input=None,
                generate_count=8,
                input_format="auto",
                smiles_column="smiles",
                id_column=None,
                delimiter="auto",
                model="base-isomeric",
                generation_mode="de-novo",
                seed_file=None,
                seed_smiles=[],
                samples_per_seed=1,
                generator_batch_size=8,
                generator_max_batch_size=8,
                model_dir=None,
                temperature=None,
                top_k=None,
                greedy=False,
                compile_generator=False,
                compile_mode="reduce-overhead",
                generator_seed=13,
                include_seed_molecules=False,
                max_candidates=None,
                max_candidate_multiplier=None,
                stagnation_limit=None,
                generator_device="auto",
                generator_dtype="auto",
                generator_metrics=False,
                screen_name="generated-smoke",
                stream_batch_size=4,
                max_stream_batches=8,
                exclude_reference_libraries=False,
                save_policy="all",
                score_threshold=None,
                keep_embeddings=False,
                encoder_backend="optimized",
                encoder_batch_size=16,
                encoder_node_budget=512,
                encoder_workers="1",
                encoder_verify_rows=16,
                encoder_threads=1,
                encoder_device="auto",
                screen_gpus=1,
                generation_logical_shards=2,
                validation_workers=1,
                fragment_policy="reject",
                dry_run=False,
            )
            result = screen(args)
            self.assertEqual(result["admitted_unique_rows"], 8)
            self.assertEqual(result["encoded_rows"], 8)
            self.assertGreaterEqual(result["stream_batches"], 2)
            with Path(result["results"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len({row["molecule_id"] for row in rows}), 8)
            self.assertFalse(list((job / "screens/generated-smoke").glob("**/*.embeddings.npz")))


if __name__ == "__main__":
    unittest.main()

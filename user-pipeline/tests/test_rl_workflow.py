from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from igenvs_ultra.cli import build_parser, validate_args
from igenvs_ultra.rl_workflow import (
    FROZEN_PROTOCOL_SHA256,
    _gate_check,
    _publish_model,
    _verify_stage_config,
    frozen_rl_bundle,
    generate_rl,
    train_rl,
)
from igenvs_ultra.workflow import PipelineError, sha256


PROJECT = Path(__file__).resolve().parents[2]


class RlCliTests(unittest.TestCase):
    def test_rl_commands_are_explicit_and_keep_protocol_fixed(self) -> None:
        parser = build_parser()
        train = parser.parse_args(
            [
                "rl-train",
                "--complex",
                str(PROJECT / "complexes/4ag8.pdb"),
                "--output-dir",
                "/tmp/rl-job",
                "--execution",
                "apptainer",
                "--gpu-ids",
                "0,1,2,3",
            ]
        )
        validate_args(parser, train)
        self.assertEqual(train.padding, 5.0)
        self.assertEqual(train.gpu_ids, "0,1,2,3")
        self.assertFalse(hasattr(train, "learning_rate"))
        self.assertFalse(hasattr(train, "reward_mode"))

        generate = parser.parse_args(
            [
                "rl-generate",
                "--model-dir",
                "/tmp/rl-job/model",
                "--generate-count",
                "25",
                "--output",
                "/tmp/rl.csv",
            ]
        )
        self.assertEqual(generate.count, 25)
        self.assertEqual(generate.seed, 13)

    def test_rl_train_dry_run_is_mutation_free(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "new-job"
            args = parser.parse_args(
                [
                    "rl-train",
                    "--assets-dir",
                    str(PROJECT),
                    "--complex",
                    str(PROJECT / "complexes/4ag8.pdb"),
                    "--output-dir",
                    str(job),
                    "--execution",
                    "native",
                    "--dry-run",
                ]
            )
            validate_args(parser, args)

            class FakeRuntime:
                execution = "native"

                def __init__(self, *arguments, **keywords):
                    pass

            with (
                mock.patch("igenvs_ultra.rl_workflow.Runtime", FakeRuntime),
                mock.patch("igenvs_ultra.rl_workflow.visible_gpu_tokens", return_value=["0"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                plan = train_rl(args)
            self.assertEqual(plan["frozen_protocol"]["sha256"], FROZEN_PROTOCOL_SHA256)
            self.assertEqual(len(plan["frozen_protocol"]["stages"]), 4)
            self.assertNotIn("independent_validation_raw_draws_per_arm", plan["frozen_protocol"])
            self.assertFalse(job.exists())

    def test_rl_train_accepts_receptor_and_bound_ligand(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receptor = root / "receptor.pdb"
            ligand = root / "bound.sdf"
            receptor.write_text("END\n", encoding="utf-8")
            ligand.write_text("bound\n$$$$\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "rl-train",
                    "--assets-dir",
                    str(PROJECT),
                    "--receptor",
                    str(receptor),
                    "--reference-ligand",
                    str(ligand),
                    "--output-dir",
                    str(root / "job"),
                    "--execution",
                    "native",
                    "--dry-run",
                ]
            )
            validate_args(parser, args)

            class FakeRuntime:
                execution = "native"

                def __init__(self, *arguments, **keywords):
                    pass

            with (
                mock.patch("igenvs_ultra.rl_workflow.Runtime", FakeRuntime),
                mock.patch("igenvs_ultra.rl_workflow.visible_gpu_tokens", return_value=["0"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                plan = train_rl(args)
            self.assertEqual(plan["target"]["target_input_mode"], "separate-files")
            self.assertEqual(plan["target"]["target_input"]["reference_ligand"], str(ligand))

    def test_rl_padding_cannot_override_frozen_value(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "rl-train",
                "--complex",
                str(PROJECT / "complexes/4ag8.pdb"),
                "--padding",
                "6",
                "--output-dir",
                "/tmp/rl-job",
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_args(parser, args)


class FrozenRlTests(unittest.TestCase):
    def test_accepted_bundle_hashes_verify(self) -> None:
        phase, protocol_path, protocol = frozen_rl_bundle(PROJECT)
        self.assertEqual(phase.name, "phase-10-rl-dev")
        self.assertEqual(sha256(protocol_path), FROZEN_PROTOCOL_SHA256)
        self.assertEqual(protocol["acceptance"]["required_development_passes"], 8)

    def test_changed_stage_setting_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text('{"learning_rate": 0.1}\n', encoding="utf-8")
            with self.assertRaises(PipelineError):
                _verify_stage_config(config, {"learning_rate": 0.00001})

    def test_adaptive_gate_requires_consecutive_complete_passes(self) -> None:
        rule = {
            "consecutive_evaluations": 2,
            "minimum_qualified_elite_fraction": 0.9,
            "minimum_qualified_elite_unique": 200,
            "minimum_chemistry_fraction": 0.9,
            "maximum_top_molecule_fraction": 0.2,
            "maximum_positive_score_fraction": 0.01,
        }
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            with (stage / "evaluations.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "label",
                        "qualified_elite_fraction",
                        "qualified_elite_unique_count",
                        "chemistry_fraction",
                        "top_molecule_fraction",
                        "positive_score_fraction",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "label": "update-9",
                        "qualified_elite_fraction": 0.91,
                        "qualified_elite_unique_count": 250,
                        "chemistry_fraction": 0.95,
                        "top_molecule_fraction": 0.1,
                        "positive_score_fraction": 0.001,
                    }
                )
                writer.writerow(
                    {
                        "label": "update-10",
                        "qualified_elite_fraction": 0.92,
                        "qualified_elite_unique_count": 251,
                        "chemistry_fraction": 0.96,
                        "top_molecule_fraction": 0.11,
                        "positive_score_fraction": 0.002,
                    }
                )
            self.assertTrue(_gate_check(stage, rule, 10)["passed"])
            self.assertFalse(_gate_check(stage, rule, 9)["passed"])


class RlOutputTests(unittest.TestCase):
    def _model(self, root: Path) -> Path:
        model = root / "model"
        artifacts = model / "base_isomeric"
        artifacts.mkdir(parents=True)
        weights = artifacts / "iGen3_base_isomeric_256d.pth"
        vocab = artifacts / "vocab.pkl"
        weights.write_bytes(b"test-weights")
        vocab.write_bytes(b"test-vocabulary")
        (model / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": "base-isomeric",
                    "target": "unit-target",
                    "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                    "artifacts": {
                        "base_isomeric/iGen3_base_isomeric_256d.pth": sha256(weights),
                        "base_isomeric/vocab.pkl": sha256(vocab),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return model

    def test_generate_commits_exact_csv_from_existing_igen3_cli(self) -> None:
        calls: list[list[str]] = []

        class FakeRuntime:
            execution = "fake"

            def __init__(self, *args, **kwargs):
                pass

            def run_logged(self, tool, command, log_path, **kwargs):
                calls.append(list(command))
                output = Path(command[command.index("--output") + 1])
                output.write_text("CC\nCCC\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self._model(root)
            output = root / "candidates.csv"
            args = argparse.Namespace(
                assets_dir=PROJECT,
                model_dir=model,
                output=output,
                count=2,
                seed=7,
                dry_run=False,
                execution="native",
                gpu_ids=None,
                igenvs_image=None,
                igenvs_docker_image=None,
            )
            with mock.patch("igenvs_ultra.rl_workflow.Runtime", FakeRuntime):
                result = generate_rl(args)

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(result["rows"], 2)
            self.assertEqual([row["smiles"] for row in rows], ["CC", "CCC"])
            self.assertEqual([row["molecule_id"] for row in rows], ["rl_00000001", "rl_00000002"])
            self.assertEqual(calls[0][0], "igen3")
            self.assertEqual(calls[0][calls[0].index("--count") + 1], "2")
            self.assertFalse(output.with_suffix(".smi.partial").exists())
            self.assertTrue(output.with_suffix(".manifest.json").is_file())

    def test_publish_model_records_selected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "job"
            source = job / "training/final/model/base_isomeric"
            checkpoint = job / "training/final/checkpoints/best.pt"
            source.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            (source / "iGen3_base_isomeric_256d.pth").write_bytes(b"weights")
            (source / "vocab.pkl").write_bytes(b"vocab")
            checkpoint.write_bytes(b"checkpoint")
            destination = _publish_model(
                job,
                job / "training/final",
                "unit-target",
            )
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["selected_checkpoint"]["selection"],
                "best checkpoint selected by the frozen online evaluator",
            )
            self.assertTrue((destination / "base_isomeric/vocab.pkl").is_file())


if __name__ == "__main__":
    unittest.main()

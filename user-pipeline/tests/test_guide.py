"""Guided choices, preflight failures, and handoff/resume contracts."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from igenvs_ultra import cli, guide, workflow


PROJECT = Path(__file__).resolve().parents[2]
COMPLEX = PROJECT / "complexes/4ag8.pdb"
LIBRARY = PROJECT / "iGenVS/examples/library.csv"


def options(*argv):
    return cli.build_parser().parse_args(["guide", *map(str, argv)])


def command(pipeline, job, assets=PROJECT):
    argv = [pipeline, "--complex", str(COMPLEX), "--ligand-id", "A:AXI:2000",
            "--output-dir", str(job), "--execution", "docker", "--assets-dir", str(assets),
            "--gpu-ids", "0", "--igenvs-docker-image", "igenvs-ultra/igenvs:latest"]
    if pipeline != "rl-train":
        argv += ["--input", str(LIBRARY)]
    if pipeline == "run":
        argv += ["--gmolai-docker-image", "igenvs-ultra/gmolai:latest"]
    return argv


class GuideTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.job = self.root / "job with spaces"
        self.output = io.StringIO()
        for patch in (
            contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output),
            mock.patch.object(guide.platform, "system", return_value="Linux"),
            mock.patch.object(guide.platform, "machine", return_value="x86_64"),
            mock.patch.object(sys.stdin, "isatty", return_value=True),
        ):
            patch.__enter__()
            self.addCleanup(patch.__exit__, None, None, None)

    def saved_job(self, pipeline="dock"):
        argv = command(pipeline, self.job)
        args = guide.parsed_arguments(argv)
        workflow.atomic_json(guide.record_path(self.job), {
            "schema_version": 1, "argv": argv, "arguments": guide.resolved_arguments(args),
        })
        return argv, args

    def guided_preview(self, pipeline, answers):
        with mock.patch("builtins.input", side_effect=answers), mock.patch.object(
            workflow, "visible_gpu_tokens", return_value=["0"],
        ), mock.patch.object(guide, "check_runtime", return_value={
            "gpu_ids": ["0"], "gpu_name": "test GPU", "free_gib": 20,
        }), mock.patch.object(guide, "missing_assets", return_value=[]), mock.patch.object(cli, "main") as execute:
            result = guide.run_guide(options(pipeline, "--assets-dir", PROJECT, "--dry-run"))
        self.assertEqual(result, 0, self.output.getvalue())
        execute.assert_not_called()
        self.assertFalse(self.job.exists())
        self.assertFalse(guide.record_path(self.job).exists())

    def test_each_workflow_previews_with_real_cli_planner(self):
        cases = {
            "dock": [str(self.job), "complex", str(COMPLEX), "generate", "3", "", "unidock", "balance", "none"],
            "run": [str(self.job), "complex", str(COMPLEX), "2", "generate", "100", "", "0.7"],
            "rl-train": [str(self.job), "complex", str(COMPLEX)],
        }
        for pipeline, answers in cases.items():
            with self.subTest(pipeline=pipeline):
                self.guided_preview(pipeline, answers)
        output = self.output.getvalue()
        self.assertIn("360,000 physical dockings", output)
        self.assertIn("scores >= 0.7", output)
        self.assertIn("Frozen RL protocol:", output)

    def test_dock_presets_and_autodock_are_passed_unchanged(self):
        for preset in ("fast", "balance", "detail"):
            with self.subTest(preset=preset), mock.patch("builtins.input", side_effect=[
                "complex", str(COMPLEX), "generate", "1,000", "", "autodock-gpu", preset, "individual",
            ]):
                args = guide.parsed_arguments(["dock", *guide.pipeline_arguments("dock"), "--output-dir", str(self.job)])
            self.assertEqual((args.engine, args.search_mode, args.pose_output, args.num_modes),
                             ("autodock-gpu", preset, "individual", 1))
            self.assertEqual(args.generate_count, 1000)

    def test_generation_offers_all_four_models_and_keeps_pipeline_defaults(self):
        for model in ("base-isomeric", "base-nonisomeric", "rl-isomeric", "rl-nonisomeric"):
            with self.subTest(model=model), mock.patch("builtins.input", side_effect=["generate", "25", model]):
                source = guide.source_arguments()
            args = guide.parsed_arguments(["dock", "--complex", str(COMPLEX), *source, "--output-dir", str(self.job)])
            self.assertEqual(args.model, model)
            self.assertEqual(args.generate_count, 25)
        for default in ("rl-nonisomeric", "base-isomeric"):
            with self.subTest(default=default), mock.patch("builtins.input", side_effect=["generate", "3", ""]):
                self.assertEqual(guide.source_arguments(default_model=default)[-2:], ["--model", default])

    def test_run_keeps_frozen_fit_defaults_and_validates_rounds_and_threshold(self):
        with mock.patch("builtins.input", side_effect=[
            "complex", str(COMPLEX), "6", "-1", "1", "generate", "0", "ten", "10", "", "nan", "1.1", "0.6",
        ]):
            args = guide.parsed_arguments(["run", *guide.pipeline_arguments("run"), "--output-dir", str(self.job)])
        self.assertEqual(args.al_rounds, 1)
        self.assertEqual(args.score_threshold, 0.6)
        self.assertEqual(args.generate_count, 10)
        self.assertTrue(workflow.release_equivalent_docking(workflow.docking_config(args)))

    def test_receptor_and_prepared_target_inputs(self):
        ligand = self.root / "bound ligand.sdf"
        ligand.touch()
        with mock.patch("builtins.input", side_effect=["receptor", str(COMPLEX), str(ligand)]):
            self.assertEqual(guide.target_arguments(), ["--receptor", str(COMPLEX), "--reference-ligand", str(ligand)])
        with mock.patch("builtins.input", side_effect=["prepared-target", str(self.root)]):
            self.assertEqual(guide.target_arguments(), ["--prepared-target", str(self.root)])

    def test_ambiguous_ligand_asks_for_selector(self):
        with mock.patch.object(workflow, "detect_ligand_id", side_effect=workflow.PipelineError("candidate residues: A:LIG:1, B:LIG:2")), \
                mock.patch("builtins.input", side_effect=["complex", str(COMPLEX), "B:LIG:2"]):
            args = guide.target_arguments()
        self.assertEqual(args[-2:], ["--ligand-id", "B:LIG:2"])
        self.assertIn("candidate residues", self.output.getvalue())

    def test_csv_columns_delimiters_bom_and_paths_with_shell_characters(self):
        for delimiter in (",", "\t", ";"):
            with self.subTest(delimiter=delimiter):
                path = self.root / "molecules 'quoted' $(literal).csv"
                path.write_text("\ufeff" + delimiter.join(["CatalogID", "Structure"]) + "\n"
                                + delimiter.join(["one", "CCO"]) + "\n", encoding="utf-8")
                with mock.patch("builtins.input", side_effect=["file", f'"{path}"', "wrong", "Structure", "CatalogID"]):
                    args = guide.source_arguments()
                self.assertEqual(args[1], str(path))
                self.assertEqual(args[args.index("--delimiter") + 1], delimiter)
                self.assertEqual(args[args.index("--smiles-column") + 1], "Structure")
                self.assertEqual(shlex.split(shlex.join(args)), args)

    def test_empty_or_duplicate_headers_are_rejected(self):
        for header in ("", "smiles,,id\n", "smiles,SMILES\n"):
            path = self.root / "bad.csv"
            path.write_text(header)
            with self.subTest(header=header), self.assertRaises(ValueError):
                guide.csv_arguments(path)

    def test_missing_input_retries_and_smi_has_no_column_questions(self):
        path = self.root / "input.smi"
        path.write_text("CCO ethanol\n")
        with mock.patch("builtins.input", side_effect=["file", str(self.root / "missing"), str(path)]) as prompts:
            self.assertEqual(guide.source_arguments(), ["--input", str(path), "--input-format", "smi"])
        self.assertEqual(prompts.call_count, 3)

    def test_fit_assets_need_only_requested_al_embeddings(self):
        for rounds in (0, 1, 5):
            required = workflow.required_fit_assets(self.root, rounds)
            self.assertEqual(len(required), 5 + rounds)
            self.assertFalse(any("AL-set" in str(path) and path.suffix == ".csv" for path in required))
        for path in workflow.required_fit_assets(self.root, 1):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        missing = guide.missing_assets(self.root, "run", 2)
        self.assertIn(self.root / "phase-2-al-sets/embeddings/AL-set-2-embeddings.npz", missing)
        self.assertFalse(any("UDRL" in str(path) for path in missing))
        for pipeline in ("dock", "rl-train"):
            self.assertFalse(any("UDRL" in str(path) or "AL-set" in str(path) for path in guide.missing_assets(self.root, pipeline)))

    def test_assets_prompt_only_when_required_files_are_missing(self):
        with mock.patch.object(guide, "missing_assets", return_value=[]), mock.patch("builtins.input") as prompt:
            self.assertEqual(guide.select_assets(options("--assets-dir", self.root), "dock", 0), self.root)
        prompt.assert_not_called()
        with mock.patch.object(guide, "missing_assets", side_effect=[[self.root / "missing"], []]), \
                mock.patch("builtins.input", return_value=str(PROJECT)):
            self.assertEqual(guide.select_assets(options("--assets-dir", self.root), "run", 1), PROJECT)

    def test_fit_and_doctor_accept_al_bundles_without_original_al_csvs(self):
        assets = self.root / "assets"
        for path in workflow.required_fit_assets(assets, 5) + [assets / name for name in (
            "gMolAI-v2.0/inference/gmolai.py", "gMolAI-v2.0/inference/models/SHA256SUMS",
            "user-pipeline/src/igenvs_ultra/model_ops.py", "user-pipeline/src/igenvs_ultra/generation_worker.py",
        )]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        args = guide.parsed_arguments([*command("run", self.job, assets), "--al-rounds", "2"])

        def publish(_runtime, job, name, _operation):
            if name.startswith("train-"):
                stage = "initial" if name == "train-initial" else name.removeprefix("train-")
                workflow.atomic_json(job / "models/final.json", {"stage": stage, "ensemble_manifest": "models/ensemble.json"})

        runtime = mock.Mock(execution="docker")
        runtime.capture.return_value = subprocess.CompletedProcess([], 0, '{"ok": true}', "")
        with mock.patch.object(workflow, "Runtime", return_value=runtime), \
                mock.patch.object(workflow, "prepare_target"), mock.patch.object(workflow, "merge_docking"), \
                mock.patch.object(workflow, "run_docking", return_value=([], {})) as dock, \
                mock.patch.object(workflow, "model_operation", side_effect=publish):
            summary = workflow.fit(args)
        self.assertEqual(summary["al_rounds_completed"], 2)
        self.assertEqual([call.args[4] for call in dock.call_args_list], [
            assets / "phase-1-udrl/library/UDRL-train.csv", assets / "phase-1-udrl/library/UDRL-valid.csv",
            self.job / "al/round-1/acquisition/selected.csv", self.job / "al/round-2/acquisition/selected.csv",
        ])
        doctor_args = cli.build_parser().parse_args(["doctor", "--assets-dir", str(assets), "--require-fit-assets"])
        with mock.patch.object(workflow, "Runtime", return_value=runtime), \
                mock.patch.object(workflow, "docker_image_exists", return_value=True), \
                mock.patch("igenvs_ultra.rl_workflow.frozen_rl_bundle"):
            report = workflow.doctor(doctor_args)
        self.assertTrue(report["ok"])
        self.assertTrue(report["fit_assets"]["ready"])
        self.assertFalse((assets / "phase-2-al-sets/library").exists())

    def test_occupied_output_requires_a_new_folder(self):
        occupied = self.root / "occupied"
        occupied.mkdir()
        keep = occupied / "keep.txt"
        keep.write_text("keep")
        with mock.patch("builtins.input", side_effect=[str(occupied), str(self.job)]):
            self.assertEqual(guide.select_job("dock"), (self.job, False))
        self.assertEqual(keep.read_text(), "keep")

    def test_selecting_previous_output_offers_resume(self):
        self.saved_job()
        with mock.patch("builtins.input", side_effect=[str(self.job), ""]):
            self.assertEqual(guide.select_job("dock"), (self.job, True))

    def test_confirmation_decline_and_eof_do_not_create_records(self):
        for answer, expected in (("no", 0), (EOFError(), 130)):
            with self.subTest(answer=answer), mock.patch.object(guide, "select_job", return_value=(self.job, False)), \
                    mock.patch.object(guide, "pipeline_arguments", return_value=["--complex", str(COMPLEX), "--generate-count", "3"]), \
                    mock.patch.object(guide, "select_assets", return_value=PROJECT), \
                    mock.patch.object(guide, "missing_assets", return_value=[]), \
                    mock.patch.object(workflow, "visible_gpu_tokens", return_value=["0"]), \
                    mock.patch.object(guide, "check_runtime", return_value={"gpu_ids": ["0"], "gpu_name": "test", "free_gib": 10}), \
                    mock.patch("builtins.input", side_effect=[answer]), mock.patch.object(cli, "main") as execute:
                self.assertEqual(guide.run_guide(options("dock")), expected, self.output.getvalue())
            execute.assert_not_called()
            self.assertFalse(self.job.exists())
            self.assertFalse(guide.record_path(self.job).exists())

    def test_record_is_saved_beside_new_job_before_handoff_and_reused_on_resume(self):
        def execute(argv):
            self.assertTrue(guide.record_path(self.job).is_file())
            args = guide.parsed_arguments(argv)
            # Exercise the real initializer's non-empty directory and config checks.
            workflow.ensure_regular_config(self.job, workflow.make_regular_config(args, PROJECT))
            return 2

        with mock.patch.object(guide, "select_job", return_value=(self.job, False)), \
                mock.patch.object(guide, "pipeline_arguments", return_value=["--complex", str(COMPLEX), "--generate-count", "3"]), \
                mock.patch.object(guide, "select_assets", return_value=PROJECT), \
                mock.patch.object(guide, "missing_assets", return_value=[]), \
                mock.patch.object(workflow, "visible_gpu_tokens", return_value=["0"]), \
                mock.patch.object(guide, "check_runtime", return_value={"gpu_ids": ["0"], "gpu_name": "test", "free_gib": 10}), \
                mock.patch("builtins.input", return_value="yes"), mock.patch.object(cli, "main", side_effect=execute) as run:
            self.assertEqual(guide.run_guide(options("dock")), 2)
            original_argv = run.call_args.args[0]
            before = guide.record_path(self.job).read_bytes()
            self.assertEqual(guide.run_guide(options("--resume", self.job, "--yes")), 2)
            self.assertEqual(run.call_args.args[0], original_argv)
            self.assertEqual(guide.record_path(self.job).read_bytes(), before)
        self.assertIn("Resume with:", self.output.getvalue())

    def test_resume_detects_changed_defaults_and_mismatched_job(self):
        for change in ("defaults", "path", "schema", "command"):
            self.saved_job()
            path = guide.record_path(self.job)
            saved = json.loads(path.read_text())
            if change == "defaults":
                saved["arguments"]["search_mode"] = "detail"
            elif change == "path":
                saved["argv"][saved["argv"].index("--output-dir") + 1] = str(self.root / "other")
            elif change == "schema":
                saved["schema_version"] = 99
            else:
                saved["argv"][0] = "doctor"
            workflow.atomic_json(path, saved)
            with self.subTest(change=change), self.assertRaises(workflow.PipelineError):
                guide.load_record(self.job)

    def test_noninteractive_usage_and_resume_preview(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            self.assertEqual(guide.run_guide(options("dock")), 2)
            self.saved_job()
            record = guide.record_path(self.job)
            before = record.stat().st_mtime_ns
            with mock.patch.object(guide, "check_runtime", return_value={"gpu_ids": ["0"], "gpu_name": "test", "free_gib": 10}), \
                    mock.patch.object(cli, "main") as execute:
                self.assertEqual(guide.run_guide(options("--resume", self.job, "--dry-run")), 0)
            execute.assert_not_called()
            self.assertFalse(self.job.exists())
            self.assertEqual(record.stat().st_mtime_ns, before)

    def test_wrong_platform_and_invalid_resume_options_stop_before_prompts(self):
        with mock.patch.object(guide.platform, "system", return_value="Windows"), mock.patch("builtins.input") as prompt:
            self.assertEqual(guide.run_guide(options()), 2)
        prompt.assert_not_called()
        self.assertEqual(guide.run_guide(options("dock", "--yes")), 2)
        self.assertEqual(guide.run_guide(options("--resume", self.job, "--gpu-ids", "1")), 2)

    def test_disk_check_uses_existing_parent_without_creating_job(self):
        self.assertGreater(guide.disk_space(self.job / "nested"), 0)
        self.assertFalse(self.job.exists())
        with mock.patch.object(guide.shutil, "disk_usage", return_value=mock.Mock(free=0)), self.assertRaises(workflow.PipelineError):
            guide.disk_space(self.job)

    def test_preflight_is_pipeline_specific_and_requires_success(self):
        for pipeline in ("dock", "run", "rl-train"):
            args = guide.parsed_arguments(command(pipeline, self.job))
            doctor = {"ok": True, "torch_cuda": {"available": True, "device_name": "test"}}
            output = {"ok": True, "igenvs": doctor} if pipeline == "rl-train" else doctor
            runtime = mock.Mock()
            runtime.capture.return_value = subprocess.CompletedProcess([], 0, json.dumps(output), "")
            with self.subTest(pipeline=pipeline), mock.patch.object(guide.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "linux\n", "")), \
                    mock.patch.object(workflow, "docker_image_exists", return_value=True) as images, \
                    mock.patch.object(workflow, "Runtime", return_value=runtime) as constructor:
                self.assertEqual(guide.check_runtime(args)["gpu_ids"], ["0"])
            self.assertEqual(images.call_count, 2 if pipeline == "run" else 1)
            self.assertEqual(runtime.capture.call_count, 3 if pipeline == "run" else 1)
            self.assertEqual(constructor.call_args.args[2], PROJECT / "user-pipeline")
            self.assertEqual(constructor.call_args.kwargs["require_gmolai"], pipeline == "run")
        runtime.capture.return_value = subprocess.CompletedProcess([], 1, "", "GPU unavailable")
        with self.assertRaisesRegex(workflow.PipelineError, "GPU unavailable"):
            guide.checked_capture(runtime, "igenvs", ["doctor"], "check")

    def test_docker_and_missing_image_failures_are_actionable(self):
        args = guide.parsed_arguments(command("dock", self.job))
        with mock.patch.object(guide.subprocess, "run", side_effect=FileNotFoundError()), \
                self.assertRaisesRegex(workflow.PipelineError, "docker info"):
            guide.check_runtime(args)
        with mock.patch.object(guide.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "linux\n", "")), \
                mock.patch.object(workflow, "docker_image_exists", return_value=False), \
                self.assertRaisesRegex(workflow.PipelineError, "build-images.sh"):
            guide.check_runtime(args)

    def test_source_launcher_help_works_outside_repo(self):
        result = subprocess.run([sys.executable, str(PROJECT / "start"), "--help"],
                                cwd=self.root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dock,run,rl-train", result.stdout)
        self.assertIn("--resume", result.stdout)


@unittest.skipUnless(os.environ.get("IGENVS_ULTRA_GUIDE_INTEGRATION") == "1", "set IGENVS_ULTRA_GUIDE_INTEGRATION=1 with Docker/GPU")
class GuidedDockerIntegrationTests(unittest.TestCase):
    def test_guided_docking_and_resume_with_both_engines(self):
        for engine in ("unidock", "autodock-gpu"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory(prefix="guided-dock-") as temporary:
                job = Path(temporary) / "job with spaces"
                output = io.StringIO()
                answers = [str(job), "complex", str(COMPLEX), "file", str(LIBRARY), "", "", engine, "fast", "merged"]
                answers += ["2", "yes"] if engine == "unidock" else ["yes"]
                with mock.patch.object(sys.stdin, "isatty", return_value=True), mock.patch("builtins.input", side_effect=answers), \
                        contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    first = guide.run_guide(options("dock", "--assets-dir", PROJECT))
                    self.assertEqual(first, 0, output.getvalue())
                    results = job / "docking/results.csv"
                    stamp = results.stat().st_mtime_ns
                    sdf = job / "docking/poses.sdf"
                    self.assertTrue(sdf.is_file(), output.getvalue())
                    sdf_stamp = sdf.stat().st_mtime_ns
                    second = guide.run_guide(options("--resume", job, "--yes"))
                self.assertEqual(second, 0, output.getvalue())
                self.assertEqual(results.stat().st_mtime_ns, stamp)
                self.assertEqual(sdf.stat().st_mtime_ns, sdf_stamp)
                summary = json.loads((job / "regular-summary.json").read_text())
                self.assertEqual(summary["counts"]["docked"], 3)
                self.assertEqual(summary["poses_sdf"], str(sdf))
                with results.open(newline="") as handle:
                    pose_count = sum(int(row["num_poses"]) for row in csv.DictReader(handle) if row["status"] == "success")
                self.assertEqual(sdf.read_text().count("$$$$\n"), pose_count)
                saved = json.loads(guide.record_path(job).read_text())
                self.assertEqual(saved["arguments"]["engine"], engine)
                runtime = workflow.Runtime(guide.parsed_arguments(saved["argv"]), PROJECT, job, require_gmolai=False)
                check = runtime.capture("igenvs", ["python3", "-c", (
                    "import csv, sys; from rdkit import Chem\n"
                    "with open(sys.argv[2]) as handle: rows = {r['molecule_id']: r for r in csv.DictReader(handle)}\n"
                    "for mol in Chem.SDMolSupplier(sys.argv[1]):\n"
                    " assert mol is not None\n"
                    " assert Chem.MolToSmiles(mol) == rows[mol.GetProp('_Name')]['canonical_smiles']\n"
                ), str(sdf), str(results)], gpu=False)
                self.assertEqual(check.returncode, 0, check.stderr)

    def test_run_and_rl_docker_preflight_and_real_plans(self):
        for pipeline in ("run", "rl-train"):
            with self.subTest(pipeline=pipeline), tempfile.TemporaryDirectory() as temporary:
                job = Path(temporary) / "preview"
                args = guide.parsed_arguments(command(pipeline, job))
                with contextlib.redirect_stdout(io.StringIO()):
                    plan = guide.dry_plan(args)
                    ready = guide.check_runtime(args)
                self.assertEqual(plan["command"], pipeline)
                self.assertEqual(plan["runtime"], "docker")
                self.assertTrue(ready["gpu_ids"])
                self.assertFalse(guide.missing_assets(PROJECT, pipeline))
                self.assertFalse(job.exists())
                self.assertFalse(guide.record_path(job).exists())


if __name__ == "__main__":
    unittest.main()

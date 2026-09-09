from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

from igenvs_ultra.cli import build_parser, discover_fast_job, validate_args
from igenvs_ultra.workflow import (
    _merge_regular_docking_shards,
    _regular_docking_shard_plan,
    _regular_results_cover_validated,
    admit_batch,
    aligned_score_shard_lengths,
    available_memory_bytes,
    build_regular_docking_command,
    choose_stream_batch_size,
    docking_gpu_ids,
    generate_batch,
    generation_logical_shard_count,
    generation_shard_plan,
    generated_rows_from_iGen3_contract,
    merge_docking,
    merge_score_shards,
    prepare_score_shards,
    open_dedup_database,
    partition_cpu_affinity,
    process_external_screen,
    process_generated_screen,
    regular_dock,
    replay_completed_admissions,
    Runtime,
    seed_reference_identities,
    screening_gpu_ids,
    sha256,
    taskset_cpu_list,
    visible_gpu_tokens,
)


PROJECT = Path(__file__).resolve().parents[2]


class CliTests(unittest.TestCase):
    def test_docker_runtime_arguments_are_public(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "screen",
                "--job-dir",
                "/tmp/job",
                "--input",
                "/tmp/library.smi",
                "--execution",
                "docker",
                "--igenvs-docker-image",
                "example/igenvs:v1",
                "--gmolai-docker-image",
                "example/gmolai:v1",
            ]
        )
        self.assertEqual(args.execution, "docker")
        self.assertEqual(args.igenvs_docker_image, "example/igenvs:v1")
        self.assertEqual(args.gmolai_docker_image, "example/gmolai:v1")

    def test_threshold_is_shorthand_for_threshold_policy(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "screen",
                "--job-dir",
                "/tmp/job",
                "--input",
                "/tmp/library.smi",
                "--score-threshold",
                "0.75",
            ]
        )
        validate_args(parser, args)
        self.assertEqual(args.save_policy, "threshold")
        self.assertEqual(args.score_threshold, 0.75)
        self.assertFalse(args.keep_embeddings)

    def test_save_embeddings_is_opt_in(self) -> None:
        parser = build_parser()
        base = ["screen", "--job-dir", "/tmp/job", "--input", "/tmp/library.smi"]
        default = parser.parse_args(base)
        enabled = parser.parse_args([*base, "--save-embeddings"])
        alias = parser.parse_args([*base, "--keep-embeddings"])
        self.assertFalse(default.keep_embeddings)
        self.assertTrue(enabled.keep_embeddings)
        self.assertTrue(alias.keep_embeddings)

    def test_screen_uses_visible_gpus_by_default(self) -> None:
        parser = build_parser()
        base = ["screen", "--job-dir", "/tmp/job", "--input", "/tmp/library.smi"]
        automatic = parser.parse_args(base)
        limited = parser.parse_args([*base, "--screen-gpus", "2"])
        self.assertEqual(automatic.screen_gpus, "auto")
        self.assertEqual(automatic.encoder_batch_size, "auto")
        self.assertEqual(limited.screen_gpus, 2)

    def test_count_only_screen_discovers_completed_job(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(["screen-fast", "12345"])
        self.assertEqual(parsed.molecules, 12345)
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "target-job"
            nested = job / "work/subdir"
            (job / "models").mkdir(parents=True)
            nested.mkdir(parents=True)
            (job / "fit-config.json").write_text("{}\n", encoding="utf-8")
            (job / "models/final.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(discover_fast_job(nested), job)

    def test_screen_auto_respects_slurm_gpu_task_limit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "screen", "--job-dir", "/tmp/job", "--input", "/tmp/library.smi",
                "--gpu-ids", "0,1,2,3",
            ]
        )
        with mock.patch.dict("os.environ", {"SLURM_GPUS_PER_TASK": "2"}, clear=False):
            self.assertEqual(screening_gpu_ids(args), ["0", "1"])

    def test_regular_and_ultra_are_explicitly_separate(self) -> None:
        parser = build_parser()
        regular = parser.parse_args(
            [
                "dock", "--target", "/tmp/target", "--input", "/tmp/library.smi",
                "--output-dir", "/tmp/regular",
            ]
        )
        ultra = parser.parse_args(
            [
                "run", "--target", "/tmp/target", "--input", "/tmp/library.smi",
                "--output-dir", "/tmp/ultra",
            ]
        )
        self.assertEqual(regular.search_mode, "balance")
        self.assertEqual(regular.pose_output, "merged")
        self.assertEqual(regular.model, "rl-nonisomeric")
        self.assertEqual(regular.docking_gpus, "auto")
        self.assertEqual(regular.embed_max_attempts, "auto")
        self.assertEqual(regular.embed_timeout, "auto")
        self.assertEqual(ultra.search_mode, "fast")
        self.assertEqual(ultra.pose_output, "none")
        self.assertEqual(ultra.model, "base-isomeric")
        self.assertEqual(ultra.docking_logical_shards, 4)

    def test_regular_command_is_original_igenvs_screen(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dock", "--target", "/tmp/target", "--input", "/tmp/library.csv",
                "--smiles-column", "SMILES", "--id-column", "CatalogID",
                "--output-dir", "/tmp/regular", "--pose-output", "none",
            ]
        )
        source = {
            "kind": "external", "path": "/tmp/library.csv", "format": "auto",
            "smiles_column": "SMILES", "id_column": "CatalogID", "delimiter": "auto",
        }
        command = build_regular_docking_command(
            args, source=source, target=Path("/tmp/regular/target"), output=Path("/tmp/regular/docking")
        )
        self.assertEqual(command[:2], ["igenvs", "screen"])
        self.assertIn("--input", command)
        self.assertIn("--target", command)
        self.assertIn("--pose-output", command)
        self.assertNotIn("--al-rounds", command)
        self.assertNotIn("--save-embeddings", command)

    def test_regular_prevalidated_command_shards_without_revalidating(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dock", "--target", "/tmp/target", "--input", "/tmp/library.csv",
                "--output-dir", "/tmp/regular", "--pose-output", "none",
            ]
        )
        command = build_regular_docking_command(
            args,
            source={"kind": "external"},
            target=Path("/tmp/regular/target"),
            output=Path("/tmp/regular/docking/runs/shard-2"),
            prevalidated_input=Path("/tmp/regular/library/validation/validated.csv"),
            num_shards=4,
            shard_index=2,
            device_id=0,
        )
        self.assertIn("--prevalidated-input", command)
        self.assertNotIn("--input", command)
        self.assertEqual(command[command.index("--num-shards") + 1], "4")
        self.assertEqual(command[command.index("--shard-index") + 1], "2")
        self.assertEqual(command[command.index("--device-id") + 1], "0")

    def test_regular_docking_auto_respects_slurm_gpu_task_limit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dock", "--target", "/tmp/target", "--input", "/tmp/library.smi",
                "--output-dir", "/tmp/regular", "--gpu-ids", "0,1,2,3",
            ]
        )
        with mock.patch.dict("os.environ", {"SLURM_GPUS_PER_TASK": "2"}, clear=False):
            self.assertEqual(docking_gpu_ids(args), ["0", "1"])

    def test_regular_dry_run_does_not_create_job(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            (target / "manifest.json").write_text("{}\n")
            source = root / "library.smi"
            source.write_text("CCO mol-1\n")
            job = root / "job"
            args = parser.parse_args(
                [
                    "dock", "--assets-dir", str(PROJECT), "--target", str(target),
                    "--input", str(source), "--output-dir", str(job), "--dry-run",
                ]
            )
            validate_args(parser, args)
            with contextlib.redirect_stdout(io.StringIO()):
                plan = regular_dock(args)
            self.assertEqual(plan["workflow"], "regular-iGenVS-docking")
            self.assertFalse(job.exists())


class RuntimeTests(unittest.TestCase):
    def test_docker_wrap_mounts_assets_and_preserves_worker_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary).resolve()
            job = assets / "runs/job"
            job.mkdir(parents=True)
            args = argparse.Namespace(
                execution="docker",
                gpu_ids=None,
                igenvs_image=None,
                gmolai_image=None,
                igenvs_docker_image="example/igenvs:v1",
                gmolai_docker_image="example/gmolai:v1",
            )
            with mock.patch("igenvs_ultra.workflow.shutil.which", return_value="/usr/bin/docker"):
                runtime = Runtime(args, assets, job)
                wrapped = runtime._wrap(
                    "gmolai",
                    ["python3", str(assets / "worker.py"), "--serve"],
                    gpu=True,
                    cpu_affinity=[2, 3],
                )
        self.assertEqual(wrapped[:5], ["docker", "run", "--rm", "--init", "-i"])
        self.assertIn("--gpus", wrapped)
        self.assertEqual(wrapped[wrapped.index("--cpuset-cpus") + 1], "2,3")
        self.assertIn(f"{assets}:{assets}:rw", wrapped)
        self.assertEqual(wrapped[wrapped.index("--entrypoint") + 1], "python3")
        image_index = wrapped.index("example/gmolai:v1")
        self.assertEqual(wrapped[image_index + 1 :], [str(assets / "worker.py"), "--serve"])


class DedupTests(unittest.TestCase):
    def test_cross_batch_exact_dedup_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            screen = Path(temporary)
            connection = open_dedup_database(screen)
            seed_reference_identities(connection, screen, screen, enabled=False)
            rows = [
                {"molecule_id": "a", "original_smiles": "CC", "canonical_smiles": "CC", "source_row": "1"},
                {"molecule_id": "b", "original_smiles": "CCC", "canonical_smiles": "CCC", "source_row": "2"},
                {"molecule_id": "c", "original_smiles": "CC", "canonical_smiles": "CC", "source_row": "3"},
            ]
            first = admit_batch(connection, screen, 1, rows, "external", 0)
            self.assertEqual(first["accepted_rows"], 2)
            self.assertEqual(first["duplicate_rows"], 1)
            self.assertEqual(admit_batch(connection, screen, 1, rows, "external", 0), first)
            second = admit_batch(
                connection,
                screen,
                2,
                [
                    {"molecule_id": "d", "original_smiles": "CC", "canonical_smiles": "CC", "source_row": "4"},
                    {"molecule_id": "e", "original_smiles": "CO", "canonical_smiles": "CO", "source_row": "5"},
                ],
                "external",
                2,
            )
            self.assertEqual(second["accepted_rows"], 1)
            self.assertEqual(second["duplicate_rows"], 1)
            connection.close()

    def test_admission_marker_repairs_a_precommit_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            screen = Path(temporary)
            connection = open_dedup_database(screen)
            rows = [
                {
                    "molecule_id": "a",
                    "original_smiles": "CC",
                    "canonical_smiles": "CC",
                    "source_row": "1",
                }
            ]
            from igenvs_ultra import workflow

            original_atomic_json = workflow.atomic_json

            def publish_then_interrupt(path, value):
                original_atomic_json(path, value)
                if path.name == "admission.json":
                    raise KeyboardInterrupt("injected before SQLite commit")

            with mock.patch(
                "igenvs_ultra.workflow.atomic_json", side_effect=publish_then_interrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    admit_batch(connection, screen, 1, rows, "external", 0)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM smiles").fetchone()[0], 0
            )
            recovered = admit_batch(connection, screen, 1, rows, "external", 0)
            self.assertEqual(recovered["accepted_rows"], 1)
            self.assertEqual(
                connection.execute("SELECT value, owner FROM smiles").fetchall(),
                [("CC", 1)],
            )
            connection.close()

    def test_replay_removes_legacy_commit_without_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            screen = Path(temporary)
            connection = open_dedup_database(screen)
            connection.execute(
                "INSERT INTO smiles(value, owner) VALUES ('CC', 1)"
            )
            connection.commit()
            replay_completed_admissions(connection, screen)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM smiles").fetchone()[0], 0
            )
            record = admit_batch(
                connection,
                screen,
                1,
                [
                    {
                        "molecule_id": "a",
                        "original_smiles": "CC",
                        "canonical_smiles": "CC",
                        "source_row": "1",
                    }
                ],
                "external",
                0,
            )
            self.assertEqual(record["accepted_rows"], 1)
            connection.close()


class DockingMergeTests(unittest.TestCase):
    def test_cpu_affinity_is_disjoint_and_complete(self) -> None:
        groups = partition_cpu_affinity([8, 2, 6, 4, 0, 7, 1, 5, 3], 4)
        self.assertEqual(groups, [[0, 1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertEqual(taskset_cpu_list(groups[0]), "0,1,2")
        self.assertEqual(sorted(value for group in groups for value in group), list(range(9)))
        self.assertFalse(set(groups[0]).intersection(groups[1]))

    def test_merge_restores_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            source = job / "source.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["molecule_id", "smiles"])
                writer.writeheader()
                for index in range(1, 5):
                    writer.writerow({"molecule_id": f"mol-{index}", "smiles": "C" * index})
            shard_dirs = []
            fields = ["molecule_id", "original_smiles", "canonical_smiles", "source_row", "status", "docking_score"]
            for shard in range(2):
                directory = job / f"docking/UDRL-train/runs/shard-{shard}"
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text(json.dumps({"status": "complete"}))
                with (directory / "results.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for source_row in range(shard + 1, 5, 2):
                        writer.writerow(
                            {
                                "molecule_id": f"mol-{source_row}",
                                "original_smiles": "C" * source_row,
                                "canonical_smiles": "C" * source_row,
                                "source_row": source_row,
                                "status": "success",
                                "docking_score": -float(source_row),
                            }
                        )
                shard_dirs.append(directory)
            output = merge_docking(job, "UDRL-train", source, shard_dirs, "target")
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["molecule_id"] for row in rows], ["mol-1", "mol-2", "mol-3", "mol-4"])
            self.assertEqual([int(row["shard_index"]) for row in rows], [0, 1, 0, 1])

    def test_regular_multi_gpu_merge_is_ordered_and_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            shard_dirs = []
            fields = ["molecule_id", "source_row", "status", "docking_score"]
            for shard in range(2):
                directory = job / f"docking/runs/shard-{shard}"
                directory.mkdir(parents=True)
                rows = list(range(shard + 1, 7, 2))
                manifest = {
                    "status": "complete",
                    "counts": {"prepared": len(rows), "docked": len(rows)},
                    "timings": {
                        "docking_wall_seconds": 10.0 + shard,
                        "preparation_wait_seconds": 2.0 + shard,
                    },
                }
                (directory / "manifest.json").write_text(json.dumps(manifest))
                with (directory / "results.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for source_row in rows:
                        writer.writerow(
                            {
                                "molecule_id": f"mol-{source_row}",
                                "source_row": source_row,
                                "status": "success",
                                "docking_score": -float(source_row),
                            }
                        )
                shard_dirs.append(directory)
            args = argparse.Namespace(
                engine="unidock", search_mode="fast", pose_output="none"
            )
            manifest = _merge_regular_docking_shards(
                args,
                job,
                shard_dirs,
                ["GPU-a", "GPU-b"],
                launcher_seconds=12.5,
                validation={"valid": 6},
            )
            with (job / "docking/results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["molecule_id"] for row in rows],
                [f"mol-{index}" for index in range(1, 7)],
            )
            self.assertEqual(manifest["counts"], {"docked": 6, "prepared": 6})
            self.assertEqual(manifest["gpu_ids"], ["GPU-a", "GPU-b"])
            self.assertEqual(manifest["timings"]["parallel_docking_wall_seconds"], 11.0)

    def test_regular_plan_pins_logical_shards_and_skips_empty_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            validated = job / "validated.csv"
            with validated.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_row"])
                writer.writeheader()
                for source_row in (1, 2, 3):
                    writer.writerow({"source_row": source_row})
            plan = _regular_docking_shard_plan(
                job, validated, {"valid_rows": 3}, initial_gpu_count=4
            )
            self.assertEqual(plan["logical_shards"], 4)
            self.assertEqual(
                [item["input_rows"] for item in plan["shards"]], [1, 1, 1, 0]
            )
            resumed = _regular_docking_shard_plan(
                job, validated, {"valid_rows": 3}, initial_gpu_count=2
            )
            self.assertEqual(resumed, plan)

    def test_regular_plan_recovers_legacy_four_way_resume_on_two_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            validated = job / "validated.csv"
            with validated.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_row"])
                writer.writeheader()
                for source_row in range(1, 9):
                    writer.writerow({"source_row": source_row})
            legacy = job / "docking/runs/shard-0"
            legacy.mkdir(parents=True)
            (legacy / "manifest.json").write_text(
                json.dumps({"config": {"num_shards": 4, "shard_index": 0}}),
                encoding="utf-8",
            )
            plan = _regular_docking_shard_plan(
                job, validated, {"valid_rows": 8}, initial_gpu_count=2
            )
            self.assertEqual(plan["logical_shards"], 4)
            self.assertEqual(
                [item["input_rows"] for item in plan["shards"]], [2, 2, 2, 2]
            )

    def test_regular_merge_rejects_missing_partition_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            fields = ["molecule_id", "source_row", "status", "docking_score"]
            shard_dirs = []
            for shard, source_rows in enumerate(((1, 5), (2, 6))):
                directory = job / f"docking/runs/shard-{shard}"
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text(
                    json.dumps({"status": "complete", "counts": {"docked": 2}}),
                    encoding="utf-8",
                )
                with (directory / "results.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for source_row in source_rows:
                        writer.writerow(
                            {
                                "molecule_id": f"mol-{source_row}",
                                "source_row": source_row,
                                "status": "success",
                                "docking_score": -source_row,
                            }
                        )
                shard_dirs.append(directory)
            args = argparse.Namespace(
                engine="unidock", search_mode="fast", pose_output="none"
            )
            with self.assertRaisesRegex(Exception, "expected 8"):
                _merge_regular_docking_shards(
                    args,
                    job,
                    shard_dirs,
                    ["0", "1"],
                    1.0,
                    {"valid": 8},
                    logical_shards=4,
                )

    def test_completed_regular_results_are_rechecked_for_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = root / "validated.csv"
            validated.write_text(
                "source_row,smiles\n1,CC\n2,CCC\n3,CCCC\n", encoding="utf-8"
            )
            results = root / "results.csv"
            results.write_text(
                "source_row,status\n1,success\n3,success\n", encoding="utf-8"
            )
            self.assertFalse(_regular_results_cover_validated(results, validated))
            results.write_text(
                "source_row,status\n1,success\n2,failed\n3,success\n",
                encoding="utf-8",
            )
            self.assertTrue(_regular_results_cover_validated(results, validated))


class HardwareVisibilityTests(unittest.TestCase):
    def test_explicit_disable_masks_never_fall_back_to_physical_inventory(self) -> None:
        for value in ("", "-1", "NoDevFiles", "void"):
            with self.subTest(value=value), mock.patch.dict(
                "os.environ", {"CUDA_VISIBLE_DEVICES": value}, clear=False
            ), mock.patch("igenvs_ultra.workflow.subprocess.run") as discovery:
                self.assertEqual(visible_gpu_tokens(), [])
                discovery.assert_not_called()
            with self.subTest(explicit=value), mock.patch(
                "igenvs_ultra.workflow.subprocess.run"
            ) as discovery:
                self.assertEqual(visible_gpu_tokens(value), [])
                discovery.assert_not_called()

    def test_memory_detection_honors_the_cgroup_limit(self) -> None:
        original = Path.read_text

        def fake_read_text(path, *args, **kwargs):
            values = {
                "/proc/meminfo": "MemAvailable:       8388608 kB\n",
                "/proc/self/cgroup": "0::/test.slice\n",
                "/sys/fs/cgroup/test.slice/memory.max": str(2 * 1024**3),
                "/sys/fs/cgroup/test.slice/memory.current": str(1024**3),
            }
            if str(path) in values:
                return values[str(path)]
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", fake_read_text):
            self.assertEqual(available_memory_bytes(), 1024**3)

    def test_small_memory_budget_can_select_less_than_ten_thousand_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "igenvs_ultra.workflow.available_memory_bytes", return_value=16 * 1024**2
        ), mock.patch("igenvs_ultra.workflow.gpu_memory_mib", return_value=[]):
            selected, decision = choose_stream_batch_size(Path(temporary), "auto")
        self.assertLess(selected, 10_000)
        self.assertEqual(selected, decision["constraints"]["host_memory_bound_rows"])


class ScoreShardTests(unittest.TestCase):
    def test_shards_keep_encoder_batch_boundaries(self) -> None:
        lengths = aligned_score_shard_lengths(1_000_000, 4, 192)
        self.assertEqual(lengths, [249_984, 249_984, 249_984, 250_048])
        self.assertTrue(all(length % 192 == 0 for length in lengths[:-1]))
        self.assertEqual(sum(lengths), 1_000_000)

    def test_small_workload_preserves_one_encoder_batch(self) -> None:
        self.assertEqual(aligned_score_shard_lengths(3, 4, 192), [3])
        self.assertEqual(aligned_score_shard_lengths(300, 4, 192), [192, 108])

    def test_score_shards_are_contiguous_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prepared.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["molecule_id", "smiles"])
                writer.writeheader()
                for index in range(11):
                    writer.writerow({"molecule_id": f"mol-{index}", "smiles": "C" * (index + 1)})
            output = root / "scores.csv"
            plan = prepare_score_shards(
                source, output, rows=11, shards=3, alignment=4
            )
            self.assertEqual([item["input_rows"] for item in plan], [4, 4, 3])
            ids = []
            for item in plan:
                with Path(item["input"]).open(newline="") as handle:
                    ids.extend(row["molecule_id"] for row in csv.DictReader(handle))
            self.assertEqual(ids, [f"mol-{index}" for index in range(11)])
            self.assertEqual(
                prepare_score_shards(source, output, rows=11, shards=3, alignment=4),
                plan,
            )

    def test_parallel_score_merge_restores_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prepared.csv"
            source.write_text("molecule_id,smiles\na,CC\nb,CCC\n", encoding="utf-8")
            model = root / "ensemble.json"
            model.write_text("{}\n", encoding="utf-8")
            plans = []
            manifests = []
            for index, molecule in enumerate(("a", "b")):
                shard = root / f"shard-{index}"
                shard.mkdir()
                shard_input = shard / "prepared.csv"
                shard_input.write_text(
                    f"molecule_id,smiles\n{molecule},{'CC' if index == 0 else 'CCC'}\n",
                    encoding="utf-8",
                )
                scores = shard / "scores.csv"
                scores.write_text(
                    f"molecule_id,ensemble_probability\n{molecule},0.{index + 1}\n",
                    encoding="utf-8",
                )
                rejections = shard / "scores.embeddings.rejections.csv"
                rejections.write_text("input_row,input_id,input_smiles,reason\n", encoding="utf-8")
                metadata = shard / "scores.embeddings.metadata.json"
                metadata.write_text("{}\n", encoding="utf-8")
                plans.append(
                    {
                        "shard_index": index,
                        "input": str(shard_input),
                        "input_sha256": "fixture",
                        "input_rows": 1,
                        "input_row_offset": index,
                    }
                )
                manifests.append(
                    {
                        "input_rows": 1,
                        "encoded_rows": 1,
                        "retained_rows": 1,
                        "output": str(scores),
                        "output_sha256": "fixture",
                        "save_policy": "all",
                        "score_threshold": None,
                        "model_manifest_sha256": "same",
                        "members": [],
                        "inference_seconds_member_sum": 0.01,
                        "encoder": {
                            "elapsed_seconds": 0.1,
                            "dimensions": 384,
                            "embedding_space": "released_hybrid_w3",
                            "backend": "fixture",
                            "device": "cuda:0",
                            "workers": 1,
                        },
                        "encoder_rejections": str(rejections),
                        "encoder_metadata": str(metadata),
                    }
                )
            output = root / "scores.csv"
            merged = merge_score_shards(
                source,
                output,
                model,
                plans,
                manifests,
                parallel_wall_seconds=0.2,
                keep_embeddings=False,
            )
            with output.open(newline="") as handle:
                self.assertEqual(
                    [row["molecule_id"] for row in csv.DictReader(handle)], ["a", "b"]
                )
            self.assertEqual(merged["encoded_rows"], 2)
            self.assertEqual(merged["encoder"]["shards"], 2)


class GenerationShardTests(unittest.TestCase):
    def test_iGen3_contract_elides_redundant_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "generated.smi"
            source.write_text("CCO\nc1ccccc1\n", encoding="utf-8")
            rows, record = generated_rows_from_iGen3_contract(
                source, root / "validation"
            )
            self.assertEqual([row["canonical_smiles"] for row in rows], ["CCO", "c1ccccc1"])
            self.assertEqual(record["validated_rows"], 2)
            self.assertEqual(record["rejected_rows"], 0)

    def test_fixed_logical_plan_is_hardware_independent(self) -> None:
        expected = generation_shard_plan(
            100_000, batch_number=7, base_seed=13, logical_shards=4
        )
        self.assertEqual([item["requested"] for item in expected], [25_000] * 4)
        self.assertEqual([item["seed"] for item in expected], [37, 38, 39, 40])
        self.assertEqual(sum(item["requested"] for item in expected), 100_000)

    def test_generation_uses_at_least_one_logical_shard_per_gpu(self) -> None:
        automatic = argparse.Namespace(generation_logical_shards="auto")
        fixed = argparse.Namespace(generation_logical_shards=4)
        invalid = argparse.Namespace(generation_logical_shards=2)
        self.assertEqual(generation_logical_shard_count(automatic, ["0", "1"]), 2)
        self.assertEqual(generation_logical_shard_count(fixed, ["0", "1"]), 4)
        with self.assertRaisesRegex(RuntimeError, "some selected GPUs would be idle"):
            generation_logical_shard_count(invalid, ["0", "1", "2", "3"])

    def test_persistent_resume_keeps_logical_shards_on_original_lanes(self) -> None:
        class FixturePool:
            def __init__(self):
                self.calls = []

            def run_wave(self, requests, lane_indices=None):
                lanes = list(lane_indices or [])
                self.calls.append(lanes)
                results = []
                for request, lane in zip(requests, lanes):
                    Path(request["output"]).write_text(
                        f"lane-{lane}\n", encoding="utf-8"
                    )
                    results.append({"generated": 1, "seconds": 0.01})
                return results

        with tempfile.TemporaryDirectory() as temporary:
            screen = Path(temporary)
            completed = screen / "batches/batch-000001/generation-shards/shard-0000"
            completed.mkdir(parents=True)
            completed_output = completed / "generated.smi"
            completed_output.write_text("completed-lane-0\n", encoding="utf-8")
            (completed / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "logical_shard": 0,
                        "requested": 1,
                        "seed": 13,
                        "produced": 1,
                        "output": str(completed_output),
                        "sha256": sha256(completed_output),
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                gpu_ids="0,1",
                screen_gpus="auto",
                encoder_device="auto",
                generation_logical_shards=4,
                generator_seed=13,
                seed_file=None,
                model_dir=None,
                generator_metrics=False,
            )
            pool = FixturePool()
            _, record = generate_batch(
                argparse.Namespace(execution="docker"),
                args,
                screen,
                1,
                4,
                pool,
            )

        self.assertEqual(pool.calls, [[1], [0, 1]])
        self.assertEqual(
            [item["gpu"] for item in record["shards"] if item["logical_shard"] > 0],
            ["1", "0", "1"],
        )

    def test_persistent_generated_screen_prefetches_during_scoring(self) -> None:
        generation_started = threading.Event()
        score_started = threading.Event()

        def generate(_runtime, _args, screen, batch, requested, _pool):
            raw = screen / f"batch-{batch}.smi"
            if batch == 2:
                generation_started.set()
                self.assertTrue(score_started.wait(timeout=2))
            return raw, {"produced": requested}

        def score(_runtime, _args, _job, _assets, _screen, batch, admission, _models, _pool):
            if batch == 1:
                self.assertTrue(generation_started.wait(timeout=2))
                score_started.set()
            return {"encoded_rows": admission["accepted_rows"]}

        args = argparse.Namespace(
            generate_count=4,
            max_stream_batches=None,
            fragment_policy="reject",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "igenvs_ultra.workflow.generate_batch", side_effect=generate
            ), mock.patch(
                "igenvs_ultra.workflow.generated_rows_from_iGen3_contract",
                return_value=([{"canonical_smiles": "CC"}], {"validated_rows": 1}),
            ), mock.patch(
                "igenvs_ultra.workflow.admit_batch",
                return_value={"accepted_rows": 2},
            ), mock.patch(
                "igenvs_ultra.workflow.score_batch", side_effect=score
            ):
                records = process_generated_screen(
                    mock.sentinel.runtime,
                    args,
                    root,
                    root,
                    root,
                    mock.sentinel.connection,
                    2,
                    root / "models.json",
                    mock.sentinel.generation_pool,
                    mock.sentinel.score_pool,
                )
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["prefetched_next_generation"])
        self.assertFalse(records[1]["prefetched_next_generation"])

    def test_generated_admission_overlaps_scoring_without_changing_counts(self) -> None:
        score_started = threading.Event()
        next_admitted = threading.Event()
        requests = []
        molecules = iter(["CC", "CCC", "CCCC", "CCCCC", "CCCCCC"])
        real_admit = admit_batch

        def generate(_runtime, _args, screen, batch, requested, _pool):
            requests.append(requested)
            raw = screen / f"generated-{batch}.smi"
            raw.write_text("".join(next(molecules) + "\n" for _ in range(requested)))
            return raw, {"produced": requested}

        def admit(*args):
            batch = args[2]
            if batch == 2:
                self.assertTrue(score_started.wait(timeout=2))
            result = real_admit(*args)
            if batch == 2:
                next_admitted.set()
            return result

        def score(_runtime, _args, _job, _assets, _screen, batch, admission, *_rest):
            if batch == 1:
                score_started.set()
                self.assertTrue(next_admitted.wait(timeout=2))
            return {"encoded_rows": admission["accepted_rows"] - int(batch == 1)}

        args = argparse.Namespace(generate_count=4, max_stream_batches=None, fragment_policy="reject")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            connection = open_dedup_database(root)
            try:
                with mock.patch("igenvs_ultra.workflow.generate_batch", side_effect=generate), mock.patch(
                    "igenvs_ultra.workflow.admit_batch", side_effect=admit,
                ), mock.patch("igenvs_ultra.workflow.score_batch", side_effect=score):
                    records = process_generated_screen(
                        mock.sentinel.runtime, args, root, root, root, connection, 2,
                        root / "models.json", mock.sentinel.generation_pool, mock.sentinel.score_pool,
                    )
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM smiles").fetchone()[0], 5)
            finally:
                connection.close()
        self.assertEqual(requests, [2, 2, 1])
        self.assertEqual(sum(row["score"]["encoded_rows"] for row in records), 4)
        self.assertEqual([row["prefetched_next_admission"] for row in records], [True, True, False])

    def test_external_admission_overlaps_prior_batch_scoring(self) -> None:
        score_started = threading.Event()
        second_admitted = threading.Event()

        def admit(_connection, screen, batch, _rows, _kind, _offset):
            if batch == 2:
                self.assertTrue(score_started.wait(timeout=2))
                second_admitted.set()
            prepared = screen / f"prepared-{batch}.csv"
            prepared.write_text("molecule_id,smiles\n", encoding="utf-8")
            return {"accepted_rows": 1, "prepared": str(prepared)}

        def score(_runtime, _args, _job, _assets, _screen, batch, *_rest):
            if batch == 1:
                score_started.set()
                self.assertTrue(second_admitted.wait(timeout=2))
            return {"encoded_rows": 1}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = root / "validated.csv"
            validated.write_text(
                "molecule_id,canonical_smiles\na,CC\nb,CCC\n", encoding="utf-8"
            )
            args = argparse.Namespace(
                input=str(root / "source.csv"),
                input_format="csv",
                smiles_column="smiles",
                id_column=None,
                delimiter=",",
            )
            with mock.patch(
                "igenvs_ultra.workflow.validate_library",
                return_value={"validated": str(validated)},
            ), mock.patch(
                "igenvs_ultra.workflow.admit_batch", side_effect=admit
            ), mock.patch(
                "igenvs_ultra.workflow.score_batch", side_effect=score
            ):
                records = process_external_screen(
                    mock.sentinel.runtime,
                    args,
                    root,
                    root,
                    root,
                    mock.sentinel.connection,
                    1,
                    root / "model.json",
                    mock.sentinel.score_pool,
                )
        self.assertEqual([item["batch"] for item in records], [1, 2])


if __name__ == "__main__":
    unittest.main()

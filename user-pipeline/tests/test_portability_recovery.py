from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from igenvs_ultra import rl_workflow, workflow

PROJECT = Path(__file__).resolve().parents[2]


class PortabilityTests(unittest.TestCase):
    def test_disable_masks_reach_real_children(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mask in ("", "-1", "NoDevFiles", "void"):
                with self.subTest(mask=mask), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
                    runtime = workflow.Runtime(
                        argparse.Namespace(execution="native", gpu_ids=mask), PROJECT, root,
                        require_gmolai=False,
                    )
                    command = ["python3", "-c", "import os; print(repr(os.environ['CUDA_VISIBLE_DEVICES']))"]
                    result = runtime.capture("igenvs", command)
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(result.stdout.strip(), "''")
                    log = root / "mask.log"
                    runtime.run_logged("igenvs", command, log)
                    self.assertEqual(log.read_text().strip(), "''")

    def test_core_commands_use_mounted_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = workflow.Runtime(
                argparse.Namespace(execution="native", gpu_ids=None), PROJECT, Path(temporary),
                require_gmolai=False,
            )
            for entry in ("igenvs", "igen3"):
                command = runtime._wrap("igenvs", [entry, "--help"], gpu=False)
                self.assertEqual(command, ["python3", str(PROJECT / "user-pipeline/src/igenvs_ultra/core_runtime.py"), entry, "--help"])

    def test_all_cgroup_ancestors_bound_available_ram(self):
        gib = 2**30
        for version in (1, 2):
            for leaf_limit in ("max", str(6 * gib)):
                with self.subTest(version=version, leaf_limit=leaf_limit):
                    base = "/sys/fs/cgroup" if version == 2 else "/sys/fs/cgroup/memory"
                    limit = "memory.max" if version == 2 else "memory.limit_in_bytes"
                    usage = "memory.current" if version == 2 else "memory.usage_in_bytes"
                    values = {
                        "/proc/meminfo": f"MemAvailable: {8*gib//1024} kB\n",
                        "/proc/self/cgroup": "0::/job/step\n" if version == 2 else "7:memory:/job/step\n",
                        f"{base}/job/step/{limit}": str(2**63-4096) if version == 1 and leaf_limit == "max" else leaf_limit,
                        f"{base}/job/step/{usage}": str(gib//2),
                        f"{base}/job/{limit}": str(2*gib),
                        f"{base}/job/{usage}": str(gib),
                    }
                    def read(path, *args, **kwargs):
                        if str(path) not in values:
                            raise FileNotFoundError(str(path))
                        return values[str(path)]
                    with mock.patch.object(Path, "read_text", read):
                        self.assertEqual(workflow.available_memory_bytes(), gib)

    def test_legacy_zero_success_docking_is_reopened(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel), tempfile.TemporaryDirectory() as temporary:
                job = Path(temporary)
                output = job / "docking"
                output.mkdir()
                results = output / "results.csv"
                results.write_text("source_row,status\n1,docking_failed\n")
                manifest = {
                    "status": "complete", "counts": {"prepared": 1, "docked": 0},
                    "outputs": {"results": str(results), "poses": None},
                }
                workflow.atomic_json(output / "manifest.json", manifest)
                workflow.atomic_json(job / "regular-summary.json", manifest)
                validated = job / "library/validation/validated.csv"
                validated.parent.mkdir(parents=True)
                validated.write_text("source_row,smiles\n1,CCO\n")
                if parallel:
                    (output / "runs").mkdir()
                def complete(*args, **kwargs):
                    output.mkdir(exist_ok=True)
                    results.write_text("source_row,status\n1,success\n")
                    repaired = {**manifest, "counts": {"prepared": 1, "docked": 1}}
                    workflow.atomic_json(output / "manifest.json", repaired)
                    return repaired
                runtime = mock.Mock()
                runtime.run_logged.side_effect = complete
                with mock.patch.multiple(
                    workflow, resolve_assets=mock.Mock(return_value=PROJECT),
                    make_regular_config=mock.Mock(return_value={"source": {"kind": "external"}, "target": {"target_name": "fixture"}}),
                    ensure_regular_config=mock.Mock(), prepare_target=mock.Mock(), Runtime=mock.Mock(return_value=runtime),
                    build_regular_docking_command=mock.Mock(return_value=["igenvs", "screen"]),
                    _regular_extra_paths=mock.Mock(return_value=[]),
                    docking_gpu_ids=mock.Mock(return_value=["0", "1"] if parallel else ["0"]),
                    prepare_shared_docking_library=mock.Mock(return_value=(validated, {})),
                    _launch_regular_docking_shards=mock.Mock(return_value=([], 0, {"logical_shards": 4}, ["0"])),
                    _merge_regular_docking_shards=mock.Mock(side_effect=complete),
                ):
                    result = workflow.regular_dock(argparse.Namespace(
                        output_dir=job, num_shards=1, shard_index=0, engine="unidock", dry_run=False,
                    ))
                self.assertEqual(result["counts"]["docked"], 1)
                self.assertEqual(len(list(job.glob("regular-summary.incomplete-*.json"))), 1)


class AdaptiveRecoveryTests(unittest.TestCase):
    def test_checkpoint_recovery_precedes_every_stopping_decision(self):
        stage = json.loads((PROJECT / "phase-10-rl-dev/protocol.json").read_text())["stages"][2]
        for legacy, stopped in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(legacy=legacy, stopped=stopped), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                stage_dir = root / stage["name"]
                stage_dir.mkdir()
                (stage_dir / "history.csv").write_text("update,seconds\n9,1\n10,1\n")
                if not legacy:
                    workflow.atomic_json(stage_dir / "progress.json", {"status": "complete", "completed_updates": 10, "model_latest_exported": False})
                if stopped:
                    workflow.atomic_json(stage_dir / "stopping.json", {
                        "rule": stage["adaptive_stopping"], "stopped_at_update": 10, "gate_met": True,
                    })
                calls = []
                def publish(update):
                    workflow.atomic_json(stage_dir / "progress.json", {"status": "complete", "completed_updates": update, "model_latest_exported": True})
                def recover(*args):
                    calls.append("recover")
                    publish(0 if legacy else 10)
                    return 0 if legacy else 10
                def train(**kwargs):
                    calls.append("train")
                    self.assertEqual(kwargs["requested_total"], 10)
                    publish(10)
                def gate(*args):
                    self.assertEqual(calls[0], "recover")
                    self.assertTrue(json.loads((stage_dir / "progress.json").read_text())["model_latest_exported"])
                    return {"passed": True}
                with mock.patch.object(rl_workflow, "_recover_stage", side_effect=recover), mock.patch.object(
                    rl_workflow, "_run_training_to", side_effect=train,
                ), mock.patch.object(rl_workflow, "_gate_check", side_effect=gate):
                    result = rl_workflow._run_adaptive_stage(
                        runtime=mock.Mock(), assets=PROJECT, job=root, stage_dir=stage_dir, stage=stage,
                        timing_records=[], gpu_count=1, timing_path=root / "timing.json", target="fixture", stages=[stage],
                    )
                self.assertTrue(result["gate_met"])
                self.assertEqual(calls, ["recover", "train"] if legacy else ["recover"])


if __name__ == "__main__":
    unittest.main()

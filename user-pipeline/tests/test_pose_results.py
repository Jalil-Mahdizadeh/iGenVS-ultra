"""Pose-level result rows must survive merges, export, and resume checks."""
import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from igenvs_ultra.workflow import (
    _docking_result_groups, _merge_regular_docking_shards,
    _regular_results_cover_validated, merge_docking, PipelineError,
)


FIELDS = ["molecule_id", "source_row", "status", "docking_score", "pose_rank", "num_poses"]


def pose_rows(source):
    return [dict(zip(FIELDS, (f"mol-{source}", str(source), "success", str(-8 + rank), str(rank), "3")))
            for rank in (1, 2, 3)]


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class PoseResultsTests(unittest.TestCase):
    def test_merge_keeps_all_poses_and_counts_molecules(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary)
            validated = job / "validated.csv"
            validated.write_text("source_row\n1\n2\n3\n4\n")
            source = job / "input.csv"
            source.write_text("molecule_id,smiles\n" + "".join(f"mol-{i},{'C' * i}\n" for i in range(1, 5)))
            shards = []
            for shard in range(2):
                directory = job / f"docking/runs/shard-{shard}"
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text(json.dumps({
                    "status": "complete", "counts": {"docked": 2},
                }))
                write_rows(directory / "results.csv", [row for i in range(shard + 1, 5, 2) for row in pose_rows(i)])
                shards.append(directory)
            manifest = _merge_regular_docking_shards(
                argparse.Namespace(engine="unidock", search_mode="fast", pose_output="none"),
                job, shards, ["0", "1"], 1.0, {"valid": 4}, validated=validated,
            )
            results = job / "docking/results.csv"
            with results.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(int(r["source_row"]), int(r["pose_rank"])) for r in rows],
                             [(source, rank) for source in range(1, 5) for rank in (1, 2, 3)])
            self.assertEqual(manifest["counts"]["docked"], 4)
            self.assertTrue(_regular_results_cover_validated(results, validated))
            # Model fitting still needs one best-score label per molecule.
            labels = merge_docking(job, "UDRL-train", source, shards, "target")
            with labels.open() as handle:
                label_rows = list(csv.DictReader(handle))
            self.assertEqual(len(label_rows), 4)
            self.assertTrue(all(row["pose_rank"] == "1" for row in label_rows))
            write_rows(results, rows[:-1])
            self.assertFalse(_regular_results_cover_validated(results, validated))

    def test_invalid_groups_are_not_accepted_as_complete(self):
        valid = pose_rows(1)
        for rows in (valid[:2], [valid[0], valid[0], valid[2]], valid[::-1], valid + pose_rows(1)):
            with self.subTest(rows=rows), self.assertRaises(PipelineError):
                list(_docking_result_groups(rows))
        failure = {**valid[0], "status": "preparation_failed", "pose_rank": "", "num_poses": "0", "docking_score": ""}
        self.assertEqual(list(_docking_result_groups([failure, *pose_rows(2)])), [[failure], pose_rows(2)])


if __name__ == "__main__":
    unittest.main()

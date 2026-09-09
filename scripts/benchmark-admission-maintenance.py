#!/usr/bin/env python3
"""Replay a generated screen's admitted library through a fresh identity DB."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "user-pipeline/src"))
from igenvs_ultra.workflow import admit_batch, atomic_json, open_dedup_database, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-screen", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT):
        parser.error("output must be a new directory outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    connection = open_dedup_database(output)
    count = 0
    timings = []
    try:
        for path in sorted(args.source_screen.glob("batches/batch-*/admission.json")):
            original = json.loads(path.read_text())
            if original["source_kind"] != "iGen3" or original["duplicate_rows"]:
                raise ValueError("this stage benchmark requires globally unique generated batches")
            with Path(original["prepared"]).open(newline="") as handle:
                rows = [{**row, "canonical_smiles": row["smiles"]} for row in csv.DictReader(handle)]
            started = time.perf_counter()
            result = admit_batch(connection, output, int(original["batch"]), rows, "iGen3", count)
            elapsed = time.perf_counter() - started
            assert result["prepared_sha256"] == original["prepared_sha256"]
            assert result["dedup_rejections_sha256"] == original["dedup_rejections_sha256"]
            assert result["accepted_rows"] == original["accepted_rows"]
            count += int(result["accepted_rows"])
            timings.append(elapsed)
            print(f"batch={original['batch']} rows={len(rows)} seconds={elapsed:.3f} byte-identical", flush=True)
        assert connection.execute("SELECT COUNT(*) FROM smiles").fetchone()[0] == count
    finally:
        connection.close()
    if not timings:
        raise ValueError("source screen contains no admission batches")
    record = {"status": "complete", "classification": "isolated_admission_replay_not_end_to_end",
              "admitted_rows": count, "batch_seconds": timings, "admission_seconds": sum(timings),
              "prepared_and_rejected_csvs_byte_identical": True,
              "workflow_sha256": sha256(ROOT / "user-pipeline/src/igenvs_ultra/workflow.py")}
    atomic_json(output / "summary.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

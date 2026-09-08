# Frozen five-receptor RL benchmark

Status: **complete and passed**. All 5/5 held-out receptors passed every frozen
independent docking, diversity, and chemistry-safety endpoint. Together with the
8/8 development result, the protocol passed all 13 curated receptors without a
benchmark-driven change.

This directory applies the byte-audited operational recipe accepted on all eight
development receptors to the five targets in `targets.txt`. The benchmark is
one-shot: its outcomes are reported as generalization evidence and do not change
the frozen protocol.

Only target metadata differs from development: benchmark receptors use the
already prepared targets under `phase-7-benchmark-docking/targets`. Training,
sampling, reward, chemistry, stopping, validation, and acceptance settings have
the same operational recipe hash recorded in `benchmark-freeze.json`.

Launch one four-GPU node per target:

```bash
sbatch phase-10-rl-bench/slurm/benchmark_array.sbatch
```

After all array tasks finish:

```bash
python3 phase-10-rl-bench/scripts/aggregate.py
```

Results are summarized in `REPORT.md`. Machine-readable outputs are
`docking-summary.csv`, `molecular-structural-summary.csv`, `training-times.csv`,
`model-manifest.csv`, and `integrity-audit.json`. Each target's `validation`
directory contains its 10,000-row paired base-versus-RL CSV.

The frozen recipe is now available through the public
`./igenvs-ultra rl-train` command. See [`../docs/CLI.md`](../docs/CLI.md) for
target inputs, container behavior, SIF/Docker execution, model export, and CSV
generation.

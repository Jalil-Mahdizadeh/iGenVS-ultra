# Full cold speed benchmark

Status: **complete**. All 12 docking cases and all three exact-10-million-score
screening cases finished on 2026-09-06. See the
[`REPORT.md`](REPORT.md) for the final 1/2/4-GPU results.

This directory contains the locked, one-sample benchmark described in
[`plan.md`](plan.md). The public `igenvs-ultra` wrappers select every performance
setting automatically.

From the repository root on an Arrhenius GH200 GPU allocation:

```bash
python speed-bench/scripts/prepare_fixed_library.py
python speed-bench/scripts/freeze_protocol.py
python speed-bench/scripts/submit.py --dry-run
python speed-bench/scripts/submit.py
```

After all submitted jobs finish successfully:

```bash
python speed-bench/scripts/collect_report.py
```

`protocol.json` hashes the plan, inputs, source trees, model assets, target, and
container images before execution. `audit/submission.json` maps every benchmark
case to its Slurm job. Each result directory contains one cold `summary.json` and
the scheduler and pipeline logs are retained under `logs/`.

For maintained source, do not re-freeze or overwrite this historical benchmark.
Use the separate operational runner with an output directory outside the repo:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/benchmark-screening-maintenance.py \
  --output-dir /tmp/igenvs-maintenance-screen --count 10000000
```

This invokes the public count-only command with both dedicated SIFs, a fresh
profile cache, and the same released 4ag8 target heads. It validates exact finite
score counts and records source hashes, hardware, and stage timings. It is not
a new scientific acceptance run or a replacement for the frozen 1/2/4-GPU
matrix. See [September 9 validation](MAINTENANCE-2026-09-09.md).

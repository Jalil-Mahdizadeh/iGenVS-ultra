# Full cold speed benchmark

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

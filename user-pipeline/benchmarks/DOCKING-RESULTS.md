# Portable regular-docking optimization results

Status: one-GPU smoke/medium qualification and the requested cold 1/2/4-GPU
matrix are complete as of 2026-09-06. A separate larger sustained/cache/pose
study remains optional follow-on work, not an unfinished release benchmark.

## Boundary and hardware

Every smoke/medium case below invokes the public `igenvs-ultra dock` command,
not an isolated kernel. Wall time begins before the wrapper/container startup
and ends after the durable result and terminal manifests exist. The fixture is
the prepared 4ag8 target and a deterministic prefix of
`speed-bench/inputs/fixed-20k.csv`, using standard preparation, one score,
refinement, no pose persistence, and automatic batching/workers.

The qualification node was `n191`: one visible NVIDIA GH200 120GB (97,871 MiB
reported), 72 CPUs in the process affinity, driver 580.159.04, and Slurm job
2071150. These measurements establish correctness and a medium performance
signal. The later full cold matrix used separate 1/2/4-GPU allocations; other
hardware should still use its own automatic plan.

## Smoke tests

| Engine / mode | Input | Finite scores | Complete wall |
| --- | ---: | ---: | ---: |
| Uni-Dock fast | 8 | 8 | 5.859 s |
| Uni-Dock balance | 8 | 8 | 6.273 s |
| Uni-Dock detail | 8 | 8 | 6.416 s |
| AutoDock-GPU fast | 8 | 8 | 4.525 s |

AutoDock-GPU automatically selected a 4,096 outer batch and six workers,
started a run-private CUDA MPS daemon, and stopped it at completion. No MPS
process or private pipe directory remained.
An additional default-product Uni-Dock balance smoke wrote one merged pose for
each of eight successful rows; every result `pose_ref` resolved into the
durable merged pose stream.

## Medium tests

The medium suite uses the same first 4,096 input rows for every case:

| Engine / mode | Prepared | Successful | Prep wait | Engine wall | Complete wall | Input/h | Pre-opt input/h | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uni-Dock fast | 4,066 | 4,046 | 5.875 s | 66.789 s | 74.380 s | 198,248 | 166,502 | +19.1% |
| Uni-Dock balance | 4,066 | 4,036 | 5.791 s | 208.351 s | 215.865 s | 68,309 | 63,263 | +8.0% |
| Uni-Dock detail | 4,066 | 4,038 | 5.845 s | 260.562 s | 268.136 s | 54,993 | 51,342 | +7.1% |
| AutoDock-GPU fast | 4,066 | 4,066 | 5.885 s | 387.637 s | 395.521 s | 37,281 | 33,392 | +11.6% |

The comparison rates in the last two columns are from the pre-optimization
20,000-row one-GPU record. Rate normalization is useful, but a 4,096-row prefix
and a 20,000-row library do not have identical chemistry or amortization. The
completed controlled matrix is reported separately below.

A separate exact-library Uni-Dock-fast check used all 20,000 locked rows. Its
final automatic plan used a 2,048-ligand ramp while preparing the remaining
17,952 behind the first engine invocation. The approximately 247k/hour value
is **not a new raw-engine rate**: the pre-optimization record measured 246,377
successful molecules/hour over engine wall. The like-for-like boundaries are:

| Metric | Pre-optimization run | Optimized run | Delta |
| --- | ---: | ---: | ---: |
| Input rows/hour, complete wall | 166,502 | 246,932 | +48.3% |
| Successful rows/hour, complete wall | 164,046 | 242,809 | +48.0% |
| Successful rows/hour, engine wall only | 246,377 | 248,666 | +0.9% |

Complete wall fell from 432.427 to 291.578 seconds while engine wall was
essentially constant (287.925 versus 284.709 seconds). The improvement closes
the old gap between end-to-end and engine-only throughput by overlapping
preparation with docking; it does not make Uni-Dock itself materially faster.
The bounded-preparation run without a ramp took 307.710 seconds, so the ramp
supplied a further 5.5% wall-time improvement.

## Completed full cold scaling matrix

The final controlled benchmark used 20,000 fixed input molecules per GPU,
score-only output, the public wrapper, and automatic performance settings.
Every row is one cold sample with no warm-up or repeat.

| Engine / mode | GPUs | Input | Yield | Complete wall | Input/hour | Successful/hour | Engine-only successful/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uni-Dock fast | 1 | 20,000 | 98.395% | 302.967 s | 237,649 | 233,835 | 242,218 |
| Uni-Dock fast | 2 | 40,000 | 98.345% | 301.827 s | 477,095 | 469,199 | 485,652 |
| Uni-Dock fast | 4 | 80,000 | 98.365% | 319.341 s | 901,859 | 887,113 | 931,350 |
| Uni-Dock balance | 1 | 20,000 | 98.380% | 1,002.441 s | 71,825 | 70,661 | 71,401 |
| Uni-Dock balance | 2 | 40,000 | 98.305% | 1,020.373 s | 141,125 | 138,733 | 140,307 |
| Uni-Dock balance | 4 | 80,000 | 98.374% | 1,011.047 s | 284,853 | 280,221 | 283,773 |
| Uni-Dock detail | 1 | 20,000 | 98.250% | 1,262.779 s | 57,017 | 56,019 | 56,491 |
| Uni-Dock detail | 2 | 40,000 | 98.310% | 1,283.153 s | 112,224 | 110,327 | 111,373 |
| Uni-Dock detail | 4 | 80,000 | 98.355% | 1,308.415 s | 220,114 | 216,493 | 218,568 |
| AutoDock-GPU fast | 1 | 20,000 | 99.090% | 1,993.594 s | 36,116 | 35,787 | 36,013 |
| AutoDock-GPU fast | 2 | 40,000 | 99.037% | 2,021.262 s | 71,243 | 70,557 | 71,097 |
| AutoDock-GPU fast | 4 | 80,000 | 99.066% | 2,044.646 s | 140,856 | 139,540 | 140,864 |

The full stage timing table and audit metadata are in
[`../../speed-bench/REPORT.md`](../../speed-bench/REPORT.md). Four-GPU input
rate scaled by 3.79x for Uni-Dock fast, 3.97x for balance, 3.86x for detail,
and 3.90x for AutoDock-GPU fast relative to each one-GPU case.

## Hard-molecule policy

The old exact-library run allowed 159 ultimately failed embeddings to consume
3,524.439 CPU-seconds and wait as long as 98.185 seconds. With a 50-attempt,
3-second primary ETKDG budget and a smaller conditional fallback:

| Metric | Old | Optimized | Change |
| --- | ---: | ---: | ---: |
| Preparation wait | 129.489 s | 4.028 s | -96.9% |
| Preparation CPU sum | 4,820.803 s | 1,345.587 s | -72.1% |
| Failed-molecule CPU sum | 3,524.439 s | 252.351 s | -92.8% |
| p99.5 preparation duration | 12.411 s | 1.358 s | -89.1% |
| Maximum preparation duration | 98.185 s | 4.217 s | -95.7% |
| Preparation failures/timeouts | 159 / 0 | 179 / 4 | +24 total |

The deliberate yield cost is 24 rows, or 0.12% of input. Larger calibrated
budgets consumed substantially more CPU for only a handful of recoveries and
did not rescue any of the old 159 final failures. The final ramp changes
Uni-Dock's seeded batch stream: among 19,559 molecules with a successful old
and final score, Spearman rank correlation was 0.920 and top-100 overlap was
71%. The non-ramped bounded run had correlation 0.932.
The accepted-population tradeoff is therefore explicit and configurable rather
than hidden behind an unbounded fallback.

## AutoDock-GPU balance

The six medium worker spans were 369.817, 366.736, 380.114, 368.233, 387.223,
and 371.262 seconds: a 1.036 maximum/mean ratio. The previous static-stride
20k run averaged 1.084 maximum/mean over five batches and reached 1.199 in its
tail batch. The new deterministic longest-work-first torsion assignment reduces
idle tail without changing LGA runs, heuristics, autostop, local search, AD4
scoring, or per-ligand preparation.

## Neural-screening non-regression

Docking changes must not take resources or throughput from the separate ultra
screen. Two post-change checks used the frozen base-isomeric/4ag8 three-head
screening protocol:

| Check | Frozen wall | Post-docking wall | Frozen/post hash | Outcome |
| --- | ---: | ---: | --- | --- |
| 8,192, fixed 2,048 stream plan | 52.705 s | 50.234 s | `18c299...f85c7b` | exact, no slowdown |
| 200,000, cached auto plan | 75.170 s | 75.536 s | `980335...d9e3` | exact, -0.48% rate |

The medium post-change rate was 9,531,919 scored molecules/hour versus
9,578,259/hour at code freeze. A 0.48% single-run difference is ordinary timing
noise, while exact result hashes, row counts, worker plans, and component times
show no functional or material performance regression. The corresponding
records are `results/smoke-post-docking-opt-fixed-plan` and
`results/medium-post-docking-opt-cached`.

## Reproduce and evidence

Run `benchmarks/docking_optimization.py` with a new label. Each result directory
contains `summary.json`, the exact input prefix, command log, wrapper job,
iGenVS manifest, engine logs, terminal CSV, hardware record, source hashes, and
result checksum. The authoritative runs are:

- `results/docking-smoke-wrapper-unidock-fast-final`
- `results/docking-smoke-wrapper-unidock-{balance,detail}-v1`
- `results/docking-smoke-wrapper-autodock-gpu-fast-codefreeze`
- `results/docking-smoke-wrapper-unidock-balance-poses-final`
- `results/docking-medium-wrapper-unidock-{fast,balance,detail}-v1`
- `results/docking-medium-wrapper-unidock-fast-ramp-v1` (exact 20k final plan)
- `results/docking-medium-wrapper-autodock-gpu-fast-v1`
- `results/docking-medium-unidock-fast-v1` (exact 20k comparison)

The requested 20,000-ligand/GPU cold matrix is complete. Optional follow-on
work can test at least 65,536 ligands per GPU for longer steady-state
amortization, warm preparation caches, and pose-writing products. Those are
distinct protocols and must not replace or be merged with the locked cold
results.

# iGenVS-ultra regular-docking speed audit and optimization blueprint

Status: portable speed implementation, one-GPU smoke/medium qualification, and
the requested cold 1/2/4-GPU matrix completed on 2026-09-06. The original
pre-optimization audit is retained as historical rationale; larger
sustained/cache/pose experiments remain optional follow-on work.

## Scope

This audit covers both docking routes in `user-pipeline`:

1. The user-facing `igenvs-ultra dock` command, which automatically creates one
   upstream `igenvs screen` lane per selected GPU and merges the results.
2. Reference, validation, and active-learning docking launched by `fit`, which
   uses four fixed logical shards and later merges their results.

It traces target preparation, generated or external molecule ingress,
validation and deduplication, RDKit/Meeko ligand preparation, Uni-Dock and
AutoDock-GPU execution, pose/result persistence, sharding, restart behavior,
and hardware auto-configuration. Neural final screening is covered separately
by `speed-opt-screening.md`.

## Executive conclusion

The implemented portable multi-GPU path removes most avoidable outer
orchestration and ligand-preparation tail overhead. On the completed
base-isomeric 4ag8 cold fixture, four GH200 GPUs processed 80,000 rows at
901,859 input molecules/hour and 887,113 successful molecules/hour end to end;
the concurrent Uni-Dock engine boundary was 931,350 successful/hour.

The remaining fast-mode gap is now small and primarily inside the engine
boundary: engine critical time was 304.173 of 319.341 complete seconds, while
preparation critical time was only 5.740 seconds. Balance, detail, and
AutoDock-GPU are even more engine dominated. Further work should therefore
focus on measured engine setup/kernel behavior, longer sustained workloads,
and optional cache/pose products rather than re-solving the completed
multi-GPU orchestration problem.

## Implementation checkpoint: 2026-09-06

The following high-priority changes are now implemented in the source-tree
execution path and covered by automated or real-engine tests:

- `igenvs-ultra dock` discovers all CUDA-visible GPUs by default, launches one
  docking lane per GPU with disjoint CPU affinity, and k-way merges shard CSVs
  back into global source order. Explicit legacy `--num-shards`/
  `--shard-index` operation remains available and disables the automatic
  fan-out.
- Multi-GPU regular docking generates an iGen3 library only once and validates
  and globally deduplicates only once. Each engine lane consumes a trusted,
  checksummed validation artifact and applies the original source-row modulo
  partition, avoiding G-fold input scans and SQLite/RDKit validation.
- UDRL and active-learning docking also validate once per library rather than
  once per logical shard. Fixed logical shard identities remain independent of
  the number of physical GPUs used to execute them.
- Ligand preparation now uses physical cores, current affinity, and available
  RAM rather than blindly treating every logical CPU as a safe RDKit process.
  Workers and GPU identity are resolved inside the actual CUDA visibility
  boundary, including numeric and UUID `CUDA_VISIBLE_DEVICES` forms.
- The hard-molecule policy is deterministic and bounded. Primary ETKDG gets at
  most 50 attempts and a native 3-second guard. A quick non-timeout failure can
  use one random-coordinate rescue with half the attempt budget and a 2-second
  guard; a primary timeout is never sent through the expensive fallback.
  Likely difficult molecules are submitted first so the bounded tail drains
  early, while returned records stay in source order. Molecules already over
  the engine atom limit are rejected before 3D construction.
- Large Uni-Dock jobs use a workload/CPU-aware first preparation ramp (2,048
  ligands on the measured 64-worker lane), then submit a full steady batch
  before the first GPU invocation. Small jobs avoid the extra launch. This
  hides almost all remaining preparation without shrinking sustained batches.
- AutoDock-GPU automatically selects a capacity-class outer batch and MPS
  worker prior, starts a run-private CUDA MPS daemon when concurrency is
  selected, records the decision, and always tears the daemon down. Unsupported
  machines fall back safely to one process. Its file lists use deterministic
  longest-work-first torsion balancing rather than static striding.
- Only the selected docking engine and actually requested generator are probed;
  unused Uni-Dock, AutoDock-GPU, AutoGrid, and iGen3 cold probes no longer tax
  every run.

The neural `screen`/`screen-fast` data plane is separate. None of its resident
iGen3 or gMolAI worker, batching, encoding, inference, or result-policy code is
called by these changes. A post-change smoke/medium regression is nevertheless
part of qualification, because source-level isolation is not accepted as the
only evidence of non-regression.

### Hard-molecule evidence

The locked 20,000-row one-GPU fast fixture exposed the original pathology:
159 ultimately failed embeddings consumed 3,524.439 CPU-seconds, 73.1% of all
preparation CPU, and the slowest molecule consumed 98.185 seconds. Under the
bounded policy:

| Quantity | Original | Optimized | Change |
| --- | ---: | ---: | ---: |
| Preparation wait | 129.489 s | 4.028 s | -96.9% |
| Total preparation CPU | 4,820.803 CPU-s | 1,345.587 CPU-s | -72.1% |
| Failed-molecule preparation CPU | 3,524.439 CPU-s | 252.351 CPU-s | -92.8% |
| p99 preparation time | 1.176 s | 0.931 s | -20.8% |
| p99.5 preparation time | 12.411 s | 1.358 s | -89.1% |
| Maximum preparation time | 98.185 s | 4.217 s | -95.7% |
| Preparation failures/timeouts | 159 / 0 | 179 / 4 | +24 total |

Calibration on the complete old hard cohort found that larger budgets spent
substantially more CPU for only a few rescues, while none of the old 159 final
failures was rescued. The selected policy therefore makes the tradeoff explicit:
24 extra preparation drops (0.12% of input) eliminate the destructive tail.
The final 2,048-ligand ramp changes Uni-Dock's seeded batch stream: of 19,559
common successful results, old/final score ranks had Spearman correlation
0.920 and top-100 overlap was 71%. The bounded non-ramped run had correlation
0.932. Because Uni-Dock's stochastic stream also depends on ligand position,
dropping a ligand or changing batch boundaries can change subsequent scores
even though each prepared conformer seed is molecule-stable; users requiring
an older accepted population can explicitly raise `--embed-max-attempts` and
`--embed-timeout`.

### Smoke and medium qualification

All smoke cases used the public `igenvs-ultra dock` wrapper, the prepared 4ag8
target, standard preparation, automatic hardware planning, one GH200, and
score-only output. Eight of eight inputs produced finite scores in each of
Uni-Dock fast/balance/detail and AutoDock-GPU fast. AutoDock-GPU selected six
workers, created its private MPS service, and left no daemon or pipe directory
after completion.

The medium suite used the same deterministic first 4,096 rows of the locked
20,000-row base-isomeric library. It is intentionally a one-GPU qualification,
distinct from the later completed full scaling benchmark:

| Engine / mode | Input | Prepared | Successful | Complete wall | Input/h | Locked 20k input/h | Rate delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uni-Dock fast | 4,096 | 4,066 | 4,046 | 74.380 s | 198,248 | 166,502 | +19.1% |
| Uni-Dock balance | 4,096 | 4,066 | 4,036 | 215.865 s | 68,309 | 63,263 | +8.0% |
| Uni-Dock detail | 4,096 | 4,066 | 4,038 | 268.136 s | 54,993 | 51,342 | +7.1% |
| AutoDock-GPU fast | 4,096 | 4,066 | 4,066 | 395.521 s | 37,281 | 33,392 | +11.6% |

The percentage comparisons normalize against the prior locked 20k run but do
not make the two chemistry/amortization windows identical. A separate exact
20,000-row Uni-Dock-fast rerun supplies the like-for-like anchor. Its final
automatic plan uses a 2,048-ligand ramp, then prepares the remaining 17,952
while the first GPU invocation runs.

The approximately 247k/hour value is not a new Uni-Dock engine-rate result.
The pre-optimization record already contained 246,377 successful rows/hour
over engine wall. These are the correct like-for-like comparisons:

| Metric | Locked run | Optimized run | Delta |
| --- | ---: | ---: | ---: |
| Input rows/hour, complete wall | 166,502 | 246,932 | +48.3% |
| Successful rows/hour, complete wall | 164,046 | 242,809 | +48.0% |
| Successful rows/hour, engine wall only | 246,377 | 248,666 | +0.9% |

Wall fell from 432.427 to 291.578 seconds and preparation wait fell to 4.028
seconds; engine wall was essentially unchanged (287.925 versus 284.709
seconds). The measured gain is removal of orchestration/preparation idle time,
not materially faster Uni-Dock computation. The bounded non-ramped
intermediate took 307.710 seconds, so adaptive overlap contributed another
5.5% wall reduction.

For AutoDock-GPU, the optimized medium run's six worker spans were
366.736--387.223 seconds (maximum/mean 1.036). The old static-stride 20k run
averaged 1.084 maximum/mean across its five batches and reached 1.199 in the
tail batch. This supports torsion-aware balancing while leaving the LGA run,
evaluation, local-search, autostop, and scoring protocol untouched.

Machine-readable evidence is stored under
`../user-pipeline/benchmarks/results/docking-{smoke,medium}-wrapper-*`, with the
exact command, hardware, source hashes, manifest timings, terminal counts, and
result checksum. `../user-pipeline/benchmarks/docking_optimization.py`
reproduces the bounded suite. The requested cold 1/2/4-GPU matrix was completed
after this qualification and is summarized below.

The isolation regression also passed. An exact-plan 8,192-row neural smoke run
reproduced checksum `18c29950572bcb1d8c6fef3f27ef4734148d216dc1b6bb98c80f48aa2fc85c7b`
and completed in 50.234 seconds versus 52.705 at code freeze. The cached
200,000-row medium run reproduced checksum
`980335c823b8cc13380347bb8e7ca0b18b369986c5361284832f513ae718d9e3`
and completed in 75.536 seconds versus 75.170, a 0.49% single-run difference.
This is timing noise rather than material degradation; the docking path did
not modify neural generation, encoding, head inference, or their resource plan.

### Completed full cold 1/2/4-GPU benchmark

All cases used 20,000 fixed inputs per GPU, score-only output, one cold sample,
and automatic performance controls. The complete stage table and audit are in
[`speed-bench/REPORT.md`](../speed-bench/REPORT.md).

| Engine / mode | 1-GPU input/h | 2-GPU input/h | 4-GPU input/h | 4-GPU successful/h | 4-GPU engine-only successful/h |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uni-Dock fast | 237,649 | 477,095 | 901,859 | 887,113 | 931,350 |
| Uni-Dock balance | 71,825 | 141,125 | 284,853 | 280,221 | 283,773 |
| Uni-Dock detail | 57,017 | 112,224 | 220,114 | 216,493 | 218,568 |
| AutoDock-GPU fast | 36,116 | 71,243 | 140,856 | 139,540 | 140,864 |

Relative to one GPU, four-GPU complete-wall input throughput scales 3.79x,
3.97x, 3.86x, and 3.90x respectively. Yields are 98.25-98.40% for Uni-Dock
and 99.04-99.09% for AutoDock-GPU across the matrix.

## Pre-optimization execution paths (historical)

The execution-path and bottleneck sections below describe the implementation
that motivated the checkpoint above. They are retained to explain the design;
statements in these sections are not descriptions of the released optimized
path.

### Standalone regular docking

`user-pipeline/src/igenvs_ultra/workflow.py::regular_dock` performs these steps:

1. Resolve assets and create the job configuration.
2. Prepare the target once.
3. Build one `igenvs screen` command.
4. Launch that one process with `gpu=True`.
5. Accept an already-complete manifest, but rename any incomplete output and
   restart the whole screen.

The upstream `iGenVS/src/igenvs/pipeline.py` process then:

1. probes versions for Uni-Dock, AutoDock-GPU, AutoGrid, and iGen3;
2. completes the entire generated library, if generation was requested;
3. completes validation and deduplication and writes a validated file;
4. reads that file into outer batches;
5. prepares every ligand in the first batch and waits for the whole batch;
6. submits preparation for the next batch while docking the current batch;
7. parses results, persists optional poses, and writes result rows;
8. repeats until complete and then writes the terminal manifest.

This overlaps preparation and docking only after the first complete
preparation batch. Generation, validation, initial preparation, and the first
GPU invocation remain serial barriers.

### Docking inside fitting and active learning

`workflow.py::run_docking` uses four fixed logical modulo shards by default.
It starts at most one process per selected GPU and assigns additional logical
shards to that GPU in later waves. This preserves hardware-count-independent
logical shard identities, but each logical shard is a fresh `igenvs screen`
process with its own:

- input scan and validation database;
- tool probes and container startup;
- RDKit/Meeko process pool;
- first-batch preparation barrier;
- Uni-Dock process and temporary files;
- teardown.

On one GPU, the four logical shards execute as four sequential cold processes;
on two GPUs they execute in two waves; on four GPUs they execute concurrently.
This is reproducible, but it multiplies cold work and makes completion depend
on the slowest preparation tail.

## Pre-optimization measured baseline (historical)

The original audit used these older measurements. They are not the completed
results in `speed-bench/REPORT.md`:

| Engine / mode | GPUs | Input | Complete wall | Input/h | Engine successful/h |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uni-Dock fast | 1 | 20,000 | 432.427 s | 166,502 | 246,377 |
| Uni-Dock fast | 2 | 40,000 | 420.436 s | 342,502 | 513,185 |
| Uni-Dock fast | 4 | 80,000 | 504.513 s | 570,847 | 980,565 |
| Uni-Dock balance | 1 | 20,000 | 1,138.114 s | 63,263 | 71,363 |
| Uni-Dock balance | 4 | 80,000 | 1,227.252 s | 234,671 | 280,660 |
| Uni-Dock detail | 1 | 20,000 | 1,402.366 s | 51,342 | 56,206 |
| Uni-Dock detail | 4 | 80,000 | 1,497.472 s | 192,324 | 222,156 |
| AutoDock-GPU fast | 1 | 20,000 | 2,156.219 s | 33,392 | 34,885 |
| AutoDock-GPU fast | 4 | 80,000 | 2,140.806 s | 134,529 | 141,316 |

`Input/h` includes every input row and complete launcher wall. `Engine
successful/h` uses successful finite scores and the slowest concurrent shard's
docking wall, excluding validation, preparation, launcher startup, and result
bookkeeping. They intentionally expose different boundaries and must not be
silently interchanged.

The published 1,340,509 attempted/hour four-GH200 result used 262,144
RL-nonisomeric ligands against 1iep, or 65,536 ligands per GPU. This historical
benchmark used 80,000 base-isomeric ligands against 4ag8, or 20,000 per GPU.
Those ligands are larger and more flexible on average, and the shorter
run amortizes startup and preparation less. The rate difference is therefore
not evidence that all of the published rate can be recovered by orchestration
alone.

## Bottleneck 1: standalone `dock` is not an automatic multi-GPU command

The regular wrapper builds and launches exactly one upstream screen command.
It passes one `--device-id`, one shard index, and one shard count. Making
several GPUs visible does not create one lane per GPU; device 0 remains the
selected engine device.

Manual `--num-shards` and `--shard-index` values describe only one shard for
one invocation. A user must launch several commands, allocate separate output
directories, and merge results outside the ordinary command. That violates the
goal of automatic maximum-speed operation.

Generated input makes manual sharding worse. Separate shard invocations each
generate the requested complete library with the same seed and then retain
only their modulo shard, multiplying generation work by the number of GPUs. A
single invocation with `num_shards > 1` generates all requested molecules but
docks only one shard.

Required design:

- discover the GPUs actually visible to CUDA;
- create one persistent docking lane per selected GPU;
- validate/deduplicate and partition the source once;
- route each accepted molecule to exactly one logical shard and physical lane;
- automatically merge terminal result partitions;
- retain fixed logical identities independently of physical GPU count when
  cross-hardware byte identity is part of the contract.

## Bottleneck 2: long-tail ligand preparation is on the critical path

Ligand preparation uses one `ProcessPoolExecutor` future per molecule. The
collector calls `future.result()` in submission order and does not release a
partially ready GPU batch. Docking begins only after every future in the outer
batch has terminated.

For the one-GPU 20,000-row fast run:

| Quantity | Value |
| --- | ---: |
| Preparation wait | 129.489 s |
| Docking wall | 287.925 s |
| Total preparation CPU | 4,820.803 CPU-s |
| Prepared | 19,841 |
| Preparation failures | 159 |
| Successfully docked | 19,705 |

The preparation duration distribution was strongly heavy-tailed:

| Percentile | Time |
| --- | ---: |
| p50 | 0.0371 s |
| p95 | 0.1355 s |
| p99 | 1.1763 s |
| p99.5 | 12.4107 s |
| maximum | 98.185 s |

All 159 preparation failures ended with the same RDKit ETKDG status `-1`.
Together they consumed 3,524.439 CPU-s, or approximately 73.1% of all
preparation CPU time, despite representing only 0.795% of input rows. Their
mean duration was 22.166 seconds. The expensive path is the automatic second
embedding attempt with `useRandomCoords=True`.

Only seven successfully prepared molecules took at least five seconds. This
shows that the dominant tail is not useful slow work; it is mostly a long
fallback that ultimately fails.

Across the four-GPU fast run, shard preparation waits ranged from 112.9 to
205.7 seconds, while docking times stayed tightly grouped at 282.8–289.1
seconds. Preparation imbalance, rather than Uni-Dock GPU imbalance, selected
the slowest shard and launcher wall.

Required design:

- keep a persistent topology-bound preparation pool;
- submit a bounded window instead of tens of thousands of futures at once;
- separate the common fast embedding path from exceptional random-coordinate
  fallback work so failures cannot block an otherwise ready GPU batch;
- start docking with a smaller ramp batch, then use large steady-state batches;
- schedule predicted-complex molecules early to drain the tail sooner;
- retain deterministic per-molecule seeds independent of completion order;
- establish a scientifically documented retry/time policy before changing the
  accepted/failure population.

Simply disabling the random-coordinate retry could improve speed but would
change the preparation-success contract. It is not a free optimization.

## Bottleneck 3: the outer pipeline has coarse stage barriers

For generated work, iGen3 must finish the entire library before validation.
Validation and SQLite-backed deduplication must finish before the validated
file is reopened for preparation. The GPU cannot dock early molecules while
later molecules are generated or validated.

For external libraries, the generation barrier is absent, but full validation
still precedes ligand preparation. At very large input sizes, time-to-first-GPU
work and temporary storage grow unnecessarily.

Required data flow:

```text
source reader or generator
        |
bounded validation + global exact dedup
        |
fast preparation queue ---- slow fallback queue
        |
ready prepared-ligand blocks
        |
one persistent docking lane per GPU
        |
partitioned result/checkpoint writer
```

Backpressure must bound RAM and scratch usage. Completed records can flow
forward without waiting for the entire input, while a global identity index
continues to enforce exact deduplication.

## Bottleneck 4: fixed outer batches optimize engine rate, not end-to-end time

The existing overlap submits preparation of batch `k+1` immediately before
docking batch `k`. This is useful once steady state has begun, but the first
batch is fully cold. Selecting a 32,768-ligand batch because it gives the
highest raw Uni-Dock rate can make short jobs wait for the entire preparation
tail before any GPU work begins.

The optimum should depend on requested molecule count:

- a small first ramp batch minimizes time to first GPU work;
- large middle batches maximize sustained Uni-Dock efficiency;
- a deliberately sized final batch avoids a poorly filled tail;
- ready-queue depth should be large enough to hide preparation but bounded by
  host RAM, scratch capacity, and failure-recovery cost.

The AutoDock-GPU runs incidentally demonstrate the effect: their 4,096-ligand
outer batches allowed most later preparation to hide beneath much slower GPU
docking, so measured preparation wait was lower despite identical chemistry.

## Bottleneck 5: hardware planning is not portable enough

Current automatic worker counts are derived from scheduler affinity or logical
CPU count and capped. They do not distinguish physical cores from SMT threads
or account for:

- total and currently available host RAM;
- NUMA distance between CPU cores, memory, scratch, and each GPU;
- per-worker RDKit/Meeko memory;
- other processes in the allocation;
- shared-filesystem bandwidth and inode pressure;
- target box dimensions and scoring protocol;
- ligand atom/torsion distributions;
- heterogeneous GPU performance.

A fixed 64-worker preparation pool was effective on the measured GH200 node,
but it is not a safe universal default. It can oversubscribe an x86 workstation
with SMT, starve the writer or driver threads, or exhaust RAM.

The planner should inventory physical cores, logical siblings, NUMA nodes,
available RAM, visible GPUs, free VRAM, and local scratch. It should reserve
control/I/O capacity, assign disjoint physical-core sets to GPU lanes, and cap
workers using both memory and measured preparation throughput.

## Bottleneck 6: visible-GPU identity and provenance are unreliable

`selected_gpu()` indexes the node-wide `nvidia-smi` result and assumes it has
been remapped by `CUDA_VISIBLE_DEVICES`. In the current Apptainer environment,
`nvidia-smi` continued to expose the node-wide order. Four separate shard
manifests consequently recorded the same GPU-0 UUID even though the launcher
assigned four distinct GPUs and CUDA docking used the remapping correctly.

This is more than a reporting defect. On heterogeneous or partially occupied
GPUs, the auto batch planner can consult the wrong model or free-memory value,
leading to a suboptimal selection or an out-of-memory failure.

Required fix:

- parse visible identifiers as indices or UUIDs;
- resolve each token against NVML or node inventory explicitly;
- distinguish host/nvidia-smi index from process-local CUDA ordinal;
- record both identities and verify them from the CUDA runtime;
- key performance profiles by the resolved device, not assumed list position.

## Bottleneck 7: every logical shard repeats ingress and validation

Modulo sharding occurs while each shard reads the source. With `G` shard
processes, the same large CSV or SMI file is opened and scanned `G` times.
Each process constructs its own validation outputs and SQLite database.

Consequences:

- input parsing and shared-filesystem traffic scale with GPU count;
- RDKit validation infrastructure is repeatedly initialized;
- deduplication is only local to a shard, so identical molecules assigned to
  different shards can both be docked;
- global counts and duplicate policy are harder to reason about;
- a generated library may be regenerated redundantly under manual sharding.

Validation and global exact deduplication should happen once. The admitted
stream should then be deterministically partitioned into lane-ready records.
For very large sources, partition indexes or compact Arrow/Parquet partitions
are preferable to `G` complete CSV passes.

## Bottleneck 8: no prepared-ligand cache

Prepared PDBQT depends on canonical SMILES, molecule identity/seed,
preparation mode, limits, and RDKit/Meeko behavior, but not on the docking
target. The current pipeline regenerates the same ligand conformer and PDBQT
for every target and every rerun.

This is especially expensive in `fit`:

- the fixed UDRL training and validation libraries are docked for each target;
- active-learning rounds can revisit chemistry across stages;
- tuning prepares pilot sets that may later be prepared again;
- a failed screen restarts preparation from the beginning.

A content-addressed cache key should include at least:

- canonical isomeric SMILES;
- molecule ID and deterministic base seed, if identity affects coordinates;
- preparation mode and atom/torsion limits;
- RDKit and Meeko versions;
- relevant preparation implementation/version hash.

The durable cache should store compact immutable objects and status metadata.
Hot runs should materialize files into node-local scratch efficiently, avoiding
millions of random small-file reads from a shared filesystem. Negative caching
of deterministic preparation failures can prevent repeated multi-second
fallbacks, but only when the complete key and failure policy match.

## Bottleneck 9: Uni-Dock fast contains a second CPU/setup ceiling

The current one-GPU fast log divides approximately as follows:

| Uni-Dock component | Time |
| --- | ---: |
| CUDA search kernels | 213.519 s |
| serial post-search pose/refinement/rescoring loop | 21.431 s |
| complete internal classified batches | 280.429 s |
| external Uni-Dock process invocation | 287.111 s |

The log label `poses saveing time` is misleading for score-only work. The
associated source loop removes redundant poses, performs explicit-receptor
refinement/rescoring where enabled, sorts results, computes RMSD fields, and
prepares outputs one ligand at a time after GPU search.

The same run split 19,841 prepared ligands into size classes and sequential
internal GPU sub-batches:

- 14,338 small ligands in batches of 5,604, 5,592, and 3,142;
- 5,339 medium ligands in batches of 3,155 and 2,184;
- 158 large ligands;
- 6 extra-large ligands.

The sparse large and extra-large tails require separate underfilled launches.
Class and batch processing is sequential, and CPU postprocessing of the current
batch is not overlapped with GPU search for the next one.

In `balance` and `detail`, CUDA search dominates and this overhead is only a
small percentage of engine wall. In `fast`, it is material because the kernels
finish quickly.

Candidate engine changes:

- double-buffer setup and postprocessing against the next GPU search;
- parallelize independent per-ligand refinement/rescoring after removing or
  isolating shared mutable state;
- reuse receptor data, device allocations, and transfer buffers across
  internal and outer batches;
- reduce sparse size-class tail launches, potentially with safe mixed packing
  or concurrent streams;
- specialize score-only output to avoid pose serialization and RMSD work that
  is not needed by the result contract;
- keep one engine process resident rather than reparsing the receptor and
  rebuilding state for every outer batch.

Every change requires exact outcome, failure-yield, score, ranking, and
determinism checks. Removing refinement is a different scientific protocol,
not an implementation optimization.

## Bottleneck 10: the automatic tuner measures the wrong boundary

The docking tuner prepares one pilot set and times `dock_batch_resilient`.
It therefore selects raw engine batch throughput without including:

- cold validation and ligand preparation;
- the first-batch barrier;
- preparation/GPU overlap;
- process/container startup;
- pose policy and result writing;
- requested total molecule count;
- tail-batch efficiency;
- retries and failure recovery.

It also uses score-only timing even if the intended run writes poses. A batch
profile is validated primarily by engine name; it is not strongly keyed by
GPU, target, box, search mode, chemistry, output policy, driver, or software
artifact hashes.

The replacement tuner should minimize predicted invocation-to-durable-result
wall for the actual `N`. Its profile key should include hardware topology,
engine/binary hash, target/receptor/box, scoring and search protocol, ligand
complexity sample, preparation versions, output policy, and scratch class.
Bounded calibration should run only when its cost amortizes for `N`.

## Bottleneck 11: AutoDock-GPU defaults leave the GPU underfilled

Without a profile, AutoDock-GPU resolves to an outer batch of 512 and one
same-GPU process. The measured GH200 optimum was 4,096 ligands with six
independent processes under CUDA MPS and four CPU threads per process.

Measured exploration showed approximately 5.0--5.5 successful molecules/s for
one process and 12.45/s for six MPS processes at batch 512. The corrected-map
batch sweep improved from 11.098/s at 512 to 12.084/s at 4,096. These GH200
values must not be hard-coded on other hardware, but they prove that the safe
generic default is not a maximum-speed default.

The ordinary wrapper does not start and own an MPS control daemon. More than
one worker requires a pre-existing MPS environment, making the tuned mode
nonautomatic. Each outer batch also creates fresh host threads and fresh
AutoDock-GPU subprocesses, and static strided work assignment leaves some
worker imbalance.

Required design:

- probe whether MPS is supported and beneficial on the current GPU;
- create a run-private MPS directory and lifecycle when qualified;
- calibrate worker count, CPU threads, work-item width, and outer batch as a
  joint configuration;
- keep workers persistent and feed them from a dynamic queue;
- use chemistry/complexity-aware load balancing when it preserves semantics;
- fall back safely to one process where MPS is unavailable or slower.

## Bottleneck 12: pose output performs redundant work

The regular command defaults to `balance` with merged poses, while the maximum
speed benchmark uses `fast` with no persistent poses. These are different
products and must be reported separately.

For merged output, the current path allows the engine to write each pose file,
reads the file to parse scores, then reopens and reads it again to append to the
merged pose file. Individual-pose mode additionally creates or copies one file
per molecule, causing severe inode and metadata pressure at large scale.

Potential improvements:

- parse and append each produced pose in one pass;
- move result/pose writing to an asynchronous bounded writer;
- provide a direct engine-to-merged stream where recoverability permits;
- keep score-only as the explicitly named maximum-throughput product;
- benchmark merged and individual pose contracts independently.

The planner must never silently change a requested pose policy to inflate the
headline rate.

## Bottleneck 13: restart is stage-level, not batch-level

The upstream screen requires a fresh output directory. The wrapper reuses a
complete manifest, but an incomplete directory is renamed and all completed
batches are recomputed. This becomes costly for multi-million-molecule jobs and
lowers effective throughput whenever a node, process, or filesystem fails.

Add transactional batch partitions containing:

- logical shard and batch identity;
- input range or identity digest;
- preparation/cache keys;
- protocol and artifact hashes;
- result count, failure count, and checksum;
- terminal commit marker.

Resume should validate committed partitions, reconstruct the pending queue,
and continue without changing deterministic seeds or emitting duplicates.

## Bottleneck 14: fixed-shard merge is serial and memory-heavy

The fitting pipeline loads the entire source CSV into a Python list, allocates
a result list of the same length, reads every shard into dictionaries, writes a
single CSV, fsyncs it, and hashes complete files. This is acceptable at the
current 100K--300K reference scale but will not scale cleanly to very large
docking campaigns.

A scalable merge should stream ordered shard partitions or use an external
merge keyed by source row. Partition manifests can preserve source order and
permit zero-copy concatenation when each partition covers a known ordered
range. Hashing can use per-partition Merkle-style manifests rather than
re-reading a monolithic output.

## Bottleneck 15: small fixed overheads accumulate across shards

Each upstream process probes Uni-Dock, AutoDock-GPU, AutoGrid, and iGen3 even
when only one docking engine and an external source are used. Target creation
also prepares AutoDock grid assets even for a Uni-Dock-only run. These are not
dominant in a sustained screen, but repeated logical shard processes and short
active-learning batches multiply them.

Tool and target work should be lazy and keyed by the selected engine. Immutable
version/artifact metadata can be verified once by the supervisor and inherited
by persistent lanes.

## Bottleneck 16: node-local scratch and topology are performance features

The upstream screen sensibly prefers `SLURM_TMPDIR` and then `/tmp`, but an
explicit scratch path may place millions of small PDBQT and pose files on a
shared filesystem. Published iGenVS runs also showed that CPU/GPU binding was
critical: a short unbound/underallocated four-GPU run achieved 185.47
successful/s, while 72 cores per GPU with binding achieved 307.13/s. A larger
two-batch-per-GPU workload reached 370.42/s through additional amortization.

The portable planner should:

- prefer capacity-checked node-local storage;
- estimate temporary bytes/inodes before accepting a plan;
- bind each GPU lane to nearby physical CPU cores and memory;
- avoid sharing a preparation pool across distant NUMA nodes;
- record the resolved topology and storage class in the manifest.

## Scientific and correctness constraints

The following changes are not interchangeable speed optimizations:

- `balance` to `fast` search;
- merged or individual poses to score-only output;
- standard to fast ligand preparation;
- refined to no-refinement docking;
- Vina scoring to AD4 scoring;
- changing target, box, seed, ligand policy, or retry behavior.

The published fast-preparation comparison did not improve full throughput and
changed ranking materially: Spearman rho was 0.771 and exact top-1% overlap was
39.9% relative to standard preparation. Standard preparation must remain the
qualified default unless a separate scientific protocol is explicitly chosen.

Implementation-level concurrency can also change outcomes if engine RNG is
batch-position-dependent. Repartitioning, mixed size groups, dynamic ordering,
or persistent processes must be tested for:

- identical admitted identities and source association;
- preparation success/failure equivalence;
- finite docking yield;
- score equality or a predeclared numeric tolerance;
- rank correlation and top-fraction overlap;
- deterministic restart and cross-hardware behavior where promised.

## Original prioritized implementation blueprint

The checkpoint and completed benchmark above supersede this original ordering.
Automatic GPU resolution and fan-out, shared validation, bounded preparation
and ramp overlap, physical-core-aware planning, lazy probes, and managed
AutoDock-GPU MPS are implemented. A prepared-ligand cache, persistent Uni-Dock
engine state, deeper asynchronous output, and telemetry remain follow-on work.

### Original P0 targets for the full regular-docking benchmark

1. Fix visible-GPU resolution and record process-local and physical identities.
2. Add automatic multi-GPU standalone docking with one persistent lane per GPU.
3. Perform validation/global dedup once and partition the admitted stream once.
4. Build a persistent, NUMA-aware preparation service with bounded queues.
5. Start with a ramp batch and isolate slow ETKDG fallback work from the normal
   preparation path.
6. Add a deterministic prepared-ligand cache, including negative cache entries
   for fully keyed deterministic failures.
7. Replace logical-CPU worker heuristics with physical-core/RAM/topology-aware
   planning that is safe on different hardware.
8. Make tuning end-to-end and workload-aware, keyed by the scientific and
   hardware configuration.
9. Automatically qualify and manage AutoDock-GPU MPS when that engine is used.

### P1: important fast-mode and resilience work

1. Keep Uni-Dock state resident across outer batches.
2. Overlap Uni-Dock CPU setup/postprocessing with subsequent CUDA search.
3. Parallelize safe per-ligand refinement/rescoring and reduce sparse class
   tails without changing results.
4. Reuse device and host buffers.
5. Add transactional batch checkpoints and true resume.
6. Use one-pass asynchronous score/pose persistence.

### P2: scaling and polish

1. Stream or partition-aware merge instead of whole-dataset Python lists.
2. Lazy-probe only required tools and lazily build engine-specific target
   artifacts.
3. Use compact partition formats internally while retaining requested exports.
4. Add continuous telemetry for queue depth, GPU starvation, preparation
   percentiles, scratch traffic, and per-lane utilization.

## Automatic planning objective

The planner should minimize predicted invocation-to-durable-result time for
the requested molecule count subject to the selected scientific protocol. A
useful model is:

```text
T_total = T_fixed
        + max over GPU lanes(
              T_ingress_not_hidden
            + T_preparation_not_hidden
            + T_engine
            + T_output_not_hidden
          )
        + T_final_commit
```

It should plan per GPU rather than applying one global batch to heterogeneous
devices. Calibration is justified only when expected savings exceed its cost.
The first invocation can use conservative architecture-class priors, bounded
micro-calibration, and online measurements; subsequent runs can reuse a
strongly keyed profile.

The ordinary interface should not require users to understand preparation
workers, batch sizes, MPS processes, CPU affinity, or logical shards. Advanced
overrides can remain diagnostic, but every automatic decision must be written
to a plan/manifest for reproducibility.

## Completed benchmark and optional extensions

The requested release benchmark is complete: 20,000 ligands/GPU, identical
4ag8/base-isomeric chemistry, one/two/four GPUs, Uni-Dock
`fast`/`balance`/`detail`, AutoDock-GPU `fast`, score-only output, automatic
public wrappers, and exactly one cold sample per case. It records:

- all input rows/hour by complete scheduler or launcher wall;
- successful finite scores/hour end to end;
- engine-only successful scores/hour;
- generation, validation, preparation wait/CPU, docking, result, and commit
  timings;
- exact target, chemistry, search, scoring, pose, preparation, refinement, and
  artifact identities.

See [`speed-bench/REPORT.md`](../speed-bench/REPORT.md). Larger sustained runs,
warm prepared-ligand-cache trials, pose-writing products, preparation latency
percentiles, and hardware utilization telemetry remain useful optional
extensions; they are not missing cases from the requested cold benchmark.

Do not compare the current base-isomeric 4ag8 20K/GPU result directly with the
published RL-nonisomeric 1iep 65,536/GPU result without preserving both fixture
and timing-boundary differences.

## Observed improvement and remaining headroom

For the current four-GPU fast fixture, complete-wall input throughput improved
from the historical 570,847/hour to 901,859/hour. The completed run's engine
boundary is 931,350 successful/hour, versus 887,113 successful/hour end to end.
The remaining outer-orchestration gap is therefore about 4.7% on the successful
rate boundary, not the much larger gap diagnosed before optimization.

A larger sustained workload or a warm prepared-ligand cache may improve
amortization, but those are different protocols from the completed cold run.
Validation, unavoidable preparation, transfers, refinement, result durability,
failures, and terminal tails cannot all disappear.

For `balance` and `detail`, the engine consumes most wall time, so outer
orchestration provides little percentage headroom. Uni-Dock kernel work or
scientific search settings dominate those modes. AutoDock-GPU likewise spends
almost all tuned wall inside the engine; the automatic wrapper now selects and
activates its qualified concurrent execution mode.

## Final decision

The requested cold 1/2/4-GPU benchmark is complete and is the definitive
release result for this fixture and protocol. Preserve score-only and
pose-writing products as separate protocols in any extension. Prepared-ligand
caching, longer sustained runs, persistent-engine experiments, deeper NUMA
calibration, and Uni-Dock internal setup work are optional follow-on
optimization rather than outstanding parts of this benchmark.

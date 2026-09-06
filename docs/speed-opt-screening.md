# iGenVS-ultra maximum-speed screening design

Status: Phase-1 persistent orchestration and portable automatic planning are
implemented and smoke/medium-qualified on one GH200. The full 1/2/4-GPU speed
benchmark and the deeper shared-memory/kernel phases remain pending.

## Implementation checkpoint (2026-09-06)

The first optimization pass is now active in `user-pipeline`:

- `screen-fast N` is the count-only entry point inside a completed target job.
- Every selected GPU owns one persistent iGen3 process and one persistent
  gMolAI/three-head process for the screen lifetime.
- iGen3 generation, gMolAI encoding, and head inference all use every selected
  screening GPU; CPU affinities are disjoint between GPU lanes.
- Generated surplus is retained between stream blocks, and completion is
  defined by exactly `N` successfully encoded and scored molecules. Encoder
  rejections are replenished automatically.
- Generated default-policy input bypasses the redundant iGenVS validation
  container, exact dedup uses an in-memory set plus one bulk SQLite commit, and
  final CSV partitions are concatenated without reparsing every row.
- iGen3 canonicalization is handled by a persistent, order-preserving RDKit
  process pool. Its size is derived from physical cores inside each lane's CPU
  affinity (one core reserved, maximum 64); smoke-size work stays serial.
- Generator batch selection is bounded by current free GPU memory, workload,
  the user safety ceiling, and the current CUDA SDPA launch limit. Capacity
  failures including OOM and invalid kernel configuration halve and retry.
- Expensive four-point generator calibration runs only when its proposal work
  is below approximately 1% of the requested per-GPU workload. Ordinary runs
  choose the largest safe batch immediately, avoiding a non-amortizing cold
  benchmark.
- The gMolAI encoder measures the released qualified batches
  64/128/192/256/512 on a representative sustained block and caches the fastest
  successful value. Tiny-block measurements cannot poison a sustained profile.
- Profiles are keyed by GPU architecture/resources, CPU model/topology and
  affinity, Python/PyTorch/CUDA/RDKit/backend versions, model artifacts,
  scientific generation settings, and worker topology. A cached generator
  choice is rechecked against current free memory before use.

The final one-GH200 all-score medium check produced exactly 200,000 durable
scores in 75.17 seconds (9.58 million/hour) with a warm encoder profile. The
fresh encoder-profile run took 85.16 seconds. The prior best automatic medium
run took 104.37 seconds, so this pass reduced reusable wall time by 28.0% and
increased throughput by 38.8%. A compile probe produced identical output but
was slightly slower end to end, so compilation is not enabled automatically.
Detailed evidence is in `user-pipeline/benchmarks/RESULTS.md`.

Implemented here does not mean the complete blueprint below is finished.
Shared-memory transport, cross-stage queue overlap, fused GPU
encoder/calibrator/head execution, reusable KV buffers, asynchronous token
transfer, and decoder-kernel research remain the next optimization phases.

## Goal

The final-screening interface should require exactly one user value:

```text
igenvs-ultra screen-fast 10000000
```

The value is the number of molecules the user wants **successfully screened**.
Everything else should be discovered or inherited automatically: target-head
artifacts, generation policy, visible GPUs, CPU allocation, NUMA topology,
memory, scratch, execution batches, worker counts, precision, compilation,
queue depths, checkpoint cadence, output layout, and tail handling.

This design assumes the command is run inside, or points implicitly to, an
already fitted target job. A target and target-specific head cannot logically
be inferred from a molecule count. Target preparation, reference docking,
active learning, and fitting are one-time prerequisites and should be reused;
the count-only interface begins at the large final iGen3 screen.

The optimization objective is:

> Minimize invocation-to-durable-result wall time for exactly `N` globally
> unique, policy-valid, successfully encoded and scored molecules, without
> changing the frozen generator settings, gMolAI representation, or target
> model merely to claim a higher speed.

## Executive conclusion

Ultra should become a persistent streaming service for the lifetime of one
screen, not a launcher that repeatedly executes short container commands.

The desired data plane is:

```text
                                      one long-lived supervisor
                                                 |
                       +-------------------------+-------------------------+
                       |                         |                         |
                 GPU worker 0              GPU worker 1             GPU worker G-1
              iGen3 + gMolAI + heads     iGen3 + gMolAI + heads          ...
                       |                         |                         |
                       +------ raw-token blocks in shared-memory queues --+
                                                 |
                           persistent topology-bound CPU chemistry pool
                decode -> policy/canonicalize -> exact global dedup -> graph pack
                                                 |
                           accepted packed-graph blocks with backpressure
                                                 |
                    GPU gMolAI -> standardize -> three fused heads, without
                       materializing 384-D embeddings in host memory
                                                 |
                         asynchronous partitioned result/checkpoint writer
```

Each GPU worker should load iGen3, gMolAI, the gMol calibrator, the input
standardizer, and all three target heads once. It should retain its CUDA
context, KV-cache buffers, graph preprocessing machinery, and model weights
until completion. The supervisor should schedule short generation and scoring
quanta according to queue pressure rather than destroying and recreating the
models between stages.

Once process churn is removed, generation should dominate. One GH200 has
already demonstrated about 2,076 valid unique base-isomeric output SMILES/s in
the local 98,304-row fixture, whereas the published hot gMolAI encoder reaches
58,330 molecules/s on **one** GH200 at batch 512. Target-head forward inference
is faster again. The final pipeline should therefore approach the sustained
globally unique iGen3 rate, not spend tens of minutes starting encoders.

## What the current measurements say

The locked 10-million-molecule results are in
[`speed-bench/REPORT.md`](../speed-bench/REPORT.md). The four-GH200 run measured:

| Quantity | Current result |
|---|---:|
| End-to-end wall | 9,197.00 s |
| End-to-end screening | 3.914 million/h |
| Generation critical path | 6,195.86 s |
| Parallel score-stage wall | 2,320.07 s |
| Remaining validation/dedup/finalization wall, approximately | 681.07 s |
| Reported encoding critical path | 927.63 s |
| Reported head-forward critical path | 11.27 s |
| Locally unique generated rows needed for 10M globally admitted | 10,663,693 |
| Successfully encoded rows | 9,999,995 |

The approximate end-to-end fractions are 67.4% generation, 25.2% scoring, and
7.4% other orchestration. Pure head forward is only about 0.12% of wall time.
This makes the optimization order clear: remove process and data-boundary
overhead, then optimize iGen3. Optimizing the tiny head forward in isolation
cannot materially change end-to-end speed.

The current run is not measuring intrinsic model limits:

- Four logical iGen3 shards across 110 outer batches caused **440 fresh iGen3
  processes**. This is also 440 processes on one GPU, where four are run
  serially for every outer batch.
- The four-GPU score path caused **436 fresh scoring processes**. Each process
  verifies and hashes artifacts, reloads gMolAI, reads CSV, starts a policy
  pool, starts a graph pool, reloads three target heads, writes and hashes
  output, shuts down its pools, and empties CUDA state.
- Validation adds another 110 container commands. The four-GPU screen therefore
  crosses roughly 986 short process/container boundaries in its hot path.
- The current generator auto-tuner selects the largest cache allocation that
  fits, capped at 32,768. It does not select the batch with the best accepted
  molecules per second.
- The current generator requests a full internal candidate batch even when
  considerably fewer accepted molecules remain, then discards valid surplus.
- The execution is strictly `generate -> validate -> admit -> score` for every
  outer batch. There is no cross-stage overlap.
- The speed benchmark uses `--score-threshold 1.0`, so almost no full result
  rows are saved. A production benchmark that promises scores for all `N`
  molecules must include durable result writing in its primary wall time.

The source confirms the boundaries:

- [`generation.py`](../iGenVS/iGen3/src/igen3/generation.py) allocates KV caches
  per batch, synchronously decodes, canonicalizes and writes each result, and
  checks `bool(finished.all())` from the host inside the token loop.
- [`workflow.py`](../user-pipeline/src/igenvs_ultra/workflow.py) launches fresh
  generation, validation, and score commands and serializes several CSV
  representations per outer batch.
- [`model_ops.py`](../user-pipeline/src/igenvs_ultra/model_ops.py) starts its
  reported encoder timer before artifact verification, model loading, CSV
  parsing, and policy-worker creation. It then closes the encoder and empties
  CUDA before loading and executing the heads.

Reference hot-path evidence is deliberately kept separate from the current
end-to-end metric:

| Reference | Hardware | Boundary | Rate |
|---|---|---|---:|
| Local iGen3 base-isomeric fixed library | 1 GH200 | valid unique output including RDKit work, model already loaded | 2,076.5/s |
| Published iGen3 base-isomeric | RTX PRO 2000 Blackwell Laptop, 8 GB | stored benchmark generation boundary | 1,463.1/s |
| Published gMolAI optimized, batch 512 | 1 GH200 | warmed canonical SMILES in RAM to FP32 host vectors | 58,330.4/s |
| Current ultra head forward | 4 GH200 aggregate | forward-only critical path | about 887,500/s |

The iGen3 result is documented in the
[`iGen3` benchmark](https://github.com/Jalil-Mahdizadeh/iGen3/tree/main/benchmarks/latest),
and the gMolAI result and timing boundary are documented in
[`extra-benchmark/speed/RESULTS.md`](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/extra-benchmark/speed/RESULTS.md)
and [`PROTOCOL.md`](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/extra-benchmark/speed/PROTOCOL.md).
These rates are not directly interchangeable, but they prove that the current
ultra rates are dominated by orchestration rather than the frozen models.

## Correctness contract

Speed must be optimized subject to a fixed contract. The planner may change
execution, not scientific meaning.

1. `N` means exactly `N` successfully scored molecules, not `N` generated,
   locally unique, admitted, or attempted molecules.
2. Every committed molecule is valid under the released policy and globally
   unique by exact canonical isomeric SMILES.
3. Encoding failures do not reduce the final count. The producer replenishes
   until `N` scores have been durably committed.
4. Every identity is paired with the correct three target-head outputs,
   ensemble probability, and mutual information.
5. Model, vocabulary, checkpoint, calibrator, standardizer, and head hashes are
   verified once per run and recorded.
6. A crash and resume cannot duplicate, omit, or silently change committed
   rows. RNG position and queue/checkpoint state must be recoverable.
7. Faster numerical paths are enabled only after their own equivalence record
   passes. No approximate chemistry or precision change is silently selected.
8. The planner never changes generation model, temperature, top-k, isomeric
   policy, maximum sequence length, or target model merely because another
   scientific workload is faster.

The current benchmark stopping at 10,000,000 admissions but producing
9,999,995 encodings is a useful example of why the committed-score counter must
be authoritative.

## The count-only user experience

The normal command should expose no performance knobs:

```text
igenvs-ultra screen-fast N
```

The fitted job supplies the target, final ensemble, frozen generation protocol,
base seed, and result schema. The runtime automatically:

- discovers the GPUs actually granted to the process, including Slurm/cgroup
  limits rather than trusting node-wide visibility;
- discovers CPU affinity, physical cores, NUMA/GPU locality, RAM and scratch;
- validates artifacts and loads a compatible cached performance profile;
- decides how many visible GPUs minimize completion time for this `N`;
- selects a different generator batch and score batch for each heterogeneous
  GPU if necessary;
- decides whether compilation or calibration can amortize for this run;
- creates the output location, resumes an identical interrupted run, and emits
  a complete decision manifest.

Developer-only diagnostic overrides can exist, but they should not be part of
the ordinary user contract. Every automatic choice must remain inspectable in
`plan.json`; automatic must not mean opaque.

The default output contract should also be fixed because the user does not
choose it. For maximum useful throughput, the recommended primary output is a
partitioned Parquet or Arrow dataset containing all `N` identities and scores,
plus a small manifest. A CSV export can be a separate lazy compatibility step.
If the product only needs a fixed top fraction, that can be a different
predeclared product profile, but silently retaining zero rows or only a
thresholded subset is not a valid all-molecule screening benchmark.

## Replace the outer batch with independent execution scales

One outer `stream_batch_size` currently controls unrelated concerns. The new
engine should separate at least five scales:

1. **Generator microbatch:** GPU sequences sampled together.
2. **Chemistry block:** raw strings sent to CPU validation/canonicalization.
3. **Graph batch:** accepted molecules packed for one gMolAI forward, bounded
   by graphs and nodes.
4. **Result partition:** rows written as one durable output object.
5. **Checkpoint interval:** amount of completed work between resumable commits.

These values have different optima. A generator batch may be 8K-32K, a graph
batch around 512, a shared-memory chemistry block a few thousand, and a result
partition hundreds of thousands or millions. None requires a new process.

The old 100K/250K/500K/1M stream-size heuristic can remain only as a fallback
checkpoint/output-partition estimate. It should not determine model lifetime or
GPU kernel batch size.

## Automatic planner

### 1. Inventory

At startup the planner should collect, without asking the user:

- visible GPU UUID, model, compute capability, SM count, VRAM, current free
  memory, MIG status, driver and CUDA runtime;
- PyTorch, RDKit, Python, container, model and source hashes;
- CPU model, affinity/cgroup allowance, physical-core count, NUMA nodes,
  memory bandwidth hints, and `nvidia-smi topo -m`/hwloc locality;
- available host memory, local scratch and output-filesystem capacity and
  measured sequential write bandwidth;
- whether GPUs are heterogeneous, thermally throttled, or unexpectedly busy.

The planner must inspect each GPU independently. Equal shards based on the
smallest GPU waste heterogeneous hardware and make every wave wait for the
slowest lane.

### 2. Profile cache

Reuse a qualified profile keyed by at least:

```text
GPU architecture + SM count + VRAM
driver + CUDA + PyTorch + container hash
iGen3 model/vocabulary/source hashes
gMolAI source/checkpoint/calibrator hashes
target-head architecture and precision mode
CPU model + core allowance + NUMA layout
generation mode and semantic settings
```

Do not invalidate a profile merely because the requested `N` changes; the
planner can use the same measured rates in its completion-time model. Do
invalidate or requalify it when any executable, model, runtime, topology, or
scientific setting changes.

### 3. Optimize time-to-result, not batch size

For each qualified candidate plan, predict:

```text
T_plan(N) = T_boot
          + T_required_calibration
          + T_steady(N)
          + T_tail(N)
          + T_durable_finalize(N)
```

Choose the plan with the lowest predicted wall time. The throughput numerator
must be committed scored molecules, not proposal slots, locally valid SMILES,
or encoded rows.

The planner should enable an optimization only when its predicted saved time
comfortably exceeds its setup cost:

```text
N * (1 / R_old - 1 / R_new) > T_setup + uncertainty_margin
```

This naturally makes small screens skip compilation and extensive tuning,
while large screens pay a one-time cost for the fastest steady-state path.

### 4. Bounded calibration

If there is no exact cached profile:

- use a conservative known-safe configuration immediately;
- spend a bounded fraction of predicted wall time, for example no more than
  1%, comparing a small candidate set;
- warm before timing and use at least two full batches so startup is not
  mistaken for steady-state throughput;
- retain valid production outputs from safe trials when reproducibility rules
  permit, rather than throwing all calibration work away;
- cache the result for later screens.

For tiny `N`, the fastest choice may be one GPU and no tuning. For large `N`,
all healthy granted GPUs will normally win. The decision should be predicted
and then verified online, not hard-coded.

### 5. Online control

Static calibration cannot predict every molecular distribution. Maintain an
exponentially weighted rate and queue-delay estimate for each stage and GPU.
Every few seconds, adjust only cheap controls:

- generation versus encoding duty cycle per GPU;
- CPU workers assigned to decoding/policy/graph construction;
- queue high/low watermarks;
- result partition and checkpoint cadence within memory limits;
- per-GPU work share.

Expensive changes such as recompilation or a precision mode switch should not
occur in the middle of a run unless a documented fallback is triggered.

## Persistent process and container architecture

This is the highest-confidence optimization.

### Preferred architecture

Build one unified ultra runtime containing iGen3, gMolAI, RDKit and target-head
dependencies. Start one supervisor and one persistent CUDA worker per active
GPU. A GPU worker owns exactly one CUDA context and keeps both model families
resident. The models are small relative to GH200 memory; generator KV buffers,
not model weights, should determine the working-set limit.

### Transitional architecture

If combining the two released environments is initially too risky, start one
long-lived iGen3 daemon and one long-lived gMolAI/head daemon per GPU in their
existing containers. Communicate using Unix sockets plus shared-memory block
handles. This still removes almost all Apptainer, Python, import, hash, model
load, and process-pool churn. An Apptainer instance alone is insufficient if
every operation still launches a new Python process and reloads the model.

### Lifetime rules

- Verify each immutable artifact once at supervisor startup.
- Load and warm each model once per GPU.
- Create topology-bound RDKit/graph workers once.
- Preallocate reusable GPU and pinned-host buffers.
- Never call `torch.cuda.empty_cache()` in the ordinary hot loop.
- Tear resources down only after the durable final manifest, an unrecoverable
  failure, or an explicit reconfiguration.

This turns hundreds of load/setup cycles into `O(number_of_GPUs)` setup work.

## iGen3 generation optimization

Generation is the eventual bottleneck and deserves the deepest work.

### Immediate, low-risk changes

#### Keep generators resident

Maintain one loaded base-isomeric generator per active GPU for the full run.
Logical random streams should be state objects inside workers, not separate
processes. Save their RNG state/candidate counters at checkpoints.

#### Tune sustained accepted throughput

The current auto-tuner binary-searches the largest one-step KV allocation that
fits. Replace it with a two-stage tuner:

1. Estimate a safe memory ceiling.
2. Benchmark a short geometric ladder below that ceiling, such as nearby
   4K/8K/12K/16K/24K/32K values, using complete warmed sequences.

Select the batch minimizing expected time to globally new policy-valid output,
including token transfer/decode and CPU chemistry backpressure. The fastest
batch is not necessarily the largest that fits. Keep several qualified tail
batches as well as the main steady-state batch.

#### Do not discard surplus

Generation should produce candidate blocks into a persistent queue. Valid
unique molecules beyond the current result-partition boundary remain queued
for the next partition. This removes the present full-batch rounding waste.

Near completion, forecast proposal demand from the observed committed yield:

```text
p_commit = scored_unique / proposal_slots
next_proposals ~= remaining_scores / lower_confidence_bound(p_commit)
```

Shrink the reservation margin as `remaining_scores` approaches zero. Keep the
fastest prequalified partial-batch size unless retaining a small amount of
surplus is faster than a poorly utilized tiny GPU batch.

#### Remove token-loop host synchronization

`bool(finished.all())` inside every autoregressive position forces a device to
host decision. With a large batch, at least one sequence usually reaches the
maximum length, so the checks add synchronization without ending early. Use
one of:

- a fixed-length device loop for the initial implementation;
- checks only at coarse positions;
- a device-resident active-count flag consumed without synchronizing every
  token;
- the active-slot design below.

Measure this change independently because it is simple and potentially large.

#### Reuse buffers

Preallocate and reuse outputs, finished flags, SOS vectors, position data, KV
caches and pinned transfer buffers for the selected batch-size ladder. The
current allocator creates cache tensors for every generated batch. CUDA's
caching allocator helps, but persistent typed buffers remove Python object,
allocation, fragmentation and initialization work.

#### Decouple GPU sampling from CPU chemistry

The current iGen3 writer synchronously transfers tokens, decodes Python rows,
runs serial RDKit canonicalization, performs local deduplication and writes a
file before the next GPU batch. Instead:

- copy compact token buffers into pinned host memory asynchronously;
- hand them to a persistent CPU decoder/chemistry pool;
- immediately launch the next GPU generation quantum when a free host buffer
  exists;
- perform global uniqueness once downstream.

Vocabulary size is below 256, so retained output tokens can use `uint8` rather
than copying an `int64 [batch, max_len]` result, provided conversion is verified.
A vectorized/native token decoder can replace the nested Python row/token loop.

### High-potential decoder work

These ideas require measurement and a distribution-equivalence audit.

#### Active-sequence compaction

Finished sequences currently continue through every layer until all sequences
finish. At large batch sizes the longest outlier can keep tens of thousands of
already-finished rows active. Periodically compact unfinished rows, preserve
their original output indices, and operate on smaller KV views. Good checkpoint
positions can be selected from observed length quantiles, for example after
32/48/64/96 tokens. Compacting every token is likely too expensive.

#### Continuous slot recycling

The strongest version replaces finished slots immediately with new candidate
IDs, keeping a fixed number of active sequences instead of running rectangular
batches to the longest sequence. This needs ragged or paged KV-cache kernels
and per-slot positions, but it can remove both finished-row waste and tail
batches. It is likely the largest kernel-level opportunity.

#### Fused decoder kernels

- Concatenate the separate Q/K/V projections into one GEMM using the unchanged
  weights.
- Fuse top-k, temperature, Gumbel/categorical sampling and argmax for the small
  vocabulary instead of launching several general PyTorch operators per token.
- Fuse residual, normalization, activation and cache-update operations where
  numerical qualification allows.
- Evaluate a specialized single-query attention kernel rather than assuming
  the generic scaled-dot-product path is optimal at very large batch and short
  sequence length.

#### Compilation and CUDA graphs

Benchmark eager, `torch.compile` modes, and CUDA graphs on each supported
architecture. The current Python `pos_idx` can cause position specialization,
and graph-captured random sampling needs graph-safe RNG handling, so simply
turning compilation on is not automatically faster. Cache compiled artifacts
and include compilation cost in the `N`-dependent plan.

#### Precision candidates

BF16 is already the normal CUDA choice where supported. FP16, TF32 choices,
and Blackwell FP8 may be explored, but FP8 is not a default optimization: it
can alter token probabilities and therefore chemistry. It needs distribution,
validity, novelty, descriptor and downstream-score qualification, not merely a
kernel timing.

### Reproducible asynchronous sampling

The best long-term RNG design is counter based: derive every stochastic draw
from `(base_seed, global_candidate_id, token_position)`. Then GPU count,
microbatch size, compaction and work stealing do not change candidate identity.
This likely requires a custom Philox-based sampling kernel. Until that exists,
persist one RNG stream per fixed lane, record the chosen plan, and guarantee
resume identity only under the same qualified plan.

## Chemistry, validation and exact deduplication

An accepted molecule currently crosses several redundant chemistry boundaries:

1. iGen3 parses and canonicalizes it for local valid-unique output.
2. iGenVS validates and canonicalizes it again.
3. gMol policy parses it, canonicalizes it, and reparses the canonical string.
4. gMol graph construction parses the canonical string again.

Depending on the exact path, that is up to five RDKit parses plus multiple
string/file round trips. The optimized engine should have one authoritative
chemistry pipeline.

### Fused chemistry worker

A persistent worker should receive decoded raw SMILES and perform:

```text
parse/sanitize
-> fragment and allowed-element policy
-> atom-count policy
-> canonical isomeric SMILES
-> exact identity key
-> graph features/packed arrays for newly admitted identities
```

Ideally the same RDKit molecule is reused for graph packing. If canonical atom
ordering is required to reproduce the frozen encoder exactly, retain only the
necessary canonical reparse and reuse that reparsed molecule for graph
construction. Prove equivalence on the existing policy-edge corpus and a large
generated fixture before removing any reparse.

Do not serialize RDKit molecule objects between processes. Parse and pack in
the same persistent worker, returning canonical bytes, counts and compact graph
arrays. If building graphs before the central dedup wastes too much work, use a
two-phase worker protocol; if duplicates remain near the observed 6%, doing the
graph work once per proposal may be faster than another IPC round trip. Let the
planner benchmark both.

### Exact global uniqueness

For the common 10M scale, an in-memory exact canonical-string set or a native
robin-hood hash table will be far faster than one Python-to-SQLite call per row.
The durable state should be an append-only checkpoint journal or partition
manifest, not a synchronous database lookup for every molecule.

For screens too large for the configured RAM fraction:

- shard identities by a stable 128-bit/256-bit digest;
- use a disk-backed LSM/LMDB/RocksDB-style exact store;
- place a Bloom filter in front only as a safe negative lookup accelerator;
- resolve every possible hash collision against canonical bytes.

A Bloom filter must never be the authority because false positives would
silently discard unique molecules. Python's randomized hash must not define a
reproducible durable identity.

### Concurrent deterministic admission

Assign every proposal a monotonically increasing global candidate ID. The
dedup coordinator defines the lowest candidate ID as the owner of a duplicated
canonical identity and emits accepted rows in candidate-ID order. A bounded
reorder window permits asynchronous CPU work without making results depend on
worker completion order.

Use two-phase states—reserved, scored, committed—so a crash after dedup but
before scoring does not permanently consume an identity or reduce the final
count. Resume should finish pending reservations before generating more.

## gMolAI encoding optimization

### Keep the exact proven implementation hot

The ultra checkout contains the same optimized `fast_graph.py`,
`fast_inference.py`, checkpoint and calibrator hashes as the published
58,330/s run. Start with that implementation rather than rewriting the GNN.

For each GPU:

- validate/hash and load once;
- create and warm the RDKit graph pool once;
- retain the pool across every block;
- begin with batch 512 and the existing 16,384-node safety limit;
- tune graph-count and node budgets on representative generated chemistry;
- form node-balanced batches and restore result order;
- use asynchronous packed-batch prefetch so CPU workers prepare `k+1` while
  the GPU executes `k`.

Batch 512 is a strong starting point, not an unconditional constant. Generated
iGen3 chemistry can have a different atom-count distribution from the locked
49,844-row panel. The planner should test 256/384/512 and larger qualified
candidates where memory permits, optimizing complete packed-graph throughput.

### Eliminate host embedding round trips

This is the most important encoder specialization for ultra. The generic
encoder currently:

1. computes a GPU raw embedding;
2. copies 384 floats per molecule to CPU;
3. applies gMol calibration in NumPy;
4. applies the target input standardizer in NumPy;
5. copies the same 384 floats back to GPU three times for separately loaded
   ensemble heads.

For 10M molecules, one 384-D FP32 matrix is 15.36 GB. Moving and
materializing it repeatedly is unnecessary because ultra does not save
embeddings by default.

Create an `encode_and_score` path:

```text
packed graph -> gMol raw embedding on GPU
             -> gMol calibration on GPU
             -> frozen input standardization on GPU
             -> three resident target heads on GPU
             -> copy only IDs and score columns to host
```

The optional embedding-retention workflow can keep the generic host-vector
path. The fast default should never construct a full host embedding matrix.

### Pinned buffers and streams

Use a small ring of reusable pinned graph buffers. On each GPU, overlap:

- host packing of batch `k+2`;
- nonblocking H2D for batch `k+1`;
- GINE and head compute for batch `k`;
- nonblocking D2H of the preceding score vectors.

One or two transfer streams plus one compute stream should be benchmarked.
More streams can add contention and should not be assumed faster.

### Encoder numerical experiments

The historical TorchInductor encoder was slower and added drift, so it should
not be revived by default. New PyTorch/CUDA architectures may change that
result; re-test behind equivalence gates. FP16/BF16/TF32 or alternative scatter
kernels are second-wave experiments and require downstream score/rank
qualification, not just embedding cosine similarity.

## Target-head inference optimization

Head forward is already fast, but its surrounding setup is wasteful.

- Load all three 263,042-parameter checkpoints once.
- Verify their hashes once.
- Stack ensemble weights and execute them with batched GEMMs/`vmap`, or build
  a mathematically equivalent fused ensemble module.
- Transfer each embedding zero times in the fused encoder-to-head path; if a
  host matrix is supplied, transfer it once rather than once per member.
- Apply the frozen input standardizer on GPU.
- Compute only the classifier output during screening; the training-only rank
  output is not needed.
- Compute sigmoid, mean probability, entropy and mutual information in the
  same GPU batch, returning only final/member score columns.

The current forward-only speed means this work mainly removes model loading and
data movement. It should not take priority over persistent generation,
chemistry or encoding.

## Stage overlap and GPU scheduling

The current wall is approximately a sum of stage walls:

```text
T_current ~= T_generate + T_validate/dedup + T_score + T_files
```

With bounded queues, the steady-state wall approaches the slowest service:

```text
T_target ~= T_startup + max(T_generation_service,
                            T_chemistry_service,
                            T_encoding/head_service,
                            T_writer_service)
                         + T_tail
```

### CPU/GPU overlap

While GPUs generate block `k+1`, CPUs should decode, validate, canonicalize,
deduplicate and graph-pack block `k`. While GPUs score accepted graphs, CPUs
should continue chemistry on already transferred token blocks and write prior
scores. Bounded queues provide backpressure rather than allowing RAM to grow
without limit.

### Generation versus encoding on the same GPUs

The proven one-GPU encoder rate is tens of times higher than base-isomeric
generation. Permanently reserving one of four GPUs for encoding could sacrifice
25% of generation capacity even though the encoder needs only a small duty
cycle. Better candidates are:

1. Keep both models resident and alternate short generate/score quanta on each
   GPU according to graph-queue pressure.
2. Let all GPUs generate until the graph queue reaches a high watermark, then
   let all GPUs drain it quickly.
3. Smooth this asynchronously: temporarily switch the GPU with the largest
   local scoring backlog while the others continue generating.
4. Benchmark concurrent low-priority encoding streams only after exclusive
   alternation; simultaneous kernels may reduce iGen3 throughput more than the
   overlap saves.

The controller should estimate the required score duty cycle from measured
rates. With approximately 2K generated outputs/s and 58K encodings/s on one
GPU, the nominal encoding duty is only a few percent before chemistry/output
costs. Queue feedback, not a fixed GPU reservation, should decide it.

### No global barriers

Do not make heterogeneous GPU workers finish equally sized waves before any can
continue. Give each GPU its own calibrated batch and weighted candidate-ID
range. Shared queues and deterministic IDs preserve ordering while fast GPUs do
more work. Synchronize only at durable checkpoint/final completion boundaries.

## CPU, NUMA and memory placement

The four-GH200 node exposes abundant CPU cores, but contiguous CPU-number
partitions are not guaranteed to match GPU locality.

- Discover GPU-to-NUMA topology and bind each GPU worker, pinned buffers, and
  its graph workers to the closest cores/memory.
- Keep BLAS/OpenMP nesting at one inside RDKit workers.
- Reserve supervisor/writer cores rather than allowing graph pools to consume
  every core.
- Merge validation and graph-construction pools where that avoids duplicate
  RDKit initialization and parsing.
- Dynamically allocate CPU workers according to queue pressure, within fixed
  affinity sets; do not spawn or destroy workers to resize every block.
- Allocate shared-memory rings on the NUMA node that consumes them.
- Use large pages only if a measured packing/transfer benefit justifies their
  operational complexity.

The optimal number of RDKit workers is a throughput measurement, not always 48
per GPU. Four simultaneous 48-worker pools may be ideal for a brief encoder
drain, while fewer combined chemistry workers may be enough during generation.

## IPC and data representation

The hot path should not contain:

```text
generated.smi -> validated.csv -> prepared.csv -> per-GPU CSV shards
-> host embedding NPZ/arrays -> per-shard scores.csv -> merged results.csv
```

Use bounded shared-memory rings or memory-mapped Arrow-compatible blocks. Pass
small descriptors through Unix sockets/queues; do not pickle the payload. A
minimal block carries:

- candidate IDs and RNG provenance;
- compact token bytes or raw SMILES byte offsets;
- canonical SMILES byte offsets and policy status;
- packed node/edge arrays and source-order indices;
- score vectors and commit status.

Files are checkpoint/result products, not inter-stage transport.

## Result writing and finalization

For maximum useful speed:

- write independent ordered Parquet/Arrow partitions asynchronously;
- select compression by a tiny storage microbenchmark—uncompressed Arrow or
  Snappy may beat stronger compression on fast local storage;
- compute each partition digest while writing rather than rereading it;
- derive a Merkle-style manifest root from partition digests;
- do not concatenate all partitions into one final file;
- avoid Python `csv.DictWriter` and per-row float formatting for large screens;
- fsync durable partitions/checkpoints, not every transient shard file;
- record rejection counts by reason and write detailed rejections in compressed
  partitions asynchronously.

If CSV compatibility is mandatory, use a vectorized/native writer and produce
partitioned CSV. A single final merged CSV creates an avoidable serial tail.

The primary end-to-end benchmark must stop only after all required score
partitions and their manifest are durable. It should not hide serialization
outside the headline time.

## Exact completion and tail control

The last percent of a screen can be disproportionately expensive. The planner
should explicitly manage it.

1. Track proposal, decoded, policy-valid, globally new, graph-ready, scored,
   and committed counts separately.
2. Reserve candidate-ID ranges based on a conservative live yield estimate.
3. Stop launching main-size generator work when queued/reserved molecules are
   likely to fill the remaining score count.
4. Drain queues and launch a qualified smaller generator batch only if the
   projected committed count remains short.
5. Select exactly the first `N` committed candidate IDs; surplus remains
   uncommitted and is not written.
6. Replenish all policy, dedup, encoding and scoring failures.
7. Shrink active GPU count near the tail when distributing a tiny batch across
   all GPUs costs more than it saves.

This logic removes both the current tiny terminal stream batches and the
off-by-encoding-rejection final count.

## Failure recovery without hot-path drag

Resumability is valuable but should be block based:

- append a compact journal record when a result partition commits;
- checkpoint RNG/candidate counters, dedup state generation, queued reservation
  ranges and output digest state at the same boundary;
- atomically publish the partition then its journal commit;
- on resume, verify only the last/incomplete boundary plus manifest hashes,
  not every immutable artifact and file for every microbatch;
- retry an OOM at the same candidate IDs with the next smaller qualified batch;
- never fall back silently to CPU or a scientifically different model;
- detect sustained throughput substantially below the profile and perform one
  bounded retune or report thermal/contention diagnostics.

## One-time target setup

Although it is outside the count-only final screen, setup should never be
repeated unnecessarily:

- content-address and reuse prepared targets, UDRL/AL docking results,
  standardizers and fitted heads;
- use the existing one-task-per-GPU, topology-bound long Uni-Dock shards rather
  than short launcher-heavy docking jobs;
- reuse the fixed precomputed gMolAI embeddings for reference and AL libraries;
- load the final head ensemble directly from the completed fit manifest;
- when fitting is needed, train independent seeds concurrently or with a
  stacked implementation when this preserves the frozen training protocol.

The fast screening command should refuse to redo docking or fitting merely
because `N` changed.

## Numerical and scientific qualification tiers

Classify optimizations so the planner only selects qualified paths.

### Tier A: execution-equivalent default

- persistent processes and pools;
- artifact validation once per run;
- in-memory/shared-memory transport;
- retained surplus and exact tail accounting;
- asynchronous CPU/GPU stages;
- pinned buffers and model residency;
- exact global deduplication;
- batch sizes already inside established encoder equivalence gates;
- stacked heads proven equivalent.

### Tier B: tolerance-qualified

- new gMol graph batch/node boundaries;
- fused calibration/standardization/heads;
- compiled or alternative equivalent kernels;
- active-row compaction with changed operation grouping;
- TF32/BF16 encoder paths.

Require at minimum the existing gMol cross-batch gates, finite output, tight
head-score deltas, rank correlation, and top-fraction overlap. Threshold-edge
molecules need an explicit deterministic policy.

### Tier C: distribution-changing research, never automatic by default

- reduced maximum generation length;
- changed temperature/top-k/model/isomeric policy;
- FP8 generation without distribution qualification;
- approximate deduplication;
- a distilled/draft generator or speculative decoding;
- skipping validation or saving fewer result rows.

These can be separate scientific modes, not hidden speed optimizations.

## Metrics required to optimize honestly

The supervisor should collect low-overhead counters and timing for:

- invocation, inventory, artifact verification, model load and warm-up;
- candidate slots and candidate slots/s per GPU;
- token generation, D2H, token decode and average generated length;
- parse/sanitize, policy, canonicalization and graph packing;
- local validity, policy acceptance, within-block uniqueness, global-new yield;
- dedup lookup/insert and memory use;
- H2D, gMol forward, calibration, head forward and D2H scores;
- result formatting, compression, write, fsync and finalization;
- queue occupancy, producer blocking and consumer starvation;
- GPU utilization/memory, CPU utilization, NUMA traffic if available, and
  per-device imbalance;
- retries, OOM fallback, surplus proposals and tail waste;
- final committed molecules divided by complete durable wall.

Report distinct numerators clearly:

```text
proposal slots/s
raw decoded SMILES/s
policy-valid canonical SMILES/s
globally new molecules/s
encoded molecules/s
committed scored molecules/s
```

Never label a timer containing model loads, hashing and policy validation as
pure encoder throughput. Never sum concurrent shard times to report a critical
path. Include both cold-start and warmed steady-state rates.

The primary KPI is:

```text
durable_screening_rate = exactly_N_committed_scores / invocation_to_manifest_wall
```

Secondary system targets for sufficiently large `N` are:

- end-to-end rate at least 80-90% of measured sustained globally unique
  generator capacity on the same allocation;
- multi-GPU parallel efficiency above 85% where generation remains dominant;
- one model load and one warm-up per model per GPU;
- no intermediate SMILES CSV, score-shard CSV, or embedding NPZ in the default
  hot path;
- less than 1% avoidable proposal surplus at completion, apart from the chosen
  minimum efficient tail batch.

## Prioritized optimization backlog

| Priority | Change | Likely system impact | Complexity | Scientific risk |
|---:|---|---|---|---|
| 0 | Persistent iGen3 worker per GPU | Very high | Medium | Low |
| 0 | Persistent gMolAI pool/model and resident heads | Very high for score stage | Medium | Low |
| 0 | Count committed scores and replenish to exact `N` | Correctness plus tail speed | Medium | Low |
| 0 | Throughput-based generator batch tuner | High | Medium | Low if settings unchanged |
| 0 | Retain surplus; remove tiny outer batches | High | Medium | Low |
| 1 | Shared-memory queues; eliminate hot-path files | High | High | Low |
| 1 | Persistent parallel token decode/chemistry | High | Medium | Low |
| 1 | One policy/canonicalization boundary | High | High | Medium; prove equivalence |
| 1 | In-memory/native exact dedup with durable journal | Medium-high | High | Low if collision-safe |
| 1 | Batch-512 gMol starting profile and auto-tuning | Medium-high | Low | Already tolerance-qualified |
| 1 | GPU-resident encoder calibration -> fused heads | Medium | High | Medium; qualify numerics |
| 1 | Partitioned vectorized result writer | Medium for real all-score runs | Medium | Low |
| 1 | Queue-based stage overlap and weighted GPU work | High | High | Low |
| 2 | Remove per-token host synchronization | Medium-high | Low | Low |
| 2 | Reusable KV/token/pinned buffers | Medium | Medium | Low |
| 2 | Periodic active-sequence compaction | Potentially high | High | Medium |
| 2 | Fused QKV and sampling kernels | Potentially high | High | Medium |
| 2 | Cached compile/CUDA-graph candidates | Unknown until measured | High | Medium |
| 3 | Continuous slot recycling/paged KV | Potentially very high | Very high | Medium-high |
| 3 | Qualified FP8/alternative decoder runtime | Unknown | Very high | High |

## Recommended implementation sequence

### Phase 0: make a trustworthy baseline

1. Add the complete timing/counter schema above without changing results.
2. Preserve a representative base-isomeric candidate fixture and an exact
   end-to-end result fixture.
3. Measure model load, pool startup, chemistry, graph forward, head setup,
   writing, and container wall separately.
4. Add an all-scores durable-output benchmark; retain the current no-output
   benchmark only as a computational profile.

Exit gate: every second of current four-GPU wall is assigned to a stage, and
the sum/overlap accounting reconciles with process wall.

### Phase 1: remove cold starts

1. Implement one persistent gMolAI/head service per GPU.
2. Load, hash, warm and retain models/pools once.
3. Use batch 512 as the initial encoder candidate.
4. Implement one persistent iGen3 service per GPU with reusable buffers.
5. Store logical RNG streams inside those services.
6. Feed existing file-backed blocks through the services initially, minimizing
   simultaneous changes.

Exit gate: model and pool creation counts are `O(GPUs)`, output matches the
locked fixture, and both generator and encoder approach their standalone hot
rates on long blocks.

### Phase 2: create the streaming data plane

1. Introduce candidate IDs and bounded shared-memory block queues.
2. Run token decoding and chemistry concurrently with subsequent generation.
3. Replace repeated validation/policy paths with the qualified fused path.
4. Replace per-row SQLite admission with exact in-memory/native dedup plus a
   durable journal.
5. Remove generated, validated, prepared, and score-shard files from normal
   transport.
6. Implement exact `N` scored completion and adaptive tail control.

Exit gate: a forced crash at every block boundary resumes to the exact same
committed result with no duplicate, and steady-state GPUs are not waiting on
files or process startup.

### Phase 3: fuse encoding and scoring

1. Expose GPU tensors from the optimized GINE core.
2. Move calibrator and standardizer operations to GPU.
3. load/stack the three heads once and compute all scores per graph batch.
4. Copy only compact score columns to host.
5. Add pinned multi-buffer graph transfer and asynchronous result transfer.
6. Introduce the partitioned result writer and streaming hashes.

Exit gate: output probabilities/ranks pass the frozen equivalence suite and no
default path constructs a host 384-D screen-wide embedding matrix.

### Phase 4: automatic planning and adaptive scheduling

1. Add hardware/topology discovery and qualified profile caching.
2. Add the `N`-aware completion-time model.
3. Tune per-GPU generator/encoder batches and worker allocations.
4. Add queue-feedback GPU duty cycling and heterogeneous weighted scheduling.
5. Select active GPU count, compile policy, buffers and checkpoint cadence
   without user input.

Exit gate: the same count-only command runs efficiently on one, two and four
GH200s and on supported x86-64 Blackwell/Ada/Ampere GPUs, and records every
decision.

### Phase 5: decoder kernel research

1. Remove/coarsen the token-loop host sync and measure.
2. Test periodic active-row compaction.
3. Test fused QKV and fused small-vocabulary sampling.
4. Test cached compilation and CUDA graphs.
5. Prototype continuous slot recycling only if profiling shows finished-row
   computation remains dominant.

Promote one change at a time behind output/distribution gates. Do not delay the
much larger persistence and pipeline wins while pursuing speculative kernels.

## Acceptance test matrix

Every release candidate should cover:

- `N` smaller than one generator batch, exactly one batch, and a large
  multi-partition screen;
- one, two and four GPUs;
- heterogeneous device speeds or an artificially throttled lane;
- CPU-constrained and RAM-constrained allocations;
- duplicate-heavy and low-validity candidate fixtures;
- gMol policy edge cases and encoding rejection replenishment;
- OOM shrink/retry at generation and encoding;
- termination during every reservation/score/partition commit state;
- resume with identical artifacts and refusal after artifact changes;
- exactly `N` finite score rows, exact canonical uniqueness and ID alignment;
- per-stage performance regression thresholds;
- all-score durable output, not only a threshold that retains zero rows.

For numerical candidates, compare against the frozen implementation with:

- exact identity/order/count checks;
- finite embeddings and scores;
- same-boundary exact comparison where possible;
- established cosine and relative-L2 embedding gates across batch boundaries;
- maximum/quantile probability deltas;
- Spearman rank correlation and top-0.1%/1% overlap;
- explicit treatment of values near any operational threshold;
- generated validity, uniqueness, length and descriptor distributions when a
  decoder execution change can affect stochastic output.

## Performance opportunity and realistic target

The current four-GPU run spends 9,197 seconds for 10M admitted molecules. If
the current launch-heavy generation path alone were the lower bound, removing
all other serialized work would cap useful throughput near 5.81M final
molecules/h. That is not the real hardware ceiling because the generation path
itself starts 440 processes.

Using the observed one-GH200 hot base-isomeric result as a reference:

```text
10,663,693 locally unique generated rows / (4 * 2,076.5 rows/s)
    ~= 1,284 seconds of ideally scaled hot generation
```

That corresponds to about 28.0M final globally admitted molecules/h before
multi-GPU inefficiency, output work and tail effects. At 85% of that reference,
the result is about 23.8M/h. Separately, four ideal 58,330/s encoders would
encode 10M molecules in about 43 seconds, and the heads are faster still.

Therefore a sensible first large-screen engineering objective on four GH200s
is **20-25 million durable scored molecules/h**, followed by decoder-kernel
work if the measured generation ceiling supports more. This is a target to
validate, not a promised result: generated-molecule chemistry, all-score output,
CPU policy throughput, stochastic yield, GPU scaling and filesystem behavior
must be measured in the integrated engine.

The more portable success criterion is stronger than a fixed number:

> For a sufficiently large screen, durable end-to-end throughput should reach
> at least 80-90% of the same allocation's measured sustained globally unique
> iGen3 capacity, while returning exactly `N` valid unique scores.

## Anti-patterns to avoid

- Do not equate maximum memory-fitting batch with maximum throughput.
- Do not launch a generator, validator or encoder process per stream block.
- Do not reload or rehash immutable artifacts per block.
- Do not destroy worker pools or empty CUDA after every stage.
- Do not use CSV/NPZ as inter-stage IPC.
- Do not parse and canonicalize the same molecule in every component.
- Do not use a per-row SQLite call when an exact memory-resident set fits.
- Do not use a Bloom filter or truncated hash as the uniqueness authority.
- Do not split equal synchronized shards across unequal GPUs.
- Do not synchronize the host on every generated token.
- Do not optimize head forward before fixing generation and cold setup.
- Do not benchmark with zero retained scores and call it production end to end.
- Do not improve speed by silently changing the scientific generation policy.

## Final recommendation

The first implementation should not begin with exotic kernels. Build the
persistent engine, retain all models and pools, use throughput-selected batches,
remove redundant chemistry/files, and count exact committed scores. Those
changes directly attack almost every unexplained minute in the current result
and allow the proven standalone model rates to become relevant.

Once the new engine is demonstrably generation-bound, profile the iGen3 token
loop. The best next candidates are removing its per-token host synchronization,
overlapping CPU decode/chemistry, reusing KV buffers, and compacting finished
sequences. Only after those measurements should custom Triton/CUDA, continuous
batching, compilation or FP8 become the focus.

The guiding design rule is simple:

> Load once, parse once, keep data in memory, overlap independent work, tune
> accepted committed throughput, and never make the user choose a performance
> parameter.

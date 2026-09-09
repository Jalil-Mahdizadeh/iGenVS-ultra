# iGenVS-ultra maximum-speed screening design

Status: Phase-1 persistent orchestration, portable automatic planning, and the
full cold 1/2/4-GPU benchmark are complete as of 2026-09-06. Deeper
shared-memory, fusion, and decoder-kernel phases remain optional follow-on
work.

September 9 maintenance supersedes the in-memory identity implementation
described in the historical checkpoint below: identities/surplus are durable
SQLite state, and generated admission now overlaps current scoring with bounded
look-ahead. See [maintenance validation](../speed-bench/MAINTENANCE-2026-09-09.md).
The frozen September 6 timings and scientific settings are not rewritten.

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
Detailed evidence is in `../user-pipeline/benchmarks/RESULTS.md`.

The later full cold benchmark committed exactly 10,000,000 finite scores per
case at 31.27, 50.21, and 71.09 million scores/hour on one, two, and four
GH200 GPUs. It used the public count-only command, automatic planning,
all-score durable output, and cross-batch overlap. See
[`speed-bench/REPORT.md`](../speed-bench/REPORT.md).

Implemented here does not mean the complete blueprint below is finished.
Cross-batch generation/scoring overlap is active. Finer-grain shared-memory
transport, fused GPU encoder/calibrator/head execution, reusable KV buffers,
asynchronous token transfer, and decoder-kernel research remain possible next
optimization phases.

## Goal

The final-screening interface now requires exactly one user value:

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

Ultra is now a persistent streaming service at the process level for the
lifetime of one screen. The architecture below is the longer-term data-plane
target for removing the remaining file/IPC and kernel overhead.

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

The implemented transitional design gives each GPU one persistent iGen3
worker and one persistent gMolAI/head worker, loading and validating models
once per screen. A future unified worker could additionally share buffers and
schedule finer generation/scoring quanta through memory queues.

With process churn removed, generation is the largest scalable GPU stage. One
GH200 has already demonstrated about 2,076 valid unique base-isomeric SMILES/s
in the local 98,304-row fixture, whereas the published hot gMolAI encoder
reaches 58,330 molecules/s on **one** GH200 at batch 512. Target-head inference
is faster again. The completed pipeline no longer spends tens of minutes
restarting encoders; remaining gains depend on admission/chemistry overhead,
multi-GPU scheduling efficiency, and the generator itself.

## What the completed measurements say

The locked 10-million-score results are in
[`speed-bench/REPORT.md`](../speed-bench/REPORT.md). Every case is one cold
sample with no warm-up or repeat. All performance controls remained automatic,
all scores were durably written, and generation/scoring overlap was enabled.

| GPUs | Complete wall | Screening/hour | Batches | Generate | Score stage | Admit | Candidate yield |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,151.134 s | 31,273,519 | 11 | 923.744 s | 784.202 s | 122.122 s | 91.182% |
| 2 | 716.952 s | 50,212,570 | 9 | 470.202 s | 361.338 s | 120.669 s | 91.182% |
| 4 | 506.403 s | 71,089,653 | 8 | 242.118 s | 163.451 s | 121.345 s | 91.182% |

All three cases committed exactly 10,000,000 finite scores with 100% encoding
yield. Stage sums exceed complete wall because adjacent batches overlap. The
four-GPU generation critical path is 3.82x faster than one GPU, while complete
wall improves 2.27x. Admission remains almost fixed at about 121 seconds, so
CPU admission, scheduling/tail behavior, and startup/finalization now explain
much of the multi-GPU efficiency gap. Head inference is negligible at 2.075,
1.229, and 0.228 seconds for one, two, and four GPUs.

### Historical pre-optimization diagnosis

Before the persistent worker and overlap implementation, the earlier four-GPU
prototype needed 9,197 seconds (3.914 million/hour), repeatedly launched
hundreds of generator/scorer processes, used a threshold that retained almost
no rows, and stopped with only 9,999,995 successful encodings. Those values are
historical design evidence, not the current `speed-bench/REPORT.md` result.
The completed four-GPU path is 18.16x faster by complete wall and satisfies the
exact-score/all-output contract.

Low-level generator opportunities remain in
[`generation.py`](../iGenVS/iGen3/src/igen3/generation.py), including per-batch
KV allocation and token-loop host synchronization. Reference standalone rates
remain useful but have different boundaries: the local hot iGen3 fixture
reached 2,076.5 valid unique SMILES/s on one GH200; the published RTX PRO 2000
Blackwell Laptop iGen3 result was 1,463.1/s; and published warm gMolAI encoding
reached 58,330.4/s on one GH200. See the
[`iGen3` benchmark](https://github.com/Jalil-Mahdizadeh/iGen3/tree/main/benchmarks/latest)
and gMolAI [`RESULTS.md`](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/extra-benchmark/speed/RESULTS.md)
and [`PROTOCOL.md`](https://github.com/Jalil-Mahdizadeh/gMolAI-v2.0/blob/main/extra-benchmark/speed/PROTOCOL.md).

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

The completed benchmark confirms this contract: every 1/2/4-GPU case stopped
at exactly 10,000,000 committed finite scores, not at an upstream admission or
generation counter.

## The count-only user experience

The normal command exposes no performance knobs:

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

The implemented default output contract is now fixed: all `N` identities and
scores are durably committed to CSV plus a manifest. The completed benchmark
measures that real product. Partitioned Parquet or Arrow with a lazy CSV export
remains a possible writer optimization, provided it preserves the same public
data contract. Silently retaining zero rows or only a thresholded subset is
not a valid all-molecule screening benchmark.

## Replace the outer batch with independent execution scales

The Phase-1 engine separates generator and encoder microbatches from the outer
durable stream block. A fuller data plane should independently control all five
scales:

1. **Generator microbatch:** GPU sequences sampled together.
2. **Chemistry block:** raw strings sent to CPU validation/canonicalization.
3. **Graph batch:** accepted molecules packed for one gMolAI forward, bounded
   by graphs and nodes.
4. **Result partition:** rows written as one durable output object.
5. **Checkpoint interval:** amount of completed work between resumable commits.

These values have different optima. A generator batch may be 8K-32K, a graph
batch around 512, a shared-memory chemistry block a few thousand, and a result
partition hundreds of thousands or millions. None requires a new process.

The resource-based stream-size model now acts as a checkpoint/output-partition
estimate; it no longer determines persistent model lifetime or the generator
and encoder kernel batches.

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

The transitional two-service architecture below is implemented and is the
largest reason the completed four-GPU result improved over the historical
prototype. The unified shared-memory design remains a potential next step.

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

Generator residency, throughput-aware safe batch selection, retained surplus,
and persistent parallel canonicalization are implemented. The remaining
subsections distinguish completed design rationale from lower-level follow-on
work.

#### Keep generators resident

Maintain one loaded base-isomeric generator per active GPU for the full run.
Logical random streams should be state objects inside workers, not separate
processes. Save their RNG state/candidate counters at checkpoints.

#### Tune sustained accepted throughput

The original auto-tuner binary-searched the largest one-step KV allocation that
fit. The implemented planner now bounds from live memory/backend limits and
uses amortization-aware performance profiles. A deeper tuner can still:

1. Estimate a safe memory ceiling.
2. Benchmark a short geometric ladder below that ceiling, such as nearby
   4K/8K/12K/16K/24K/32K values, using complete warmed sequences.

Select the batch minimizing expected time to globally new policy-valid output,
including token transfer/decode and CPU chemistry backpressure. The fastest
batch is not necessarily the largest that fits. Keep several qualified tail
batches as well as the main steady-state batch.

#### Do not discard surplus

Generation now retains surplus between persistent stream blocks. A future
shared-memory queue can retain the same behavior while removing the remaining
file-backed boundary.

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
generator still creates cache tensors for each generated batch. CUDA's caching
allocator helps, but persistent typed buffers can remove Python object,
allocation, fragmentation, and initialization work.

#### Decouple GPU sampling from CPU chemistry

The remaining iGen3 generation core synchronously transfers tokens, decodes Python rows,
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

The pre-optimization path crossed several redundant chemistry boundaries:

1. iGen3 parses and canonicalizes it for local valid-unique output.
2. iGenVS validates and canonicalizes it again.
3. gMol policy parses it, canonicalizes it, and reparses the canonical string.
4. gMol graph construction parses the canonical string again.

Depending on the path, that was up to five RDKit parses plus multiple
string/file round trips. The generated Phase-1 path now bypasses redundant
iGenVS validation and uses persistent canonicalization; fully fusing policy and
graph construction remains future work.

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

This motivated the implemented in-memory scorer. The standalone generic
encoder path:

1. computes a GPU raw embedding;
2. copies 384 floats per molecule to CPU;
3. applies gMol calibration in NumPy;
4. applies the target input standardizer in NumPy;
5. copies the same 384 floats back to GPU three times for separately loaded
   ensemble heads.

For 10M molecules, one 384-D FP32 matrix is 15.36 GB. Moving and
materializing it repeatedly is unnecessary because ultra does not save
embeddings by default.

The default persistent scorer now avoids writing embedding NPZ files and keeps
the encoder and heads resident. A fully GPU-fused `encode_and_score` path could
go further:

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

The completed benchmark confirms that head forward is negligible. Further
head fusion would mainly remove data movement and should not take priority over
generation, admission, or multi-GPU scheduling.

## Stage overlap and GPU scheduling

The pre-optimization wall was approximately a sum of stage walls:

```text
T_current ~= T_generate + T_validate/dedup + T_score + T_files
```

The current implementation overlaps generation and scoring across durable
batches, which is why measured stage sums exceed complete wall. With finer
bounded queues, steady-state wall can approach the slowest service more
closely:

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

Phase 1 implements exact replenishment, surplus retention, and exact committed
counting. More adaptive tail sizing could further reduce the last partial
batch without reintroducing an off-by-rejection result.

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
path. The completed release benchmark intentionally reports one cold sample;
separate engineering profiles may additionally compare warmed steady state.

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

The completed four-GPU run has 56.8% end-to-end parallel efficiency relative
to one GPU, while its generation stage scales 3.82x. The 85% end-to-end target
therefore remains aspirational even though the original absolute throughput
target was exceeded.

## Prioritized optimization backlog

This table preserves the original priority logic. Persistent per-GPU workers,
exact committed-score replenishment, generator batch planning, surplus
retention, persistent parallel chemistry, batch-512 encoder qualification, and
cross-batch overlap are implemented. Shared-memory IPC, fully fused policy and
GPU scoring, vectorized partition output, adaptive weighted scheduling, and
decoder-kernel items remain open.

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

## Implementation sequence and current status

### Phase 0: make a trustworthy baseline

Status: **complete** for the released cold protocol. The report includes exact
counts, all-score durable output, full stage timings, automatic plans, and
1/2/4-GPU cases.

1. Add the complete timing/counter schema above without changing results.
2. Preserve a representative base-isomeric candidate fixture and an exact
   end-to-end result fixture.
3. Measure model load, pool startup, chemistry, graph forward, head setup,
   writing, and container wall separately.
4. Maintain the all-scores durable-output benchmark as the primary result;
   retain any historical no-output measurement only as a computational profile.

Exit gate: every second of current four-GPU wall is assigned to a stage, and
the sum/overlap accounting reconciles with process wall.

### Phase 1: remove cold starts

Status: **core work complete**. One persistent iGen3 and one persistent
gMolAI/head worker run per selected GPU, with model/pool lifetime bounded by
the screen rather than the stream batch.

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

Status: **partially complete**. Exact `N`, retained surplus, in-memory exact
admission with bulk durable commits, persistent chemistry, and cross-batch
overlap are active. Shared-memory blocks, fused graph construction, and a
fully journaled queue remain open.

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

Status: **partially complete**. The default path avoids embedding NPZ
round-trips and keeps the encoder and heads resident. GPU-side
calibration/standardization fusion, pinned buffer rings, and partitioned output
remain open.

1. Expose GPU tensors from the optimized GINE core.
2. Move calibrator and standardizer operations to GPU.
3. load/stack the three heads once and compute all scores per graph batch.
4. Copy only compact score columns to host.
5. Add pinned multi-buffer graph transfer and asynchronous result transfer.
6. Introduce the partitioned result writer and streaming hashes.

Exit gate: output probabilities/ranks pass the frozen equivalence suite and no
default path constructs a host 384-D screen-wide embedding matrix.

### Phase 4: automatic planning and adaptive scheduling

Status: **portable automatic planning and 1/2/4-GPU execution complete**.
Hardware/model-keyed profiles, resource bounds, automatic generator/encoder
batches, and all-visible-GPU selection are active. Heterogeneous weighted
scheduling and queue-feedback duty cycling remain open.

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

The completed cold benchmark covers the large exact-count, one/two/four-GPU,
all-score durable-output cases. Future release candidates should retain those
regressions and add the remaining failure/topology cases below:

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

The completed four-GPU run spends 506.403 seconds for exactly 10M committed
scores, or 71.09 million/hour. It exceeds the original 20-25 million/hour
engineering objective and is 18.16x faster than the historical 9,197-second
prototype. That original estimate must no longer be treated as a current
target.

The remaining opportunity is visible in scaling and stage boundaries. Four-GPU
candidate generation reaches 163.07 million candidate slots/hour with 91.182%
end-to-end candidate yield, but complete durable throughput is 71.09
million/hour. Generation scales well; admission remains about 121 seconds at
all GPU counts, and startup, scoring duty, synchronization, tail behavior, and
finalization limit end-to-end scaling. Further targets should be set only after
profiling these current boundaries rather than extrapolating the retired
launch-heavy implementation.

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

The first implementation followed the intended order: persistent workers,
retained models/pools, automatic batches, less redundant chemistry/I/O,
cross-batch overlap, and exact committed-score accounting came before exotic
kernels. The completed benchmark validates those decisions.

The next pass should profile admission and the iGen3 token
loop. The best next candidates are removing its per-token host synchronization,
overlapping CPU decode/chemistry, reusing KV buffers, and compacting finished
sequences. Only after those measurements should custom Triton/CUDA, continuous
batching, compilation or FP8 become the focus.

The guiding design rule is simple:

> Load once, parse once, keep data in memory, overlap independent work, tune
> accepted committed throughput, and never make the user choose a performance
> parameter.

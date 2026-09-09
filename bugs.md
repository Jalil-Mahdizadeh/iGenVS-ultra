# Current maintenance status — 2026-09-09

This section supersedes the historical audit/implementation comments retained
below. The remaining operational fixes have been implemented without changing
scientific protocols; full-scale validation is recorded separately in
[the maintenance report](speed-bench/MAINTENANCE-2026-09-09.md).

| Point | Current implementation and regression coverage |
| --- | --- |
| 1 — RL launch | Runtime bootstrap forwards the requested action; entry-point and CLI smoke tests pass. |
| 2 — Screening admission recovery | Marker-before-commit publication and SQLite replay survive process kills; no whole-library hot set can retain stale identities. |
| 3 — Docking coverage on resume | Logical shard plans remain pinned across GPU-count changes; cached and merged outputs undergo exact input-coverage checks. Four-way jobs resume on one/two GPUs. |
| 4 — RL recovery | Checkpoint-only recovery runs before stopping/skip decisions, repairs latest/selected exports, reconciles legacy history/evaluations, and invalidates uncommitted stopping decisions. Reference initialization/copying and torn legacy CSV tails are recoverable. |
| 5 — Empty docking shards | Empty logical partitions are skipped; small/sparse input tests pass. |
| 6 — All-rejected encoder chunks | Retained and ephemeral paths return empty scores plus rejection records; real gMol tests pass. |
| 7 — Zero-success docking | New jobs fail; cached zero-success top-level/shard/fit manifests cannot be reused as successful. Mounted core bootstraps prevent an older SIF's installed package from bypassing the guard. Actual SIF failure injection passes without a source PYTHONPATH override. |
| 8 — GPU masks | Explicit disable masks are forwarded even when normalized to an empty string; shared runtime and RL oracle honor them. Real child-environment tests pass. |
| 9 — Generator resume | Seen identities, surplus, and idempotent requests are durable SQLite state; stable lanes and rollback/restart tests preserve outputs. Unsafe legacy state requires a new screen name. |
| 10 — Admission bottleneck | Bulk SQLite admission with chunk-bounded lookup and generated/external admission–scoring overlap are implemented with bounded look-ahead. Counts, ordering, and replenishment tests pass; fresh large-run measurements are in the maintenance report. No unmeasured 2/4-GPU scaling claim is made. |
| 11 — Host RAM portability | Library-sized sets are removed. Both RAM detectors take the tightest remaining visible cgroup v1/v2 ancestor allowance, including sibling usage and zero headroom. |
| 12 — RL GPU memory | Backward retains full-batch normalization with OOM-safe gradient resets. Sampling retries smaller cache/model chunks with the original full-batch token-major RNG stream. Actual 4,096 draws under a 2 GiB allocator limit matched full-batch tokens and RNG state. |

Scientific invariants: protocol/freeze hashes are unchanged; docking/search,
sampling settings/seeds, gMol chemistry and ensemble operations, encoder
boundaries, reward/stopping rules, optimizer, precision, and effective RL batch
are unchanged. Operational implementation hashes are maintained separately in
`phase-10-rl-dev/maintenance.json`. AMD64 Docker images remain deferred.

Final validation: 185 distinct tests passed across the dedicated SIF suites and
SIF-backed public-workflow integration runs. All ten initial environment/opt-in
skips were exercised separately. Release checks, model/protocol hashes, and
`git diff --check` passed. The maintenance report records the real 10-million-score
run and the byte-identical full-library replay through the final admission code.

Test outputs and benchmark jobs are outside the repository. Whole-job operation
on every low-memory GPU and fresh 2/4-GPU scaling are not implied by the tests.

---

## Historical audit and implementation notes

• I found several concrete issues, including a broken RL entry point and two resume paths that can silently omit molecules. No repository files were changed.

  I traced all three user workflows through their generation, docking, encoding, training, and recovery code.

  ## Bugs and red flags

  1. [P1] RL training cannot launch. _runtime_command() references undefined variable action, immediately raising NameError. Both stage initialization and training use this helper; dry-run avoids it, masking the failure. Reproduced directly. user-
     pipeline/src/igenvs_ultra/rl_workflow.py:74

  2. [P1] Screening resume can silently discard unscored molecules. Admission commits identities to SQLite before publishing its completion marker. After an interruption in that window, resume loads those identities into hot_seen; retry deletes their
     database entries but leaves them in memory, incorrectly rejecting them as duplicates. Fault injection reproduced an admitted molecule becoming a duplicate without ever being scored. Admission/retry logic (user-pipeline/src/igenvs_ultra/
     workflow.py:2984)

  3. [P1] Docking resume can report completion with missing input rows when GPU count changes. Completed shards are reused without checking their original partition count. With automatic GPU selection, resuming a four-GPU job on two GPUs can merge only
     two old quarter-partitions. My isolated reproduction reported complete with rows 1,2,5,6 from an eight-row input. The merger checks duplicates, but not coverage. Shard reuse (user-pipeline/src/igenvs_ultra/workflow.py:1356), merge validation (user-
     pipeline/src/igenvs_ultra/workflow.py:1477)

  4. [P2] Interrupted RL updates are not safely resumable. An update creates its directory with exist_ok=False. If interrupted before checkpointing, resume retries the same update and hits FileExistsError. Additionally, history is written before the
     checkpoint, while the wrapper derives progress from history—so recorded progress can exceed saved training state. Update directory (phase-10-rl-dev/src/igenvs_rl/trainer.py:288), checkpoint ordering (phase-10-rl-dev/src/igenvs_rl/trainer.py:831)

  5. [P2] Small docking inputs can fail simply because more GPUs are available. The launcher creates one shard per GPU without ensuring each partition contains valid molecules. For example, the bundled three-molecule library produces an empty fourth
     shard on a four-GPU host; the underlying pipeline treats that as fatal. Sparse valid source-row partitions can trigger the same issue. Partition count (user-pipeline/src/igenvs_ultra/workflow.py:1377), empty-shard failure (iGenVS/src/igenvs/
     pipeline.py:536)

  6. [P2] One entirely encoder-rejected chunk aborts screening. If every molecule in a nonempty scoring chunk—or GPU shard—fails the gMol inference policy, encoding raises an exception instead of returning zero scores and rejection records. Consequently,
     otherwise valid later chunks never run, and resume encounters the same failure. user-pipeline/src/igenvs_ultra/model_ops.py:1309

  7. [P2] Complete docking-engine failure can still produce a successful job. Repeated Uni-Dock failures become per-ligand docking_failed records, but the pipeline marks the job complete even with zero successful dockings. This is visible in counts, but
     automation sees success and resume skips the completed work after the underlying GPU problem is resolved. Failure conversion (iGenVS/src/igenvs/docking.py:338), unconditional completion (iGenVS/src/igenvs/pipeline.py:799)

  8. [P2] Explicit GPU-disable masks are ignored. CUDA_VISIBLE_DEVICES=-1, NoDevFiles, and an explicitly empty value fall through to physical GPU discovery via nvidia-smi. Workers subsequently set their own visibility, potentially using GPUs the caller
     intended to exclude. Reproduced with mocked GPU discovery. user-pipeline/src/igenvs_ultra/workflow.py:203

  9. [P2] Generated screening libraries change across resume despite identical seeds. Persistent generation carries unused candidates and previously generated identities between requests, but neither state is checkpointed. Restarted workers skip cached
     requests without rebuilding that state. An isolated test of the actual generation method produced different next-batch outputs for uninterrupted versus restarted execution using the same seed. Persistent state (user-pipeline/src/igenvs_ultra/
     generation_worker.py:498), surplus consumption (user-pipeline/src/igenvs_ultra/generation_worker.py:524)

  ## Performance and scalability

  - Screening’s serial admission stage is a demonstrated scaling bottleneck. The checked-in 10-million-molecule benchmark shows admission remaining around 121 seconds on one, two, and four GPUs. Four-GPU head inference takes only 0.228 seconds; overall
    scaling is approximately 2.27×, not 4×. Admission and generation deserve substantially more attention than the classifier heads. Recorded timings (speed-bench/REPORT.md:43)

  - Streaming does not provide bounded identity memory. Screening materializes the entire deduplication database as a Python set; generator workers also retain growing identity sets. Separately, automatic batch sizing reads host MemAvailable without
    accounting for container/cgroup limits. Large libraries or constrained allocations therefore have a credible OOM risk. This is code-derived, not a measured OOM. Identity loading (user-pipeline/src/igenvs_ultra/workflow.py:2489), memory detection
    (user-pipeline/src/igenvs_ultra/workflow.py:238)

  - RL training has a substantial single-GPU memory constraint. Later stages process a full 4,096-sequence training batch in FP32 without microbatching. Additional GPUs parallelize docking, not policy training. Smaller-GPU portability is therefore
    questionable, although I did not measure the memory requirement. FP32 policy loading (phase-10-rl-dev/src/igenvs_rl/policy.py:159), full-batch backward path (phase-10-rl-dev/src/igenvs_rl/trainer.py:733)

  Validation: 33 unit tests passed; 7 were skipped. I also ran isolated fault-injection and orchestration checks. Full GPU/chemistry execution was not performed because the host Python environment lacks Torch and RDKit.

##############################################

  • After a second, stricter audit, I reject none of the core mechanisms. Points 1–8 are reproducible as written, while point 9 needs conditional wording. The three performance findings are supported, but two remain risk assessments rather than measured
  failures.

  ## Point-by-point verdict

  1. Confirm — RL training cannot launch. Calling _runtime_command(..., "init") inside the iGenVS SIF produced NameError: name 'action' is not defined. There is no imported or module-level action; the arguments already contain init or train. Both live
     call paths use this function. user-pipeline/src/igenvs_ultra/rl_workflow.py:70

  2. Confirm — screening admission has a data-loss crash window. Fault injection immediately after the SQLite commit produced:
      - Database after crash: CC, owner batch 1
      - Resume hot set: CC
      - Retried batch: 0 accepted, 1 duplicate
      - Database after retry: empty

     Thus an external molecule can disappear without being scored. Generated screening can replace it to recover the requested count, but not the original molecular library. Hot-set construction (user-pipeline/src/igenvs_ultra/workflow.py:2984), commit-
     before-publication (user-pipeline/src/igenvs_ultra/workflow.py:3041)

  3. Confirm, with precise scope — decreasing automatic GPU count can omit docking partitions. Exercising the actual launcher and merger with two completed shards from an interrupted four-way run yielded status=complete, but only source rows 1,2,5,6 from
     eight validated rows. The bug applies when resuming with fewer GPUs while still using multi-GPU mode; resuming with one GPU follows a different restart path. Shard reuse (user-pipeline/src/igenvs_ultra/workflow.py:1356), coverage-blind merge (user-
     pipeline/src/igenvs_ultra/workflow.py:1477)

  4. Confirm — RL update recovery is unsafe. Retrying an existing update-0001 directory produced FileExistsError. Separately, a fixture with history reporting update 1 but no checkpoint caused the wrapper to report one completed update. This matches the
     real write order: history precedes checkpoint publication. Non-resumable directory creation (phase-10-rl-dev/src/igenvs_rl/trainer.py:288), history/checkpoint order (phase-10-rl-dev/src/igenvs_rl/trainer.py:831), history-based progress (user-
     pipeline/src/igenvs_ultra/rl_workflow.py:77)

  5. Confirm — small or sparse libraries can create fatal empty docking shards. The actual partitioner returns counts [1,1,1,0] for three validated rows across four GPUs. The fourth pipeline then rejects its zero-row shard. This affects the bundled
     three-molecule example on a four-GPU allocation and sparse source-row residue classes. Four-way launch (user-pipeline/src/igenvs_ultra/workflow.py:1353), zero-row rejection (iGenVS/src/igenvs/pipeline.py:536)

  6. Confirm — an all-rejected scoring chunk aborts screening. Inside the gMol SIF, iGenVS accepted C as a valid one-heavy-atom molecule. The actual screening policy rejected it as too_few_atoms, after which encode_ephemeral() raised ModelOperationError.
     No caller converts that condition into a zero-encoded manifest, so later batches are abandoned. user-pipeline/src/igenvs_ultra/model_ops.py:1309

  7. Confirm the behavior; classify it as a contract red flag. With /bin/false as Uni-Dock, both attempts returned code 1 and the ligand became docking_failed without an exception. The pipeline subsequently marks a fully processed run complete regardless
     of docked == 0. This may reflect an intentional “all rows reached a terminal status” definition, but it is unsafe for job automation. The benchmark harness has an explicit zero-success guard that the public workflow lacks. Failure conversion
     (iGenVS/src/igenvs/docking.py:338), completion status (iGenVS/src/igenvs/pipeline.py:799), benchmark-only guard (speed-bench/scripts/run_case.py:203)

  8. Confirm — GPU-disable masks can be bypassed. With CUDA_VISIBLE_DEVICES=-1 and mocked physical discovery returning GPUs 0 and 1, visible_gpu_tokens() returned both GPUs. The same fall-through exists for NoDevFiles and a set-but-empty variable.
     Whether this exposes devices in a particular runtime depends on device-node isolation, but the helper itself violates the mask. user-pipeline/src/igenvs_ultra/workflow.py:203

  9. Partially confirm — change “libraries change” to “libraries can change.” The actual persistent state machine, supplied with a deterministic candidate producer, emitted:
      - First batch: S13_0
      - Continuous second batch with seed 14: S13_1
      - Restarted second batch with seed 14: S14_0

     This confirms that uncheckpointed surplus and seen-state can alter resumed output. It is conditional: if a completed request leaves no relevant surplus or seen-state effect, outputs may still match. Uncheckpointed state (user-pipeline/src/
     igenvs_ultra/generation_worker.py:498), surplus-first emission (user-pipeline/src/igenvs_ultra/generation_worker.py:524)

  10. Confirm, but call it an observed bottleneck rather than statistically demonstrated. Admission is serial in the controller, and the recorded times remain 122.122, 120.669, and 121.345 seconds across one, two, and four GPUs. It occupies roughly 24%
     of four-GPU wall time, whereas head inference is 0.228 seconds. Overall speedup is 2.27×. However, the report explicitly says each configuration is only one cold sample. Benchmark methodology (speed-bench/REPORT.md:5), stage timings (speed-bench/
     REPORT.md:51)

  11. Confirm the resource risk, not an observed OOM. Identity memory is mathematically unbounded with library size: the complete SQLite identity table becomes a Python set, while every generation worker retains its own growing set. Batch sizing reads /
     proc/meminfo but not cgroup memory limits. The wording in bugs.md already correctly calls OOM a credible code-derived risk rather than a reproduced failure. Identity materialization (user-pipeline/src/igenvs_ultra/workflow.py:2489), host-only memory
     detection (user-pipeline/src/igenvs_ultra/workflow.py:238)

  12. Confirm the architecture and risk; not a measured bottleneck. Stages 2–4 use batches of 4,096. Policy and prior are forced to FP32, and sequence_statistics() processes the full batch before one backward pass. Device resolution selects one CUDA
     device; additional GPUs are used for docking shards, not distributed policy training. Smaller-GPU portability therefore is legitimately questionable, but no memory threshold was experimentally established. FP32 models (phase-10-rl-dev/src/igenvs_rl/
     policy.py:159), full-batch statistics (phase-10-rl-dev/src/igenvs_rl/trainer.py:733)

  Validation used both dedicated images with the repository mounted read-only:

  - iGenVS SIF: 59 core tests passed; 33 user-pipeline tests passed and 7 skipped.
  - gMolAI SIF: all 11 selected chemistry/fast-inference tests passed.
  - Focused recovery and failure probes wrote only to temporary directories.
  - No full production GPU workload was started.

#################################################################

• Implemented all 12 findings without changing scientific protocols.

  1. Fixed RL launch NameError.
  2. Made screening admission crash-safe with marker-first publication and SQLite replay.
  3. Pinned docking shard plans across hardware changes and added exact input-coverage validation, including repair of legacy false-complete jobs.
  4. Made RL progress checkpoint-authoritative; interrupted update/evaluation directories are recoverable.
  5. Empty docking shards are skipped.
  6. Fully gMol-rejected chunks now produce valid empty results and continue.
  7. Zero-success docking now fails the job clearly.
  8. GPU-disable masks are strictly honored.
  9. Generator seen/surplus state is persisted in SQLite, with deterministic lane assignment and idempotent request replay.
  10. Admission now uses set-based SQLite operations and overlaps external scoring. A synthetic 100k-row probe took approximately 0.27 seconds with exact deduplication.
  11. Removed whole-library identity sets and added cgroup v1/v2-aware RAM sizing.
  12. RL backward processing now uses adaptive microbatches while retaining FP32, the 4,096 effective batch, and the identical mathematical objective.

  Scientific integrity:

  - RL protocol SHA-256 remains 4b9aa7fad0e563bddb28de5edc96d061ac24b1165b4f324a9711e60416d0c03b.
  - Historical freeze SHA-256 remains 32bb9d74ef4354c9fab44a9f01bfe037e2d889b39043854c27f662040e08e3e2.
  - No docking/search/scoring parameters, gMol policy, ensemble logic, reward definition, optimizer, or precision were changed.
  - Added phase-10-rl-dev/maintenance.json to record the operational implementation overlay without altering the historical freeze.

  Validation:

  - iGenVS SIF: 129 passed, 7 intentional opt-in integration skips.
  - gMolAI SIF: 12 passed.
  - Release checker: 54 tests passed, hashes and metadata valid.
  - Actual zero-docking-success failure probe passed.
  - git diff --check passed.
  - No test outputs or new cache artifacts remain. bugs.md was untouched.
  - AMD64 Docker images were not changed. Full production-scale GPU workloads remain outside this validation.

############################################################

• I cannot confirm that all 12 points are fully resolved. Several fixes are sound, but additional fault-injection tests exposed remaining recovery and portability gaps.

   Point                                         Workflow                              Verdict
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1. RL launch failure                          RL training                           Fixed; the runtime entry point launches successfully.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   2. Admission crash loses molecules            Screening                             Fixed; process-kill tests around marker publication and database commit passed.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   3. GPU-count changes omit docking rows        Docking                               Fixed; four-way jobs resumed on one/two GPUs with exact coverage.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   4. Interrupted RL recovery                    RL training                           Incomplete: adaptive-stage and reference-initialization recovery remain unsafe.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   5. Empty docking shards                       Docking                               Fixed; small and sparse inputs passed.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   6. Entirely encoder-rejected chunks           Screening                             Fixed; actual gMol tests passed for retained and ephemeral embeddings.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   7. Zero-success docking reports completion    Docking                               Incomplete: cached jobs and the installed SIF still exhibit this behavior.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   8. GPU-disable masks                          Shared runtime / RL generation        Incomplete: detection is fixed, but child-process forwarding can discard explicit masks.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   9. Generator state changes across resume      Generated screening                   Fixed for the reported mechanism; surplus restoration and replay tests passed.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   10. Serial admission bottleneck               Screening                             Improved, but not demonstrated resolved at the original benchmark scale.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   11. Unbounded identities / RAM detection      Screening / shared hardware sizing    Identity storage fixed; nested cgroup limits remain mishandled.
  ────────────────────────────────────────────  ────────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────
   12. Single-GPU RL memory constraint           RL training                           Backward microbatching works; full-batch sampling remains a portability constraint.

  The outstanding findings are:

  - #4 — Adaptive resume can bypass recovery. At a stopping boundary, the wrapper can accept evaluation results without invoking checkpoint/export reconciliation—even when model_latest_exported is false or legacy history exceeds saved state. This can
    leave a missing or stale model export. Separately, interrupting reference docking leaves a directory that causes FileExistsError on retry. Both were reproduced. See adaptive-stage handling (user-pipeline/src/igenvs_ultra/rl_workflow.py:538) and
    reference initialization (phase-10-rl-dev/src/igenvs_rl/trainer.py:129).

  - #7 — Failure handling is not effective everywhere. The wrapper still reuses complete manifests containing prepared=1, docked=0, in both single- and multi-GPU paths. Also, the dedicated iGenVS SIF loads its installed core unless explicitly redirected
    to repository source. With identical input and /bin/false as the docking executable, the installed core returned complete; the patched repository core correctly returned failed. See cached-job acceptance (user-pipeline/src/igenvs_ultra/
    workflow.py:1910). This is separate from the deferred AMD64 Docker work.

  - #8 — Explicit disable arguments can disappear. Runtime normalizes disable masks to an empty string, then treats that string as “no override.” My probes showed children retaining inherited CUDA_VISIBLE_DEVICES=0,1. The rl-generate path can reach this
    behavior. See environment forwarding (user-pipeline/src/igenvs_ultra/workflow.py:508).

  - #11 — Parent cgroup limits are ignored. Both RAM detectors miss intermediate ancestors. In v1 and v2 fixtures with only 1 GiB remaining under a parent limit, they reported the host’s 8 GiB instead. See wrapper detection (user-pipeline/src/
    igenvs_ultra/workflow.py:258) and core detection (iGenVS/src/igenvs/hardware.py:74).

  - #12 — Sampling can fail before microbatching helps. Backward gradient-equivalence tests pass, but sampling still requests all 4,096 sequences simultaneously, without an OOM fallback. Its FP32 key/value caches alone imply approximately 5.72 GiB,
    before other allocations. This is a code-derived requirement, not a measured GPU threshold. See sampling (phase-10-rl-dev/src/igenvs_rl/policy.py:208).

  - #10 — Generated-screen admission remains synchronous. SQLite improvements and external-screen overlap are present, but generated screening still completes admission before launching subsequent generation/scoring. The synthetic 100k-row timing does
    not establish that the original 10-million-molecule scaling bottleneck is resolved. See generated-screen scheduling (user-pipeline/src/igenvs_ultra/workflow.py:4202).

  Validation: repository suites produced 129 passed / 7 skipped in the iGenVS SIF and 12 passed in the gMol SIF. Release checks passed. Additional audit probes produced 8 passes and 14 failing parameterized cases covering the gaps above; reproductions
  remain outside the repository (/tmp/igenvs-review-CJfyPXjs/test_audit_regressions.py).

  No repository files or scientific protocols were changed. The final worktree snapshot and protocol/freeze hashes exactly match the starting state.

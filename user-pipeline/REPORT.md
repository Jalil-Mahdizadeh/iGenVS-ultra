# User pipeline report

Status: complete

## Delivered workflows

- `igenvs-ultra dock`: a separate regular-iGenVS path that prepares or reuses a
  target and delegates external-library or iGen3 input to the released
  `igenvs screen` workflow. Native docking scores, manifests, and optional
  merged/individual poses are preserved. Original regular defaults are kept:
  RL-nonisomeric generation, balanced search, and merged poses.
- `igenvs-ultra run`: the end-to-end ultra path: target preparation, fixed
  UDRL-train/UDRL-valid docking, frozen three-seed TH0 fitting, optional 1-5
  released AL rounds, and bounded final model screening.
- `fit` and `screen`: split ultra operation for fitting once and screening many
  libraries. `doctor` and `status` cover both job types.

The regular and ultra workflows do not invoke one another implicitly. An ultra
shortlist can be docked explicitly by supplying its `results.csv` to `dock`.
The regular wrapper exposes 53 of the original screen CLI's 56 flag names. The
three omitted flags are the expert hand-authored geometry path (`--center`,
`--size`, and `--adgpu-grid`); target geometry is deliberately derived from the
user's bound ligand or loaded from a prepared target.

## User-facing behavior

- Target input accepts a complex PDB, separate receptor PDB plus aligned 3D
  ligand SDF, or an existing prepared iGenVS target.
- Final ultra input accepts CSV/TSV/SMI libraries or streamed iGen3 generation.
- Stream size is selected from GPU memory, available host memory, and disk, and
  remains independently overrideable from iGen3's internal batch tuner.
- Exact canonical-SMILES deduplication persists across stream batches.
- Ultra output can retain all scored rows or only rows at/above a 0-1 ensemble
  threshold.
- Embeddings remain in memory and are not persisted by default;
  `--save-embeddings` retains the original released NPZ path.
- Atomic manifests support identical-command resume and reject incompatible
  configuration reuse.
- Reference/AL docking always uses four hardware-independent logical shards,
  matching the released Phase-9 execution. Available GPUs schedule those
  unchanged units with disjoint CPU affinity, so GPU count cannot alter seeded
  UniDock batch composition.
- Generated streaming and scoring use all visible GPUs by default. iGen3
  logical shards and contiguous encoder-batch-aligned score shards receive
  disjoint CPU/GPU affinity; score shards are merged in source order.
  `--generation-logical-shards` can be fixed across hardware-count benchmarks,
  while `--screen-gpus 1` reserves the remaining devices when desired.
- `screen-fast N` is a count-only generated-screen interface. Persistent iGen3
  and gMolAI/head workers, amortization-aware batch planning, exact-score
  replenishment, surplus retention, affinity-derived parallel RDKit work, and
  hardware/software/model-keyed profiles require no performance input from the
  user and re-plan safely on a different machine.

## Speed optimization qualification

The code-frozen real one-GH200 smoke produced exactly 8,192 durable scores in
52.71 seconds. The final 200,000-molecule all-score medium run took 85.16
seconds while creating a new encoder profile and 75.17 seconds when reusing it,
or 9.58 million molecules/hour. Cold and cached outputs were byte-identical. The prior
best automatic medium run took 104.37 seconds; reusable wall therefore fell
28.0% and throughput rose 38.8%.

The planner selected an iGen3 batch of 65,280 from live memory and backend
limits, 64 canonicalization workers from the 72-core affinity, and the released
gMolAI batch 512 after measuring 53,113 rows/s on its 49,152-row calibration
panel. An explicit compile probe produced identical output but was slower end
to end, so compilation remains off. These are smoke/medium qualification
results, not the pending full one-/two-/four-GPU benchmark. See
`benchmarks/RESULTS.md` for timings and boundaries.

## Verification

- Twenty-five lightweight unit/regression tests passed (seven environment-gated
  integration tests skipped in the host-only invocation).
- Real regular iGenVS integration passed: two molecules prepared and docked,
  native `results.csv` produced, and an identical rerun resumed without
  modifying results.
- A second physical smoke run prepared a fresh 1IEP target directly from its
  complex PDB and ligand selector (including the 5 A pocket), then docked all
  three test molecules successfully.
- Real ultra external-screen integration passed through the released iGenVS
  and gMolAI SIFs: duplicate and invalid inputs were rejected, two molecules
  were encoded/scored, and no embedding bundle remained by default.
- Real ultra iGen3 streaming integration passed across two independent
  generation/validation/encoding/scoring batches with eight globally unique
  final molecules.
- Real gMolAI model-operation integration passed for both default embedding
  non-persistence and explicit embedding retention.
- A 200,000-row one-GH200 regression compared the former temporary-NPZ path
  with the new inference-only in-memory path. Results were byte-identical and
  full screen wall fell from 167.69 to 37.20 seconds (4.51x); filesystem output
  fell 78.8%, with a measured 1.50x peak-RSS tradeoff. Policy-edge cases were
  also checked directly against the released gMol canonicalizer.
- A fresh three-member 1ERR TH0 was trained for all seven epochs from the
  released UDRL docking tables (197,048 successful training rows), evaluated
  on UDRL-valid, and resumed without rewriting any checkpoint.
- A real Round-1 acquisition scored all one million AL-set-1 embeddings and
  produced 30,000 unique selections with exact 15,000/7,500/7,500
  exploitation/uncertainty/diversity quotas. Every parsed selection field
  matched the released Phase-9 1ERR acquisition, and its rerun resumed
  identically.
- Using the released Round-1 docking labels, the cumulative 226,707-row ALTH1
  refit completed; every parameter tensor in all three resulting checkpoints
  exactly matched the corresponding released Phase-9 1ERR checkpoint.
- `doctor --no-gpu` passed for released assets, both SIFs, iGenVS, and the model
  self-test.
- Mutation-free dry runs passed for both target input layouts, regular iGen3
  generation, and a 10-million-molecule ultra screen. On the current GH200,
  the latter selected ten one-million-row outer batches.

The expensive future-user operation (300,000 reference dockings plus up to
150,000 AL dockings for a new target) was not redundantly launched as a test.
Its orchestration is covered by the released phase artifacts, dry-run plans,
alignment checks, and component/integration tests above.

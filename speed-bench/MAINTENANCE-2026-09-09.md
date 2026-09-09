# Operational maintenance validation — 2026-09-09

This supplements, and does not replace, the frozen September 6 benchmark or RL
scientific acceptance evidence. The scientific protocol, freeze, model weights,
chemistry policy, docking/search settings, encoder boundaries, rewards, and
optimizer/precision are unchanged.

## Correctness and resource checks

Final validation covers **185 distinct passing tests**. All ten environment/
opt-in skips in the main suite were exercised successfully in the supplemental
runs below; release-check tests overlap these tests and are not counted again.

| Test set | Environment | Result |
| --- | --- | ---: |
| Core, user-workflow, and RL regression suites | `iGenVS.SIF` | 164 passed; 10 initially skipped |
| gMol chemistry, inference, policy equivalence, training, acquisition, and resume | `gmolai-pyg-25.09-arm64.sif` | 15 passed |
| Public docking/resume, external/generated screening, and source-bootstrap failure injection | Standard-library host driver; computation in the dedicated SIFs | 4 passed |
| Real-weight RL CLI recovery and constrained-memory sampling | `iGenVS.SIF` with CUDA | 2 passed |
| Release checker, model/protocol hashes, metadata, and whitespace | `iGenVS.SIF` | Passed; 53 tests passed and 8 environment/opt-in skips |

The gMol checks include all-rejected retained and ephemeral chunks. Released
acquisition rows and retrained ensemble weights match the checked-in reference.
JUnit results are under `/tmp/igenvs-fixes-ppuyFR` (`core-results.xml`,
`gmol-all-results.xml`, and `rl-real-results.xml`). Test jobs and outputs remain
outside the repository.

- Real process-kill tests cover admission marker/database publication; logical
  four-shard resumes cover one/two GPUs, small/sparse inputs, and exact coverage.
- Real SIF launch, without a source `PYTHONPATH` override, prepares ethanol and
  substitutes `/bin/false` for Uni-Dock: the job correctly returns failure with
  `prepared=1, docked=0` and a failed manifest.
- Adaptive stopping tests cover new progress, legacy history ahead of checkpoint,
  and pre-existing stopping decisions in both public and development wrappers.
  CPU-only recovery does not sample or train. Reference and torn-CSV recovery
  tests cover interrupted initialization and publication.
- A real-weight, tiny **synthetic-oracle** CLI smoke job checks training, missing
  export repair, and idempotent absolute-update resume. Recovered export bytes
  and checkpoint hashes match; this is not a scientific acceptance experiment.
- Actual 4,096-draw RL sampling fits a 2 GiB PyTorch allocator limit on the GH200
  after retrying physical chunks at 2,048 then 1,024. Peak reserved allocation:
  2,128,609,280 bytes. Tokens and final RNG state exactly match the original
  full-batch sampler on that device. This is not a total-job VRAM guarantee:
  the driver, docking, and other phases need additional resources.
- Backward tests compare full-batch gradients and verify that an OOM retry clears
  partial gradients. Cgroup v1/v2 tests cover unlimited/loose children, constrained
  parents, sibling usage, and exhausted limits. GPU masks reach real children.

## Generated-screen performance and scale

The maintained controller overlaps next-batch validation/admission with current
scoring, retaining a one-batch look-ahead and controller-only SQLite access.
Stable IDs, input order, encoder grouping, exact counts, and rejection
replenishment are regression-tested. Admission timing now includes identity
insertion and database commit.

The public 10-million-score run completed on the single available GH200 with a
fresh automatic-profile cache, the two dedicated SIFs, and the released 4ag8
heads. It admitted 10,000,009 unique molecules, replenished nine encoder
rejections, and independently validated all five numeric score fields for
exactly 10,000,000 output rows.

| Measurement | Result |
| --- | ---: |
| Public-command wall time | 1,194.584 s (19.91 min) |
| Stream batches | 11 |
| Generation critical-stage sum | 1,003.807 s |
| Encoding critical-stage sum | 488.514 s |
| Admission sum, before the final bulk-index refinement | 112.570 s |
| Final admission implementation, isolated replay of the same 10,000,009 rows | 74.023 s |
| Prepared and rejection CSVs in that replay | All byte-identical |

The final refinement uses a **chunk-bounded** first-occurrence map, sorted bulk
SQLite inserts, and the owner index instead of a temporary B-tree plus repeated
candidate/index joins. Identities remain uncommitted until after the durable
marker is published. Process-kill, duplicate/reference, resume, ordering, and
real multi-batch screening tests pass after this refinement. A fresh database
replay of the entire real library verifies both ownership count and every
prepared/rejection CSV hash. The full expensive GPU pipeline was measured before
this final admission-only refinement, not rerun afterwards; the 74.023 s replay
is **not** a measured new end-to-end wall time or a controlled speedup ratio.

For context, the historical one-GPU wall time was 1,151.134 s. This operational
sample does not establish an end-to-end speedup; it adds durable recovery/state
guarantees and tests the revised scheduling. No new two-/four-GPU scaling result
is claimed. Both the original matrix and this run are single samples, and the
isolated admission replay has different contention from the full pipeline.

Raw evidence for this run is outside the repo under
`/tmp/igenvs-fixes-ppuyFR/screening-10m` and
`/tmp/igenvs-fixes-ppuyFR/admission-10m`. Result CSV SHA-256:
`d4f617c4c116fc86c2d08e0282b0b46b80cb307ead2ace92b146bc38ae21f9fe`.

Reproduce outside the repository:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/benchmark-screening-maintenance.py \
  --output-dir /tmp/igenvs-maintenance-screen --count 10000000
```

The runner preserves logs, source hashes, hardware, counts, and stage timings
under the chosen output directory. It does not overwrite frozen benchmark data.

To repeat just admission in the iGenVS SIF against that real generated library,
bind the repository and source/output directories, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/benchmark-admission-maintenance.py \
  --source-screen /tmp/igenvs-maintenance-screen/job/screens/fast-10000000 \
  --output-dir /tmp/igenvs-maintenance-admission
```

## Integrity

- RL protocol SHA-256:
  `4b9aa7fad0e563bddb28de5edc96d061ac24b1165b4f324a9711e60416d0c03b`.
- Historical freeze SHA-256:
  `32bb9d74ef4354c9fab44a9f01bfe037e2d889b39043854c27f662040e08e3e2`.
- Operational source overrides live in `phase-10-rl-dev/maintenance.json`, whose
  own hash is checked by the public RL launcher. AMD64 Docker images are deferred.

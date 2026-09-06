# Uni-Dock 1.2.0 GH200 performance benchmark — 2026-08-30

## Result

For the tested 1iep workload, the fastest production configuration that keeps
Uni-Dock's explicit-receptor refinement is:

- `search-mode=fast`
- `num-modes=1`
- `refine-step=3`
- `prep-mode=standard`
- `pose-output=none`
- outer batch `32768`
- 64 preparation workers and 32 validation workers per GPU
- node-local scratch
- one Slurm task and 72 bound CPU cores per GH200

It sustained **85.82 successful ligands/s by process wall time** on one GH200
and **370.42 successful ligands/s by complete Slurm wall time** on four GH200s
for the sustained two-batch-per-GPU run. The raw one-GPU docking invocation
peaked at **98.43 successful ligands/s**; the four concurrent docking streams
reached **404.66 successful ligands/s**. The scheduler-visible sustained rate is
94.1% of four times the measured one-GPU raw peak.

The no-refinement ceiling is 92.87 successful ligands/s end-to-end on one GPU,
but it is not a production recommendation: it changes rankings and emitted
pathological scores. Reducing refinement from three steps to one did not make
the run faster and increased docking failures.

## Hardware and protocol

| Item | Value |
| --- | --- |
| Cluster/node | Arrhenius HPU, interactive job 1790333 plus one-node Slurm runs |
| GPU | NVIDIA GH200 120GB, compute capability 9.0 |
| Single-GPU CPU allocation | 72 cores |
| Four-GPU CPU allocation | 4 tasks x 72 bound cores, one task/GPU |
| Container | ARM64 CUDA 12.8 SIF |
| Uni-Dock | 1.2.0, commit `95e409172b15dec0989aea70b0f2328e8ca52025` |
| Receptor | Uni-Dock 1iep test receptor, SHA-256 `761469710e3915b89b274483076dfe49754be7956661bc42f4cc36a182399d59` |
| Box center | 15.190, 53.903, 16.917 A |
| Box size | 20 x 20 x 20 A |
| Scoring/search | Vina, Uni-Dock `fast` (exhaustiveness 128, max step 20) |
| Output | Best score only; one mode; explicit-receptor refinement step 3 |
| Ligands | iGen3 RL non-isomeric valid unique SMILES |
| Main seeds | docking 401; integrated generation 405 |

This is a throughput engineering fixture, not a target-specific enrichment or
pose-accuracy validation. Batch optima can change with receptor, box, ligand
complexity, GPU, search mode, and requested output.

## One-GPU measurements

Rates below use successful molecules, not merely attempted inputs.

| Run | Successful | Process wall (s) | Manifest rate (successful/s) | Process-wall rate (successful/s) |
| --- | ---: | ---: | ---: | ---: |
| Baseline external 20k: 16K batches and pose output | 19,897 | 258.65 | 77.21 | 76.93 |
| Optimized external 20k: standard prep, refined, score only, 32K batch | 19,875 | 231.58 | 85.99 | **85.82** |
| Fast-prep variant, otherwise optimized | 19,892 | 231.71 | 86.02 | 85.85 |
| Refinement step 1, standard prep | 19,666 | 231.54 | 85.09 | 84.93 |
| No-refinement ceiling, fast prep | 19,983 | 215.18 | 93.06 | **92.87** |
| Integrated iGen3 generation -> validation -> docking, 20k | 19,907 | 271.99 | 73.47 | **73.19** |

The refined production path is 11.6% faster than baseline by external process
wall. The result payload fell from approximately 60.5 MB with poses/log volume
to 8.0 MB for ranking-only output, about an 87% reduction.

The integrated run spent 24.06 s in iGen3 generation. This confirms that
docking, not generation, controls steady-state single-GPU throughput.

## Docking batch sweep

The tuner prepares one representative ligand set and times the Uni-Dock
invocation. Only measurements with at least 99% finite scores are eligible.

| Search mode | Outer batch | Successful/attempted | Seconds | Successful/s |
| --- | ---: | ---: | ---: | ---: |
| fast | 8,192 | 8,162 / 8,192 | 95.14 | 85.79 |
| fast | 16,384 | 16,317 / 16,384 | 168.34 | 96.93 |
| fast | 32,768 | 32,596 / 32,736 | 331.15 | **98.43** |
| balance | 8,192 | 8,144 / 8,192 | 279.74 | 29.11 |
| balance | 16,384 | 16,306 / 16,384 | 550.26 | 29.63 |

The measured peak is 32,768. A 16K near-peak choice required two invocations for
the 20K full run and achieved only 81.57/s, versus 85.99/s with one 32K
invocation. The tuner and GH200 hardware heuristic now select 32,768. The
profile retains every measurement so the decision is auditable.

`fast` is about 3.3x faster than `balance` in this fixture. It is an official
Uni-Dock search preset, but its target-specific enrichment and pose quality must
be validated before it replaces `balance` in scientific production.

## CPU preparation sweep

| Preparation | Workers | Prepared/attempted | Seconds | Prepared/s |
| --- | ---: | ---: | ---: | ---: |
| standard | 32 | 8,187 / 8,192 | 18.72 | 437.28 |
| standard | 64 | 8,187 / 8,192 | 16.86 | **485.54** |
| fast | 32 | 8,187 / 8,192 | 15.89 | 515.23 |
| fast | 64 | 8,187 / 8,192 | 15.87 | 515.91 |

Fast preparation skips separate force-field minimization, but it did not improve
20K end-to-end time because docking dominates. It materially changed rankings:
Spearman rho was 0.771 and top-1% overlap was 39.9% versus standard preparation.
Therefore standard preparation remains the optimized default.

Preparation of the next outer batch is double-buffered with GPU docking. The
single-batch 20K fixture spent about 26 s waiting for standard preparation and
about 198 s in Uni-Dock; during search, the GH200 was observed at 100%
utilization and about 393 W.

## Refinement and score-output checks

- Three refinement steps versus one had Spearman rho 0.997 for common finite
  results and 99.0% top-1% overlap, but one step did not reduce runtime and
  increased docking failures from 108 to 317.
- Refined versus `--no-refine` had Spearman rho 0.983 and 83.4% top-1% overlap.
  The no-refinement run emitted a maximum score of 38,047,160 versus 270 for its
  refined comparator, so no-refinement is only a throughput ceiling.
- Compact score-only output preserves search, rescoring, and refinement. On a
  1,024-ligand cross-check, all 1,018 finite compact scores matched pose-file
  scores to pose-output rounding (maximum difference 0.000499).
- Two independent 1,024-ligand score-only runs were byte-identical after the
  deterministic, lock-free Uni-Dock loader patch.
- Score-only output did not materially reduce GPU kernel time; its benefit is
  deterministic result handling and avoiding pose serialization, parsing, and
  persistent storage.

## Four-GPU scale-out

The definitive sustained run, job 1793012, screened 262,144 inputs as two exact
32,768-input batches per GPU:

| Shard | Successful | Manifest elapsed (s) | Docking (s) | End-to-end successful/s |
| --- | ---: | ---: | ---: | ---: |
| 0 | 65,194 | 689.92 | 639.78 | 94.49 |
| 1 | 65,196 | 692.12 | 644.43 | 94.20 |
| 2 | 65,168 | 692.36 | 637.83 | 94.12 |
| 3 | 65,215 | 679.36 | 633.02 | 96.00 |
| Aggregate, complete Slurm wall | **260,773** | **704.00** | — | **370.42** |

The slowest in-process shard span gives 376.64/s and the concurrent Uni-Dock
invocation span gives 404.66/s. The conservative scheduler-visible rate is
370.42/s for successful scores and 372.36/s for attempted inputs, with a 99.48%
overall success fraction. The four shards were chemically balanced: mean heavy
atoms ranged only 22.07–22.10 and mean rotatable bonds 4.06–4.07.

Generation of the complete 262,144-molecule library took 65.22 s by outer
process wall. Adding that observed time to the 704 s Slurm screen gives
**339.01 generated-and-screened successful molecules/s** for the two-stage
four-GPU workflow.

Affinity and amortization both matter:

| Node run | Inputs | Slurm wall (s) | Successful/s |
| --- | ---: | ---: | ---: |
| Unbound/underallocated CPU topology, job 1792935 | 80,000 | 429 | 185.47 |
| 72 cores/task plus CPU/GPU binding, job 1792955 | 80,000 | 259 | 307.13 |
| Same binding, two full batches/GPU, job 1793012 | 262,144 | 704 | **370.42** |

Core/GPU binding improved the short node run by 65.6%; amortizing startup and
initial preparation added another 20.6%. Do not remove `--cpu-bind=cores`, the
one-task-per-GPU layout, or the 72-core task allocation in
`slurm/screen_4gpu.sbatch`.

## iGen3 generation and four-model verification

Large-batch RL non-isomeric generation is substantially faster than docking:

| Requested | Sampled candidates | Generator time (s) | Outer process wall (s) | Output/s, generator |
| ---: | ---: | ---: | ---: | ---: |
| 40,000 | 65,536 | 20.16 | 31.16 | 1,984.3 |
| 80,000 | 98,304 | 24.52 | 33.94 | 3,262.7 |
| 262,144 | 294,912 | 53.50 | 65.22 | 4,899.4 |

All four embedded checkpoints were loaded on the GH200 and each produced 32/32
RDKit-valid, unique SMILES: `base-isomeric`, `base-nonisomeric`,
`rl-isomeric`, and `rl-nonisomeric`.

## What changed

- Added compact score-only Uni-Dock output without disabling normal refinement.
- Replaced critical-section batch loading with deterministic indexed OpenMP
  loading, preserving input/result order.
- Added standard/fast preparation modes, persistent worker-local Meeko setup,
  bounded 64-worker preparation, and double-buffered CPU/GPU execution.
- Added ranking-only, merged-pose, and individual-pose output policies.
- Suppressed hot-path engine verbosity and made refinement controls explicit.
- Extended tuning to 32K and made the reliable measured peak the selected batch.
- Made the GH200 auto batch 32K.
- Added exact sharding and topology-aware four-GPU Slurm launch scripts.
- Kept generation in a separate process so PyTorch releases GPU memory before
  docking.
- Preserved typed preparation/docking failures and finite-score validation.

## Reproduce the production path

Tune once with representative chemistry:

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs tune-docking \
  --input pilot.smi \
  --input-format smi \
  --receptor receptor.pdbqt \
  --center 15.190 53.903 16.917 \
  --size 20 20 20 \
  --search-mode fast \
  --prep-mode standard \
  --batch-sizes 8192,16384,32768 \
  --profile gh200-fast.json
```

Run the external-library path:

```bash
OMP_NUM_THREADS=72 apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input library.csv \
  --receptor receptor.pdbqt \
  --center 15.190 53.903 16.917 \
  --size 20 20 20 \
  --search-mode fast \
  --batch-size auto \
  --batch-profile gh200-fast.json \
  --prep-workers 64 \
  --validation-workers 32 \
  --prep-mode standard \
  --pose-output none \
  --output-dir outputs/screen
```

For four GPUs, export the receptor, box, input, output, and project variables
documented in `slurm/screen_4gpu.sbatch`, then submit that file. Use
`SLURM_TMPDIR` or node-local scratch for temporary ligand PDBQTs.

## Artifact map

- `summary.json`: machine-readable headline results and container provenance.
- `baseline/`: original tuner profiles and 20K manifest.
- `optimized/batch-fast-final.json`: final peak-selecting 32K batch profile.
- `optimized/preparation-workers.json`: CPU preparation sweep.
- `optimized/quality-comparisons.json`: preparation/refinement rank comparisons.
- `optimized/screen-fast-20k-standard/`: recommended one-GPU result and manifest.
- `optimized/screen-generated-20k/`: integrated iGen3-to-docking result.
- `optimized/screen-fast-80k-4gpu-local/`: short topology-aware four-GPU run.
- `optimized/screen-fast-262144-4gpu/`: definitive sustained four-GPU run and summary.
- `model-smoke/`: valid unique output from all four iGen3 models.
- `final-sif-smoke/`: post-rebuild end-to-end refined score-only GPU check.
- `logs/`: compact process timing records; bulky stdout/stderr remains local.

Raw generated libraries, result tables, validation databases, and scheduler logs
remain in this benchmark directory locally but are excluded from Git.

Docking scores are ranking features under one fixed protocol. They are not proof
of binding affinity, activity, selectivity, or safety.

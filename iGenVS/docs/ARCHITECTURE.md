# Architecture

## Target preparation

Two mutually exclusive inputs converge on the same immutable target contract:

```text
complex PDB + exact HETATM residue selector ─┐
                                             ├─> reference heavy-atom bounds
receptor PDB + aligned single-ligand SDF ────┘       -> padded box + receptor PDB
                                                       +-> Meeko receptor PDBQT -> Uni-Dock
                                                       +-> Meeko GPF -> AutoGrid maps/FLD
                                                       |                  -> AutoDock-GPU
                                                       +-> checksummed target manifest
```

Complex mode removes only the selected reference residue and rejects ambiguous,
multi-model, or explicitly covalent inputs. Pair mode requires the SDF
coordinates to already share the receptor coordinate frame. Both modes retain
the source geometry, pocket definition, receptor, grid maps, and file hashes in
one target bundle. Screening and tuning resolve `--target` to the correct
engine assets, verify all checksums and geometry, and record target provenance
in their own manifests. This work happens once and is outside the timed
high-throughput loop.

AD4 maps have even 0.375 Angstrom grid intervals. Their physical dimensions
are rounded outward to cover the requested ligand-derived box; the loader
independently verifies center, spacing, point limits, coverage, checksums, and
manifest geometry before dispatch.

## Screening execution


The single-GPU execution path is deliberately staged:

```text
iGen3 subprocess (optional, any of 4 checkpoints)
             |
             v
raw SMILES -> RDKit sanitize/canonicalize -> exact deduplicate
                                               |
                                               v
                    bounded CPU preparation queue (RDKit ETKDG + Meeko)
                                               |
                                               v
                         +---------------------+--------------------+
                         |                                          |
                         v                                          v
         Uni-Dock memory-aware batch                 AutoDock-GPU file-list shard
                         |                             (optional MPS workers)
                         +---------------------+--------------------+
                                               v
                 finite-score check -> typed result + optional pose stream
```

Generation and docking do not share a GPU concurrently. Generation is much
faster than scientifically useful docking and its process exit releases all
PyTorch allocations. Ligand preparation is instead overlapped with docking
through a two-batch window and a persistent CPU process pool. Both backends
return the same typed result contract while naming their engine and score family.
Preparation workers follow physical cores, inherited affinity, and available
RAM. ETKDG has bounded attempt/time budgets and likely stragglers enter the
queue first, preventing ultimately failing 3D construction from creating an
unbounded first-GPU barrier.
For a sufficiently large Uni-Dock job, a CPU/outer-batch-aware ramp is emitted
first and the steady batch is submitted before that first engine call. Small
jobs retain one batch so fixed engine startup cannot outweigh preparation
overlap.

## Batch controls

There are four distinct controls:

1. iGen3 `auto` probes the largest generator KV-cache batch that fits CUDA.
2. iGenVS chooses an outer ligand count to amortize receptor/map and process
   overhead while bounding temporary files and CPU queue depth.
3. Uni-Dock estimates its own per-size-class GPU occupancy and splits the outer
   batch according to free memory.
4. AutoDock-GPU streams one file-list shard per process; optional same-GPU
   workers use an owned run-private CUDA MPS service to fill a large GPU without
   changing ligand-level search. Torsion-aware assignment balances predicted
   work across those processes.

The empirical tuner measures layer 2 under a fixed engine protocol without
including one-time ligand preparation. Saved profiles are engine-specific and
restore both batch size and the AutoDock-GPU worker count. The patched Uni-Dock
backend preserves source order during parallel PDBQT loading, removes the loader
critical section, and emits compact best scores without disabling search or
refinement. The AutoDock-GPU adapter disables DLG/clustering output, requests
only the best pose, strictly parses XML, preserves input order across concurrent
workers, and retries missing outputs once.

## Scale-out

`--num-shards N --shard-index I` assigns source row `r` to
`(r - 1) % N == I`. Each shard owns its validation database, result CSV, logs,
and pose stream. This makes tasks failure-isolated and avoids shared writes.
Concatenate result CSV bodies and pose streams after all shards reach terminal
status; retain each manifest for provenance.

The four-GPU launcher gives each Slurm task one GPU and one output shard. For
AutoDock-GPU concurrency it also gives each task a private CUDA MPS pipe/log
directory and cleans up that GPU's daemon when the task exits.

For de novo generation at scale, generate a durable library once and then run
the external-library sharded path. Independently generating the same seed on
each task wastes generator work and complicates cross-shard deduplication.

## Chemistry boundary

The production contract is rigid-receptor, noncovalent docking after RDKit and
Meeko preparation.

| Engine | Scoring | Prepared atoms | Active torsions |
| --- | --- | ---: | ---: |
| Uni-Dock 1.2.0 | Vina or Vinardo | 300 | 48 |
| AutoDock-GPU 1.6 | AutoDock4 | 256 | 57 |

Limits are enforced before dispatch according to the selected engine. Scores
also remain typed by engine because Vina/Vinardo and AD4 values cannot be merged
into one numeric ranking. Flexible receptors, macrocycles, metals, covalent
ligands, and unusual chemistry need separate qualified protocols rather than
silent fallback.

The matched benchmark found moderate cross-engine agreement (Pearson 0.302,
Spearman 0.395, and 12.92% top-1% overlap); the complete reproducible analysis
is in [`benchmarks/correlations`](../benchmarks/correlations).

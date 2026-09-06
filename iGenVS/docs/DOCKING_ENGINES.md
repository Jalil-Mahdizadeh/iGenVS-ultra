# Docking engines

iGenVS 0.2 provides two GPU backends behind the same validation, ligand
preparation, failure handling, result CSV, target provenance, and sharding
contract.

| Engine | Pinned source | Score | Intended role |
| --- | --- | --- | --- |
| Uni-Dock | 1.2.0, commit `95e409172b15dec0989aea70b0f2328e8ca52025` | Vina or Vinardo | Primary high-throughput engine |
| AutoDock-GPU | 1.6, commit `e63e6f6280ebfad18caa3e8f48afdc269e79e063` | AutoDock4 | Independent second engine and AD4 lineage |

AutoGrid is pinned at 4.2.9, commit
`6d2847beaeac8ff43ca99094707fd74e3ca1ff37`. AutoDock-GPU and AutoGrid are
installed from their official source repositories without changing their
search or scoring equations.

The v4.2.9 AutoGrid source still embeds the upstream banner `AutoGrid 4.2.7.x`.
Runtime manifests record that actual banner, while SIF labels record the exact
v4.2.9 tag commit, avoiding a false version substitution.

## Shared workflow

```text
iGen3 (any of four models) or CSV/TSV/SMI
                    |
                    v
       RDKit validation and exact deduplication
                    |
                    v
          RDKit 3D embedding plus Meeko PDBQT
                    |
             +------+------+
             |             |
             v             v
      Uni-Dock/Vina   AutoDock-GPU/AD4 maps
             |             |
             +------+------+
                    v
      finite score, typed status, optional pose
```

Select a backend with `--engine unidock` or `--engine autodock-gpu`. The
default remains Uni-Dock. `--scoring auto` resolves to Vina for Uni-Dock and
AD4 for AutoDock-GPU.

## One reusable target supports both engines

`igenvs prepare-target` now creates the receptor PDBQT, ligand-derived box,
AutoGrid parameter file, AD4 affinity/electrostatic/desolvation maps, FLD
descriptor, and checksummed provenance in one directory. This applies to both
accepted input modes:

```bash
igenvs prepare-target \
  --complex complex.pdb \
  --ligand-id A:LIG:501 \
  --output-dir targets/example

igenvs prepare-target \
  --receptor receptor.pdb \
  --reference-ligand bound_ligand.sdf \
  --output-dir targets/example
```

Screening verifies every target artifact before launching either backend.
Legacy schema-1 targets remain usable with Uni-Dock but lack the maps required
by AutoDock-GPU and should be regenerated for dual-engine use.

The AD4 grid uses 0.375 Angstrom spacing and even interval counts. Because
Meeko's raw conversion rounds dimensions inward, iGenVS first rounds every map
axis outward to the next 0.75 Angstrom multiple and then verifies the emitted
GPF. Requested and physical grid geometry are both checksummed and recorded;
under-covered, off-center, oversized, or inconsistent targets fail closed.

## Uni-Dock

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input library.csv \
  --target targets/example \
  --engine unidock \
  --search-mode fast \
  --batch-size auto \
  --pose-output none \
  --output-dir outputs/unidock
```

Uni-Dock batches many ligands in one GPU invocation. The container retains the
narrow iGenVS patches for explicit GPU-memory caps, deterministic parallel
loading, and compact score-only output with refinement preserved.

## AutoDock-GPU

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input library.csv \
  --target targets/example \
  --engine autodock-gpu \
  --search-mode balance \
  --batch-size auto \
  --pose-output none \
  --output-dir outputs/autodock-gpu
```

The effort presets are explicit LGA run counts: `fast=10`, `balance=20`, and
`detail=50`. The default `balance` protocol uses ligand-based evaluation
heuristics, convergence autostop, and the AD local-search method. Override
these only as a deliberate protocol change with `--adgpu-runs`,
`--adgpu-evaluations`, `--adgpu-no-heuristics`, `--adgpu-no-autostop`, or
`--adgpu-local-search`.

The image contains CUDA builds with 64, 128, and 256 work items. The generic
`autodock_gpu` command selects the measured 64-work-item GH200 build; experts
can benchmark another one with `--adgpu-executable autodock_gpu_128wi` or
`autodock_gpu_256wi`. On the current GH200, the fast score-only pilot reached
12.08 successful ligands/s with the corrected full-coverage grid, six MPS
workers, four CPU threads per worker, and a 4,096-ligand outer batch. The full
20,000-input run sustained 12.05 successful dockings/s and 11.90 successful
molecules/s end-to-end. Treat this as a hardware/protocol profile, not a
universal constant. The complete tuning and matched Uni-Dock comparison are in
the [AutoDock-GPU 1.6 vs Uni-Dock 1.2.0 GH200 benchmark](../benchmarks/autodock-gpu-v1.6-vs-unidock-v1.2.0-gh200-20260830).

AutoDock-GPU processes a file list while OpenMP overlaps ligand setup and XML
result handling. iGenVS requests only the best pose, disables DLG and clustering
output, parses every XML score strictly, and retries missing outputs once in
isolated shards. Current prepared-ligand limits are 256 atoms and 57 torsions.

## Saturating large GPUs with CUDA MPS

A single 10- or 20-run ligand does not fill a GH200. iGenVS can split an outer
batch across independent AutoDock-GPU processes using `--adgpu-workers N`.
For `screen`, `auto` selects a conservative capacity-class prior from the
actual visible GPU. When more than one worker is selected, iGenVS starts a
run-private CUDA MPS daemon, exposes its private pipe only to child processes,
and stops it during cleanup. If MPS is unavailable, an automatic plan falls
back to one process; an unsupported explicit multi-worker request fails.
Flexible ligands are assigned by deterministic longest-work-first torsion
balancing to reduce the slowest-worker tail.

Experts can still tune and screen under the same pre-existing MPS setup:

```bash
export CUDA_MPS_PIPE_DIRECTORY="${SLURM_TMPDIR:-/tmp}/igenvs-mps-${SLURM_JOB_ID:-manual}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY}-logs"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

apptainer exec --nv containers/iGenVS.SIF igenvs tune-docking \
  --input pilot.csv \
  --target targets/example \
  --engine autodock-gpu \
  --search-mode fast \
  --adgpu-workers 6 \
  --batch-sizes 512,1024,2048,4096 \
  --profile gh200-adgpu-fast.json

apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input library.csv \
  --target targets/example \
  --engine autodock-gpu \
  --search-mode fast \
  --batch-profile gh200-adgpu-fast.json \
  --pose-output none \
  --output-dir outputs/adgpu-tuned

printf 'quit\n' | nvidia-cuda-mps-control
```

The profile records and automatically restores both the selected outer batch
and the measured worker count. Worker count, CPU threads, ligand chemistry,
maps, search protocol, GPU, and pose-output policy can all move the optimum;
retune after changing them. The profile's hardware snapshot also records the
MPS pipe, log, and active-thread-percentage environment so capped and uncapped
runs cannot be mistaken. The four-GPU launcher manages a separate MPS daemon
per assigned GPU when `IGENVS_ADGPU_MPS=1`.

The measured GH200 `fast` profile is included at
[`profiles/gh200-adgpu-fast.json`](../benchmarks/autodock-gpu-v1.6-vs-unidock-v1.2.0-gh200-20260830/profiles/gh200-adgpu-fast.json).
It is a reproducibility artifact and convenient starting point; tune again for
a different target, ligand distribution, GPU, or search protocol.

## Comparing results correctly

Uni-Dock Vina/Vinardo scores and AutoDock-GPU AD4 binding energies are produced
by different search and scoring protocols. Do not compare their numeric values
directly, merge them into one score column for ranking, or interpret a speed
ratio as an accuracy result. Compare throughput and finite-output yield for
engineering; compare each engine's ranking, enrichment, and pose recovery
against target-specific experimental controls for scientific qualification.

On the matched 80,000-input benchmark, the 79,606 paired finite results had
Pearson score correlation 0.302, Spearman rank correlation 0.395, and 12.92%
top-1% overlap. The relationship replicated on the 20,000-input run. See the
[`cross-engine correlation analysis`](../benchmarks/correlations) for
the source hashes, full rank-overlap table, molecular-size associations, and
reproduction command. Agreement does not establish accuracy.

Every result row names `docking_engine` and `scoring_function`, and every run
manifest records the exact engine controls, tool versions, target checksums,
worker count, summed worker time, and concurrency-correct docking wall time.

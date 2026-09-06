# AutoDock-GPU 1.6 vs Uni-Dock 1.2.0 — GH200 benchmark — 2026-08-30

## Result

iGenVS now exposes pinned AutoDock-GPU 1.6 as a second docking backend while
retaining Uni-Dock 1.2.0 as the default high-throughput engine. AutoGrid maps
are generated once with every schema-2 target, checksummed, geometry-verified,
and reused across batches and runs.

The optimized AutoDock-GPU `fast` configuration on one Arrhenius GH200 is:

- AutoDock-GPU 1.6 commit
  `e63e6f6280ebfad18caa3e8f48afdc269e79e063`;
- AutoGrid 4.2.9 commit
  `6d2847beaeac8ff43ca99094707fd74e3ca1ff37`;
- CUDA `sm_90`, `OVERLAP=ON`, 64 work items;
- six same-GPU processes under CUDA MPS;
- four OpenMP CPU threads per process;
- 4,096 prepared ligands per iGenVS outer batch;
- AD local search, ligand heuristics, convergence autostop, and 10 LGA runs;
- standard RDKit/Meeko preparation and no persistent pose output.

The corrected tuning sweep peaked at **12.0836 successful dockings/s**. A full
20,000-input run sustained **12.0451 successful dockings/s** and **11.8990
successful molecules/s end-to-end**, with a finite AD4 result for all 19,983
prepared ligands.

## Matched comparison

Both engines received the same deterministic external iGen3 RL non-isomeric
library, standard ligand preparation, receptor PDBQT, requested box, seed,
score-only policy, and one requested best result. The one-GPU input had 20,000
SMILES. The four-GPU input had 80,000 SMILES split deterministically into four
20,000-row shards.

| One GH200 | Uni-Dock 1.2.0 / Vina | AutoDock-GPU 1.6 / AD4 |
| --- | ---: | ---: |
| Input | 20,000 | 20,000 |
| Prepared | 19,983 | 19,983 |
| Successful docking results | 19,906 | **19,983** |
| Docking failures/non-finite outputs | 77 | **0** |
| Docking wall seconds | 198.483 | 1,659.013 |
| Successful docking/s | **100.2906** | 12.0451 |
| End-to-end successful/s | **86.7743** | 11.8990 |
| Successful/input yield | 99.530% | **99.915%** |

On this workload, Uni-Dock was 8.33 times faster by docking wall and 7.29
times faster end-to-end. AutoDock-GPU had the higher finite-output yield.

| Four GH200s, complete Slurm wall | Uni-Dock 1.2.0 / Vina | AutoDock-GPU 1.6 / AD4 |
| --- | ---: | ---: |
| Slurm job | 1799196 | 1799455 |
| Input | 80,000 | 80,000 |
| Prepared | 79,952 | 79,952 |
| Successful docking results | 79,606 | **79,952** |
| Docking failures/non-finite outputs | 346 | **0** |
| Complete Slurm wall seconds | 259 | 1,707 |
| Successful/s by Slurm wall | **307.3591** | 46.8377 |
| Successful/input yield | 99.5075% | **99.9400%** |

The four-GPU Uni-Dock rate was 6.56 times the AutoDock-GPU rate. AutoDock-GPU
scaled to 3.94 times its one-GPU end-to-end rate, or 98.4% parallel efficiency
by this conservative scheduler-wall comparison.

These rates do **not** establish that one engine is scientifically superior.
Uni-Dock `fast` and AutoDock-GPU `fast` are engine-specific protocols:
AutoDock-GPU uses 10 LGA runs, while Uni-Dock uses its own search preset and
explicit-receptor refinement. Vina scores and AD4 binding energies have
different equations and numeric scales. Compare throughput and finite-output
yield for engineering; compare enrichment, ranking, and pose recovery against
target-specific experimental controls for science.

## Cross-engine score and ranking correlation

The 79,606 molecules with finite scores from both engines in the 80K run had
Pearson score correlation 0.3021, Spearman rank correlation 0.3950, and Kendall
rank correlation 0.2730. Exact top-1%, top-5%, and top-10% overlap was 12.92%,
21.28%, and 27.85%, respectively. The independent 20K run reproduced the
global coefficients.

The full [correlation report](../correlations) contains source-file hashes,
tie-aware top-rank analysis, score distributions, molecular-size associations,
limitations, and the deterministic analysis script. This measures agreement,
not scientific accuracy, and does not make Vina and AD4 scores numerically
interchangeable.

## Hardware and target

| Item | Value |
| --- | --- |
| Cluster | Arrhenius HPU |
| GPU | NVIDIA GH200 120GB, compute capability 9.0 |
| CPU allocation | 72 cores per GPU/task |
| Driver/CUDA | NVIDIA 580.159.04; CUDA 12.8 container |
| Architecture | ARM64; docking binaries built for `sm_90` |
| Receptor PDBQT SHA-256 | `882874a97ba4f12dfa286963b6139a7b153051e10a6b393d8333ddd87b193e17` |
| Pocket metadata SHA-256 | `9cced296d520709af54209aeef2eda9f656cca030a8c6d00a1d5802278afdf3c` |
| Requested center | 15.190, 53.9025, 16.917 A |
| Requested size | 18.664 x 26.739 x 23.526 A |
| AD4 grid | 0.375 A spacing; 50 x 72 x 64 intervals |
| Physical AD4 size | 18.75 x 27.0 x 24.0 A |
| FLD SHA-256 | `b655e2aa80a1df55058957d826c568259124439ff9aed365761a0f16eafb5c00` |
| Input chemistry | iGen3 RL non-isomeric valid unique SMILES |
| Preparation | RDKit ETKDGv3 plus standard force-field relaxation, Meeko PDBQT |
| Main seed | 181129 |

The AD4 map is slightly larger than the requested box because AutoGrid requires
even 0.375 A interval counts. iGenVS rounds each axis outward and independently
verifies the GPF and manifest so the maps never silently under-cover the
requested pocket.

The matched Uni-Dock controls used an earlier copy of the same target bundle.
Its receptor PDBQT and pocket metadata are byte-identical to the corrected
bundle above; only the AD4 maps changed, and Uni-Dock does not consume them.

## Optimization sweep

Initial single-process tests showed that work-item width alone was not enough
to fill the GH200. The useful gain came from independent file-list processes
under CUDA MPS, followed by a larger outer batch to amortize startup.

Selected exploratory measurements used the earlier map geometry:

| Change | Successful/s | Interpretation |
| --- | ---: | --- |
| One 128-WI process, 64 ligands, 4 CPU threads | 5.497 | Under-filled GPU |
| One 64-WI process, 64 ligands, 4 CPU threads | 5.008 | Width alone was slower |
| One 256-WI process, 64 ligands, 4 CPU threads | 3.254 | Rejected |
| Four MPS workers, 64 WI, batch 512 | 11.610 | 2.11x over one 128-WI process |
| Six MPS workers, 64 WI, batch 512 | 12.452 | Near saturation |
| Eight MPS workers, 64 WI, batch 512 | 12.461 | No material gain over six |
| Four workers with 50% MPS cap | 11.160 | Cap reduced throughput |
| Four workers with 25% MPS cap | 8.314 | Cap rejected |

Six workers were selected over eight because their measured rates were within
0.1%, while six use fewer host processes, CPU threads, file descriptors, and
temporary files. Four CPU threads per worker beat the measured two- and
eight-thread variants.

After fixing map geometry to guarantee complete pocket coverage, the definitive
batch sweep was:

| Outer batch | Successful/attempted | Docking wall (s) | Successful/s |
| --- | ---: | ---: | ---: |
| 512 | 512 / 512 | 46.135 | 11.0980 |
| 1,024 | 1,024 / 1,024 | 88.372 | 11.5874 |
| 2,048 | 2,048 / 2,048 | 172.278 | 11.8878 |
| 4,096 | 4,093 / 4,093 | 338.724 | **12.0836** |

The 4,096 batch is 8.9% faster than 512. The earlier inward-rounded map
measured 12.6957/s at 4,096; the correct full-coverage map costs 4.8%
throughput and is the only result used as the production figure.

The first full integration baseline used four workers, a 512 batch, and the
pre-correction map, sustaining 9.6500 docking/s. The final full run sustains
12.0451/s despite the larger corrected grid. That 24.8% gain is useful
engineering history but is not a controlled map-to-map comparison.

## Per-shard four-GPU results

| Engine/shard | Prepared | Successful | Docking failures | Process elapsed (s) | Docking wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AutoDock-GPU 0 | 19,988 | 19,988 | 0 | 1,680.169 | 1,653.292 |
| AutoDock-GPU 1 | 19,992 | 19,992 | 0 | 1,679.909 | 1,664.978 |
| AutoDock-GPU 2 | 19,985 | 19,985 | 0 | 1,694.805 | 1,658.794 |
| AutoDock-GPU 3 | 19,987 | 19,987 | 0 | 1,683.828 | 1,654.421 |
| Uni-Dock 0 | 19,988 | 19,908 | 80 | 244.802 | 207.835 |
| Uni-Dock 1 | 19,992 | 19,898 | 94 | 246.350 | 199.764 |
| Uni-Dock 2 | 19,985 | 19,887 | 98 | 245.429 | 200.621 |
| Uni-Dock 3 | 19,987 | 19,913 | 74 | 231.761 | 199.835 |

## Reproduce

Prepare one dual-engine target:

```bash
apptainer exec containers/iGenVS.SIF igenvs prepare-target \
  --receptor receptor.pdb \
  --reference-ligand aligned_ligand.sdf \
  --output-dir targets/example
```

Tune AutoDock-GPU under CUDA MPS with representative chemistry:

```bash
export CUDA_MPS_PIPE_DIRECTORY="${SLURM_TMPDIR:-/tmp}/igenvs-mps-${SLURM_JOB_ID:-manual}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY}-logs"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

apptainer exec --nv containers/iGenVS.SIF igenvs tune-docking \
  --input pilot.smi \
  --target targets/example \
  --engine autodock-gpu \
  --search-mode fast \
  --adgpu-workers 6 \
  --adgpu-cpu-threads 4 \
  --adgpu-executable autodock_gpu_64wi \
  --batch-sizes 512,1024,2048,4096 \
  --profile gh200-adgpu-fast.json
```

Use the measured profile:

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input library.csv \
  --target targets/example \
  --engine autodock-gpu \
  --search-mode fast \
  --batch-size auto \
  --batch-profile gh200-adgpu-fast.json \
  --pose-output none \
  --output-dir outputs/adgpu-fast
```

For the four-GPU launcher set `IGENVS_ENGINE=autodock-gpu`,
`IGENVS_ADGPU_WORKERS=6`, `IGENVS_ADGPU_MPS=1`, and either
`IGENVS_BATCH_SIZE=4096` or `IGENVS_BATCH_PROFILE` before submitting
`slurm/screen_4gpu.sbatch`.

## Artifact map

- `summary.json`: machine-readable definitive metrics and provenance.
- `profiles/gh200-adgpu-fast.json`: directly loadable measured batch/worker
  profile.
- `raw/`: local libraries, full manifests, result tables, tuning logs, and
  Slurm logs; intentionally excluded from Git because of size.

The final SIF was also checked with `igenvs doctor`, fresh target generation,
real AutoDock-GPU score and merged-pose output, real Uni-Dock score-only
output, and all 47 packaged tests. All four embedded iGen3 checkpoints produced
32 valid unique molecules per model in the release-candidate check and were
loaded again after final promotion, producing 8 valid unique molecules each.

Docking scores are ranking features under fixed, scientifically validated
protocols. They are not proof of binding affinity, activity, selectivity, or
safety.

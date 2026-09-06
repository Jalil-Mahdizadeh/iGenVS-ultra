# iGenVS

iGenVS couples all four iGen3 transformer checkpoints to RDKit validation,
RDKit/Meeko ligand preparation, and either of two GPU docking engines. Uni-Dock
1.2.0 is the default high-throughput backend; AutoDock-GPU 1.6 is an independent
AutoDock4-scoring backend. The same workflow accepts generated molecules or an
external CSV/TSV/SMI library and records every rejected, failed, and
successfully docked molecule.

The engine sources are pinned exactly:

- Uni-Dock 1.2.0: `95e409172b15dec0989aea70b0f2328e8ca52025`
- AutoDock-GPU 1.6: `e63e6f6280ebfad18caa3e8f48afdc269e79e063`
- AutoGrid 4.2.9: `6d2847beaeac8ff43ca99094707fd74e3ca1ff37`

See [Docking engines](docs/DOCKING_ENGINES.md) for protocol controls, tuning,
CUDA MPS, limits, and the rules for comparing the two score families.

## Supported iGen3 models

Every model shipped by iGen3 is selectable with `igenvs screen --model`:

| Model | Family | SMILES representation |
| --- | --- | --- |
| `base-isomeric` | base transformer | isomeric |
| `base-nonisomeric` | base transformer | non-isomeric |
| `rl-isomeric` | QED/SA RL-tuned | isomeric |
| `rl-nonisomeric` | QED/SA RL-tuned | non-isomeric |

Both container formats embed all four vocabularies and checkpoints. The default
is `rl-nonisomeric`; choosing a model is explicit and does not require
rebuilding the image.

## Build a container

The repository provides two architecture-specific builds:

- `containers/iGenVS.def` produces the ARM64 Arrhenius GH200 SIF with native
  `sm_90` docking engines.
- `Dockerfile` produces a Linux AMD64 image with native engines for Ampere,
  Ada, Hopper, and Blackwell GPUs, including the NVIDIA RTX PRO 2000 Blackwell.

Both install CUDA PyTorch, iGen3, RDKit, Meeko, Open Babel, Uni-Dock,
AutoDock-GPU, AutoGrid, and iGenVS. A normal clone includes the required iGen3
sources, all four model bundles, and the pinned docking-engine sources. Large
generated files under `iGen3/benchmarks/` are deliberately excluded.

Both build helpers verify required source paths and fail if an image input has
tracked or untracked changes. Developer trees that retain the upstream nested
Git metadata additionally enforce the recorded upstream commits.

### ARM64 GH200 Apptainer image

```bash
./containers/build.sh
apptainer exec --nv containers/iGenVS.SIF igenvs doctor
```

The build is native to the current ARM64 node. An ARM64 SIF does not run on an
x86-64 host; rebuild the definition on the target architecture when moving to a
different cluster. The generated `containers/iGenVS.SIF` is intentionally
ignored by Git.

### AMD64 NVIDIA Docker image

BuildKit and an x86-64 build host are required to build the image. A
CUDA-capable host also needs a compatible NVIDIA
driver and the NVIDIA Container Toolkit to run it:

```bash
./containers/build-docker.sh igenvs-ultra/igenvs:latest
docker run --rm --gpus all igenvs-ultra/igenvs:latest doctor
```

The Docker entrypoint is `igenvs`. Bind the directory containing inputs and
outputs to `/work`; using the host UID/GID keeps generated files user-owned:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/work" \
  igenvs-ultra/igenvs:latest screen \
  --input examples/library.csv \
  --target targets/my-target \
  --batch-size auto \
  --output-dir outputs/docker-screen
```

The defaults compile Uni-Dock and AutoDock-GPU for Ampere, Ada, Hopper, and
Blackwell (`sm_80`, `86`, `89`, `90`, `100`, and `120`). A smaller
architecture-specific image can be selected at build time; see
[`containers/README.md`](containers/README.md). Batch and worker optima do not
transfer from GH200 to a smaller workstation GPU, so run `tune-docking` again
on the target hardware.

## Prepare a reusable docking target

Target preparation is a one-time CPU step. It writes a prepared receptor,
ligand-derived docking box, reference structure, AutoGrid maps for
AutoDock-GPU, checksums, and provenance into one directory. The same bundle is
then reusable by either engine for screening and tuning.

The input receptor must still be curated for the intended biological state:
resolve missing residues and alternate locations, choose protonation, and
decide which waters, metals, ions, and cofactors to retain. iGenVS removes only
the selected reference ligand from a complex; it does not silently remove other
hetero residues.

### Mode 1: protein-ligand complex PDB

```bash
apptainer exec containers/iGenVS.SIF igenvs prepare-target \
  --complex complex.pdb \
  --ligand-id A:LIG:501 \
  --padding 5 \
  --output-dir targets/my-target
```

The exact selector format is `CHAIN:RESNAME:RESSEQ`; an insertion code can be
appended to the residue number. A residue name alone, such as `--ligand-id LIG`,
is accepted only when it identifies exactly one HETATM residue. Ambiguous
selectors fail and list the available residues. Multi-model and covalently
connected complexes are rejected.

iGenVS removes the selected ligand, preserves the remaining receptor records,
uses the selected ligand's heavy-atom coordinates to calculate the box, and
then converts the receptor to PDBQT with Meeko.

### Mode 2: receptor PDB plus aligned ligand SDF

```bash
apptainer exec containers/iGenVS.SIF igenvs prepare-target \
  --receptor receptor_with_hydrogens.pdb \
  --reference-ligand bound_ligand.sdf \
  --padding 5 \
  --output-dir targets/my-target
```

The SDF must contain exactly one molecule with 3D coordinates in the same
coordinate frame as the receptor. A generic 2D SDF or an independently
generated conformer cannot identify the receptor pocket. iGenVS rejects an SDF
whose closest heavy atom is more than 8 Angstrom from the receptor, as well as
an exact overlap that indicates the ligand is still present. The receptor PDB
in this mode is expected to be ligand-free.

For both modes, each box axis is the reference ligand's coordinate extent plus
twice `--padding`; the center is the midpoint of its minimum and maximum
coordinates. The default padding is 5 Angstrom.

AutoGrid requires an even number of 0.375 Angstrom intervals. iGenVS rounds
each AD4 map dimension outward to the next valid 0.75 Angstrom multiple, never
inward, while retaining the requested box for Uni-Dock. The target manifest
records requested and physical map centers, sizes, spacing, and point counts;
loading fails if the maps do not fully cover the requested pocket.

The resulting bundle is:

```text
targets/my-target/
├── receptor.pdb
├── receptor.pdbqt
├── reference_ligand.pdb or reference_ligand.sdf
├── pocket.json
├── receptor.gpf
├── receptor.maps.fld
├── receptor.<atom-type>.map
├── receptor.e.map and receptor.d.map
├── receptor.maps.xyz and receptor.box.pdb
├── boron-silicon-atom_par.dat
├── autogrid.log
└── manifest.json
```
Every target and grid artifact checksum is verified whenever `--target` is
loaded. A modified
receptor, reference ligand, or pocket fails closed and must be prepared again.


For advanced protocols, the original explicit interface remains available:
prepare a receptor with `igenvs prepare-receptor`, then pass
`--receptor receptor.pdbqt --center X Y Z --size X Y Z` directly to
`screen` or `tune-docking`.

## Screen an external CSV library

The default CSV columns are `id` (auto-detected) and `smiles`; both can be
overridden. If “CVS” was intended, name the file with the standard `.csv` suffix.

```csv
id,smiles
aspirin,CC(=O)Oc1ccccc1C(=O)O
caffeine,Cn1c(=O)c2c(ncn2C)n(C)c1=O
```

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input examples/library.csv \
  --target targets/my-target \
  --engine unidock \
  --batch-size auto \
  --output-dir outputs/external
```

Disconnected structures are rejected by default so a salt/fragment operation
cannot silently change molecular identity. Use `--fragment-policy largest`
only when that policy is intentional. Validation uses canonical isomeric
SMILES and exact disk-backed deduplication.

Run the same admitted library through the AD4 backend by changing the engine;
the target's checksummed maps are selected automatically:

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input examples/library.csv \
  --target targets/my-target \
  --engine autodock-gpu \
  --scoring auto \
  --batch-size auto \
  --pose-output none \
  --output-dir outputs/external-ad4
```

## Generate with any iGen3 model, then dock

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --generate-count 100000 \
  --model base-isomeric \
  --generator-batch-size auto \
  --target targets/my-target \
  --engine unidock \
  --output-dir outputs/base-isomeric
```

Substitute any model from the table above. Derivative generation is also
available with `--generation-mode derivative --seed-file seeds.smi`.

iGen3 generation runs as a separate process and exits before docking starts.
This releases PyTorch GPU memory and avoids running two unrelated GPU kernels in
contention on one device. During either engine's docking, CPU preparation of
batch N+1 overlaps GPU docking of batch N.

Automatic preparation uses physical cores in the current affinity plus an
available-RAM guard. Hard 3D construction is bounded to 50 ETKDG attempts and
a 3-second primary timeout; only a quick non-timeout failure receives a smaller
random-coordinate rescue. The exact choice and timeout count are recorded in
the run manifest and can be overridden with `--embed-max-attempts` and
`--embed-timeout`. This deliberately prevents a tiny ultimately failing tail
from holding the GPU idle for tens of seconds per molecule.

Large Uni-Dock runs also start with an automatically sized preparation ramp,
then submit a full steady-state batch before the first GPU invocation. This
hides remaining CPU preparation behind useful GPU work; small inputs keep one
engine invocation when a ramp would not amortize.

For AutoDock-GPU, `auto` can select several same-GPU workers on a capable
device. iGenVS owns a private CUDA MPS daemon for that run, safely falls back to
one worker when automatic MPS is unavailable, and torsion-balances file lists
to reduce worker-tail idle time. A measured batch profile still takes priority.

## Determine the best docking batch on a user's hardware

`--batch-size auto` uses an engine-specific outer-batch heuristic. Uni-Dock then
performs an exact internal memory split; AutoDock-GPU streams its file list. For
a measured optimum, tune once using a representative pilot library, target,
engine, and search mode:

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs tune-docking \
  --input pilot.csv \
  --target targets/my-target \
  --engine unidock \
  --search-mode fast \
  --batch-sizes 8192,16384,32768 \
  --profile gh200-fast.json

apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input library.csv \
  --target targets/my-target \
  --engine unidock \
  --batch-profile gh200-fast.json \
  --output-dir outputs/tuned
```

The tuner measures successful ligands per second, requires at least 99% finite
outputs, and selects the reliable batch with the highest measured successful
throughput. Profiles are engine-specific and record the GPU, target, protocol,
worker count, and all measurements. A pilot must contain at least as many
prepared unique molecules as the largest tested size. AutoDock-GPU concurrency
should be tuned under CUDA MPS; follow the measured workflow in
[Docking engines](docs/DOCKING_ENGINES.md#saturating-large-gpus-with-cuda-mps).
When `--batch-sizes` is omitted, the AutoDock-GPU tuner tests 512, 1,024,
2,048, and 4,096 ligands; provide explicit sizes when a smaller pilot or a
different latency/resilience tradeoff is required.

iGen3 has its own `--generator-batch-size auto` CUDA allocation tuner because
its memory model is independent of either docking engine.

For maximum Uni-Dock refined-score throughput after tuning, omit pose
coordinates and keep explicit-receptor refinement enabled:

```bash
apptainer exec --nv containers/iGenVS.SIF igenvs screen \
  --input library.csv \
  --target targets/my-target \
  --engine unidock \
  --search-mode fast \
  --batch-profile gh200-fast.json \
  --prep-mode standard \
  --pose-output none \
  --output-dir outputs/fast-scores
```

`--pose-output none` uses the patched compact Uni-Dock score table. It preserves
the same search, rescoring, and refinement as pose output while avoiding pose-file
serialization and parsing. `--no-refine` is a separate, lower-fidelity option and
is never enabled automatically.

## Output contract

Each run directory contains:

| Path | Contents |
| --- | --- |
| `manifest.json` | Configuration, hardware, target/receptor hashes, timing, counts, and terminal status. |
| `validation/validated.csv` | Canonical RDKit-valid unique molecules. |
| `validation/rejected.csv` | Typed invalid, fragment, duplicate-ID, and duplicate-SMILES failures. |
| `results.csv` | One terminal preparation/docking status per admitted molecule. |
| `poses.pdbqt` | Optional merged pose stream with `REMARK IGENVS MOLECULE_ID` boundaries. |
| `logs/` | iGen3 and selected-engine stdout/stderr. |

Missing outputs, parse errors, and non-finite scores never become numeric
docking scores. Missing ligands are retried once in split batches to isolate a
poison input. Use `--pose-output none` for ranking-only screens, `--pose-output merged` when
coordinates are needed, and `--pose-output individual` only for small/debug runs.
The ranking-only path avoids millions of persistent small files on a parallel
filesystem.

Every result row names its docking engine and scoring function. Vina/Vinardo
and AD4 values are not numerically interchangeable; keep their rankings and
scientific validation separate.

## Throughput and SLURM

On a matched 20,000-input GH200 fixture using the same prepared receptor,
requested pocket, ligand library, standard preparation, and score-only
`fast` mode, the two backends measured:

| Engine | Successful/input | Docking wall (s) | Successful docking/s | End-to-end successful/s |
| --- | ---: | ---: | ---: | ---: |
| Uni-Dock 1.2.0 / Vina | 19,906 / 20,000 | 198.48 | **100.29** | **86.77** |
| AutoDock-GPU 1.6 / AD4 | 19,983 / 20,000 | 1,659.01 | **12.05** | **11.90** |

On the matched 80,000-input four-GH200 run, conservative complete Slurm-wall
rates were 307.36 successful/s for Uni-Dock and 46.84 successful/s for
AutoDock-GPU. AutoDock-GPU produced a finite result for every prepared ligand
in both runs. These are throughput measurements of different search and
scoring protocols, not an accuracy comparison; Vina and AD4 values must not be
compared numerically. See the
[`AutoDock-GPU 1.6 vs Uni-Dock 1.2.0 GH200 benchmark`](benchmarks/autodock-gpu-v1.6-vs-unidock-v1.2.0-gh200-20260830)
for the full protocol, tuning sweep, yields, and reproducible profile.

Across the 79,606 molecules with finite scores from both engines in the matched
80K run, Pearson score correlation was 0.302, Spearman rank correlation was
0.395, and the top-1% rankings overlapped by 12.92%. This is meaningful but
limited agreement; raw Vina and AD4 scores must not be merged. See the
[`cross-engine correlation report`](benchmarks/correlations) for the
complete method, top-rank overlaps, score distributions, and reproducible
analysis.

The earlier sustained Uni-Dock-only production benchmark used a larger
262,144-input workload and therefore amortized startup more fully.

On the measured Arrhenius GH200, `fast`, `--num-modes 1`, score-only output,
explicit-receptor refinement, and a 32,768 outer batch delivered 85.82
successful ligands/s end-to-end on one GPU. A sustained four-GH200 run delivered
370.42 successful ligands/s by full Slurm wall time. See
[`Uni-Dock 1.2.0 GH200 benchmark`](benchmarks/unidock-v1.2.0-gh200-20260830) for exact manifests,
protocol, and quality tradeoffs. Batch optima remain receptor-, box-, ligand-,
and hardware-specific.

Use node-local scratch for temporary PDBQT files. iGenVS automatically prefers
`$SLURM_TMPDIR`, then `/tmp`; override with `--scratch-dir`. Keep `--num-modes 1`
for bulk ranking. `balance` is the conservative default; promote `fast` only
after target-specific pose/enrichment validation.

For one four-GPU node, provide the prepared target directory to the launcher:

```bash
export IGENVS_PROJECT=/path/to/iGenVS
export IGENVS_INPUT=/path/to/library.csv
export IGENVS_TARGET=/path/to/targets/my-target
export IGENVS_OUTPUT=/path/to/results
sbatch slurm/screen_4gpu.sbatch
```

For the tuned AutoDock-GPU path, additionally set:

```bash
export IGENVS_ENGINE=autodock-gpu
export IGENVS_ADGPU_WORKERS=6
export IGENVS_ADGPU_MPS=1
```

The expert fallback environment variables are documented in
[`slurm/screen_4gpu.sbatch`](slurm/screen_4gpu.sbatch). Four Slurm tasks
deterministically shard an external library by source row, with one GPU and a
separate output directory per task.

## CLI reference

The complete command-line reference, including every flag, accepted value,
default, and description, is maintained in [`docs/CLI.md`](docs/CLI.md).
The installed command also provides version-matched help:

```bash
igenvs --help
igenvs COMMAND --help
```

## Development checks

```bash
python -m pip install -e '.[test]'
pytest
python -m compileall -q src tests
```

Inside the finished image:

```bash
apptainer exec --nv containers/iGenVS.SIF pytest -q /opt/igenvs/tests
```

Docking scores are ranking features under a fixed, validated protocol; they are
not proof of binding affinity, biological activity, selectivity, or safety.

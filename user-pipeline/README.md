# iGenVS-ultra user pipeline

`igenvs-ultra` exposes two explicit, independent user workflows:

1. `dock` is regular iGenVS: generate or ingest molecules, validate, prepare,
   and physically dock them. It returns ordinary docking scores and poses.
2. `run` is iGenVS-ultra: dock the fixed UDRL references, fit a frozen
   target-specific ensemble, optionally perform 1-5 active-learning rounds,
   and stream-score a generated or external molecular library.

The ultra workflow's final large library is **not docked**. It is validated,
exactly deduplicated, encoded with gMolAI, and ranked by the final target-head
ensemble. This is what makes a multi-million-molecule final pass practical.
The two workflows never launch one another implicitly.

## Quick start

From the repository root, build both AMD64 Docker images and run the portable
launcher:

```bash
git lfs pull
./scripts/build-images.sh
./igenvs-ultra doctor
```

Or install the lightweight driver (the scientific dependencies remain in the
released containers):

```bash
python -m pip install -e ./user-pipeline
igenvs-ultra doctor
```

The driver automatically discovers project assets and selects complete local
SIFs first, then the locally built Docker images, then native commands. Docker
image names can be overridden with `--igenvs-docker-image` and
`--gmolai-docker-image`, or `IGENVS_DOCKER_IMAGE` and
`GMOLAI_DOCKER_IMAGE`. SIF paths remain configurable with `--igenvs-image`,
`--gmolai-image`, `IGENVS_IMAGE`, and `GMOLAI_IMAGE`; use `--assets-dir` or
`IGENVS_ULTRA_ASSETS` for a separate release-asset root.

The source release includes the small standardizer and a ready-to-screen 4AG8
example head. The fixed UDRL/AL fitting corpus is an optional multi-gigabyte
asset bundle. `doctor` reports its availability without failing by default;
use `doctor --require-fit-assets` before fitting a new target.

## Workflow 1: regular iGenVS docking

Use `dock` when actual docking scores and poses are wanted for the supplied
library. It is a resumable, hardware-aware wrapper around the released
`igenvs screen` command and retains its scientific defaults and options.

### Dock an external library

```bash
./user-pipeline/igenvs-ultra dock \
  --complex target-complex.pdb \
  --ligand-id A:LIG:501 \
  --input vendor-library.csv \
  --smiles-column SMILES \
  --id-column CatalogID \
  --output-dir runs/my-regular-docking
```

### Generate with iGen3 and dock

```bash
./user-pipeline/igenvs-ultra dock \
  --receptor receptor.pdb \
  --reference-ligand bound-ligand.sdf \
  --generate-count 100000 \
  --model rl-nonisomeric \
  --output-dir runs/my-generated-docking
```

Regular mode follows the original defaults: `--search-mode balance` and
`--pose-output merged`. Use `--pose-output none` for a large ranking-only
docking run, or `--pose-output individual` for separate ligand pose files.
Uni-Dock and AutoDock-GPU, generation, CSV/SMI parsing, validation,
deduplication, preparation, batch profiles, manual shards, scratch, and pose
options are exposed by `igenvs-ultra dock --help`.

`--docking-gpus auto` is the default and uses every GPU visible to the process
(while respecting Slurm's per-task limit). The wrapper generates at most one
shared library, validates and globally deduplicates it once, starts one docking
lane per GPU with disjoint CPU affinity, and restores global source order in
the merged result. Set `--docking-gpus 1` to reserve other visible devices.

Automatic ligand preparation is bounded so a tiny hard-embedding tail cannot
hold every GPU idle: ETKDG gets a deterministic 50-attempt/3-second primary
budget, and only quick failures receive a smaller random-coordinate rescue.
The selected budgets and timeout count are recorded in the manifest; expert
overrides are `--embed-max-attempts` and `--embed-timeout`. Worker count uses
physical cores, inherited affinity, and available RAM. Large Uni-Dock jobs
start with an automatically sized ramp batch so preparation of the full steady
batch overlaps useful GPU work; small jobs avoid the extra invocation.
AutoDock-GPU additionally chooses a capacity-class batch/worker plan, owns a
private CUDA MPS lifecycle when concurrent workers are selected, and balances
flexible ligands across workers. These choices do not alter
`screen`/`screen-fast` generation, encoding, or inference resources.

The intentionally omitted original screen flags are the expert raw-geometry
path (`--center`, `--size`, and `--adgpu-grid`). This wrapper keeps target input
unambiguous: use a bound complex, receptor plus aligned ligand, or a prepared
target bundle. Users who specifically need a hand-authored PDBQT box can still
invoke the underlying `igenvs screen` command directly.

The regular outputs remain native iGenVS outputs:

```text
runs/my-regular-docking/
  target/                    reusable prepared target
  docking/results.csv        terminal result for every admitted molecule
  docking/poses.pdbqt        merged poses (with the default pose policy)
  docking/manifest.json      protocol, counts, timings, hardware, and paths
  regular-summary.json       compact wrapper summary
```

## Workflow 2: iGenVS-ultra

### Complex PDB, three AL rounds, 10M generated molecules

```bash
./user-pipeline/igenvs-ultra run \
  --complex target-complex.pdb \
  --ligand-id A:LIG:501 \
  --al-rounds 3 \
  --generate-count 10000000 \
  --output-dir runs/my-target
```

If the complex contains exactly one plausible non-solvent HETATM residue,
`--ligand-id` can be omitted and is inferred. If the choice is ambiguous, the
CLI lists the candidates and asks for the same `RESNAME` or
`CHAIN:RESNAME:RESSEQ` selector used by iGenVS.

### Separate receptor PDB and bound 3D ligand SDF, no AL

```bash
./user-pipeline/igenvs-ultra run \
  --receptor receptor.pdb \
  --reference-ligand bound-ligand.sdf \
  --al-rounds 0 \
  --input vendor-library.csv \
  --smiles-column SMILES \
  --id-column CatalogID \
  --output-dir runs/my-target
```

The receptor and ligand must already share the same coordinate frame. The SDF
must contain one 3D molecule. Target padding defaults to the released 5 A.

### Save only scores at or above a threshold

```bash
./user-pipeline/igenvs-ultra run \
  --complex target-complex.pdb \
  --ligand-id LIG \
  --al-rounds 1 \
  --input library.smi \
  --score-threshold 0.80 \
  --output-dir runs/my-target
```

`--score-threshold` is shorthand for
`--save-policy threshold --score-threshold VALUE`. “Better” means a larger
ensemble top-1%-hit score; the valid range is 0-1. These are classifier scores,
not predicted docking energies and not calibrated binding probabilities.

Without a threshold, `--save-policy all` is the default and every successfully
encoded molecule is written with all three member scores, the ensemble mean,
and ensemble mutual information.

## Saving embeddings

Embeddings are **not saved by default**. The released float32 encoder output is
passed directly to the target heads in memory, so a temporary compressed `.npz`
is not written, read, and deleted. This avoids substantial I/O for large
screens. To retain the gMolAI embedding bundles:

```bash
./user-pipeline/igenvs-ultra screen \
  --job-dir runs/my-target \
  --input another-library.csv \
  --screen-name another-library \
  --save-embeddings
```

`--keep-embeddings` is accepted as an alias.

The in-memory path changes no chemistry or model operation: it applies the
same gMol acceptance and canonical-SMILES policy, the same promoted encoder
with the same batch boundaries, and the same three target heads. A real
200,000-row check produced a byte-identical result table while reducing full
screen wall time from 167.69 to 37.20 seconds on one GH200. Peak resident memory
rose 1.50x for that batch; the hardware-aware stream-size bound contains this
tradeoff. The retained-NPZ path remains the original
implementation.

## Docking an ultra shortlist

Regular docking is intentionally a separate command. To physically dock rows
retained by an ultra screen, pass its result table to `dock` and reuse the
prepared target:

```bash
./user-pipeline/igenvs-ultra dock \
  --target runs/my-target/target \
  --input runs/my-target/screens/final/results.csv \
  --smiles-column smiles \
  --id-column molecule_id \
  --output-dir runs/my-target-shortlist-docking
```

This explicit handoff makes the computational boundary visible: `run`/`screen`
produce model rankings, and `dock` produces docking scores and poses. Select a
threshold in the ultra screen first if only a shortlist should be docked.

## Fit once, screen many libraries

The end-to-end `run` command is a convenience composition of `fit` and
`screen`. Splitting them avoids redocking when several libraries use one
target:

```bash
./user-pipeline/igenvs-ultra fit \
  --complex target-complex.pdb \
  --ligand-id A:LIG:501 \
  --al-rounds 2 \
  --output-dir runs/my-target

./user-pipeline/igenvs-ultra screen \
  --job-dir runs/my-target \
  --input library-a.csv \
  --screen-name library-a

./user-pipeline/igenvs-ultra screen \
  --job-dir runs/my-target \
  --input library-b.smi \
  --screen-name library-b \
  --score-threshold 0.90
```

External inputs use the original iGenVS `csv`, `tsv`, and `smi` conventions,
including `--input-format`, `--smiles-column`, `--id-column`, `--delimiter`,
and `--fragment-policy`.

## Active learning behavior

`--al-rounds 0` disables AL. Values 1-5 execute the corresponding number of
fixed rounds:

```text
UDRL-train docking -> TH0
AL-set-1 rank/select/dock -> cumulative from-scratch fit -> ALTH1
AL-set-2 rank/select/dock -> cumulative from-scratch fit -> ALTH2
...
AL-set-5 rank/select/dock -> cumulative from-scratch fit -> ALTH5
```

Every head is a three-seed `wide_mlp_rank_aux` ensemble trained for seven
epochs. Each AL round selects 30,000 rows: 15,000 exploitation, 7,500
deep-ensemble mutual-information uncertainty, and 7,500 approximate cosine
MaxMin diversity from a deterministic 200,000-row representative pool.
Docking failures consume budget, are never imputed, and are excluded from
supervised fitting. UDRL-valid is used only for a fixed diagnostic report.

The default docking labels reproduce the released protocol: Uni-Dock/Vina,
`--search-mode fast`, seed 181129, one score-only mode, standard preparation,
and 5 A target padding. The other original iGenVS engine, scoring, preparation,
batching, pose, and AutoDock-GPU flags remain available. A changed label
protocol is recorded and visibly marked as custom/non-release-equivalent.

## Streaming and hardware selection

After fitting a target, the ordinary maximum-speed interface needs only the
number of molecules:

```bash
cd runs/my-target
/path/to/igenvs-ultra screen-fast 10000000
```

The same command can be launched elsewhere with `IGENVS_ULTRA_JOB` set to the
completed target job. It creates the resumable screen `fast-10000000` and means
exactly 10,000,000 successfully encoded and scored molecules, not merely that
many generation attempts.

The automatic planner uses the GPUs visible to the process, including Slurm
limits, and derives each lane from its actual GPU memory and CPU affinity. It
keeps one iGen3 worker and one gMolAI/three-head worker resident per GPU. iGen3
batch size is the minimum of the current memory bound, workload size, safety
ceiling, and CUDA backend launch bound; recoverable OOM/launch-capacity errors
halve and retry. Long-running RDKit generation workers use physical cores in
that lane's affinity, while smoke-size work avoids process startup. The gMolAI
encoder selects the fastest successful released batch among
64/128/192/256/512.

Tiny JSON performance profiles are reused only when GPU, CPU topology,
affinity, model artifacts, scientific settings, and Python/PyTorch/CUDA/RDKit
and backend versions match. Cached generator batches are still checked against
current free memory. On a new computer the safe planner runs without a cache;
only calibrations that can amortize for the requested `N` are performed.

`--stream-batch-size auto` is a separate checkpoint/output resource model. It
scales with selected GPU count, GPU memory, available host RAM, scratch space,
and requested molecule count. Extra batches are generated only when duplicates
or encoder rejections must be replenished. Developer overrides remain on the
full `screen` command, but are not needed by `screen-fast`.

Exact canonical-SMILES deduplication is persistent across all stream batches.
External input is also deduplicated by iGenVS before encoding. Add
`--exclude-reference-libraries` if the screen should additionally exclude
exact identities in the fixed UDRL, five AL sets, and test set; this is optional
because many users intentionally rescore known molecules.

`--screen-gpus auto` (the default) uses every visible GPU for iGen3 generation,
encoding, and target-head scoring. Generated batches are split into
deterministic logical shards and scheduled over the selected GPUs with distinct
seeds and disjoint CPU affinity. Each validated/deduplicated scoring batch is
then split into contiguous shards aligned to the frozen encoder batch size and
merged back in input order. Use `--screen-gpus 1` to reserve the other visible
GPUs. `--generation-logical-shards auto` matches the selected GPU count; fix it
to the same value across controlled 1/2/4-GPU scaling measurements to keep both
generated molecules and process granularity hardware-independent. With
`--save-embeddings`, embeddings are retained as one NPZ bundle per score shard
and listed in the batch manifest; they are still not saved by default.
Inside Slurm, `auto` also respects the task's `SLURM_GPUS_PER_TASK` limit even
if the site leaves additional node devices visible.

For UDRL and AL docking, four deterministic logical shards are fixed
independently of hardware, matching the released four-GPU protocol. One GPU
runs all four serially; 2-4 visible GPUs schedule those unchanged shards over
the available devices. The input is validated and globally deduplicated once,
then each active GPU receives a disjoint inherited CPU partition and consumes
the same trusted validation artifact. This prevents repeated chemistry work,
preparation-worker contention, and the scientific drift that would result from
changing seeded UniDock batch composition with GPU count. Keep
`--docking-logical-shards 4` for release-equivalent work;
changing it is recorded as a custom docking protocol. The CLI does not request
an HPC allocation: run it inside the desired interactive or batch allocation.
`--gpu-ids 0,1,2,3`, `--docking-gpus 4`, and `--screen-gpus 4` provide explicit
hardware control.

## Resuming and outputs

Every costly step writes an atomic manifest. Re-run the identical command to
resume. Incomplete iGenVS target/docking directories are retained with an
`incomplete-<timestamp>` suffix before a clean retry. A changed target,
docking protocol, library, filtering policy, model checkpoint, or stream size
requires a new job directory or `--screen-name`; incompatible state is never
silently mixed.

Useful commands:

```bash
./user-pipeline/igenvs-ultra dock --help
./user-pipeline/igenvs-ultra status --job-dir runs/my-target
./user-pipeline/igenvs-ultra run ... --dry-run
./user-pipeline/igenvs-ultra screen --help
```

Key outputs are:

```text
runs/my-target/
  target/                         prepared iGenVS target bundle
  docking/UDRL-{train,valid}/     sharded runs and merged terminal scores
  models/initial/                 TH0 checkpoints and validation metrics
  al/round-N/                     acquisition and docking records
  models/round-N/                 cumulative ALTHN checkpoints
  models/final.json               final head pointer
  screens/<name>/results.csv      requested final scores
  screens/<name>/rejections.csv   validation/dedup/encoding rejections
  screens/<name>/manifest.json    complete counts, policy, and hashes
```

The design rationale and frozen invariants are summarized in [plan.md](plan.md).
Qualified neural-screen measurements are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md); regular physical-docking
qualification is in
[benchmarks/DOCKING-RESULTS.md](benchmarks/DOCKING-RESULTS.md).

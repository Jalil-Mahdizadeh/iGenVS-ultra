# iGenVS-ultra

iGenVS-ultra provides two GPU workflows behind one hardware-aware launcher:

- physical molecular docking with iGenVS, using Uni-Dock or AutoDock-GPU; and
- target-specific ultra screening with iGen3 generation, gMolAI encoding, and
  a three-member target-head ensemble.

The public launcher detects the GPUs and available CPU/memory resources, then
chooses its worker counts, batches, shards, and overlap policy automatically.
Both application environments are reproducible Linux AMD64 Docker images.

## Requirements

- an x86-64 Linux host;
- Docker Engine with BuildKit;
- an NVIDIA GPU supported by CUDA 12.8/13 and a compatible NVIDIA driver;
- NVIDIA Container Toolkit configured for `docker run --gpus all`; and
- Git LFS (the frozen gMolAI decoder is an LFS object).

The iGenVS image contains native engine code for Ampere (`sm_80`, `sm_86`),
Ada (`sm_89`), Hopper (`sm_90`), and Blackwell (`sm_100`, `sm_120`), including
RTX PRO 2000 Blackwell. Building is intentionally AMD64-only.

## Clone and build

```bash
git clone <repository-url> iGenVS-ultra
cd iGenVS-ultra
git lfs pull
./scripts/build-images.sh
```

This produces:

```text
igenvs-ultra/igenvs:latest
igenvs-ultra/gmolai:latest
```

Override the tags with `IGENVS_DOCKER_IMAGE` and `GMOLAI_DOCKER_IMAGE`. To make
a smaller iGenVS image for one known GPU, also set
`IGENVS_CUDA_ARCHITECTURES` (semicolon separated) and `IGENVS_ADGPU_TARGETS`
(space separated) before building.

Validate the installation and GPU exposure:

```bash
./igenvs-ultra doctor
```

See the [complete CLI reference](docs/CLI.md) for every command, accepted
value, default, and concise description, organized by docking, SMILES
generation, fitting, and screening.

`auto` execution prefers complete local Apptainer images when present, then
the two Docker images above, then a native installation. Pass
`--execution docker` to require Docker explicitly.

## Start docking

The repository includes a three-molecule input and a 4AG8 complex, so this is
an immediate end-to-end smoke run:

```bash
make dock
```

For a real library:

```bash
./igenvs-ultra dock \
  --complex /absolute/path/complex.pdb \
  --ligand-id A:LIG:501 \
  --input /absolute/path/library.csv \
  --smiles-column smiles \
  --id-column molecule_id \
  --output-dir runs/my-docking
```

All visible GPUs are used by default. The default docking engine is Uni-Dock
in `balance` mode; use `--engine autodock-gpu` or `--search-mode fast|detail`
as needed. Outputs and resume state are written below the selected job folder.

## Start ultra screening

A compact released 4AG8 round-5 target head is included so a cloned repository
can screen immediately. `N` is the exact number of finite scores to commit:

```bash
make screen N=10000
```

Equivalently:

```bash
IGENVS_ULTRA_JOB="$PWD/examples/4ag8-screen" \
  ./igenvs-ultra screen-fast 10000
```

The only screening input in this maximum-speed path is the molecule count. The
planner uses every visible GPU for iGen3 generation, gMolAI encoding, and head
inference, while overlapping generation with scoring across stream batches.

To score an external library with the same example target head:

```bash
./igenvs-ultra screen \
  --job-dir examples/4ag8-screen \
  --input /absolute/path/library.csv \
  --smiles-column smiles \
  --id-column molecule_id \
  --screen-name my-library
```

Fitting a new target-specific head uses the separate `fit` or `run` workflow
and the optional UDRL/active-learning release asset bundle. Those multi-GB
research arrays are deliberately not part of the source clone; place the
bundle at the repository root or pass its root with `--assets-dir`. See
[the pipeline guide](user-pipeline/README.md) for target preparation, fitting,
active learning, screening, resume behavior, and all expert controls.

## Direct image use

The images also expose their native CLIs:

```bash
docker run --rm --gpus all igenvs-ultra/igenvs:latest doctor

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/work" \
  igenvs-ultra/gmolai:latest encode \
    --input /work/gMolAI-v2.0/inference/data/example_smiles.csv \
    --output /work/runs/example-embeddings.npz \
    --device cuda
```

## Repository layout

```text
iGenVS/                 docking/generation source and AMD64 Dockerfile
gMolAI-v2.0/            released encoder source/models and AMD64 Dockerfile
user-pipeline/          portable orchestration CLI
docs/                   CLI reference and performance engineering notes
examples/4ag8-screen/   compact ready-to-screen example target head
complexes/              small docking examples
speed-bench/            frozen benchmark protocol, scripts, and report
```

Generated runs, benchmark payloads, caches, SIFs, and internal study
workspaces are excluded from Git. Run `make check` before committing changes.

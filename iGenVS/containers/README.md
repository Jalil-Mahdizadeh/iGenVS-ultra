# Containers

iGenVS has two reproducible, architecture-specific container paths:

| Recipe | CPU architecture | Default GPU target | Output |
| --- | --- | --- | --- |
| `iGenVS.def` | ARM64 | GH200/Hopper, `sm_90` | `containers/iGenVS.SIF` |
| repository-root `Dockerfile` | AMD64 | Ampere through Blackwell, `sm_80`-`sm_120` | `igenvs-ultra/igenvs:latest` |

Both use CUDA 12.8, PyTorch 2.11.0 cu128, all four iGen3 checkpoints,
Uni-Dock 1.2.0, AutoDock-GPU 1.6, and AutoGrid 4.2.9. The build applies the same
two recorded Uni-Dock patches and leaves AutoDock-GPU's search and scoring
source unmodified.

## Source integrity

The repository vendors the required iGen3 files, all four checkpoints, and the
pinned engine sources. `verify-sources.sh` is shared by both build helpers and
fails if copied paths have tracked or untracked modifications. In developer
trees that retain the upstream nested Git metadata, it also verifies the exact
recorded revisions:

```bash
./containers/verify-sources.sh
```

The AutoGrid v4.2.9 source retains an upstream `AutoGrid 4.2.7.x` executable
banner; image labels carry the exact tag commit and runtime manifests carry the
actual banner.

## ARM64 GH200 SIF

`iGenVS.def` compiles both docking engines natively for `sm_90`:

```bash
./containers/build.sh
apptainer exec --nv containers/iGenVS.SIF igenvs doctor
```

The SIF is native to the ARM64 build host and cannot run on x86-64.

## AMD64 NVIDIA Docker image

The RTX PRO 2000 Blackwell has CUDA compute capability 12.0. CUDA 12.8 is the
first toolkit release with the corresponding `sm_120` compiler target, and
PyTorch's cu128 builds support Blackwell. The Dockerfile therefore uses NVIDIA's
CUDA 12.8 AMD64 images and builds both native docking engines for Ampere
(`80`, `86`), Ada (`89`), Hopper (`90`), and Blackwell (`100`, `120`).
See NVIDIA's
[compute-capability table](https://developer.nvidia.com/cuda/gpus),
[CUDA 12.8 feature list](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html),
and [Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html).

Build on an x86-64 Linux host with Docker BuildKit:

```bash
./containers/build-docker.sh igenvs-ultra/igenvs:latest
```

The script always requests `--platform linux/amd64`. The Dockerfile also checks
`TARGETARCH` and fails instead of silently producing an ARM64 image. A narrow
`.dockerignore` admits only the application, tests, model bundle, engine
sources, and patches; the SIF, generated benchmarks, Git metadata, and runtime
outputs do not enter the build context.

At runtime, install a Blackwell-capable NVIDIA driver and configure the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Then expose the GPU and bind input/output data:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/work" \
  igenvs-ultra/igenvs:latest doctor
```

The image entrypoint is `igenvs`, so append any command and flags directly.
Target preparation itself is CPU-only; `screen`, `tune-docking`, and iGen3
generation require GPU exposure.

### Smaller architecture-specific image

The default is the portable modern-GPU image. To reduce build time and binary
size for one known Blackwell workstation, override both target lists:

```bash
IGENVS_CUDA_ARCHITECTURES='120' \
IGENVS_ADGPU_TARGETS='120' \
./containers/build-docker.sh igenvs:amd64-sm120
```

The CMake list is semicolon-separated; the AutoDock-GPU list is
whitespace-separated. CUDA 12.8 must support every requested target. Native
`sm_120` compilation is preferable to relying on older PTX JIT for Blackwell.

## Runtime contents and tuning

Each image contains:

- all four iGen3 model checkpoints and vocabularies;
- RDKit, Meeko, Gemmi, Open Babel, SciPy, NumPy, and pandas;
- the patched Uni-Dock score-only and explicit-memory-cap paths;
- AutoDock-GPU CUDA binaries with 64, 128, and 256 work items;
- AutoGrid and the reusable dual-engine target workflow; and
- the iGenVS CLI and tests.

AutoDock-GPU's default symlink remains `autodock_gpu_64wi`, the GH200 benchmark
winner. Work-item width, same-GPU worker count, generator batch, and docking
batch can differ substantially on RTX PRO 2000 Blackwell. Run the packaged
tests, `igenvs doctor`, and target-specific `tune-docking` before production;
do not reuse the GH200 batch profile.

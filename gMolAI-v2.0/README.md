# gMolAI v2.0 release runtime

This directory contains the frozen gMolAI molecular representation runtime
used by iGenVS-ultra. Its public inference CLI can:

- encode a CSV of SMILES into calibrated 384-dimensional
  `released_hybrid_w3` embeddings;
- decode those embeddings into seed-conditioned analogue candidates; and
- validate the frozen model bundle and runtime.

The Linux AMD64 Docker image is the supported portable environment. It is
based on NVIDIA PyG 25.09 and contains CUDA, PyTorch, PyTorch Geometric, RDKit,
the exact encoder/calibrator, and the frozen decoder.

## Build

From this directory:

```bash
git lfs pull
docker build --platform linux/amd64 -t igenvs-ultra/gmolai:latest .
```

From the parent iGenVS-ultra repository, `./scripts/build-images.sh` builds
this image and the matching iGenVS image together.

The Dockerfile is a native Docker/OCI recipe; it does not invoke Apptainer.
Its AMD64 base manifest is pinned by digest, and the build runs a CPU artifact
validation before the image is emitted.

## Validate GPU access

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all \
  igenvs-ultra/gmolai:latest validate --device cuda
```

The image entrypoint is `gmolai`, so commands and flags follow the image name.

## Encode SMILES

Input is CSV with a SMILES column and, optionally, an identifier column:

```csv
molecule_id,smiles
ethanol,CCO
aspirin,CC(=O)Oc1ccccc1C(=O)O
```

Run the optimized CUDA encoder while retaining host ownership of outputs:

```bash
mkdir -p output
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/work" \
  igenvs-ultra/gmolai:latest encode \
    --input /work/inference/data/example_smiles.csv \
    --output /work/output/embeddings.npz \
    --smiles-column smiles \
    --id-column molecule_id \
    --backend optimized \
    --device cuda
```

The `.npz` output retains embeddings, canonical SMILES, input identifiers,
source rows, hashes, and release provenance. Rejected inputs and execution
metadata are written as adjacent sidecars. Existing outputs are never replaced
unless `--overwrite` is supplied.

The release policy accepts single-fragment molecules with 2-256 atoms from C,
N, O, F, P, S, Cl, Br, I, H, B, and Si. Canonical isomeric SMILES preserve
stereochemistry.

## Decode embeddings

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD:/work" \
  igenvs-ultra/gmolai:latest decode \
    --embeddings /work/output/embeddings.npz \
    --output-dir /work/output/candidates \
    --proposal-budget 1000 \
    --device cuda
```

The proposal budget is a frozen raw-decoder prefix, not a promised output row
count. Invalid strings, policy rejections, duplicate molecular identities, and
the reconstructed seed are removed according to the release contract.

See [`inference/README.md`](inference/README.md) for the complete encode/decode
schema, all supported flags, and the frozen generation policy.

## iGenVS-ultra integration

The parent launcher keeps one gMolAI encoder plus target-head worker resident
on every selected GPU. It streams iGen3 output directly into encoding and
inference without writing intermediate embedding archives by default:

```bash
cd ..
make screen N=10000
```

Use the standalone `gmolai` CLI when embeddings or analogue candidates are the
desired product. Use the parent `igenvs-ultra screen`/`screen-fast` commands for
target-specific virtual screening.

## Model integrity

`inference/models/SHA256SUMS` records every frozen artifact. The decoder is a
Git LFS object because it exceeds GitHub's normal per-file limit. If validation
reports a tiny text pointer instead of a model, run `git lfs pull` in the clone
and rebuild.

For source development, `pyproject.toml` exposes the `gmolai-retrain` package
and the tests remain under `tests/`; production screening should use the pinned
Docker image.

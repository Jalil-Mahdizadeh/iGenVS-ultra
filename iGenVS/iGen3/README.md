# iGen3

iGen3 is a transformer-based de novo SMILES generator with base and reinforcement-learning tuned checkpoints. The package supports both de novo generation and derivative generation, where randomized seed SMILES are used as conditioning prefixes and the model samples derivative molecules from those prefixes.

Generation commands write canonical SMILES that are both RDKit-valid and unique. In derivative mode, molecules identical to the input seed structures are excluded by default.

## Models

| CLI model id | Family | SMILES type | Sequence length | Default de novo temperature | Default derivative temperature |
| --- | --- | --- | ---: | ---: | ---: |
| `base-isomeric` | Base transformer | Isomeric | 122 | 1.0 | 1.5 |
| `base-nonisomeric` | Base transformer | Non-isomeric | 112 | 1.0 | 1.5 |
| `rl-isomeric` | RL tuned | Isomeric | 122 | 1.2 | 2.0 |
| `rl-nonisomeric` | RL tuned | Non-isomeric | 112 | 1.2 | 2.0 |

## Quick Start With Docker

Docker images contain the code and dependencies only. Mount this repository's
`models/` directory at `/models` whenever a command needs checkpoints.

Blackwell GPU image:

```bash
docker build -f Dockerfile.blackwell -t igen3:blackwell .
docker run --rm --gpus all -v "${PWD}/models:/models:ro" igen3:blackwell list-models
```

Older GPU image:

```bash
docker build -f Dockerfile.legacy-gpu -t igen3:legacy-gpu .
docker run --rm --gpus all -v "${PWD}/models:/models:ro" igen3:legacy-gpu list-models
```

Generate de novo SMILES:

```bash
docker run --rm --gpus all \
  -v "${PWD}/models:/models:ro" \
  -v "${PWD}/outputs:/outputs" \
  igen3:blackwell generate \
  --model base-isomeric \
  --mode de-novo \
  --count 100000 \
  --output /outputs/base_isomeric_100k.smi
```

Prepare randomized seed SMILES for derivative generation:

```bash
docker run --rm \
  -v "${PWD}/outputs:/outputs" \
  igen3:blackwell randomize-smiles \
  --smiles "N1(C(COC)=O)C(C)CN(CC1)C(c1cc(Cl)ccc1)" \
  --max-variants 10000 \
  --stagnation-limit 5000 \
  --output /outputs/reference_randomized.smi \
  --metadata-output /outputs/reference_randomized.json
```

Generate derivatives from that randomized seed file:

```bash
docker run --rm --gpus all \
  -v "${PWD}/models:/models:ro" \
  -v "${PWD}/outputs:/outputs" \
  igen3:blackwell generate \
  --model rl-isomeric \
  --mode derivative \
  --seed-file /outputs/reference_randomized.smi \
  --samples-per-seed 10 \
  --count 10000 \
  --output /outputs/rl_isomeric_derivatives.smi
```

Run benchmarks and create GitHub-ready figures:

```bash
docker run --rm --gpus all \
  -v "${PWD}/models:/models:ro" \
  -v "${PWD}/benchmarks:/benchmarks" \
  igen3:blackwell benchmark \
  --count 500000 \
  --output-dir /benchmarks/latest
```

Run the derivative-generation benchmark from a randomized seed file:

```bash
docker run --rm --gpus all \
  -v "${PWD}/models:/models:ro" \
  -v "${PWD}/benchmarks:/benchmarks" \
  igen3:blackwell benchmark-derivative \
  --seed-file /benchmarks/derivative_latest/seeds/reference_randomized.smi \
  --count 10000 \
  --output-dir /benchmarks/derivative_latest
```

## CLI

```bash
igen3 list-models
igen3 generate --model rl-nonisomeric --mode de-novo --count 1000 --output outputs/rl_noiso.smi
igen3 randomize-smiles --smiles "N1(C(COC)=O)C(C)CN(CC1)C(c1cc(Cl)ccc1)" --output outputs/reference_randomized.smi
igen3 benchmark --count 500000 --output-dir benchmarks/latest
igen3 benchmark-derivative --seed-file outputs/reference_randomized.smi --count 10000 --output-dir benchmarks/derivative_latest
```

For non-Docker installs, install the PyTorch CUDA wheel that matches your GPU first, then install this package:

```bash
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
python -m pip install .
```

If you run the CLI outside the repository root, set `IGEN3_MODEL_DIR` to the absolute `models` directory.

Important options:

| Option | Description |
| --- | --- |
| `--batch-size auto` | Autotunes the largest batch size that fits GPU memory. |
| `--temperature` | Overrides the model/mode default sampling temperature. |
| `--top-k` | Overrides the default top-k cutoff of 64. Use `0` to disable. |
| `--greedy` | Uses argmax decoding instead of sampling. |
| `--compile on` | Enables `torch.compile`; useful for very large runs, but startup can be slower. |
| `--metrics` | Computes RDKit metrics after a single generation run. |
| `--max-candidates` | Caps sampled candidates used to fill the valid unique output file. |
| `--stagnation-limit` | Stops when sampling stops finding new valid unique molecules. Use `0` to disable. |
| `--include-seed-molecules` | Allows derivative output to include molecules identical to the seed structures. |

The `randomize-smiles` command writes canonical and non-canonical RDKit SMILES
variants for one valid reference molecule. It stops when `--max-variants`,
`--max-attempts`, or `--stagnation-limit` is reached. Use this output as the
`--seed-file` for derivative generation.

## Project Layout

```text
src/igen3/              Maintained package and master CLI
models/                 Runtime-mounted model weights and vocabularies
examples/               Example derivative seed SMILES
benchmarks/             Benchmark outputs and figures
docs/                   Docker and benchmark documentation
legacy/original_scripts Original scripts kept for reference
```

## Notes

The tokenizer maps `Cl` and `Br` to the original training tokens `X` and `Y` before generation. This fixes derivative-seed handling for halogen-containing SMILES.

`requirements.txt` intentionally excludes PyTorch because CUDA wheel selection is hardware-specific. The Dockerfiles install explicit PyTorch CUDA wheels before installing the common Python dependencies.

Docker image choices follow current official PyTorch CUDA binary guidance and NVIDIA CUDA container guidance:

- [PyTorch local install matrix](https://pytorch.org/get-started/locally/)
- [PyTorch release notes for CUDA 12.8/12.6 support](https://github.com/pytorch/pytorch/releases)
- [NVIDIA CUDA container image documentation](https://nvidia.github.io/container-wiki/toolkit/container-images.html)

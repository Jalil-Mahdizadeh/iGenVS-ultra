# Docker Usage

Two Dockerfiles are provided because the best CUDA/PyTorch wheel differs by GPU generation.
The images do not include checkpoint weights or vocabularies. Mount the local
`models/` directory at `/models` for generation and benchmark commands.

## Blackwell GPUs

Use this image for Blackwell GPUs such as RTX 50-series and RTX PRO Blackwell cards.

```bash
docker build -f Dockerfile.blackwell -t igen3:blackwell .
docker run --rm --gpus all -v "${PWD}/models:/models:ro" igen3:blackwell list-models
```

## Older GPUs

Use this image for less-modern GPUs that need CUDA 12.6 PyTorch binaries.

```bash
docker build -f Dockerfile.legacy-gpu -t igen3:legacy-gpu .
docker run --rm --gpus all -v "${PWD}/models:/models:ro" igen3:legacy-gpu list-models
```

## Examples

Generate 100,000 de novo SMILES:

```bash
docker run --rm --gpus all \
  -v "${PWD}/models:/models:ro" \
  -v "${PWD}/outputs:/outputs" \
  igen3:blackwell generate \
  --model rl-isomeric \
  --mode de-novo \
  --count 100000 \
  --output /outputs/rl_isomeric_100k.smi
```

Generate randomized derivative seed SMILES:

```bash
docker run --rm \
  -v "${PWD}/outputs:/outputs" \
  igen3:blackwell randomize-smiles \
  --smiles "N1(C(COC)=O)C(C)CN(CC1)C(c1cc(Cl)ccc1)" \
  --max-variants 10000 \
  --stagnation-limit 5000 \
  --output /outputs/reference_randomized.smi
```

Generate derivatives from that randomized seed file:

```bash
docker run --rm --gpus all \
  -v "${PWD}/models:/models:ro" \
  -v "${PWD}/outputs:/outputs" \
  igen3:blackwell generate \
  --model base-nonisomeric \
  --mode derivative \
  --seed-file /outputs/reference_randomized.smi \
  --samples-per-seed 10 \
  --count 10000 \
  --output /outputs/base_nonisomeric_derivatives.smi
```

Run the full benchmark:

```bash
docker run --rm --gpus all \
  -v "${PWD}/models:/models:ro" \
  -v "${PWD}/benchmarks:/benchmarks" \
  igen3:blackwell benchmark \
  --count 500000 \
  --output-dir /benchmarks/latest
```

Run the derivative benchmark:

```bash
docker run --rm --gpus all \
  -v "${PWD}/models:/models:ro" \
  -v "${PWD}/benchmarks:/benchmarks" \
  igen3:blackwell benchmark-derivative \
  --seed-file /benchmarks/derivative_latest/seeds/reference_randomized.smi \
  --count 10000 \
  --output-dir /benchmarks/derivative_latest
```

Generation outputs are canonical, RDKit-valid, and unique. Derivative generation excludes molecules identical to the seed structures unless `--include-seed-molecules` is passed.

The images use `python:3.12-slim` plus explicit PyTorch CUDA wheels. CUDA-enabled PyTorch still dominates image size, but model weights stay outside the image and are mounted read-only at runtime.

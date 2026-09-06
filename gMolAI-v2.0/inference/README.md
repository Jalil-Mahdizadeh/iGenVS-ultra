# gMolAI release inference

This directory is the user-facing workflow for the frozen gMolAI release. It
provides two operations:

1. encode a CSV list of SMILES as calibrated 384-dimensional
   `released_hybrid_w3` embeddings; and
2. decode each unperturbed embedding into a seed-specific analogue library
   using the frozen Step-2 decoder and Step-2d
   `hybrid_b500_s500_t120` strategy.

No training, latent perturbation, property optimization, or chemistry-aware
candidate ordering occurs here.

## Quick start

From the repository root, in the pinned gMolAI environment:

```bash
python inference/gmolai.py validate

python inference/gmolai.py encode \
  --input inference/data/example_smiles.csv \
  --output inference/output/embeddings.npz

python inference/gmolai.py decode \
  --embeddings inference/output/embeddings.npz \
  --output-dir inference/output/candidates \
  --proposal-budget 1000
```

The final command creates one CSV per accepted seed molecule. `generate` is an
alias for `decode`.

## Directory layout

```text
inference/
├── gmolai.py                  # public encode/decode/validate CLI
├── _decoder.py                # frozen inference-only decoder primitives
├── generate_embeddings.py     # preserved legacy CSV encoder
├── models/
│   ├── SHA256SUMS
│   ├── representation-best.pt
│   ├── representation-calibrator.pt
│   ├── representation_selection.json
│   ├── resolved_config.json
│   └── decoder_inference.pt
├── model -> models            # compatibility for historical frozen manifests
├── data/
│   └── example_smiles.csv
└── output/
```

The compact decoder is stored with Git LFS because it exceeds GitHub's normal
100 MiB object limit. After a fresh clone, run `git lfs pull` before inference.

## Runtime

From a repository clone with Python 3.12 or newer, the standard CPU-capable
installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python inference/gmolai.py validate --device cpu
```

PyTorch and PyTorch Geometric are core package dependencies. For CUDA or another
accelerator, install the matching platform-specific PyTorch wheel before
`pip install .`. The pinned AMD64 Docker image is the authoritative portable
CUDA runtime:

```bash
docker build --platform linux/amd64 -t igenvs-ultra/gmolai:latest .
docker run --rm --gpus all igenvs-ultra/gmolai:latest validate --device cuda
```

Encoding can fall back to CPU, but the frozen 1,000-proposal decoder workflow
is intended for a GPU.

## Encode SMILES to `.npz`

The input must be a CSV with a SMILES column. An identifier column is optional.
For example:

```csv
molecule_id,smiles
ethanol,CCO
aspirin,CC(=O)Oc1ccccc1C(=O)O
```

Run:

```bash
python inference/gmolai.py encode \
  --input molecules.csv \
  --smiles-column smiles \
  --id-column molecule_id \
  --output results/embeddings.npz \
  --backend optimized \
  --device cuda
```

`.npz` is used instead of bare `.npy` so the embedding matrix remains aligned
with its row identifiers and molecular identities. The archive can be opened
without pickle:

```python
import numpy as np

with np.load("results/embeddings.npz", allow_pickle=False) as bundle:
    vectors = bundle["embeddings"]          # float32, shape (N, 384)
    smiles = bundle["canonical_smiles"]     # shape (N,)
    molecule_ids = bundle["input_id"]       # shape (N,)
```

The archive also contains `input_row`, `input_smiles`, `molecule_hash`,
`atom_count`, the representation definition, and all encoder/calibrator
identities. Two sidecars are written beside it:

- `<name>.rejections.csv` records rejected input rows and reasons;
- `<name>.metadata.json` records provenance, artifact/output hashes, runtime
  versions, execution settings, and row counts.

Encoding is collection-size memory bounded. Each completed model batch is
written to a temporary float32 file, accepted row metadata and exact uniqueness
counts are staged in SQLite, and the final compressed `.npz` is assembled from
memory-mapped arrays. RAM therefore scales with active preprocessing/model
batches and bounded compression buffers rather than with the total input row
count. Temporary staging is created beside `--output` and removed on success
or handled failure. For large collections, reserve output-filesystem scratch
space for at least 1,536 uncompressed embedding bytes per accepted molecule,
plus row metadata, the SQLite index, and the final compressed archive.

Existing outputs are never replaced unless `--overwrite` is supplied.

### Encode flags

| Flag | Meaning | Default |
|---|---|---|
| `--input PATH` | Input SMILES CSV | `inference/data/example_smiles.csv` |
| `--output PATH` | Self-describing embedding archive; must end in `.npz` | `inference/output/embeddings.npz` |
| `--smiles-column NAME` | Input SMILES column | `smiles` |
| `--id-column NAME` | ID column, `auto`, or `none` | `auto` |
| `--backend MODE` | `optimized`, `reference`, or cross-checking `verify` | `optimized` |
| `--device DEVICE` | `auto`, `cpu`, `cuda`, or `cuda:<index>` | `auto` |
| `--batch-size N` | Encoder graph batch ceiling | `192` |
| `--node-budget N` | Encoder node ceiling per batch | `16384` |
| `--workers N\|auto` | RDKit preprocessing workers; `auto` respects Slurm/affinity | `auto` |
| `--verify-rows N` | Rows compared with the reference backend in `verify` mode | `1024` |
| `--invalid-policy report\|error` | Record rejected rows or fail atomically | `report` |
| `--limit N` | Encode only the first `N` input rows | all rows |
| `--threads N` | PyTorch CPU threads | up to 8 |
| `--models-dir PATH` | Release artifact directory | `inference/models` |
| `--overwrite` | Replace the output archive and sidecars | disabled |

The release policy accepts single-fragment, 2–256 atom molecules composed of
C, N, O, F, P, S, Cl, Br, I, H, B, or Si. Canonical isomeric SMILES preserve
stereochemistry. The public vector is standardized graph-256 concatenated with
standardized mean-node-128, with the mean-node block weighted by exactly 3.

## Decode embeddings to candidate CSVs

Run the decoder only on an `.npz` bundle created by `encode`:

```bash
python inference/gmolai.py decode \
  --embeddings results/embeddings.npz \
  --output-dir results/candidates \
  --proposal-budget 1000 \
  --device cuda
```

Each seed receives a collision-resistant filename such as
`seed-000001-ethanol-ab1de819.csv`. Every retained row is:

- a valid, policy-accepted molecule;
- represented by canonical isomeric SMILES;
- unique within that seed's CSV, with the first occurrence in the frozen
  proposal order retained; and
- annotated with seed-to-candidate Morgan/Tanimoto similarity (radius 2,
  2,048 bits, chirality disabled), proposal/source ranks, decoder score, and
  generated length.

The reconstructed seed identity is excluded by default because it is not a
derivative. Pass `--include-seed` to retain it. Candidate rows are not sorted by
similarity or any property; their order is the property-free frozen decoder
order.

`generation.metadata.json` records per-seed raw, token-decodable,
policy-accepted, duplicate, excluded-seed, and retained-unique counts, plus all
artifact and strategy identities.

### Decode flags

| Flag | Meaning | Default |
|---|---|---|
| `--embeddings PATH` | `.npz` output from `encode` | `inference/output/embeddings.npz` |
| `--output-dir PATH` | Directory for seed CSVs and metadata | `inference/output/candidates` |
| `--proposal-budget N` | Frozen nested raw-proposal prefix: `50`, `100`, `250`, `500`, or `1000` | `1000` |
| `--seed-limit N` | Decode only the first `N` embedded seeds | all seeds |
| `--include-seed` | Retain a reconstructed seed identity | disabled |
| `--device DEVICE` | `auto`, `cpu`, `cuda`, or `cuda:<index>` | `auto` |
| `--threads N` | PyTorch CPU threads | up to 8 |
| `--models-dir PATH` | Release artifact directory | `inference/models` |
| `--overwrite` | Replace colliding candidate CSVs and metadata | disabled |

### Proposal budget is not a row target

`--proposal-budget 1000` means 1,000 raw decoder slots per seed, not 1,000
guaranteed CSV rows. Invalid strings, policy rejections, duplicate molecular
identities, and (by default) the seed identity are removed. The completed
Step-2d study observed a median of 182 unique policy-accepted identities at the
1,000-slot prefix. Over-generating until 1,000 unique rows would change the
frozen strategy and is therefore deliberately unsupported.

## Frozen generation contract

The decoder consumes each release embedding without perturbation. The complete
1,000-slot `hybrid_b500_s500_t120` stream is fixed as follows:

- 500 beam hypotheses from width 1,000, ordered by cumulative decoder log
  probability with length penalty 0;
- 500 sample-stream hypotheses: one greedy result followed by the first 499
  fixed-order stochastic draws from the registered 999-draw pool;
- stochastic temperature 1.2 and top-p 0.995;
- maximum 128 SMILES bytes;
- global seed 20,260,817, with the Step-2d `final` phase and molecular SHA-256
  used to derive an independent seed stream; and
- deterministic balanced beam/sample interleaving.

Temperature, top-p, beam allocation, length penalty, output ceiling, seed rule,
and merge order are intentionally not CLI flags. Changing any of them would be
a different generation baseline.

## Validate release artifacts

Run before first use or after copying the directory:

```bash
python inference/gmolai.py validate --device cpu
```

Validation hashes all five artifacts before deserialization, then checks the
checkpoint, feature schema, configuration, selection record, calibrator,
embedding definition, decoder architecture/state, conditioning space, and
frozen Step-2 source identity.

| Artifact | SHA-256 |
|---|---|
| `representation-best.pt` | `02f49a2a94ddfc9dc780cc3d5f1a3df54306ae0fdc5d4b3767e3fd2e7f27b05e` |
| `representation-calibrator.pt` | `5cbe3210b2fa6742b165c61e3562118553f567df13181d863776c9ca5527365b` |
| `representation_selection.json` | `43f1f857576f10fd8aa7ed9276f9ce899ca90d011172225d04e8cff77a9333a1` |
| `resolved_config.json` | `9ad8e4000b3dc0b7a2c3ef8631200fbfa301ef377fd7518293b8636964844628` |
| `decoder_inference.pt` | `8b4f8db04499083ea2e9d028eaaae18d629b34ce773608d8e2c80863e9121d47` |

The 113 MB decoder export contains no optimizer state and no gMolAI encoder
parameters. It is derived from the frozen Step-2 training checkpoint
`bb9623080ddaed070278c8abca39252e070c110a6611b3bd7a75caf6c37a41f6`.

## Legacy encoder

`generate_embeddings.py` is retained byte-for-byte because completed study
manifests bind its SHA-256. It continues to produce the older wide CSV format.
New applications should use `gmolai.py encode`; the `model -> models` symlink
keeps that historical entry point functional without restoring the old folder.

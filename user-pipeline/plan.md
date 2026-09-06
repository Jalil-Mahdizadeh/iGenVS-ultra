# User-facing pipeline plan

Status: implemented and verified

## Scope

Build one small CLI, `igenvs-ultra`, with two explicitly separate workflows.
The regular `dock` workflow will call original iGenVS generation/ingress,
validation, preparation, and physical docking. The ultra workflow will:

1. prepare a target from either a complex PDB or a receptor PDB plus aligned
   3D ligand SDF;
2. dock the fixed UDRL-train and UDRL-valid libraries with iGenVS;
3. train the frozen three-member `wide_mlp_rank_aux` target-head ensemble;
4. optionally execute zero to five released 50/25/25 AL rounds; and
5. stream either iGen3 output or an external CSV/SMI library through
   validation, global exact deduplication, gMolAI encoding, and final-ensemble
   scoring.

## Interface decisions

- Commands: `doctor`, `dock`, `fit`, `screen`, `screen-fast`, `run`, and
  `status`.
- `dock` is the independent regular iGenVS workflow. It does not fit or invoke
  a target head. Conversely, ultra `run` and `screen` do not physically dock
  their final scored library.
- `run` is the one-command path; `fit` plus `screen` permits repeated library
  screens without redocking the reference libraries.
- `screen-fast N`, run inside a completed target job, is the normal optimized
  final-screen interface. Molecule count is its only user input; hardware,
  workers, batches, persistent services, stream size, and profile reuse are
  planned automatically.
- `--al-rounds 0` disables AL; values 1-5 run that many fixed rounds.
- iGen3's internal GPU batch remains `--generator-batch-size auto`. A separate
  `--stream-batch-size auto` bounds validation/encoding/scoring and is selected
  from visible GPU memory, host memory, and free scratch space.
- Generated streams use distinct deterministic seeds and are replenished until
  the requested number of globally unique, prepared molecules is admitted.
- `--screen-gpus` controls iGen3 generation as well as gMolAI encoding and
  target-head inference. Generated batches use deterministic logical shards;
  `--generation-logical-shards` may be fixed across hardware-count runs so the
  generated workload does not change with GPU count.
- Final score means the arithmetic mean of the three target-head sigmoid
  scores. `--save-policy all` writes every encoded molecule;
  `--save-policy threshold --score-threshold P` writes scores >= P.
- Every expensive stage has an atomic completion manifest. Re-running the same
  command resumes; incompatible settings require a new job/screen name.
- Automatic choices are portable rather than GH200 constants: GPU batch bounds
  use live free memory and backend capacity, CPU chemistry workers use physical
  cores inside per-GPU affinity, encoder candidates are measured, and cached
  profiles include hardware/software/model fingerprints. Expensive tuning is
  skipped unless it can amortize for the requested per-GPU molecule count.

## Scientific invariants

- Frozen 384D `released_hybrid_w3` embeddings and Phase-5 standardizer.
- Frozen 384-512-128 head, three seeds 260904-260906, seven epochs.
- Original finite UDRL-train Q1% defines positives and never moves during AL.
- UDRL-valid is evaluation-only.
- AL selects exactly 15k exploitation, 7.5k ensemble-MI uncertainty, and 7.5k
  approximate cosine MaxMin diversity rows from each fixed one-million set.
- Docking failures consume acquisition budget, receive no imputed label, and
  are excluded from fitting.
- Default label protocol is iGenVS Uni-Dock/Vina `--search-mode fast`, 5 A
  target padding, and score-only output. User-selected iGenVS alternatives are
  allowed but explicitly marked as custom rather than release-equivalent.

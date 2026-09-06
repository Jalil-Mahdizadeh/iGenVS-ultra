# iGenVS command-line reference

This reference is derived from [`src/igenvs/cli.py`](../src/igenvs/cli.py). Run
`igenvs COMMAND --help` for the installed image's parser output.
`required` means there is no default; `unset` means the
option is omitted unless supplied. Boolean switches are off by default.
`auto` invokes profile- or hardware-based selection.

## Commands

| Command | Purpose |
| --- | --- |
| `doctor` | Check packaged tools, versions, target capabilities, and optionally the GPU. |
| `validate` | Parse, RDKit-validate, canonicalize, shard, and optionally deduplicate a SMILES library. |
| `prepare-receptor` | Convert a curated/protonated receptor PDB to rigid PDBQT. |
| `prepare-target` | Build a reusable dual-engine receptor, ligand-defined pocket, AutoGrid maps, and checksummed manifest. |
| `screen` | Ingest or generate molecules, validate, prepare, and dock them with either engine. |
| `tune-docking` | Benchmark candidate outer batches and write a hardware/protocol profile. |

## Global flags

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `-h`, `--help` | switch | off | Show help for the root command or selected subcommand and exit. |
| `--version` | switch | off | Print the iGenVS version and exit. |

## Shared library-input flags

These flags apply to `validate`, `screen`, and
`tune-docking`. For `validate` and
`tune-docking`, `--input` is required. For
`screen`, exactly one of `--input` and
`--generate-count` is required.

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--input` | `PATH` | required/one-of | CSV, TSV, or whitespace-delimited SMILES library. |
| `--input-format` | `auto`, `csv`, `smi` | `auto` | Select parsing explicitly or infer it from the file. |
| `--smiles-column` | column name | `smiles` | Case-insensitive CSV/TSV column containing SMILES. |
| `--id-column` | column name | unset | Optional source molecule-ID column; generated row IDs are used otherwise. |
| `--delimiter` | `auto`, one character, or literal `\t` | `auto` | Delimiter for tabular input. |
| `--fragment-policy` | `reject`, `largest` | `reject` | Reject multi-fragment SMILES or retain their largest fragment. |

## `doctor` flags

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--no-gpu` | switch | off | Run checks without requiring a visible CUDA GPU. |
| `--json` | switch | off | Emit the diagnostic report as JSON. |

## `validate`-only flags

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--output-dir` | `PATH` | required | Write validated/rejected tables and the deduplication database here. |
| `--workers` | `auto` or integer > 0 | `auto` | RDKit validation processes; auto uses available CPUs capped at 32. |
| `--no-deduplicate` | switch | off | Preserve duplicate canonical SMILES instead of rejecting repeats. |
| `--num-shards` | integer > 0 | `1` | Deterministically divide input rows into this many shards. |
| `--shard-index` | integer from 0 to `num-shards - 1` | `0` | Process only this zero-based shard. |

## `prepare-receptor` flags

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--input` | receptor `.pdb` | required | Curated, protonated receptor PDB. |
| `--output` | receptor `.pdbqt` | required | Destination rigid-receptor PDBQT. |

## `prepare-target` flags

Exactly one source mode is required: `--complex` with
`--ligand-id`, or `--receptor` with
`--reference-ligand`.

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--complex` | complex `.pdb` | one source required | Protein and bound ligand in one PDB. Mutually exclusive with `--receptor`. |
| `--receptor` | receptor `.pdb` | one source required | Curated/protonated receptor PDB. Mutually exclusive with `--complex`. |
| `--ligand-id` | `RESNAME` or `CHAIN:RESNAME:RESSEQ` | required with `--complex` | Select the bound HETATM residue used to define the pocket. |
| `--reference-ligand` | single-molecule 3D `.sdf` | required with `--receptor` | Aligned ligand coordinates used to define the pocket. |
| `--padding` | finite float >= 0, angstrom | `5.0` | Add this distance to every side of the ligand bounding box. |
| `--output-dir` | `PATH` | required | Destination reusable target directory. |

## `screen` source and iGen3-generation flags

Generation options take effect when `--generate-count` is the selected
source.

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--generate-count` | integer > 0 | one source required | Generate this many valid unique SMILES instead of reading `--input`. |
| `--model` | `base-isomeric`, `base-nonisomeric`, `rl-isomeric`, `rl-nonisomeric` | `rl-nonisomeric` | Select one of the four bundled iGen3 checkpoints. |
| `--generation-mode` | `de-novo`, `derivative` | `de-novo` | Generate from scratch or condition on seed SMILES. |
| `--seed-file` | SMILES `PATH` | required for `derivative` | Seed molecules for derivative generation. |
| `--samples-per-seed` | integer > 0 | `1` | Requested derivatives per seed molecule. |
| `--generator-batch-size` | `auto` or integer > 0 | `auto` | iGen3 sampling batch; auto probes the visible GPU. |
| `--generator-max-batch-size` | integer > 0 | `32768` | Upper bound for generator batch autotuning. |
| `--model-dir` | `PATH` | bundled models | Override the iGen3 model directory. |
| `--temperature` | float > 0 | model/mode setting | Sampling temperature: base defaults are 1.0 de novo and 1.5 derivative; RL defaults are 1.2 and 2.0. |
| `--top-k` | integer >= 0 | `64` | Sampling cutoff; `0` disables top-k filtering. |
| `--compile-generator` | switch | off | Enable the iGen3/PyTorch compiled generation path. |
| `--generator-seed` | integer | `13` | Random seed passed to iGen3 generation. |

## Shared docking target and protocol flags

These flags apply to `screen` and `tune-docking`.
`--target` is the normal path. Expert mode instead requires
`--receptor` and `--center`; AutoDock-GPU expert mode also
requires `--adgpu-grid`.

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--engine` | `unidock`, `autodock-gpu` | `unidock` | Select the docking backend. |
| `--target` | prepared target directory | unset | Load the receptor, pocket, maps, checksums, and geometry from a reusable target. Cannot be combined with expert geometry flags. |
| `--receptor` | rigid receptor `.pdbqt` | unset | Expert-mode prepared receptor. |
| `--adgpu-grid` | AutoGrid `.maps.fld` | unset | Expert-mode AD4 map descriptor; valid only with AutoDock-GPU. |
| `--center` | three floats: `X Y Z` | required in expert mode | Docking-box center in angstrom. |
| `--size` | three positive floats: `X Y Z` | target value or `22.5 22.5 22.5` | Expert-mode box dimensions in angstrom. |
| `--search-mode` | `fast`, `balance`, `detail` | `balance` | Engine effort preset; AutoDock-GPU maps these to 10, 20, or 50 LGA runs. |
| `--scoring` | `auto`, `vina`, `vinardo`, `ad4` | `auto` | Auto selects Vina for Uni-Dock and AD4 for AutoDock-GPU; other combinations are rejected. |
| `--num-modes` | integer > 0 | `1` | Requested output poses; AutoDock-GPU currently requires exactly one. |
| `--energy-range` | float | `3.0` | Uni-Dock-only output energy window. |
| `--refine-step` | integer > 0 | `3` | Uni-Dock-only explicit-receptor refinement steps. |
| `--no-refine` | switch | off | Uni-Dock only: skip explicit-receptor refinement. |
| `--unidock-verbosity` | `0`, `1`, `2` | `0` | Uni-Dock engine log verbosity. |
| `--seed` | integer | `181129` | Docking random seed. |
| `--device-id` | integer >= 0 | `0` | Visible CUDA device index. |
| `--max-gpu-memory` | integer >= 0, MiB | `0` | Uni-Dock-only memory cap; zero keeps the engine default. |
| `--adgpu-runs` | integer > 0 | search preset | Override AutoDock-GPU LGA runs instead of 10/20/50. |
| `--adgpu-evaluations` | integer > 0 | unset | Set a hard maximum score-evaluation count per LGA run. |
| `--adgpu-no-heuristics` | switch | off; heuristics on | Disable ligand-based evaluation heuristics. |
| `--adgpu-no-autostop` | switch | off; autostop on | Disable convergence-based early stopping. |
| `--adgpu-local-search` | `sw`, `sd`, `fire`, `ad`, `adam` | `ad` | Select the AutoDock-GPU local-search method. |
| `--adgpu-cpu-threads` | integer > 0 | `4` | CPU threads per AutoDock-GPU file-list process. |
| `--adgpu-workers` | `auto` or integer 1-64 | `auto` | Concurrent same-GPU processes. Screen auto prefers a profile, otherwise selects a capacity-class prior and owns a private CUDA MPS lifecycle; unsupported automatic concurrency falls back to one. Tuning auto means one. |
| `--adgpu-executable` | command or path | `autodock_gpu` | Override the optimized container executable/symlink. |

## `screen` execution and output flags

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--output-dir` | `PATH` | required | Destination manifest, validation tables, scores, logs, and optional poses. |
| `--batch-size` | `auto` or integer > 0 | `auto` | Outer docking batch; auto uses the supplied profile or a hardware heuristic. |
| `--batch-profile` | JSON `PATH` | unset | Profile written by `tune-docking`; can supply batch and AutoDock-GPU worker optima. |
| `--prep-workers` | `auto` or integer > 0 | `auto` | Ligand-preparation processes; auto uses physical cores in the current affinity, reserves control capacity, applies an available-RAM bound, and caps at 64. |
| `--prep-mode` | `standard`, `fast` | `standard` | Standard minimizes ETKDG conformers; fast skips separate force-field minimization. |
| `--embed-max-attempts` | `auto` or integer > 0 | `auto` (50) | Bound deterministic ETKDG attempts for hard molecules. |
| `--embed-timeout` | `auto` or integer > 0 | `auto` (3 seconds) | Native RDKit wall guard for the primary ETKDG phase; the optional fallback receives a smaller budget. |
| `--validation-workers` | `auto` or integer > 0 | `auto` | RDKit validation processes; auto uses available CPUs capped at 32. |
| `--no-deduplicate` | switch | off | Preserve duplicate canonical SMILES. |
| `--num-shards` | integer > 0 | `1` | Deterministically divide external input rows into this many shards. |
| `--shard-index` | integer from 0 to `num-shards - 1` | `0` | Process only this zero-based shard. |
| `--scratch-dir` | `PATH` | `$SLURM_TMPDIR`, then system temp | Override temporary ligand/work storage. |
| `--keep-work` | switch | off | Preserve temporary prepared ligands and engine work files. |
| `--pose-output` | `none`, `merged`, `individual` | `merged` | Save scores only, one merged pose stream, or one pose file per ligand. |
| `--individual-poses` | switch | off | Compatibility alias that forces `--pose-output individual`. |

## `tune-docking`-only flags

The tuner also accepts all shared library-input and docking-protocol flags.

| Flag | Accepted value(s) | Default | Description |
| --- | --- | --- | --- |
| `--profile` | JSON `PATH` | required | Destination measured batch/worker profile. |
| `--batch-sizes` | comma-separated positive integers | hardware candidates | Candidate outer batches; omission uses engine/GPU-specific defaults. |
| `--prep-workers` | `auto` or integer > 0 | `auto` | Ligand-preparation processes; auto is physical-core and available-RAM bounded, capped at 64. |
| `--prep-mode` | `standard`, `fast` | `standard` | Preparation protocol used for every timed candidate. |
| `--scratch-dir` | `PATH` | system temp | Override tuning work storage. |

Return to the [project README](../README.md) for installation and workflow
examples.

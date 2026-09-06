# Model Artifacts

These four checkpoint directories are mounted into Docker containers at runtime.
Use `-v "${PWD}/models:/models:ro"` or set `IGEN3_MODEL_DIR` to this directory
when running outside the repository root.

| Directory | CLI model id | Description |
| --- | --- | --- |
| `base_isomeric` | `base-isomeric` | Base transformer trained for isomeric SMILES. |
| `base_nonisomeric` | `base-nonisomeric` | Base transformer trained for non-isomeric SMILES. |
| `rl_isomeric` | `rl-isomeric` | RL-tuned generator for isomeric SMILES. |
| `rl_nonisomeric` | `rl-nonisomeric` | RL-tuned generator for non-isomeric SMILES. |

Each directory contains one `.pth` state dict and its matching `vocab.pkl`.

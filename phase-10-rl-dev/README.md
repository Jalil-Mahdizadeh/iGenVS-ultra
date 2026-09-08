# Universal RL protocol development

Status: **accepted and frozen**. The final protocol passed all predeclared gates on
all 8/8 development receptors. Its byte-exact protocol and evidence hashes are in
`freeze.json`; the five-receptor benchmark is a one-shot application of that
frozen recipe. That held-out benchmark subsequently passed 5/5 without changing
the recipe; see `../phase-10-rl-bench/REPORT.md`.

This directory is the clean replacement for the archived Phase-10 experiments.
It develops one target-independent training algorithm on the eight receptors in
`targets.txt`. The five receptors in `complexes/bench-panel.txt` are excluded until
the development acceptance contract passes 8/8 and the protocol is frozen.

The implementation reuses the existing iGen3 model and sampler, Phase-4 prepared
targets, `igenvs screen`, and Uni-Dock. The RL layer supplies policy gradients,
an immutable base-model KL prior, target-local score caching, chemistry-qualified
rewards, anti-collapse gradient weighting, and fresh checkpoint evaluation.

The frozen protocol and its predeclared acceptance criteria are in
`protocol.json`; the design rationale and prior evidence are in `plan.md`.

The accepted implementation is exposed to users by
`./igenvs-ultra rl-train`; see [`../docs/CLI.md`](../docs/CLI.md). The public
orchestrator verifies the frozen protocol and implementation hashes before it
runs and does not expose scientific override flags.

Run the focused tests:

```bash
apptainer exec --bind "$PWD:$PWD" \
  --env "PYTHONPATH=$PWD/phase-10-rl-dev/src:$PWD/iGenVS/iGen3/src:$PWD/iGenVS/src" \
  /nobackup/proj/disk/theo-storage/personal/jalil/iGenVS/containers/iGenVS.SIF \
  python3 -m pytest -q phase-10-rl-dev/tests
```

Launch one four-GPU node per development receptor:

```bash
mkdir -p phase-10-rl-dev/logs
sbatch phase-10-rl-dev/slurm/develop_array.sbatch
```

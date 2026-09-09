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

Operational implementation updates are recorded in `maintenance.json`, an
explicit hash-checked overlay on the historical freeze. Neither `protocol.json`
nor `freeze.json` is rewritten. Before a stopping decision, `recover --job-dir`
repairs checkpoint-derived history/evaluations and selected/latest exports on
CPU without generating molecules or training. Partial reference directories are
retained before retry; reference score publication and stage copying are atomic.

Training retains FP32 and the full effective batch. Backward is microbatched;
sampling retries allocation failures with smaller cache/model chunks while
preserving the released full-batch token-major RNG stream. Tests compare tokens,
final RNG state, and gradients against the original full-batch operations.

Run the focused tests:

```bash
apptainer exec --bind "$PWD:$PWD:ro" --pwd /tmp \
  --env "PYTHONPATH=$PWD/phase-10-rl-dev/src:$PWD/iGenVS/iGen3/src:$PWD/iGenVS/src" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  /nobackup/proj/disk/theo-storage/personal/jalil/iGenVS/containers/iGenVS.SIF \
  python3 -m pytest -q -p no:cacheprovider "$PWD/phase-10-rl-dev/tests"
```

Add `--nv --env IGENVS_RL_CUDA_TEST=1` to run the opt-in real-model 4,096-draw
test with a 2 GiB PyTorch allocator limit. This tests allocation fallback on the
available GPU, not the total memory requirements of an entire docking/RL job.
Add `--env IGENVS_RL_REAL_MODEL_TEST=1` for the real-weight CLI recovery smoke
test. Its tiny synthetic-oracle fixture is isolated from the frozen protocol;
it verifies export repair and idempotent resume, not scientific acceptance.

Launch one four-GPU node per development receptor:

```bash
mkdir -p phase-10-rl-dev/logs
sbatch phase-10-rl-dev/slurm/develop_array.sbatch
```

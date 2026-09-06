# Portable ultra-screen optimization results

Regular physical-docking smoke/medium qualification is reported separately in
[`DOCKING-RESULTS.md`](DOCKING-RESULTS.md).

Status: smoke and medium qualification complete on one GPU; full 1/2/4-GPU
benchmark intentionally deferred.

## Boundary

These are real invocation-to-durable-result measurements, not isolated kernel
rates. Each run loaded the released base-isomeric iGen3 generator, gMolAI
encoder, calibrator, standardizer, and three 4AG8 round-5 target heads through
the two released containers. Generation used temperature 1.0, top-k 64, and
seed 2026090601. `--save-policy all` wrote every member score, ensemble score,
and uncertainty value to the final CSV. The harness rejected a run unless the
manifest certified exactly the requested number of successfully encoded and
scored rows.

Hardware was one visible NVIDIA GH200 120GB (97,871 MiB reported), 72 physical
CPU cores in the task affinity, driver 580.159.04, PyTorch 2.11.0+cu128 in the
iGen3 container, and Slurm job 2071150. Rates below must not be projected to
another GPU without running its automatic planner.

## Final measurements

| Run | Rows | Profile state | Wall (s) | Rows/s | Rows/hour | Generation (s) | Score wall (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| `smoke-codefreeze` | 8,192 | smoke-local tuning | 52.705 | 155.43 | 559,550 | 27.257 | 2.913 |
| `medium-codefreeze-fresh` | 200,000 | fresh encoder profile | 85.163 | 2,348.43 | 8,454,342 | 45.102 | 16.982 |
| `medium-codefreeze-cached` | 200,000 | reusable encoder profile | **75.170** | **2,660.63** | **9,578,259** | 44.754 | 7.320 |

The two final medium runs have the same result SHA-256:
`980335c823b8cc13380347bb8e7ca0b18b369986c5361284832f513ae718d9e3`.
The final smoke hash
`18c29950572bcb1d8c6fef3f27ef4734148d216dc1b6bb98c80f48aa2fc85c7b`
also matches the pre-parallel smoke fixture. Thus cache state and persistent
parallel chemistry did not alter a fixed execution plan's result.

All three code-frozen records contain the on-disk source hashes
`9ccdf961cadefe3e67e2a117effd0d31984878aed16f9b3a1c88cb92a5271a3c`
for `generation_worker.py`,
`ca903b26bf672e0c0c917787eaa49062b6243eb4455e83da7e13ff990ad76e06`
for `model_ops.py`, and
`4120f013faa1b4de7d3fb4eb7456d7e546505f56159b43458c6d3aaa65c90bdc`
for the then-frozen `workflow.py`. The workflow hash later changed to add the
isolated physical-docking control plane; generation and model-worker hashes
remain unchanged.

Post-docking-optimization regressions reproduced both reference result hashes.
The exact-plan 8,192-row smoke completed in 50.234 seconds with
`18c29950572bcb1d8c6fef3f27ef4734148d216dc1b6bb98c80f48aa2fc85c7b`.
The cached 200,000-row medium completed in 75.536 seconds (9,531,919/hour) with
`980335c823b8cc13380347bb8e7ca0b18b369986c5361284832f513ae718d9e3`.
Its 0.49% rate difference from the frozen 75.170-second run is within ordinary
single-run timing noise and shows no material screening-speed regression.

The previous best cached automatic medium run was 104.370 seconds and
6,898,564 rows/hour. The final reusable plan reduces wall by 28.0% and raises
throughput by 38.8%. The comparison is like-for-like: one GPU, 200,000 requested
successful scores, real generation and encoding, and all scores durably saved.

## Planner decisions and component evidence

- The final automatic generator batch was 65,280: the minimum of live
  memory-fit capacity (about 67,120 here), workload, the 131,072 safety ceiling,
  and the current CUDA SDPA launch bound. A diagnostic sweep measured accepted
  rates of 12,749/s at 16,640, 13,142/s at 33,536, and 13,491/s at 50,176;
  67,072 produced `CUDA error: invalid configuration argument` and was safely
  rejected. A direct 65,280 end-to-end probe was fastest before it became the
  automatic safe choice.
- The final generator used 64 persistent RDKit canonicalization processes,
  selected as physical cores in its lane's affinity minus one, capped at 64.
  An 80,000-row probe fell from 10.98 seconds serial to 0.69 seconds at 16
  workers, 0.35 at 32, 0.24 at 48, and 0.18 at 64. The actual warmed generated
  stage fell from 49.63 seconds with serial chemistry to 18.50 seconds with the
  final process pool in the cold diagnostic run.
- Fresh gMolAI qualification selected batch 512. Its 49,152-row measurements
  were 13,672/s (64), 23,578/s (128), 31,373/s (192), 42,948/s (256), and
  53,113/s (512). The rate includes the promoted encoder path used by the
  scorer and is kept distinct from the published standalone 58,330/s boundary.
- An explicit `torch.compile` medium probe produced the same result hash but
  took 77.534 seconds versus 76.965 seconds for the comparable eager run.
  Compilation was therefore rejected for automatic use at this workload.
- One gMol policy rejection in several earlier seeded plans was replenished;
  the terminal manifest still contained exactly 200,000 scores. The final
  65,280 plan happened to contain no policy rejection and completed in one
  outer stream partition.

## Portability behavior

No device name or GH200 batch table selects the production plan. Each machine
uses its visible/cgroup-granted GPUs, live free memory, compute capability,
physical CPU cores and affinity, host RAM, scratch capacity, workload size,
and measured encoder candidates. Profile keys include GPU resources, CPU model
and topology, worker count, Python/PyTorch/CUDA/NumPy/RDKit/backend versions,
model artifact hashes, and scientific generation settings. A cached generator
choice is rejected if current memory headroom is smaller.

Generator calibration is attempted only when its four warmed/measured
candidates account for under approximately 1% of the per-GPU requested work;
otherwise the largest safe bound is used immediately. OOM, allocation, launch
resource, and invalid-configuration capacity failures automatically shrink the
batch. This is why a new workstation can run `screen-fast N` without carrying
a profile from this GH200.

## Reproduce

The harness is `speed_optimization.py`. The authoritative machine-readable
records are:

- `results/smoke-codefreeze/summary.json`
- `results/medium-codefreeze-fresh/summary.json`
- `results/medium-codefreeze-cached/summary.json`

This pass does not claim a theoretical decoder maximum or multi-GPU scaling
result. Shared-memory transport, cross-stage overlap, fused GPU
encoding/calibration/heads, and decoder-kernel work remain in the blueprint.
The requested full benchmark should be run separately after this code is
frozen.

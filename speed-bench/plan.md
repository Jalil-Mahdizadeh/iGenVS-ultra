Tools available:
- iGenVS is a fast molecular docking tool: It has two engines, iGen3, a tranformer-based de novo SMILES generator, Uni-Dock as the main docking engine and AutoDock-GPU as alternative docking engine.
- gMolAI is a fast graph-based moleculae encoding tool.
- iGenVS-ultra is a combination of iGenVS and gMolAI + optional active learning (AL) for ultra large molecular screening.the

SIF files for running tools:
iGenVS: /nobackup/proj/disk/theo-storage/personal/jalil/iGenVS/containers/iGenVS.SIF
gMolAI: /nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/gmolai-pyg-25.09-arm64.sif

The goal is to conduct a speed benchmarking in two separate parts A and B as outlined below:

A- Regular docking benchmark (iGenVS):
Gerentate fixed 20k, 40k, and 80k molecules using iGen3, base-isomeric and deduplicate. Run the following benchmarking using '4ag8'.
Apply 20K for 1GPU, 40k for 2GPUs, 80k for 4GPU.

1- Regular docking (iGenVS), fixed 20k molecules, Uni-Dock 1.2.0 / Vinak, fast, 1GPU
2- Regular docking (iGenVS), fixed 40k molecules, Uni-Dock 1.2.0 / Vinak, fast, 2GPU
3- Regular docking (iGenVS), fixed 80k molecules, Uni-Dock 1.2.0 / Vinak, fast, 4GPU
4- Regular docking (iGenVS), fixed 20k molecules, Uni-Dock 1.2.0 / Vinak, balence, 1GPU
5- Regular docking (iGenVS), fixed 40k molecules, Uni-Dock 1.2.0 / Vinak, balence, 2GPU
6- Regular docking (iGenVS), fixed 80k molecules, Uni-Dock 1.2.0 / Vinak, balence, 4GPU
7- Regular docking (iGenVS), fixed 20k molecules, Uni-Dock 1.2.0 / Vinak, detail, 1GPU
8- Regular docking (iGenVS), fixed 40k molecules, Uni-Dock 1.2.0 / Vinak, detail, 2GPU
9- Regular docking (iGenVS), fixed 80k molecules, Uni-Dock 1.2.0 / Vinak, detail, 4GPU
10- Regular docking (iGenVS), fixed 20k molecules, AutoDock-GPU 1.6 / AD4, 1GPU
11- Regular docking (iGenVS), fixed 40k molecules, AutoDock-GPU 1.6 / AD4, 2GPU
12- Regular docking (iGenVS), fixed 80k molecules, AutoDock-GPU 1.6 / AD4, 4GPU

B- Screening benchmark (iGenVS-ultra):
1- Screening (iGenVS-ultera), use models from '4ag8' AL round-5, stream batches of smiles using iGen3 base-isomeric, screan the batches using the models, continue until exactly 10M unique molecules have finite committed scores, 1GPU.
2- Screening (iGenVS-ultera), use models from '4ag8' AL round-5, stream batches of smiles using iGen3 base-isomeric, screan the batches using the models, continue until exactly 10M unique molecules have finite committed scores, 2GPU.
3- Screening (iGenVS-ultera), use models from '4ag8' AL round-5, stream batches of smiles using iGen3 base-isomeric, screan the batches using the models, continue until exactly 10M unique molecules have finite committed scores, 4GPU.

- All steps (smiles generation, encoding, target-specific head inference, etc.) must be devided between GPUs for a fair comparison.
- Preserve generate > encode > inference dependencies within each batch, but enable cross-batch overlap.
- Use the public optimized wrappers. All performance settings, including batch sizes and worker counts, must be determined automatically by the pipeline.
- For the screening benchmark, I want the speed of smiles generation, encoding, target-specific head inference, and e2e.
- Run only one cold timing sample per case; do not run warm or repeated timing samples.
- For docking report complete-wall input/hour, complete-wall successful/hour, engine-only successful/hour, yield, and all stage timings.
- For screening report exactly committed finite scores/hour end to end, yield, and all stage timings.

Computetional resource:
- You are running on a single GPU interactive node on Arrhenius HPC; meaning that you can run jobs directly on this interactive node and/or submit/cancel jobs on separate nodes (up to 4x GPUs per node).
- You can submit multiple jobs at the same time. No need to use exactly the same node.

In the end, generate a concise report + docking/h and screening/h for each benchmark.

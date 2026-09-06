# Benchmark artifacts

Engine-run directory names identify the docking engine versions, GPU, and
benchmark date (`YYYYMMDD`). The correlation directory contains a
derived cross-engine analysis of the matched runs.

| Directory | Scope |
| --- | --- |
| [`unidock-v1.2.0-gh200-20260830`](unidock-v1.2.0-gh200-20260830) | Uni-Dock 1.2.0 throughput optimization, quality checks, iGen3 generation measurements, and one-/four-GH200 scaling. |
| [`autodock-gpu-v1.6-vs-unidock-v1.2.0-gh200-20260830`](autodock-gpu-v1.6-vs-unidock-v1.2.0-gh200-20260830) | AutoDock-GPU 1.6 optimization plus matched one-/four-GH200 comparisons with Uni-Dock 1.2.0. |
| [`correlations`](correlations) | Reproducible Uni-Dock/Vina versus AutoDock-GPU/AD4 score correlation, rank correlation, and top-hit overlap. |

The reports, summaries, profiles, and compact provenance records are tracked in
Git. Bulky result tables, validation databases, generated libraries, maps, and
raw logs remain local and are excluded by `.gitignore`.

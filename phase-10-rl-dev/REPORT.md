# Eight-target universal RL protocol development

## Decision

**Accepted: all 8/8 development targets passed every predeclared online, independent docking, diversity, and chemistry-safety gate.**

The same receptor-agnostic recipe was used independently for every target. Each run began from `base-isomeric`; only the receptor and its target-local base docking references differed. A bounded percentile warm-up was followed by binary top-1% Uni-Dock/Vina `balance` concentration and binary top-0.5% `fast` refinement. Every stage retained an immutable base KL prior; the binary stages used chemistry qualification, repeat-capped reward gradients, and uncached checkpoint docking.

Final evaluation used matched independent 10,000-draw base and RL samples. Invalid strings, repeats, docking failures, non-elites, and positive scores remain in raw denominators. Every distinct molecule was freshly docked in both Uni-Dock `fast` and `balance`.

## Headline results

| Target | Pass | Online stop | Stop / selected | Training min | Fast q-elite | Balance q-elite | Fast distinct | Top molecule | Chemistry | Fast gains best10 / median / p95 | Balance gains best10 / median / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1iep | yes | yes | 9 / 9 | 126.0 | 93.58% | 97.59% | 1,405 | 3.39% | 98.65% | 1.966/4.208/5.177 | 1.812/4.177/5.831 |
| 1nyx | yes | yes | 12 / 11 | 135.5 | 96.67% | 97.28% | 2,758 | 3.09% | 98.28% | 1.561/3.715/5.224 | 1.591/3.614/5.351 |
| 1t7r | yes | yes | 4 / 4 | 76.7 | 95.49% | 95.24% | 2,174 | 3.89% | 98.48% | 1.366/4.573/26.043 | 1.385/4.291/23.547 |
| 2zv2 | yes | yes | 4 / 3 | 92.8 | 95.27% | 94.78% | 3,828 | 2.47% | 97.28% | 3.123/4.374/5.802 | 2.933/4.086/5.399 |
| 4f8h | yes | yes | 10 / 10 | 122.7 | 92.63% | 98.13% | 965 | 9.41% | 99.03% | 1.517/3.408/3.640 | 1.440/3.208/4.294 |
| 4yay | yes | yes | 8 / 8 | 134.4 | 93.24% | 96.14% | 2,094 | 5.52% | 98.06% | 2.575/3.819/4.806 | 2.424/3.993/5.342 |
| 5ek0 | yes | yes | 8 / 7 | 127.9 | 95.17% | 96.09% | 4,171 | 3.15% | 98.39% | 2.451/3.165/4.509 | 2.265/3.066/4.557 |
| 5mzj | yes | yes | 3 / 3 | 89.8 | 95.94% | 95.30% | 3,008 | 5.25% | 97.68% | 2.238/4.895/19.071 | 2.244/4.365/13.718 |

## Acceptance details

| Target | Failed gates | Fast positive | Balance positive | Unique valid | Unique scaffolds | Internal diversity | QED | fraction-Csp3 | Aromatic rings | Formal charge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1iep | none | 0.050% | 0.070% | 1,679 | 926 | 0.470 | 0.602 | 0.158 | 3.039 | 0.792 |
| 1nyx | none | 0.040% | 0.030% | 3,001 | 1,323 | 0.514 | 0.577 | 0.169 | 3.218 | 0.011 |
| 1t7r | none | 0.230% | 0.240% | 2,537 | 1,440 | 0.565 | 0.720 | 0.604 | 0.874 | 0.002 |
| 2zv2 | none | 0.080% | 0.070% | 4,106 | 3,427 | 0.612 | 0.627 | 0.268 | 2.549 | 0.005 |
| 4f8h | none | 0.030% | 0.020% | 1,172 | 547 | 0.533 | 0.700 | 0.142 | 2.884 | -0.523 |
| 4yay | none | 0.030% | 0.060% | 2,471 | 1,884 | 0.543 | 0.552 | 0.207 | 3.218 | 0.003 |
| 5ek0 | none | 0.030% | 0.050% | 4,544 | 3,553 | 0.638 | 0.668 | 0.230 | 2.607 | -0.145 |
| 5mzj | none | 0.060% | 0.120% | 3,260 | 2,591 | 0.582 | 0.685 | 0.302 | 2.438 | 0.000 |

## Runtime and integrity

Recorded training time totals 15.09 node-hours and 60.38 allocated GPU-hours. Integrity audit: **pass**.

Docking scores are computational prioritization estimates, not measured binding affinities. The chemistry guards directly block the low-QED, high-charge, and excessively planar/aromatic failure modes seen previously, but they do not establish synthesis or activity. The single best score is reported in `docking-summary.csv` as a diagnostic; acceptance uses the best-10 mean because a one-sample minimum is an unstable extreme statistic.

## Output index

- `docking-summary.csv`: full fast/balance distribution comparisons.
- `molecular-structural-summary.csv`: molecular properties and diversity.
- `training-times.csv`: wall time and allocated GPU-hours.
- `model-manifest.csv`: accepted model paths and hashes.
- `integrity-audit.json`: protocol, target, row, checkpoint, and docking checks.
- `<target>/validation/<target>_base-vs-rl_10k.csv`: paired raw libraries.

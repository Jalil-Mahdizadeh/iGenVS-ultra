# Frozen five-target RL protocol benchmark

## Decision

**Benchmark result: all 5/5 targets passed every frozen independent docking, diversity, and chemistry-safety endpoint.**

The protocol remained frozen regardless of these outcomes; no benchmark result was used to change training, stopping, or acceptance settings.

Together with the accepted 8/8 development result, the same protocol passed 13/13 curated receptors: eight used to develop the rule and five held out until after it was frozen. This supports freezing one universal training protocol for this workflow; it does not imply one shared receptor-independent model or guarantee performance on every possible target.

The same receptor-agnostic recipe was used independently for every target. Each run began from `base-isomeric`; only the receptor and its target-local base docking references differed. A bounded percentile warm-up was followed by binary top-1% Uni-Dock/Vina `balance` concentration and binary top-0.5% `fast` refinement. Every stage retained an immutable base KL prior; the binary stages used chemistry qualification, repeat-capped reward gradients, and uncached checkpoint docking.

Final evaluation used matched independent 10,000-draw base and RL samples. Invalid strings, repeats, docking failures, non-elites, and positive scores remain in raw denominators. Every distinct molecule was freshly docked in both Uni-Dock `fast` and `balance`.

## Headline results

| Target | Pass | Online stop | Stop / selected | Training min | Fast q-elite | Balance q-elite | Fast distinct | Top molecule | Chemistry | Fast gains best10 / median / p95 | Balance gains best10 / median / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1err | yes | yes | 8 / 7 | 138.2 | 96.69% | 97.10% | 1,276 | 8.23% | 98.80% | 1.655/3.674/5.023 | 1.716/3.812/5.192 |
| 4ag8 | yes | yes | 4 / 4 | 118.8 | 95.56% | 95.83% | 2,418 | 2.99% | 96.98% | 2.255/4.430/6.051 | 2.004/4.281/5.984 |
| 5l2s | yes | yes | 7 / 7 | 123.5 | 95.85% | 95.90% | 4,888 | 2.04% | 97.97% | 2.537/3.921/5.074 | 2.522/3.748/5.007 |
| 6d6t | yes | yes | 5 / 5 | 105.2 | 96.72% | 96.09% | 2,371 | 3.07% | 98.37% | 1.777/4.016/6.452 | 1.695/3.671/5.793 |
| 6iiu | yes | yes | 13 / 12 | 137.8 | 97.34% | 98.58% | 1,009 | 3.25% | 99.19% | 1.231/3.444/6.084 | 1.192/3.410/5.949 |

## Acceptance details

| Target | Failed gates | Fast positive | Balance positive | Unique valid | Unique scaffolds | Internal diversity | QED | fraction-Csp3 | Aromatic rings | Formal charge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1err | none | 0.020% | 0.030% | 1,551 | 903 | 0.439 | 0.523 | 0.157 | 3.893 | 0.966 |
| 4ag8 | none | 0.100% | 0.060% | 2,762 | 1,957 | 0.523 | 0.619 | 0.130 | 3.385 | 0.032 |
| 5l2s | none | 0.160% | 0.140% | 5,203 | 4,579 | 0.613 | 0.582 | 0.248 | 3.041 | 0.007 |
| 6d6t | none | 0.060% | 0.080% | 2,600 | 2,047 | 0.645 | 0.611 | 0.193 | 2.530 | -0.002 |
| 6iiu | none | 0.160% | 0.060% | 1,179 | 490 | 0.419 | 0.609 | 0.160 | 3.008 | 0.001 |

## Molecular and structural comparison

Each cell is `base -> RL`; the complete property set is retained in `molecular-structural-summary.csv`.

| Target | Unique valid | Unique scaffolds | Internal diversity | QED | SA score |
|---|---:|---:|---:|---:|---:|
| 1err | 9,779 -> 1,551 | 8,274 -> 903 | 0.891 -> 0.439 | 0.603 -> 0.523 | 3.065 -> 3.611 |
| 4ag8 | 9,779 -> 2,762 | 8,274 -> 1,957 | 0.891 -> 0.523 | 0.603 -> 0.619 | 3.065 -> 2.529 |
| 5l2s | 9,779 -> 5,203 | 8,274 -> 4,579 | 0.891 -> 0.613 | 0.603 -> 0.582 | 3.065 -> 3.053 |
| 6d6t | 9,779 -> 2,600 | 8,274 -> 2,047 | 0.891 -> 0.645 | 0.603 -> 0.611 | 3.065 -> 2.862 |
| 6iiu | 9,779 -> 1,179 | 8,274 -> 490 | 0.891 -> 0.419 | 0.603 -> 0.609 | 3.065 -> 2.389 |

| Target | Molecular weight | LogP | TPSA | fraction-Csp3 | Aromatic rings | Formal charge |
|---|---:|---:|---:|---:|---:|---:|
| 1err | 359.472 -> 426.239 | 2.698 -> 3.794 | 74.601 -> 39.243 | 0.325 -> 0.157 | 2.318 -> 3.893 | 0.134 -> 0.966 |
| 4ag8 | 359.472 -> 403.495 | 2.698 -> 3.049 | 74.601 -> 81.324 | 0.325 -> 0.130 | 2.318 -> 3.385 | 0.134 -> 0.032 |
| 5l2s | 359.472 -> 415.981 | 2.698 -> 2.714 | 74.601 -> 85.094 | 0.325 -> 0.248 | 2.318 -> 3.041 | 0.134 -> 0.007 |
| 6d6t | 359.472 -> 314.418 | 2.698 -> 3.029 | 74.601 -> 59.584 | 0.325 -> 0.193 | 2.318 -> 2.530 | 0.134 -> -0.002 |
| 6iiu | 359.472 -> 417.796 | 2.698 -> 4.175 | 74.601 -> 49.684 | 0.325 -> 0.160 | 2.318 -> 3.008 | 0.134 -> 0.001 |

## Runtime and integrity

Recorded training time totals 10.39 node-hours and 41.57 allocated GPU-hours. Integrity audit: **pass**.

As a non-gating diagnostic, the observed single-molecule minimum also improved in all 10 target/mode comparisons (0.936 to 2.301 kcal/mol).

Docking scores are computational prioritization estimates, not measured binding affinities. The chemistry guards directly block the low-QED, high-charge, and excessively planar/aromatic failure modes seen previously, but they do not establish synthesis or activity. The single best score is reported in `docking-summary.csv` as a diagnostic; acceptance uses the best-10 mean because a one-sample minimum is an unstable extreme statistic.

## Output index

- `docking-summary.csv`: full fast/balance distribution comparisons.
- `molecular-structural-summary.csv`: molecular properties and diversity.
- `training-times.csv`: wall time and allocated GPU-hours.
- `model-manifest.csv`: accepted model paths and hashes.
- `integrity-audit.json`: protocol, target, row, checkpoint, and docking checks.
- `<target>/validation/<target>_base-vs-rl_10k.csv`: paired raw libraries.

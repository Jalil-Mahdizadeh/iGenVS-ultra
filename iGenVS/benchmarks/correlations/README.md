# Uni-Dock 1.2.0 versus AutoDock-GPU 1.6 correlations

## Result

The matched results show stable, moderate agreement rather than interchangeable
rankings. On the definitive 80,000-input comparison, 79,606 molecules had a
finite score from both engines:

- Pearson docking-score correlation was **0.3021** (`r² = 0.0913`);
- Spearman rank correlation was **0.3950**;
- Kendall rank correlation was **0.2730**;
- the top-1% lists overlapped by **12.92%**; and
- the median molecule moved **20.50 percentile points** between rankings.

The independent 20,000-input comparison reproduced the relationship: Pearson
`r = 0.3024` and Spearman `rho = 0.3994`. Both engines use
lower-is-better scores, so positive coefficients indicate agreement.

## Data and method

The inputs are the matched score-only runs in the
[AutoDock-GPU versus Uni-Dock benchmark](../autodock-gpu-v1.6-vs-unidock-v1.2.0-gh200-20260830).
Rows were joined by `molecule_id` and retained only when:

1. both rows contained the same canonical SMILES;
2. both terminal statuses were `success`; and
3. both docking scores were finite.

| Run | Inputs/engine | Paired finite scores | SMILES mismatches |
| --- | ---: | ---: | ---: |
| One GH200 | 20,000 | 19,906 | 0 |
| Four GH200s | 80,000 | 79,606 | 0 |

Pearson measures linear agreement between the numeric scores. Spearman and
Kendall measure rank agreement and are more appropriate for comparing these
different scoring functions. Average ranks were used for score ties.

## Global score and rank agreement

| Metric | 20K replication | 80K definitive |
| --- | ---: | ---: |
| Pearson `r` | 0.3024 | **0.3021** |
| Pearson `r²` | 0.0915 | **0.0913** |
| Pearson `r`, 1% winsorized per tail | 0.3083 | **0.3044** |
| Spearman `rho` | 0.3994 | **0.3950** |
| Kendall `tau-b` | 0.2763 | **0.2730** |
| Partial Pearson controlling heavy atoms and rotatable bonds | 0.4305 | **0.4301** |
| Partial Spearman controlling heavy atoms and rotatable bonds | 0.4531 | **0.4477** |
| Median absolute percentile displacement | 20.29 | **20.50** |
| Mean absolute percentile displacement | 24.81 | **24.90** |
| Within 10 percentile points | 28.45% | **28.73%** |
| Within 25 percentile points | 58.17% | **57.77%** |

Only about 9.1% of cross-engine score variance is captured by a simple linear
relationship. Winsorizing both score distributions changes Pearson `r`
by only 0.0023, so rare score extremes do not explain the modest global
correlation.

## Top-rank overlap

The primary table uses exactly the same number of molecules from each 80K
ranking. Score ties are resolved deterministically by `molecule_id`;
the machine-readable summary also records tie-inclusive cutoffs and overlaps.

| Rank window | Molecules/engine | Shared | Observed overlap | Random expectation | Enrichment over random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top 0.1% | 80 | 3 | 3.75% | 0.10% | 37.3x |
| Top 1% | 797 | 103 | **12.92%** | 1.00% | 12.9x |
| Top 5% | 3,981 | 847 | **21.28%** | 5.00% | 4.25x |
| Top 10% | 7,961 | 2,217 | **27.85%** | 10.00% | 2.78x |

The overlap is substantially enriched over random, but most top-ranked
molecules remain engine-specific. The extreme top-0.1% result is only three
shared molecules and should not be treated as a stable enrichment estimate.

## Score distributions and molecular-size association

Raw values are shown for auditability, not numeric cross-engine comparison.

| 80K paired set | Uni-Dock / Vina | AutoDock-GPU / AD4 |
| --- | ---: | ---: |
| Minimum | -11.671 | -12.590 |
| 1st percentile | -9.865 | -10.450 |
| 5th percentile | -9.282 | -9.840 |
| Median | -7.800 | -8.460 |
| 95th percentile | -6.117 | -7.000 |
| 99th percentile | 120.984 | -6.190 |
| Maximum | 302.490 | -0.240 |
| Unique score values | 78,687 | 701 |
| Finite positive values | 2,194 (2.76%) | 0 |

The large finite positive Uni-Dock tail is placed at the unfavorable end of the
ranking and does not drive the top-hit overlap. AutoDock-GPU scores were much
more associated with heavy-atom count in this library: Spearman
`rho = -0.5604` for AD4 versus `-0.0409` for Vina. After
controlling for heavy atoms and rotatable bonds, partial rank correlation rose
from `0.3950` to `0.4477`. This is an association, not
evidence of a causal scoring bias.

## Interpretation

- The engines agree enough that shared top hits are meaningful, but not enough
  to substitute one ranking for the other.
- Vina and AD4 values must not be compared or averaged directly. If a
  dual-engine consensus is used, combine within-engine ranks or normalized
  percentiles under a target-qualified protocol.
- A cascade that applies AutoDock-GPU only to Uni-Dock's narrowest top slice can
  miss AutoDock-GPU-specific candidates. A rank-union or deliberately broader
  first-stage cutoff preserves more complementarity.
- Agreement is not accuracy. No experimental actives, decoys, affinities, or
  reference-pose RMSDs were used here.

These findings apply to one receptor, one ligand distribution, standard ligand
preparation, and the two engine-specific `fast` protocols. Search and
scoring differ simultaneously, so this analysis does not isolate the scoring
functions from their pose-generation algorithms.

## Reproduce

The shipped SIF contains the required NumPy and SciPy versions:

```bash
apptainer exec containers/iGenVS.SIF python3 benchmarks/correlations/analyze.py --output /tmp/igenvs-correlation-summary.json

cmp /tmp/igenvs-correlation-summary.json benchmarks/correlations/summary.json
```

Full result tables remain local and Git-ignored. `summary.json`
records the path, size, and SHA-256 digest of every source CSV so the
calculation is tied to the exact raw outputs.

## Artifacts

- `summary.json`: machine-readable methods, source hashes,
  correlations, distributions, rank distances, and top-rank overlap.
- `analyze.py`: deterministic analysis used to generate the summary.

Docking scores are ranking features under fixed protocols. They are not proof
of binding affinity, activity, selectivity, or safety.

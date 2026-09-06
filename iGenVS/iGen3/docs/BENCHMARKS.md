# Benchmarks

The benchmark commands generate canonical, RDKit-valid, unique molecules for each selected model and then compute RDKit metrics on those final output files.

De novo benchmark:

```bash
igen3 benchmark --count 500000 --output-dir benchmarks/latest
```

Derivative benchmark workflow:

```bash
igen3 randomize-smiles \
  --smiles "N1(C(COC)=O)C(C)CN(CC1)C(c1cc(Cl)ccc1)" \
  --max-variants 10000 \
  --stagnation-limit 5000 \
  --output benchmarks/derivative_latest/seeds/reference_randomized.smi \
  --metadata-output benchmarks/derivative_latest/seeds/reference_randomized.json

igen3 benchmark-derivative \
  --seed-file benchmarks/derivative_latest/seeds/reference_randomized.smi \
  --count 10000 \
  --output-dir benchmarks/derivative_latest
```

Outputs:

| Path | Contents |
| --- | --- |
| `benchmark_summary.csv` | One-row-per-model speed and molecular metric summary, including candidate and accepted counts. |
| `generation_summary.csv` | Generation timing, batch size, output path, candidate count, and accepted output count. |
| `throughput_trace.csv` | Per-batch cumulative throughput for line plots. |
| `all_molecule_metrics.csv` | Per-molecule RDKit descriptor table. |
| `smiles/*.smi` | Final valid unique generated SMILES files. |
| `figures/*.png` | GitHub-ready benchmark charts. |
| `benchmark_metadata.json` | Runtime, CUDA, PyTorch, and benchmark settings. |

Metrics include validity, unique-valid fraction, duplicate fraction, QED, LogP, molecular weight, TPSA, hydrogen bond donors/acceptors, rotatable bonds, Lipinski rule-of-five pass fraction, and SA score when the RDKit SA scorer is available. The molecular-quality figure includes mean SA score in the same chart as the other quality metrics.

# Ready-to-screen 4AG8 example

This compact job contains the released round-5, three-seed target-head ensemble
for PDB target 4AG8. It is included to make a fresh Docker installation
immediately testable without downloading the multi-gigabyte fitting corpus.

From the repository root:

```bash
make screen N=10000
```

The command generates and commits exactly 10,000 finite target-head scores.
Generated state is written to `screens/`, which is intentionally ignored by
Git. Delete a named screen only when you intentionally want a cold rerun;
otherwise the pipeline resumes it safely.

The members are byte-identical copies of the released 4AG8 round-5 artifacts.
Their SHA-256 identities are recorded in `models/round-5/ensemble-manifest.json`.
This example predicts enrichment under that frozen target/protocol; its output
is not a physical docking energy or a calibrated binding probability.

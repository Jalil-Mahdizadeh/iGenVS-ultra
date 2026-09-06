"""RDKit molecular metrics and plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

try:  # RDKit wheels usually include the Contrib SA scorer, but make it optional.
    from rdkit.Contrib.SA_Score import sascorer
except Exception:  # pragma: no cover - depends on RDKit installation layout.
    sascorer = None


def read_smiles(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [line.rstrip("\r\n") for line in fh]


def _lipinski_pass(mw: float, logp: float, hbd: int, hba: int) -> bool:
    return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10


def calculate_molecule_metrics(
    smiles: Iterable[str],
    *,
    model_id: str,
    isomeric_smiles: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    valid_canonicals: list[str] = []

    for row_index, smi in enumerate(smiles):
        smi = smi.strip()
        if not smi:
            records.append(
                {
                    "model_id": model_id,
                    "row_index": row_index,
                    "smiles": smi,
                    "valid": False,
                }
            )
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append(
                {
                    "model_id": model_id,
                    "row_index": row_index,
                    "smiles": smi,
                    "valid": False,
                }
            )
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric_smiles)
        valid_canonicals.append(canonical)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        qed = QED.qed(mol)
        sa_score = sascorer.calculateScore(mol) if sascorer is not None else None

        records.append(
            {
                "model_id": model_id,
                "row_index": row_index,
                "smiles": smi,
                "valid": True,
                "canonical_smiles": canonical,
                "qed": qed,
                "logp": logp,
                "mol_weight": mw,
                "tpsa": tpsa,
                "hbd": hbd,
                "hba": hba,
                "rotatable_bonds": rotatable,
                "lipinski_ro5_pass": _lipinski_pass(mw, logp, hbd, hba),
                "sa_score": sa_score,
            }
        )

    molecule_df = pd.DataFrame.from_records(records)
    total = len(molecule_df)
    valid_df = molecule_df[molecule_df["valid"] == True].copy()  # noqa: E712
    valid_count = len(valid_df)
    unique_valid_count = len(set(valid_canonicals))

    summary: dict[str, object] = {
        "model_id": model_id,
        "total": total,
        "valid_count": valid_count,
        "valid_fraction": valid_count / total if total else 0.0,
        "unique_valid_count": unique_valid_count,
        "unique_valid_fraction": unique_valid_count / valid_count if valid_count else 0.0,
        "duplicate_valid_fraction": 1 - (unique_valid_count / valid_count) if valid_count else 0.0,
    }

    numeric_cols = ["qed", "logp", "mol_weight", "tpsa", "hbd", "hba", "rotatable_bonds", "sa_score"]
    for col in numeric_cols:
        if col in valid_df.columns and valid_df[col].notna().any():
            summary[f"{col}_mean"] = float(valid_df[col].mean())
            summary[f"{col}_median"] = float(valid_df[col].median())
        else:
            summary[f"{col}_mean"] = None
            summary[f"{col}_median"] = None

    if "lipinski_ro5_pass" in valid_df.columns and valid_count:
        summary["lipinski_ro5_pass_fraction"] = float(valid_df["lipinski_ro5_pass"].mean())
    else:
        summary["lipinski_ro5_pass_fraction"] = 0.0

    summary_df = pd.DataFrame([summary])
    return molecule_df, summary_df


def save_metrics(
    smiles_path: Path,
    *,
    model_id: str,
    isomeric_smiles: bool,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    molecules, summary = calculate_molecule_metrics(
        read_smiles(smiles_path),
        model_id=model_id,
        isomeric_smiles=isomeric_smiles,
    )
    molecules.to_csv(output_dir / f"{model_id}_molecule_metrics.csv", index=False)
    summary.to_csv(output_dir / f"{model_id}_summary_metrics.csv", index=False)
    return molecules, summary

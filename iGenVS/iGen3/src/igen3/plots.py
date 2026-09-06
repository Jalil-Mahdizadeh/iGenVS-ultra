"""Static benchmark plots for GitHub documentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_throughput(summary_df: pd.DataFrame, output_dir: Path, *, title_prefix: str = "Generation") -> Path:
    path = output_dir / "throughput_smiles_per_second.png"
    ordered = summary_df.sort_values("smiles_per_second", ascending=False)
    plt.figure(figsize=(9, 5))
    bars = plt.bar(ordered["model_id"], ordered["smiles_per_second"], color="#2d6a9f")
    plt.ylabel("SMILES per second")
    plt.xlabel("Model")
    plt.title(f"{title_prefix} Throughput")
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:,.0f}", ha="center", va="bottom", fontsize=8)
    _save_current(path)
    return path


def plot_metric_bars(summary_df: pd.DataFrame, output_dir: Path, *, title_prefix: str = "Molecular") -> Path:
    path = output_dir / "molecular_quality_summary.png"
    cols = [
        ("valid_fraction", "Valid"),
        ("unique_valid_fraction", "Unique"),
        ("lipinski_ro5_pass_fraction", "Ro5 Pass"),
        ("qed_mean", "Mean QED"),
    ]
    sa_col = "sa_score_mean" if "sa_score_mean" in summary_df.columns and summary_df["sa_score_mean"].notna().any() else None
    plot_df = summary_df[["model_id"] + [col for col, _ in cols]].copy()
    plot_df = plot_df.melt(id_vars="model_id", var_name="metric", value_name="value")
    label_map = dict(cols)
    plot_df["metric"] = plot_df["metric"].map(label_map)

    models = list(summary_df["model_id"])
    metrics = [label for _, label in cols]
    total_bars = len(metrics) + (1 if sa_col else 0)
    width = min(0.16, 0.8 / max(total_bars, 1))
    x = range(len(models))

    _, ax = plt.subplots(figsize=(11, 5.5))
    for metric_index, metric in enumerate(metrics):
        values = [
            float(plot_df[(plot_df["model_id"] == model) & (plot_df["metric"] == metric)]["value"].fillna(0).iloc[0])
            for model in models
        ]
        offsets = [pos + (metric_index - (total_bars - 1) / 2) * width for pos in x]
        ax.bar(offsets, values, width=width, label=metric)

    ax.set_ylabel("Fraction or QED score")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")
    ax.set_title(f"{title_prefix} Quality Metrics")
    ax.set_xticks(list(x), models, rotation=25, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    if sa_col:
        sa_axis = ax.twinx()
        sa_index = len(metrics)
        sa_offsets = [pos + (sa_index - (total_bars - 1) / 2) * width for pos in x]
        sa_values = [float(summary_df.loc[summary_df["model_id"] == model, sa_col].fillna(0).iloc[0]) for model in models]
        sa_bars = sa_axis.bar(
            sa_offsets,
            sa_values,
            width=width,
            color="#7a4e9f",
            hatch="//",
            label="Mean SA Score",
            alpha=0.85,
        )
        sa_axis.set_ylabel("Mean SA score (lower is easier)")
        sa_axis.set_ylim(0, max(10.0, max(sa_values) * 1.15 if sa_values else 10.0))
        handles.append(sa_bars)
        labels.append("Mean SA Score")

    ax.legend(handles, labels, ncols=min(len(labels), 5), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save_current(path)
    return path


def plot_descriptor_distributions(molecule_df: pd.DataFrame, output_dir: Path, *, title_prefix: str = "Descriptor") -> Path:
    path = output_dir / "descriptor_distributions.png"
    valid = molecule_df[molecule_df["valid"] == True].copy()  # noqa: E712
    descriptors = [
        ("qed", "QED"),
        ("logp", "LogP"),
        ("mol_weight", "Molecular weight"),
        ("tpsa", "TPSA"),
        ("sa_score", "SA score"),
    ]
    models = sorted(valid["model_id"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, title) in zip(axes.flatten(), descriptors):
        data = [valid[valid["model_id"] == model][col].dropna() for model in models]
        ax.boxplot(data, labels=models, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
    for ax in axes.flatten()[len(descriptors) :]:
        ax.axis("off")
    fig.suptitle(f"{title_prefix} Distributions for Valid Molecules")
    _save_current(path)
    return path


def plot_throughput_trace(trace_df: pd.DataFrame, output_dir: Path, *, title_prefix: str = "Generation") -> Path | None:
    if trace_df.empty:
        return None
    path = output_dir / "throughput_trace.png"
    x_col = "candidates_generated" if "candidates_generated" in trace_df.columns else "generated"
    y_col = "candidate_smiles_per_second" if "candidate_smiles_per_second" in trace_df.columns else "smiles_per_second"
    x_label = "Candidate SMILES sampled" if x_col == "candidates_generated" else "Generated SMILES"
    y_label = "Cumulative candidate SMILES per second" if y_col == "candidate_smiles_per_second" else "Cumulative SMILES per second"

    plt.figure(figsize=(9, 5))
    for model_id, group in trace_df.groupby("model_id"):
        plt.plot(group[x_col], group[y_col], marker="o", markersize=2, linewidth=1.5, label=model_id)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title_prefix} Throughput During Benchmark")
    plt.legend(fontsize=8)
    _save_current(path)
    return path


def write_plot_index(paths: list[Path], output_dir: Path) -> Path:
    path = output_dir / "README.md"
    lines = ["# Benchmark Figures", ""]
    for image_path in paths:
        rel = image_path.relative_to(output_dir).as_posix()
        title = image_path.stem.replace("_", " ").title()
        lines.extend([f"## {title}", "", f"![{title}]({rel})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

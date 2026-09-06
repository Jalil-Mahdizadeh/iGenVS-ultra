"""Benchmark iGen3 generators for speed and RDKit molecular metrics."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from .generation import (
    autotune_batch_size,
    generate_derivative_batch,
    generate_de_novo_batch,
    write_de_novo_file,
    write_derivative_file,
)
from .metrics import save_metrics
from .model import load_generator, maybe_compile_generator
from .plots import (
    plot_descriptor_distributions,
    plot_metric_bars,
    plot_throughput,
    plot_throughput_trace,
    write_plot_index,
)
from .registry import MODEL_SPECS, default_model_root, resolve_model


def _parse_batch_size(batch_size: str | int | None, generator, max_batch_size: int) -> int:
    if batch_size is None or str(batch_size).lower() == "auto":
        return autotune_batch_size(generator, max_batch_size=max_batch_size)
    value = int(batch_size)
    if value <= 0:
        raise ValueError("batch_size must be positive or 'auto'")
    return value


def _cuda_info() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(index),
        "device_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": total_bytes / 1024**3,
        "free_memory_gb": free_bytes / 1024**3,
        "torch_cuda": torch.version.cuda,
    }


def _prepare_output_dirs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles_dir = output_dir / "smiles"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    smiles_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    all_metrics_path = output_dir / "all_molecule_metrics.csv"
    if all_metrics_path.exists():
        all_metrics_path.unlink()
    return smiles_dir, metrics_dir, figures_dir, all_metrics_path


def _append_molecule_metrics(molecule_df: pd.DataFrame, output_path: Path, *, write_header: bool) -> None:
    molecule_df.to_csv(output_path, mode="w" if write_header else "a", header=write_header, index=False)


def _plot_frame(molecule_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["model_id", "valid", "qed", "logp", "mol_weight", "tpsa", "sa_score"]
    available = [col for col in cols if col in molecule_df.columns]
    return molecule_df[available].copy()


def _read_seed_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _repeat_to_count(values: list[str], count: int) -> list[str]:
    if not values:
        raise ValueError("seed file must contain at least one SMILES")
    repeats = math.ceil(count / len(values))
    return (values * repeats)[:count]


def run_benchmark(
    *,
    model_ids: list[str] | None = None,
    output_dir: Path = Path("benchmarks/latest"),
    count: int = 100_000,
    batch_size: str | int | None = "auto",
    max_batch_size: int = 32768,
    device: str = "auto",
    dtype: str = "auto",
    compile_enabled: bool = False,
    compile_mode: str = "reduce-overhead",
    top_k: int | None = None,
    greedy: bool = False,
    seed: int | None = 13,
    model_root: Path | None = None,
    max_candidates: int | None = None,
    max_candidate_multiplier: float = 50.0,
    stagnation_limit: int | None = None,
) -> pd.DataFrame:
    smiles_dir, metrics_dir, figures_dir, all_metrics_path = _prepare_output_dirs(output_dir)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    selected = model_ids or list(MODEL_SPECS)
    resolved_specs = [resolve_model(model_id) for model_id in selected]
    all_summaries: list[pd.DataFrame] = []
    plot_frames: list[pd.DataFrame] = []
    trace_records: list[dict[str, object]] = []
    generation_records: list[dict[str, object]] = []
    wrote_all_metrics = False

    for spec in resolved_specs:
        setup_start = perf_counter()
        generator = load_generator(spec, model_root=model_root or default_model_root(), device_name=device, dtype_name=dtype)
        chosen_batch_size = _parse_batch_size(batch_size, generator, max_batch_size)
        generator = maybe_compile_generator(generator, enabled=compile_enabled, compile_mode=compile_mode)

        warmup_bs = min(chosen_batch_size, count)
        _ = generate_de_novo_batch(
            generator,
            warmup_bs,
            temperature=spec.default_de_novo_temperature,
            do_sample=not greedy,
            top_k=spec.default_top_k if top_k is None else top_k,
        )
        if generator.device.type == "cuda":
            torch.cuda.synchronize()
        setup_seconds = perf_counter() - setup_start

        output_path = smiles_dir / f"{spec.model_id}_{count}.smi"
        stats = write_de_novo_file(
            generator,
            output_path=output_path,
            count=count,
            batch_size=chosen_batch_size,
            temperature=spec.default_de_novo_temperature,
            do_sample=not greedy,
            top_k=spec.default_top_k if top_k is None else top_k,
            max_candidates=max_candidates,
            max_candidate_multiplier=max_candidate_multiplier,
            stagnation_limit=stagnation_limit,
            progress=True,
        )
        if generator.device.type == "cuda":
            torch.cuda.synchronize()

        molecule_df, metric_summary = save_metrics(
            output_path,
            model_id=spec.model_id,
            isomeric_smiles=spec.is_isomeric,
            output_dir=metrics_dir,
        )

        metric_summary["family"] = spec.family
        metric_summary["stereochemistry"] = spec.stereochemistry
        metric_summary["generation_mode"] = "de-novo"
        metric_summary["temperature"] = spec.default_de_novo_temperature
        metric_summary["top_k"] = spec.default_top_k if top_k is None else top_k
        metric_summary["batch_size"] = stats.batch_size
        metric_summary["generation_seconds"] = stats.seconds
        metric_summary["smiles_per_second"] = stats.smiles_per_second
        metric_summary["candidate_count"] = stats.candidates_generated
        metric_summary["accepted_count"] = stats.generated
        metric_summary["acceptance_fraction"] = stats.generated / stats.candidates_generated if stats.candidates_generated else 0.0
        metric_summary["candidate_smiles_per_second"] = stats.candidate_smiles_per_second
        metric_summary["stop_reason"] = stats.stopped_reason
        metric_summary["setup_warmup_seconds"] = setup_seconds
        metric_summary["output_path"] = str(output_path)
        all_summaries.append(metric_summary)
        _append_molecule_metrics(molecule_df, all_metrics_path, write_header=not wrote_all_metrics)
        wrote_all_metrics = True
        plot_frames.append(_plot_frame(molecule_df))

        generation_records.append(
            {
                "model_id": spec.model_id,
                "generation_mode": "de-novo",
                "target_count": count,
                "accepted_count": stats.generated,
                "candidate_count": stats.candidates_generated,
                "batch_size": stats.batch_size,
                "generation_seconds": stats.seconds,
                "smiles_per_second": stats.smiles_per_second,
                "candidate_smiles_per_second": stats.candidate_smiles_per_second,
                "stop_reason": stats.stopped_reason,
                "setup_warmup_seconds": setup_seconds,
                "output_path": str(output_path),
            }
        )
        for row in stats.batch_trace:
            trace_records.append({"model_id": spec.model_id, **row})

        del molecule_df
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.concat(all_summaries, ignore_index=True)
    molecule_df = pd.concat(plot_frames, ignore_index=True)
    trace_df = pd.DataFrame(trace_records)
    generation_df = pd.DataFrame(generation_records)

    summary_df.to_csv(output_dir / "benchmark_summary.csv", index=False)
    generation_df.to_csv(output_dir / "generation_summary.csv", index=False)
    trace_df.to_csv(output_dir / "throughput_trace.csv", index=False)

    metadata = {
        "count_per_model": count,
        "model_ids": [spec.model_id for spec in resolved_specs],
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "device": device,
        "dtype": dtype,
        "compile_enabled": compile_enabled,
        "compile_mode": compile_mode,
        "greedy": greedy,
        "seed": seed,
        "max_candidates": max_candidates,
        "max_candidate_multiplier": max_candidate_multiplier,
        "stagnation_limit": stagnation_limit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "model_root": str(model_root or default_model_root()),
        "cuda": _cuda_info(),
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plots = [
        plot_throughput(summary_df, figures_dir, title_prefix="De Novo Generation"),
        plot_metric_bars(summary_df, figures_dir, title_prefix="De Novo Molecular"),
        plot_descriptor_distributions(molecule_df, figures_dir, title_prefix="De Novo Descriptor"),
    ]
    trace_plot = plot_throughput_trace(trace_df, figures_dir, title_prefix="De Novo Generation")
    if trace_plot is not None:
        plots.append(trace_plot)
    write_plot_index(plots, figures_dir)
    return summary_df


def run_derivative_benchmark(
    *,
    seed_file: Path,
    model_ids: list[str] | None = None,
    output_dir: Path = Path("benchmarks/derivative_latest"),
    count: int = 10_000,
    batch_size: str | int | None = "auto",
    max_batch_size: int = 32768,
    device: str = "auto",
    dtype: str = "auto",
    compile_enabled: bool = False,
    compile_mode: str = "reduce-overhead",
    top_k: int | None = None,
    greedy: bool = False,
    seed: int | None = 13,
    model_root: Path | None = None,
    exclude_seed_molecules: bool = True,
    max_candidates: int | None = None,
    max_candidate_multiplier: float = 50.0,
    stagnation_limit: int | None = None,
) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("count must be positive")

    seeds = _read_seed_file(seed_file)
    if not seeds:
        raise ValueError(f"Seed file is empty: {seed_file}")
    samples_per_seed = math.ceil(count / len(seeds))

    smiles_dir, metrics_dir, figures_dir, all_metrics_path = _prepare_output_dirs(output_dir)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    selected = model_ids or list(MODEL_SPECS)
    resolved_specs = [resolve_model(model_id) for model_id in selected]
    all_summaries: list[pd.DataFrame] = []
    plot_frames: list[pd.DataFrame] = []
    trace_records: list[dict[str, object]] = []
    generation_records: list[dict[str, object]] = []
    wrote_all_metrics = False

    for spec in resolved_specs:
        setup_start = perf_counter()
        generator = load_generator(spec, model_root=model_root or default_model_root(), device_name=device, dtype_name=dtype)
        chosen_batch_size = _parse_batch_size(batch_size, generator, max_batch_size)
        generator = maybe_compile_generator(generator, enabled=compile_enabled, compile_mode=compile_mode)

        warmup_count = min(chosen_batch_size, count)
        warmup_seeds = _repeat_to_count(seeds, warmup_count)
        _ = generate_derivative_batch(
            generator,
            warmup_seeds,
            temperature=spec.default_derivative_temperature,
            do_sample=not greedy,
            top_k=spec.default_top_k if top_k is None else top_k,
        )
        if generator.device.type == "cuda":
            torch.cuda.synchronize()
        setup_seconds = perf_counter() - setup_start

        output_path = smiles_dir / f"{spec.model_id}_derivative_{count}.smi"
        stats = write_derivative_file(
            generator,
            seeds=seeds,
            output_path=output_path,
            batch_size=chosen_batch_size,
            samples_per_seed=samples_per_seed,
            count=count,
            temperature=spec.default_derivative_temperature,
            do_sample=not greedy,
            top_k=spec.default_top_k if top_k is None else top_k,
            exclude_seed_molecules=exclude_seed_molecules,
            max_candidates=max_candidates,
            max_candidate_multiplier=max_candidate_multiplier,
            stagnation_limit=stagnation_limit,
            progress=True,
        )
        if generator.device.type == "cuda":
            torch.cuda.synchronize()

        molecule_df, metric_summary = save_metrics(
            output_path,
            model_id=spec.model_id,
            isomeric_smiles=spec.is_isomeric,
            output_dir=metrics_dir,
        )

        metric_summary["family"] = spec.family
        metric_summary["stereochemistry"] = spec.stereochemistry
        metric_summary["generation_mode"] = "derivative"
        metric_summary["temperature"] = spec.default_derivative_temperature
        metric_summary["top_k"] = spec.default_top_k if top_k is None else top_k
        metric_summary["batch_size"] = stats.batch_size
        metric_summary["generation_seconds"] = stats.seconds
        metric_summary["smiles_per_second"] = stats.smiles_per_second
        metric_summary["candidate_count"] = stats.candidates_generated
        metric_summary["accepted_count"] = stats.generated
        metric_summary["acceptance_fraction"] = stats.generated / stats.candidates_generated if stats.candidates_generated else 0.0
        metric_summary["candidate_smiles_per_second"] = stats.candidate_smiles_per_second
        metric_summary["stop_reason"] = stats.stopped_reason
        metric_summary["setup_warmup_seconds"] = setup_seconds
        metric_summary["input_seed_count"] = len(seeds)
        metric_summary["samples_per_seed"] = samples_per_seed
        metric_summary["exclude_seed_molecules"] = exclude_seed_molecules
        metric_summary["output_path"] = str(output_path)
        all_summaries.append(metric_summary)
        _append_molecule_metrics(molecule_df, all_metrics_path, write_header=not wrote_all_metrics)
        wrote_all_metrics = True
        plot_frames.append(_plot_frame(molecule_df))

        generation_records.append(
            {
                "model_id": spec.model_id,
                "generation_mode": "derivative",
                "target_count": count,
                "accepted_count": stats.generated,
                "candidate_count": stats.candidates_generated,
                "input_seed_count": len(seeds),
                "samples_per_seed": samples_per_seed,
                "exclude_seed_molecules": exclude_seed_molecules,
                "batch_size": stats.batch_size,
                "generation_seconds": stats.seconds,
                "smiles_per_second": stats.smiles_per_second,
                "candidate_smiles_per_second": stats.candidate_smiles_per_second,
                "stop_reason": stats.stopped_reason,
                "setup_warmup_seconds": setup_seconds,
                "output_path": str(output_path),
            }
        )
        for row in stats.batch_trace:
            trace_records.append({"model_id": spec.model_id, **row})

        del molecule_df
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.concat(all_summaries, ignore_index=True)
    molecule_df = pd.concat(plot_frames, ignore_index=True)
    trace_df = pd.DataFrame(trace_records)
    generation_df = pd.DataFrame(generation_records)

    summary_df.to_csv(output_dir / "benchmark_summary.csv", index=False)
    generation_df.to_csv(output_dir / "generation_summary.csv", index=False)
    trace_df.to_csv(output_dir / "throughput_trace.csv", index=False)

    metadata = {
        "generation_mode": "derivative",
        "target_count_per_model": count,
        "seed_file": str(seed_file),
        "input_seed_count": len(seeds),
        "samples_per_seed": samples_per_seed,
        "exclude_seed_molecules": exclude_seed_molecules,
        "model_ids": [spec.model_id for spec in resolved_specs],
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "device": device,
        "dtype": dtype,
        "compile_enabled": compile_enabled,
        "compile_mode": compile_mode,
        "greedy": greedy,
        "seed": seed,
        "max_candidates": max_candidates,
        "max_candidate_multiplier": max_candidate_multiplier,
        "stagnation_limit": stagnation_limit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "model_root": str(model_root or default_model_root()),
        "cuda": _cuda_info(),
    }
    (output_dir / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plots = [
        plot_throughput(summary_df, figures_dir, title_prefix="Derivative Generation"),
        plot_metric_bars(summary_df, figures_dir, title_prefix="Derivative Molecular"),
        plot_descriptor_distributions(molecule_df, figures_dir, title_prefix="Derivative Descriptor"),
    ]
    trace_plot = plot_throughput_trace(trace_df, figures_dir, title_prefix="Derivative Generation")
    if trace_plot is not None:
        plots.append(trace_plot)
    write_plot_index(plots, figures_dir)
    return summary_df

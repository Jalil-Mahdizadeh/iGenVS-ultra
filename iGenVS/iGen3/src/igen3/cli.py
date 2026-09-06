"""Command line interface for iGen3 generation and benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .benchmark import run_benchmark, run_derivative_benchmark
from .generation import (
    autotune_batch_size,
    write_derivative_file,
    write_de_novo_file,
)
from .metrics import save_metrics
from .model import load_generator, maybe_compile_generator
from .randomize import (
    enumerate_randomized_smiles,
    write_randomization_metadata,
    write_randomized_smiles,
)
from .registry import MODEL_SPECS, default_model_root, resolve_model


def _compile_flag(value: str) -> bool:
    if value in {"on", "true", "yes", "1"}:
        return True
    if value in {"off", "false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("--compile must be on or off")


def _batch_size(value: str, generator, max_batch_size: int) -> int:
    if value.lower() == "auto":
        return autotune_batch_size(generator, max_batch_size=max_batch_size)
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be positive or auto")
    return parsed


def _read_seed_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def list_models(_: argparse.Namespace) -> None:
    rows = [
        {
            "model_id": spec.model_id,
            "family": spec.family,
            "stereochemistry": spec.stereochemistry,
            "seq_len": spec.seq_len,
            "de_novo_temperature": spec.default_de_novo_temperature,
            "derivative_temperature": spec.default_derivative_temperature,
            "description": spec.description,
        }
        for spec in MODEL_SPECS.values()
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def generate(args: argparse.Namespace) -> None:
    spec = resolve_model(args.model)
    generator = load_generator(
        spec,
        model_root=args.model_dir,
        device_name=args.device,
        dtype_name=args.dtype,
        compile_model=False,
    )
    batch_size = _batch_size(args.batch_size, generator, args.max_batch_size)
    generator = maybe_compile_generator(generator, enabled=args.compile, compile_mode=args.compile_mode)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    temperature = args.temperature
    if temperature is None:
        temperature = spec.default_temperature(args.mode)
    top_k = spec.default_top_k if args.top_k is None else args.top_k

    if args.mode == "de-novo":
        stats = write_de_novo_file(
            generator,
            output_path=args.output,
            count=args.count,
            batch_size=batch_size,
            temperature=temperature,
            do_sample=not args.greedy,
            top_k=top_k,
            max_candidates=args.max_candidates,
            max_candidate_multiplier=args.max_candidate_multiplier,
            stagnation_limit=args.stagnation_limit,
            progress=not args.no_progress,
        )
    else:
        seeds: list[str] = []
        if args.seed_smiles:
            seeds.extend(args.seed_smiles)
        if args.seed_file:
            seeds.extend(_read_seed_file(args.seed_file))
        if not seeds:
            raise SystemExit("derivative mode requires --seed-smiles or --seed-file")
        stats = write_derivative_file(
            generator,
            seeds=seeds,
            output_path=args.output,
            batch_size=batch_size,
            samples_per_seed=args.samples_per_seed,
            count=args.count,
            temperature=temperature,
            do_sample=not args.greedy,
            top_k=top_k,
            exclude_seed_molecules=not args.include_seed_molecules,
            max_candidates=args.max_candidates,
            max_candidate_multiplier=args.max_candidate_multiplier,
            stagnation_limit=args.stagnation_limit,
            progress=not args.no_progress,
        )

    print(
        f"Wrote {stats.generated:,} valid unique SMILES with {spec.model_id} after sampling "
        f"{stats.candidates_generated:,} candidates in {stats.seconds:.2f}s "
        f"({stats.smiles_per_second:.1f} output SMILES/s; stop: {stats.stopped_reason}). "
        f"Output: {stats.output_path}"
    )

    if args.metrics:
        metrics_dir = args.metrics_dir or (args.output.parent / "metrics")
        _, summary = save_metrics(
            args.output,
            model_id=spec.model_id,
            isomeric_smiles=spec.is_isomeric,
            output_dir=metrics_dir,
        )
        print(summary.to_string(index=False))


def benchmark(args: argparse.Namespace) -> None:
    top_k = None if args.top_k is None else args.top_k
    summary = run_benchmark(
        model_ids=args.models,
        output_dir=args.output_dir,
        count=args.count,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        dtype=args.dtype,
        compile_enabled=args.compile,
        compile_mode=args.compile_mode,
        top_k=top_k,
        greedy=args.greedy,
        seed=args.seed,
        model_root=args.model_dir,
        max_candidates=args.max_candidates,
        max_candidate_multiplier=args.max_candidate_multiplier,
        stagnation_limit=args.stagnation_limit,
    )
    print(
        summary[
            ["model_id", "smiles_per_second", "valid_fraction", "unique_valid_fraction", "qed_mean", "sa_score_mean"]
        ].to_string(index=False)
    )
    print(f"Benchmark artifacts written to {args.output_dir}")


def benchmark_derivative(args: argparse.Namespace) -> None:
    top_k = None if args.top_k is None else args.top_k
    summary = run_derivative_benchmark(
        seed_file=args.seed_file,
        model_ids=args.models,
        output_dir=args.output_dir,
        count=args.count,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        dtype=args.dtype,
        compile_enabled=args.compile,
        compile_mode=args.compile_mode,
        top_k=top_k,
        greedy=args.greedy,
        seed=args.seed,
        model_root=args.model_dir,
        exclude_seed_molecules=not args.include_seed_molecules,
        max_candidates=args.max_candidates,
        max_candidate_multiplier=args.max_candidate_multiplier,
        stagnation_limit=args.stagnation_limit,
    )
    print(
        summary[
            ["model_id", "smiles_per_second", "valid_fraction", "unique_valid_fraction", "qed_mean", "sa_score_mean"]
        ].to_string(index=False)
    )
    print(f"Derivative benchmark artifacts written to {args.output_dir}")


def randomize_smiles(args: argparse.Namespace) -> None:
    result = enumerate_randomized_smiles(
        args.smiles,
        max_variants=args.max_variants,
        max_attempts=args.max_attempts,
        stagnation_limit=args.stagnation_limit,
        include_canonical=args.include_canonical,
        isomeric_smiles=args.isomeric_smiles,
        seed=args.seed,
    )
    write_randomized_smiles(result, args.output)
    if args.metadata_output:
        write_randomization_metadata(result, args.metadata_output)
    print(
        f"Wrote {result.variant_count:,} seed SMILES to {args.output} "
        f"after {result.attempts:,} randomized attempts (stop: {result.stop_reason})."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="igen3", description="Run iGen3 de novo and derivative SMILES generation.")
    parser.add_argument("--model-dir", type=Path, default=default_model_root(), help="Directory containing packaged model artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-models", help="List available model variants.")
    list_parser.set_defaults(func=list_models)

    gen = subparsers.add_parser("generate", help="Generate SMILES with one model.")
    gen.add_argument("--model", required=True, choices=sorted(MODEL_SPECS), help="Model variant to run.")
    gen.add_argument("--mode", choices=["de-novo", "derivative"], default="de-novo", help="Generation mode.")
    gen.add_argument("--output", type=Path, required=True, help="Output .smi/.txt file.")
    gen.add_argument("--count", type=int, default=1000, help="Valid unique output molecules requested.")
    gen.add_argument("--batch-size", default="auto", help="Batch size integer or 'auto'.")
    gen.add_argument("--max-batch-size", type=int, default=32768, help="Upper bound used by batch-size autotuning.")
    gen.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Defaults depend on model and mode.")
    gen.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff. Use 0 to disable.")
    gen.add_argument("--greedy", action="store_true", help="Use argmax decoding instead of sampling.")
    gen.add_argument("--seed", type=int, default=None, help="Torch random seed.")
    gen.add_argument("--seed-smiles", action="append", help="Seed SMILES for derivative mode. Can be repeated.")
    gen.add_argument("--seed-file", type=Path, help="Text file containing one derivative seed SMILES per line.")
    gen.add_argument("--samples-per-seed", type=int, default=1, help="Derivative-mode samples per seed in each sampling cycle.")
    gen.add_argument(
        "--include-seed-molecules",
        action="store_true",
        help="Allow derivative output to include molecules identical to the input seeds.",
    )
    gen.add_argument("--max-candidates", type=int, default=None, help="Maximum sampled candidates used to fill valid unique output.")
    gen.add_argument(
        "--max-candidate-multiplier",
        type=float,
        default=50.0,
        help="Candidate cap multiplier when --max-candidates is not provided.",
    )
    gen.add_argument(
        "--stagnation-limit",
        type=int,
        default=None,
        help="Stop after this many sampled candidates produce no new valid unique output. Use 0 to disable.",
    )
    gen.add_argument("--device", default="auto", help="Torch device, for example auto, cuda, cpu.")
    gen.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Inference dtype.")
    gen.add_argument("--compile", type=_compile_flag, default=False, help="Enable torch.compile: on/off.")
    gen.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode.")
    gen.add_argument("--metrics", action="store_true", help="Compute RDKit metrics after generation.")
    gen.add_argument("--metrics-dir", type=Path, help="Directory for generated metrics CSV files.")
    gen.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    gen.set_defaults(func=generate)

    bench = subparsers.add_parser("benchmark", help="Benchmark all or selected de novo models.")
    bench.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), help="Model variants. Defaults to all.")
    bench.add_argument("--output-dir", type=Path, default=Path("benchmarks/latest"), help="Directory for benchmark outputs.")
    bench.add_argument("--count", type=int, default=100_000, help="Valid unique output molecules requested per model.")
    bench.add_argument("--batch-size", default="auto", help="Batch size integer or 'auto'.")
    bench.add_argument("--max-batch-size", type=int, default=32768, help="Upper bound used by batch-size autotuning.")
    bench.add_argument("--device", default="auto", help="Torch device, for example auto, cuda, cpu.")
    bench.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="Inference dtype.")
    bench.add_argument("--compile", type=_compile_flag, default=False, help="Enable torch.compile: on/off.")
    bench.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode.")
    bench.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff. Defaults to the model setting.")
    bench.add_argument("--greedy", action="store_true", help="Use argmax decoding instead of sampling.")
    bench.add_argument("--seed", type=int, default=13, help="Torch random seed.")
    bench.add_argument("--max-candidates", type=int, default=None, help="Maximum sampled candidates per model.")
    bench.add_argument(
        "--max-candidate-multiplier",
        type=float,
        default=50.0,
        help="Candidate cap multiplier when --max-candidates is not provided.",
    )
    bench.add_argument(
        "--stagnation-limit",
        type=int,
        default=None,
        help="Stop after this many sampled candidates produce no new valid unique output. Use 0 to disable.",
    )
    bench.set_defaults(func=benchmark)

    derivative_bench = subparsers.add_parser(
        "benchmark-derivative",
        help="Benchmark derivative generation using a precomputed randomized seed file.",
    )
    derivative_bench.add_argument("--seed-file", type=Path, required=True, help="Seed SMILES file from randomize-smiles.")
    derivative_bench.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), help="Model variants. Defaults to all.")
    derivative_bench.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/derivative_latest"),
        help="Directory for derivative benchmark outputs.",
    )
    derivative_bench.add_argument("--count", type=int, default=10_000, help="Valid unique derivatives requested per model.")
    derivative_bench.add_argument("--batch-size", default="auto", help="Batch size integer or 'auto'.")
    derivative_bench.add_argument("--max-batch-size", type=int, default=32768, help="Upper bound used by batch-size autotuning.")
    derivative_bench.add_argument("--device", default="auto", help="Torch device, for example auto, cuda, cpu.")
    derivative_bench.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Inference dtype.",
    )
    derivative_bench.add_argument("--compile", type=_compile_flag, default=False, help="Enable torch.compile: on/off.")
    derivative_bench.add_argument("--compile-mode", default="reduce-overhead", help="torch.compile mode.")
    derivative_bench.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff. Defaults to the model setting.")
    derivative_bench.add_argument("--greedy", action="store_true", help="Use argmax decoding instead of sampling.")
    derivative_bench.add_argument("--seed", type=int, default=13, help="Torch random seed.")
    derivative_bench.add_argument(
        "--include-seed-molecules",
        action="store_true",
        help="Allow derivative output to include molecules identical to the input seeds.",
    )
    derivative_bench.add_argument("--max-candidates", type=int, default=None, help="Maximum sampled candidates per model.")
    derivative_bench.add_argument(
        "--max-candidate-multiplier",
        type=float,
        default=50.0,
        help="Candidate cap multiplier when --max-candidates is not provided.",
    )
    derivative_bench.add_argument(
        "--stagnation-limit",
        type=int,
        default=None,
        help="Stop after this many sampled candidates produce no new valid unique output. Use 0 to disable.",
    )
    derivative_bench.set_defaults(func=benchmark_derivative)

    rand = subparsers.add_parser(
        "randomize-smiles",
        help="Generate canonical and randomized seed SMILES for derivative generation.",
    )
    rand.add_argument("--smiles", required=True, help="Reference molecule SMILES.")
    rand.add_argument("--output", type=Path, required=True, help="Output seed file, one SMILES per line.")
    rand.add_argument("--max-variants", type=int, default=10_000, help="Maximum unique variants to write.")
    rand.add_argument("--max-attempts", type=int, default=None, help="Maximum randomized RDKit attempts.")
    rand.add_argument(
        "--stagnation-limit",
        type=int,
        default=5_000,
        help="Stop after this many consecutive randomized attempts produce no new SMILES.",
    )
    rand.add_argument("--seed", type=int, default=13, help="RDKit/Python random seed.")
    iso_group = rand.add_mutually_exclusive_group()
    iso_group.add_argument("--isomeric", dest="isomeric_smiles", action="store_true", default=True)
    iso_group.add_argument("--non-isomeric", dest="isomeric_smiles", action="store_false")
    canonical_group = rand.add_mutually_exclusive_group()
    canonical_group.add_argument("--include-canonical", dest="include_canonical", action="store_true", default=True)
    canonical_group.add_argument("--no-canonical", dest="include_canonical", action="store_false")
    rand.add_argument("--metadata-output", type=Path, help="Optional JSON metadata output path.")
    rand.set_defaults(func=randomize_smiles)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

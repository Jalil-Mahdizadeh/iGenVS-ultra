"""Command-line interface for the user-facing iGenVS-ultra workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .workflow import PipelineError, doctor, fit, plan_run, regular_dock, screen, status


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def auto_positive(value: str) -> Any:
    if value.lower() == "auto":
        return "auto"
    return positive_int(value)


def add_runtime_arguments(parser: argparse.ArgumentParser, *, include_gmolai: bool = True) -> None:
    group = parser.add_argument_group("execution")
    group.add_argument(
        "--execution",
        choices=("auto", "docker", "apptainer", "native"),
        default="auto",
        help=(
            "Use released SIFs when available, then locally built Docker images, "
            "otherwise native commands (default: auto)."
        ),
    )
    group.add_argument("--assets-dir", type=Path, help="iGenVS-ultra project/release-assets root.")
    group.add_argument("--igenvs-image", type=Path, help="Override the released iGenVS SIF.")
    group.add_argument(
        "--igenvs-docker-image",
        help="Override the iGenVS Docker image (default: igenvs-ultra/igenvs:latest).",
    )
    if include_gmolai:
        group.add_argument("--gmolai-image", type=Path, help="Override the released gMolAI SIF.")
        group.add_argument(
            "--gmolai-docker-image",
            help="Override the gMolAI Docker image (default: igenvs-ultra/gmolai:latest).",
        )
    group.add_argument(
        "--gpu-ids",
        help="Comma-separated visible GPU identifiers; defaults to CUDA_VISIBLE_DEVICES/hardware discovery.",
    )


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("target input (choose one mode)")
    source = group.add_mutually_exclusive_group(required=True)
    source.add_argument("--complex", type=Path, help="Protein-ligand complex in one PDB file.")
    source.add_argument("--receptor", type=Path, help="Curated/protonated receptor PDB file.")
    source.add_argument(
        "--prepared-target",
        "--target",
        dest="prepared_target",
        type=Path,
        help="Existing iGenVS prepare-target bundle.",
    )
    group.add_argument(
        "--ligand-id",
        help="Complex ligand as RESNAME or CHAIN:RESNAME:RESSEQ; inferred only when unambiguous.",
    )
    group.add_argument(
        "--reference-ligand",
        type=Path,
        help="One aligned 3D bound ligand in SDF format; required with --receptor.",
    )
    group.add_argument("--padding", type=nonnegative_float, default=5.0, help="Pocket padding in A (default: 5).")
    group.add_argument("--target-name", help="Filename-safe target label (default: input stem).")


def add_docking_arguments(parser: argparse.ArgumentParser, *, regular: bool = False) -> None:
    group = parser.add_argument_group("iGenVS docking")
    group.add_argument("--engine", choices=("unidock", "autodock-gpu"), default="unidock")
    group.add_argument(
        "--search-mode",
        choices=("fast", "balance", "detail"),
        default="balance" if regular else "fast",
    )
    group.add_argument("--scoring", choices=("auto", "vina", "vinardo", "ad4"), default="auto")
    group.add_argument("--num-modes", type=positive_int, default=1)
    group.add_argument("--energy-range", type=float, default=3.0)
    group.add_argument("--refine-step", type=positive_int, default=3)
    group.add_argument("--no-refine", action="store_true")
    group.add_argument("--unidock-verbosity", type=int, choices=(0, 1, 2), default=0)
    group.add_argument("--seed", type=int, default=181129, help="Docking seed (default: 181129).")
    group.add_argument("--max-gpu-memory", type=int, default=0, metavar="MIB")
    group.add_argument("--adgpu-runs", type=positive_int)
    group.add_argument("--adgpu-evaluations", type=positive_int)
    group.add_argument("--adgpu-no-heuristics", action="store_true")
    group.add_argument("--adgpu-no-autostop", action="store_true")
    group.add_argument("--adgpu-local-search", choices=("sw", "sd", "fire", "ad", "adam"), default="ad")
    group.add_argument("--adgpu-cpu-threads", type=positive_int, default=4)
    group.add_argument("--adgpu-workers", type=auto_positive, default="auto")
    group.add_argument("--adgpu-executable", default="autodock_gpu")
    group.add_argument("--batch-size", type=auto_positive, default="auto", help="Outer docking batch (default: auto).")
    group.add_argument("--batch-profile", type=Path)
    group.add_argument("--prep-workers", type=auto_positive, default="auto")
    group.add_argument("--prep-mode", choices=("standard", "fast"), default="standard")
    group.add_argument(
        "--embed-max-attempts",
        type=auto_positive,
        default="auto",
        help="Bound deterministic ETKDG work per molecule (default: auto).",
    )
    group.add_argument(
        "--embed-timeout",
        type=auto_positive,
        default="auto",
        metavar="SECONDS",
        help="Native RDKit time guard for each ETKDG phase (default: auto).",
    )
    group.add_argument("--validation-workers", type=auto_positive, default="auto")
    group.add_argument("--fragment-policy", choices=("reject", "largest"), default="reject")
    group.add_argument("--no-deduplicate", action="store_true")
    if regular:
        group.add_argument(
            "--docking-gpus",
            type=auto_positive,
            default="auto",
            help="Use all visible GPUs automatically (default: auto).",
        )
        group.add_argument("--device-id", type=int, default=0, help="Visible CUDA device index (default: 0).")
        group.add_argument("--num-shards", type=positive_int, default=1)
        group.add_argument("--shard-index", type=int, default=0)
    else:
        group.add_argument("--docking-gpus", type=auto_positive, default="auto", help="Locally shard over visible GPUs.")
        group.add_argument(
            "--docking-logical-shards",
            type=positive_int,
            default=4,
            help=(
                "Fixed hardware-independent docking shards (default: 4). Keep this fixed when "
                "comparing or resuming across GPU counts."
            ),
        )
    group.add_argument("--scratch-dir", type=Path)
    group.add_argument("--keep-work", action="store_true")
    group.add_argument(
        "--pose-output",
        choices=("none", "merged", "individual"),
        default="merged" if regular else "none",
    )
    group.add_argument("--individual-poses", dest="pose_output", action="store_const", const="individual")


def add_regular_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the source/generation options of the original iGenVS screen CLI."""
    group = parser.add_argument_group("library source")
    source = group.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="External CSV/TSV/SMI library.")
    source.add_argument("--generate-count", type=positive_int, help="Generate this many valid unique SMILES with iGen3.")
    group.add_argument("--input-format", choices=("auto", "csv", "smi"), default="auto")
    group.add_argument("--smiles-column", default="smiles")
    group.add_argument("--id-column")
    group.add_argument("--delimiter", default="auto", help="CSV delimiter, auto, or literal \\t.")

    generation = parser.add_argument_group("iGen3 generation (regular docking)")
    generation.add_argument(
        "--model",
        choices=("base-isomeric", "base-nonisomeric", "rl-isomeric", "rl-nonisomeric"),
        default="rl-nonisomeric",
    )
    generation.add_argument("--generation-mode", choices=("de-novo", "derivative"), default="de-novo")
    generation.add_argument("--seed-file", type=Path)
    generation.add_argument("--samples-per-seed", type=positive_int, default=1)
    generation.add_argument("--generator-batch-size", type=auto_positive, default="auto")
    generation.add_argument("--generator-max-batch-size", type=positive_int, default=32_768)
    generation.add_argument("--model-dir", type=Path)
    generation.add_argument("--temperature", type=float)
    generation.add_argument("--top-k", type=int)
    generation.add_argument("--compile-generator", action="store_true")
    generation.add_argument("--generator-seed", type=int, default=13)


def add_al_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--al-rounds",
        type=int,
        choices=range(0, 6),
        default=0,
        metavar="0..5",
        help="0 disables AL; 1-5 runs that many released AL rounds (default: 0).",
    )


def add_screening_source_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("screening source")
    source = group.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="External CSV/TSV/SMI library.")
    source.add_argument(
        "--generate-count",
        type=positive_int,
        help="Number of valid unique molecules to successfully encode and score.",
    )
    group.add_argument("--input-format", choices=("auto", "csv", "smi"), default="auto")
    group.add_argument("--smiles-column", default="smiles")
    group.add_argument("--id-column")
    group.add_argument("--delimiter", default="auto", help="CSV delimiter, auto, or literal \\t.")

    generation = parser.add_argument_group("iGen3 generation")
    generation.add_argument(
        "--model",
        choices=("base-isomeric", "base-nonisomeric", "rl-isomeric", "rl-nonisomeric"),
        default="base-isomeric",
    )
    generation.add_argument("--generation-mode", choices=("de-novo", "derivative"), default="de-novo")
    generation.add_argument("--seed-file", type=Path)
    generation.add_argument("--seed-smiles", action="append", default=[])
    generation.add_argument("--samples-per-seed", type=positive_int, default=1)
    generation.add_argument("--generator-batch-size", type=auto_positive, default="auto")
    generation.add_argument(
        "--generator-max-batch-size",
        type=positive_int,
        default=131_072,
        help="Safety ceiling for automatic tuning; actual size is hardware-measured.",
    )
    generation.add_argument("--model-dir", type=Path)
    generation.add_argument("--temperature", type=float)
    generation.add_argument("--top-k", type=int)
    generation.add_argument("--greedy", action="store_true")
    generation.add_argument("--compile-generator", action="store_true")
    generation.add_argument("--compile-mode", default="reduce-overhead")
    generation.add_argument("--generator-seed", type=int, default=13)
    generation.add_argument("--include-seed-molecules", action="store_true")
    generation.add_argument("--max-candidates", type=positive_int)
    generation.add_argument("--max-candidate-multiplier", type=float)
    generation.add_argument("--stagnation-limit", type=int)
    generation.add_argument("--generator-device", default="auto")
    generation.add_argument("--generator-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    generation.add_argument("--generator-metrics", action="store_true")


def add_screening_execution_arguments(parser: argparse.ArgumentParser, *, shared_validation: bool) -> None:
    group = parser.add_argument_group("streaming and scoring")
    group.add_argument("--screen-name", default="final", help="Name for this reusable screen (default: final).")
    group.add_argument(
        "--stream-batch-size",
        type=auto_positive,
        default="auto",
        help="Molecules per validate/encode/score chunk; auto uses hardware (default: auto).",
    )
    group.add_argument("--max-stream-batches", type=positive_int, help="Safety cap for replenished iGen3 streams.")
    group.add_argument(
        "--exclude-reference-libraries",
        action="store_true",
        help="Also exclude exact identities found in the fixed UDRL/AL/test libraries.",
    )
    group.add_argument("--save-policy", choices=("all", "threshold"), default="all")
    group.add_argument(
        "--score-threshold",
        type=float,
        help="Minimum final ensemble score to save (0-1); implies --save-policy threshold.",
    )
    group.add_argument(
        "--save-embeddings",
        "--keep-embeddings",
        dest="keep_embeddings",
        action="store_true",
        help="Save each gMolAI embedding bundle (default: score in memory without persisting it).",
    )
    group.add_argument("--encoder-backend", choices=("optimized", "reference", "verify"), default="optimized")
    group.add_argument(
        "--encoder-batch-size",
        type=auto_positive,
        default="auto",
        help=(
            "gMolAI graph batch; auto calibrates qualified batch sizes and caches "
            "the fastest hardware-specific choice (default: auto)."
        ),
    )
    group.add_argument("--encoder-node-budget", type=positive_int, default=16_384)
    group.add_argument("--encoder-workers", default="auto")
    group.add_argument("--encoder-verify-rows", type=positive_int, default=1024)
    group.add_argument("--encoder-threads", type=positive_int, default=4)
    group.add_argument("--encoder-device", default="auto")
    group.add_argument(
        "--screen-gpus",
        type=auto_positive,
        default="auto",
        help=(
            "Use this many visible GPUs for iGen3 generation and prepared-batch "
            "scoring; auto uses all visible GPUs (default: auto)."
        ),
    )
    group.add_argument(
        "--generation-logical-shards",
        type=auto_positive,
        default="auto",
        help=(
            "Deterministic iGen3 shards per generated stream batch; auto matches "
            "--screen-gpus. Fix this across hardware-count benchmarks so the generated "
            "molecules and process granularity stay constant."
        ),
    )
    if not shared_validation:
        group.add_argument("--validation-workers", type=auto_positive, default="auto")
        group.add_argument("--fragment-policy", choices=("reject", "largest"), default="reject")


def add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the execution plan only.")


def discover_fast_job(start: Path) -> Path:
    override = os.environ.get("IGENVS_ULTRA_JOB")
    candidates = [Path(override).expanduser().resolve()] if override else []
    resolved = start.expanduser().resolve()
    candidates.extend([resolved, *resolved.parents])
    for candidate in candidates:
        if (candidate / "fit-config.json").is_file() and (
            candidate / "models/final.json"
        ).is_file():
            return candidate
    raise PipelineError(
        "screen-fast must be run from inside a completed target job "
        "(or with IGENVS_ULTRA_JOB set)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="igenvs-ultra",
        description=(
            "Two explicit workflows: regular iGenVS docking, or iGenVS-ultra "
            "target-head fitting/AL and streamed scoring."
        ),
    )
    parser.add_argument("--version", action="version", version=f"iGenVS-ultra pipeline {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("doctor", help="Check released assets, containers, tools, and hardware.")
    add_runtime_arguments(check)
    check.add_argument("--no-gpu", action="store_true", help="Do not require a visible GPU for the iGenVS check.")
    check.add_argument(
        "--require-fit-assets",
        action="store_true",
        help="Also require the optional multi-GB UDRL/active-learning fit asset bundle.",
    )

    dock_parser = subparsers.add_parser(
        "dock",
        help="Run the separate, regular iGenVS generate/ingest-to-docking workflow.",
    )
    add_runtime_arguments(dock_parser, include_gmolai=False)
    add_target_arguments(dock_parser)
    add_regular_source_arguments(dock_parser)
    add_docking_arguments(dock_parser, regular=True)
    dock_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or resumable regular-docking job directory.",
    )
    add_dry_run(dock_parser)

    fit_parser = subparsers.add_parser("fit", help="Prepare target, dock UDRL, fit TH0, and optionally run AL.")
    add_runtime_arguments(fit_parser)
    add_target_arguments(fit_parser)
    add_docking_arguments(fit_parser)
    add_al_arguments(fit_parser)
    fit_parser.add_argument("--output-dir", type=Path, required=True, help="New or resumable target job directory.")
    add_dry_run(fit_parser)

    screen_parser = subparsers.add_parser("screen", help="Stream-score a library with a completed target job.")
    add_runtime_arguments(screen_parser)
    add_screening_source_arguments(screen_parser)
    add_screening_execution_arguments(screen_parser, shared_validation=False)
    screen_parser.add_argument("--job-dir", type=Path, required=True, help="Completed fit job directory.")
    add_dry_run(screen_parser)

    fast_parser = subparsers.add_parser(
        "screen-fast",
        help="Maximum-speed generated screen; from a fitted job, only provide molecule count.",
    )
    fast_parser.add_argument("molecules", type=positive_int)

    run_parser = subparsers.add_parser("run", help="One-command fit plus final library screen.")
    add_runtime_arguments(run_parser)
    add_target_arguments(run_parser)
    add_docking_arguments(run_parser)
    add_al_arguments(run_parser)
    add_screening_source_arguments(run_parser)
    add_screening_execution_arguments(run_parser, shared_validation=True)
    run_parser.add_argument("--output-dir", type=Path, required=True, help="New or resumable target job directory.")
    add_dry_run(run_parser)

    status_parser = subparsers.add_parser("status", help="Show completed and incomplete stages for a job.")
    status_parser.add_argument("--job-dir", type=Path, required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command in {"dock", "fit", "run"}:
        if args.receptor and not args.reference_ligand:
            parser.error("--receptor requires --reference-ligand")
        if not args.receptor and args.reference_ligand:
            parser.error("--reference-ligand is valid only with --receptor")
        if not args.complex and args.ligand_id:
            parser.error("--ligand-id is valid only with --complex")
    if args.command in {"screen", "run"}:
        if args.score_threshold is not None:
            if not 0.0 <= args.score_threshold <= 1.0:
                parser.error("--score-threshold must be between 0 and 1")
            args.save_policy = "threshold"
        if args.save_policy == "threshold" and args.score_threshold is None:
            parser.error("--save-policy threshold requires --score-threshold")
    if args.command in {"dock", "screen", "run"}:
        seed_smiles = getattr(args, "seed_smiles", [])
        if args.generate_count is not None and args.generation_mode == "derivative" and not (args.seed_file or seed_smiles):
            parser.error("derivative generation requires --seed-file or --seed-smiles")
        if args.generate_count is not None and args.generation_mode == "de-novo" and (args.seed_file or seed_smiles):
            parser.error("--seed-file/--seed-smiles require --generation-mode derivative")
        if args.input is not None and (args.seed_file or seed_smiles):
            parser.error("generation seed options cannot be combined with --input")
        if args.top_k is not None and args.top_k < 0:
            parser.error("--top-k must be non-negative")
        if getattr(args, "max_candidate_multiplier", None) is not None and args.max_candidate_multiplier < 1:
            parser.error("--max-candidate-multiplier must be at least 1")
        if getattr(args, "stagnation_limit", None) is not None and args.stagnation_limit < 0:
            parser.error("--stagnation-limit must be non-negative")
    if args.command == "dock":
        if args.device_id < 0:
            parser.error("--device-id must be non-negative")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            parser.error("--shard-index must be in [0, --num-shards)")


def main(argv: Any = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "screen-fast":
        job = discover_fast_job(Path.cwd())
        args = parser.parse_args(
            [
                "screen",
                "--job-dir",
                str(job),
                "--generate-count",
                str(args.molecules),
                "--screen-name",
                f"fast-{args.molecules}",
            ]
        )
    validate_args(parser, args)
    try:
        if args.command == "doctor":
            return 0 if doctor(args)["ok"] else 1
        if args.command == "status":
            status(args)
            return 0
        if args.command == "dock":
            regular_dock(args)
            return 0
        if args.command == "fit":
            fit(args)
            return 0
        if args.command == "screen":
            screen(args)
            return 0
        if args.command == "run":
            if args.dry_run:
                plan_run(args)
                return 0
            fit(args)
            args.job_dir = args.output_dir
            screen(args)
            return 0
        raise PipelineError(f"unknown command: {args.command}")
    except PipelineError as exc:
        print(f"igenvs-ultra: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("igenvs-ultra: interrupted; rerun the same command to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

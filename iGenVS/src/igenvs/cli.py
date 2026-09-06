"""Command-line interface for iGenVS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .doctor import format_doctor, run_doctor
from .errors import IGenVSError, InputError
from .generation import GenerationConfig
from .ingress import iter_source_records, validate_library
from .pipeline import ScreenConfig, run_screen
from .receptor import prepare_receptor
from .target import DEFAULT_BOX_SIZE, PreparedTarget, load_target, prepare_target
from .tuning import TuningConfig, tune_docking


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _workers(value: str) -> str | int:
    if value.lower() == "auto":
        return "auto"
    return _positive_int(value)


def _batch_size(value: str) -> str | int:
    if value.lower() == "auto":
        return "auto"
    return _positive_int(value)


def _batch_sizes(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _add_library_arguments(parser: argparse.ArgumentParser, *, input_required: bool = True) -> None:
    parser.add_argument("--input", type=Path, required=input_required, help="CSV/TSV or whitespace-delimited SMILES library.")
    parser.add_argument("--input-format", choices=["auto", "csv", "smi"], default="auto")
    parser.add_argument("--smiles-column", default="smiles", help="CSV column containing SMILES (case-insensitive).")
    parser.add_argument("--id-column", default=None, help="Optional CSV molecule-ID column.")
    parser.add_argument("--delimiter", default="auto", help="CSV delimiter, 'auto', or literal \\t.")
    parser.add_argument("--fragment-policy", choices=["reject", "largest"], default="reject")


def _add_pocket_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=["unidock", "autodock-gpu"],
        default="unidock",
        help="Docking backend (default: unidock).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help=(
            "Prepared dual-engine target containing receptor.pdbqt, pocket.json, "
            "AD4 maps, and manifest.json."
        ),
    )
    parser.add_argument(
        "--receptor",
        type=Path,
        help="Expert mode: prepared rigid receptor in PDBQT format.",
    )
    parser.add_argument(
        "--adgpu-grid",
        type=Path,
        help="Expert AutoDock-GPU mode: matching .maps.fld descriptor.",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Expert mode: docking-box center.",
    )
    parser.add_argument(
        "--size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Expert mode: docking-box size; defaults to 22.5 A per axis.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["fast", "balance", "detail"],
        default="balance",
        help=(
            "Engine effort preset: Uni-Dock preset or AutoDock-GPU 10/20/50 "
            "LGA runs."
        ),
    )
    parser.add_argument(
        "--scoring",
        choices=["auto", "vina", "vinardo", "ad4"],
        default="auto",
        help="auto selects Vina for Uni-Dock and AD4 for AutoDock-GPU.",
    )
    parser.add_argument("--num-modes", type=_positive_int, default=1)
    parser.add_argument(
        "--energy-range",
        type=float,
        default=3.0,
        help="Uni-Dock only: output energy window.",
    )
    parser.add_argument(
        "--refine-step",
        type=_positive_int,
        default=3,
        help="Uni-Dock only: explicit-receptor refinement steps.",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Uni-Dock only: skip explicit-receptor refinement.",
    )
    parser.add_argument(
        "--unidock-verbosity",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Uni-Dock-only log verbosity.",
    )
    parser.add_argument("--seed", type=int, default=181129)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--max-gpu-memory",
        type=int,
        default=0,
        metavar="MIB",
        help="Uni-Dock-only GPU-memory cap; 0 uses the engine default.",
    )
    parser.add_argument(
        "--adgpu-runs",
        type=_positive_int,
        help="Override AutoDock-GPU LGA runs (preset defaults: 10/20/50).",
    )
    parser.add_argument(
        "--adgpu-evaluations",
        type=_positive_int,
        help="Optional hard maximum score evaluations per LGA run.",
    )
    parser.add_argument(
        "--adgpu-no-heuristics",
        action="store_true",
        help="Disable AutoDock-GPU ligand-based evaluation heuristics.",
    )
    parser.add_argument(
        "--adgpu-no-autostop",
        action="store_true",
        help="Disable AutoDock-GPU convergence-based early stopping.",
    )
    parser.add_argument(
        "--adgpu-local-search",
        choices=["sw", "sd", "fire", "ad", "adam"],
        default="ad",
    )
    parser.add_argument(
        "--adgpu-cpu-threads",
        type=_positive_int,
        default=4,
        help="CPU threads for AutoDock-GPU's overlapped file-list pipeline.",
    )
    parser.add_argument(
        "--adgpu-workers",
        type=_workers,
        default="auto",
        help="Concurrent same-GPU processes or auto from profile (requires CUDA MPS above 1).",
    )
    parser.add_argument(
        "--adgpu-executable",
        default="autodock_gpu",
        help="AutoDock-GPU executable (default: optimized container symlink).",
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="igenvs", description="High-throughput iGen3/RDKit virtual screening with Uni-Dock or AutoDock-GPU.")
    parser.add_argument("--version", action="version", version=f"iGenVS {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check the container, tools, and GPU.")
    doctor.add_argument("--no-gpu", action="store_true", help="Do not require a visible CUDA GPU.")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    validate = subparsers.add_parser("validate", help="Validate and deduplicate a SMILES library with RDKit.")
    _add_library_arguments(validate)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--workers", type=_workers, default="auto")
    validate.add_argument("--no-deduplicate", action="store_true")
    validate.add_argument("--num-shards", type=_positive_int, default=1)
    validate.add_argument("--shard-index", type=int, default=0)

    receptor = subparsers.add_parser("prepare-receptor", help="Convert a curated/protonated PDB receptor to PDBQT with Meeko.")
    receptor.add_argument("--input", type=Path, required=True)
    receptor.add_argument("--output", type=Path, required=True)

    target = subparsers.add_parser(
        "prepare-target",
        help="Prepare a reusable receptor and ligand-defined docking pocket.",
    )
    target_source = target.add_mutually_exclusive_group(required=True)
    target_source.add_argument("--complex", type=Path, help="Protein-ligand complex in one PDB file.")
    target_source.add_argument("--receptor", type=Path, help="Curated/protonated receptor PDB file.")
    target.add_argument(
        "--ligand-id",
        help="Complex-PDB selector: RESNAME or CHAIN:RESNAME:RESSEQ (for example A:LIG:501).",
    )
    target.add_argument(
        "--reference-ligand",
        type=Path,
        help="Aligned single-molecule 3D SDF used to define the pocket.",
    )
    target.add_argument(
        "--padding",
        type=_nonnegative_float,
        default=5.0,
        help="Angstrom padding on every side of the reference ligand (default: 5).",
    )
    target.add_argument("--output-dir", type=Path, required=True)

    screen = subparsers.add_parser("screen", help="Generate or ingest, validate, prepare, and dock a library.")
    source = screen.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="External CSV/TSV/SMI library.")
    source.add_argument("--generate-count", type=_positive_int, help="Generate this many valid unique SMILES with iGen3.")
    source.add_argument(
        "--prevalidated-input",
        type=Path,
        help=argparse.SUPPRESS,
    )
    screen.add_argument("--input-format", choices=["auto", "csv", "smi"], default="auto")
    screen.add_argument("--smiles-column", default="smiles")
    screen.add_argument("--id-column", default=None)
    screen.add_argument("--delimiter", default="auto")
    screen.add_argument("--model", choices=["base-isomeric", "base-nonisomeric", "rl-isomeric", "rl-nonisomeric"], default="rl-nonisomeric")
    screen.add_argument("--generation-mode", choices=["de-novo", "derivative"], default="de-novo")
    screen.add_argument("--seed-file", type=Path)
    screen.add_argument("--samples-per-seed", type=_positive_int, default=1)
    screen.add_argument("--generator-batch-size", type=_batch_size, default="auto")
    screen.add_argument("--generator-max-batch-size", type=_positive_int, default=32_768)
    screen.add_argument("--model-dir", type=Path)
    screen.add_argument("--temperature", type=float)
    screen.add_argument("--top-k", type=int)
    screen.add_argument("--compile-generator", action="store_true")
    screen.add_argument("--generator-seed", type=int, default=13)
    _add_pocket_arguments(screen)
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.add_argument("--batch-size", type=_batch_size, default="auto", help="Outer docking batch or auto.")
    screen.add_argument("--batch-profile", type=Path, help="JSON profile written by tune-docking.")
    screen.add_argument("--prep-workers", type=_workers, default="auto")
    screen.add_argument("--prep-mode", choices=("standard", "fast"), default="standard", help="standard minimizes ETKDG conformers; fast skips separate force-field minimization.")
    screen.add_argument(
        "--embed-max-attempts",
        type=_workers,
        default="auto",
        help="Maximum deterministic ETKDG attempts per molecule (default: auto).",
    )
    screen.add_argument(
        "--embed-timeout",
        type=_workers,
        default="auto",
        metavar="SECONDS",
        help="Native RDKit wall guard for each ETKDG attempt phase (default: auto).",
    )
    screen.add_argument("--validation-workers", type=_workers, default="auto")
    screen.add_argument("--fragment-policy", choices=["reject", "largest"], default="reject")
    screen.add_argument("--no-deduplicate", action="store_true")
    screen.add_argument("--num-shards", type=_positive_int, default=1)
    screen.add_argument("--shard-index", type=int, default=0)
    screen.add_argument("--scratch-dir", type=Path)
    screen.add_argument("--keep-work", action="store_true")
    screen.add_argument("--pose-output", choices=("none", "merged", "individual"), default="merged")
    screen.add_argument("--individual-poses", action="store_true")

    tune = subparsers.add_parser("tune-docking", help="Empirically select the fastest reliable outer batch for the selected engine.")
    _add_library_arguments(tune)
    _add_pocket_arguments(tune)
    tune.add_argument("--profile", type=Path, required=True, help="Output JSON batch profile.")
    tune.add_argument("--batch-sizes", type=_batch_sizes, default=(), help="Comma-separated sizes; hardware defaults if omitted.")
    tune.add_argument("--prep-workers", type=_workers, default="auto")
    tune.add_argument("--prep-mode", choices=("standard", "fast"), default="standard")
    tune.add_argument("--scratch-dir", type=Path)

    return parser


def _run_validate(args: argparse.Namespace) -> dict[str, object]:
    records = iter_source_records(
        args.input,
        input_format=args.input_format,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        delimiter=args.delimiter,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    return validate_library(
        records,
        output_dir=args.output_dir,
        workers=args.workers,
        fragment_policy=args.fragment_policy,
        deduplicate=not args.no_deduplicate,
    )


def _resolve_pocket_arguments(
    args: argparse.Namespace,
) -> tuple[
    Path,
    tuple[float, float, float],
    tuple[float, float, float],
    PreparedTarget | None,
    Path | None,
]:
    if args.target is not None:
        if (
            args.receptor is not None
            or args.center is not None
            or args.size is not None
            or args.adgpu_grid is not None
        ):
            raise InputError(
                "--target cannot be combined with --receptor, --center, "
                "--size, or --adgpu-grid"
            )
        target = load_target(args.target)
        return (
            target.receptor,
            target.center,
            target.size,
            target,
            target.autodock_gpu_fld,
        )
    if args.receptor is None or args.center is None:
        raise InputError("provide --target, or provide both --receptor and --center")
    size = tuple(args.size) if args.size is not None else DEFAULT_BOX_SIZE
    return args.receptor, tuple(args.center), size, None, args.adgpu_grid


def _resolve_scoring(engine: str, scoring: str) -> str:
    if scoring == "auto":
        return "ad4" if engine == "autodock-gpu" else "vina"
    if engine == "autodock-gpu" and scoring != "ad4":
        raise InputError("AutoDock-GPU supports only --scoring ad4 (or auto)")
    if engine == "unidock" and scoring not in {"vina", "vinardo"}:
        raise InputError("Uni-Dock supports only --scoring vina or vinardo (or auto)")
    return scoring


def _validate_engine_arguments(args: argparse.Namespace, grid: Path | None) -> None:
    if args.device_id < 0:
        raise InputError("--device-id must be non-negative")
    if args.engine == "autodock-gpu":
        if grid is None:
            raise InputError(
                "AutoDock-GPU requires AD4 maps: use a new --target bundle or "
                "provide --adgpu-grid in expert mode"
            )
        if args.num_modes != 1:
            raise InputError("AutoDock-GPU currently emits exactly one best pose")
        if args.no_refine:
            raise InputError("--no-refine is a Uni-Dock-only option")
        if args.energy_range != 3.0 or args.refine_step != 3:
            raise InputError("--energy-range and --refine-step are Uni-Dock-only options")
        if args.max_gpu_memory:
            raise InputError("--max-gpu-memory is a Uni-Dock-only option")
        if args.unidock_verbosity:
            raise InputError("--unidock-verbosity is a Uni-Dock-only option")
    else:
        if args.adgpu_grid is not None:
            raise InputError("--adgpu-grid requires --engine autodock-gpu")
        if args.adgpu_workers not in {1, "auto"}:
            raise InputError("--adgpu-workers requires --engine autodock-gpu")
        adgpu_controls_changed = any(
            (
                args.adgpu_runs is not None,
                args.adgpu_evaluations is not None,
                args.adgpu_no_heuristics,
                args.adgpu_no_autostop,
                args.adgpu_local_search != "ad",
                args.adgpu_cpu_threads != 4,
                args.adgpu_executable != "autodock_gpu",
            )
        )
        if adgpu_controls_changed:
            raise InputError(
                "AutoDock-GPU protocol controls require --engine autodock-gpu"
            )

def _screen_config(args: argparse.Namespace) -> ScreenConfig:
    generation = None
    if args.generate_count is not None:
        generation = GenerationConfig(
            count=args.generate_count,
            model=args.model,
            mode=args.generation_mode,
            batch_size=args.generator_batch_size,
            max_batch_size=args.generator_max_batch_size,
            model_dir=args.model_dir,
            seed_file=args.seed_file,
            samples_per_seed=args.samples_per_seed,
            temperature=args.temperature,
            top_k=args.top_k,
            compile_model=args.compile_generator,
            seed=args.generator_seed,
        )
    receptor, center, size, target, autodock_gpu_fld = _resolve_pocket_arguments(args)
    _validate_engine_arguments(args, autodock_gpu_fld)
    scoring = _resolve_scoring(args.engine, args.scoring)
    return ScreenConfig(
        output_dir=args.output_dir,
        receptor=receptor,
        center=center,
        size=size,
        target=target.directory if target is not None else None,
        engine=args.engine,
        autodock_gpu_fld=autodock_gpu_fld,
        input_path=args.input,
        prevalidated_input=args.prevalidated_input,
        input_format=args.input_format,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        delimiter=args.delimiter,
        generation=generation,
        docking_batch_size=args.batch_size,
        batch_profile=args.batch_profile,
        prep_workers=args.prep_workers,
        prep_mode=args.prep_mode,
        embed_max_attempts=args.embed_max_attempts,
        embed_timeout_seconds=args.embed_timeout,
        validation_workers=args.validation_workers,
        fragment_policy=args.fragment_policy,
        deduplicate=not args.no_deduplicate,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        search_mode=args.search_mode,
        scoring=scoring,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        seed=args.seed,
        refine_step=args.refine_step,
        no_refine=args.no_refine,
        unidock_verbosity=args.unidock_verbosity,
        device_id=args.device_id,
        max_gpu_memory=args.max_gpu_memory,
        scratch_dir=args.scratch_dir,
        keep_work=args.keep_work,
        pose_output=args.pose_output,
        individual_poses=args.individual_poses,
        autodock_gpu_runs=args.adgpu_runs,
        autodock_gpu_evaluations=args.adgpu_evaluations,
        autodock_gpu_heuristics=not args.adgpu_no_heuristics,
        autodock_gpu_autostop=not args.adgpu_no_autostop,
        autodock_gpu_local_search=args.adgpu_local_search,
        autodock_gpu_cpu_threads=args.adgpu_cpu_threads,
        autodock_gpu_workers=args.adgpu_workers,
        autodock_gpu_executable=args.adgpu_executable,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = run_doctor(require_gpu=not args.no_gpu)
            print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_doctor(report))
            return 0 if report["ok"] else 1
        if args.command == "validate":
            print(json.dumps(_run_validate(args), indent=2, sort_keys=True))
            return 0
        if args.command == "prepare-receptor":
            result = prepare_receptor(args.input, args.output)
            print(f"Prepared receptor: {result['output']}")
            return 0
        if args.command == "prepare-target":
            target = prepare_target(
                output_dir=args.output_dir,
                complex_pdb=args.complex,
                ligand_id=args.ligand_id,
                receptor_pdb=args.receptor,
                reference_ligand_sdf=args.reference_ligand,
                padding=args.padding,
            )
            print(
                json.dumps(
                    {
                        "target": str(target.directory),
                        "receptor": str(target.receptor),
                        "center": target.center,
                        "size": target.size,
                        "autodock_gpu_fld": str(target.autodock_gpu_fld),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "screen":
            manifest = run_screen(_screen_config(args))
            print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "outputs": manifest["outputs"]}, indent=2))
            return 0
        if args.command == "tune-docking":
            receptor, center, size, target, autodock_gpu_fld = _resolve_pocket_arguments(args)
            _validate_engine_arguments(args, autodock_gpu_fld)
            scoring = _resolve_scoring(args.engine, args.scoring)
            profile = tune_docking(
                TuningConfig(
                    input_path=args.input,
                    receptor=receptor,
                    center=center,
                    size=size,
                    profile_path=args.profile,
                    target=target.directory if target is not None else None,
                    engine=args.engine,
                    autodock_gpu_fld=autodock_gpu_fld,
                    batch_sizes=args.batch_sizes,
                    input_format=args.input_format,
                    smiles_column=args.smiles_column,
                    id_column=args.id_column,
                    delimiter=args.delimiter,
                    prep_workers=args.prep_workers,
                    prep_mode=args.prep_mode,
                    fragment_policy=args.fragment_policy,
                    search_mode=args.search_mode,
                    scoring=scoring,
                    num_modes=args.num_modes,
                    energy_range=args.energy_range,
                    seed=args.seed,
                    refine_step=args.refine_step,
                    no_refine=args.no_refine,
                    unidock_verbosity=args.unidock_verbosity,
                    device_id=args.device_id,
                    max_gpu_memory=args.max_gpu_memory,
                    scratch_dir=args.scratch_dir,
                    autodock_gpu_runs=args.adgpu_runs,
                    autodock_gpu_evaluations=args.adgpu_evaluations,
                    autodock_gpu_heuristics=not args.adgpu_no_heuristics,
                    autodock_gpu_autostop=not args.adgpu_no_autostop,
                    autodock_gpu_local_search=args.adgpu_local_search,
                    autodock_gpu_cpu_threads=args.adgpu_cpu_threads,
                    autodock_gpu_workers=(
                        1
                        if args.adgpu_workers == "auto"
                        else args.adgpu_workers
                    ),
                    autodock_gpu_executable=args.adgpu_executable,
                )
            )
            print(f"Selected docking batch size: {profile['selected_batch_size']:,}")
            print(f"Profile: {args.profile}")
            return 0
        parser.error(f"unhandled command: {args.command}")
    except (IGenVSError, ValueError, OSError) as exc:
        print(f"igenvs: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

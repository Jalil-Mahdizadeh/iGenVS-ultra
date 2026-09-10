"""Small, standard-library-only guide that hands jobs to the existing CLI."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import workflow
from .workflow import PipelineError


PIPELINES = (
    ("dock", "Physical docking pipeline with Uni-Dock (vina) or AutoDock-GPU (AD4)"),
    ("run", "Active learning pipeline, fit a target-specific model, screen a library with it"),
    ("rl-train", "Reinforcement learning pipeline, train a target-specific molecule generator"),
)
GENERATION_MODELS = (
    ("base-isomeric", "Base model, with stereochemistry"),
    ("base-nonisomeric", "Base model, without stereochemistry"),
    ("rl-isomeric", "RL model, with stereochemistry"),
    ("rl-nonisomeric", "RL model, without stereochemistry"),
)


def colour(text: str, code: str, *, stream: Any = None) -> str:
    stream = sys.stdout if stream is None else stream
    if not stream.isatty() or "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return text
    return f"\033[{code}m{text}\033[0m"


def heading(text: str) -> None:
    print("\n" + colour(text, "1;36"))


def step(number: int, total: int, label: str) -> None:
    heading(f"Step {number}/{total} - {label}")


def notice(text: str, kind: str = "OK", *, stream: Any = None) -> None:
    stream = sys.stdout if stream is None else stream
    marker = colour(f"[{kind}]", {"OK": "32", "WARN": "33", "ERROR": "31"}[kind], stream=stream)
    print(f"{marker} {text}", file=stream, flush=True)


def field(label: str, value: Any) -> None:
    print(f"  {colour((label + ':').ljust(20), '2')} {value}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pipeline", nargs="?", choices=[key for key, _ in PIPELINES])
    parser.add_argument("--assets-dir", type=Path, help="Project/release root; otherwise discovered automatically.")
    parser.add_argument("--gpu-ids", help="Optional GPU restriction, for example 0 or 0,1.")
    parser.add_argument("--resume", type=Path, metavar="JOB", help="Reuse the choices saved for a guided job.")
    parser.add_argument("--dry-run", action="store_true", help="Check and preview without saving or running a job.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation with --resume (also works without a terminal).")


def ask(label: str, default: str | None = None, parse: Callable = str) -> Any:
    while True:
        hint = colour(f" [{default}]", "2") if default is not None else ""
        value = input(f"{label}{hint}{colour(': ', '36')}").strip()
        if not value and default is not None:
            value = default
        if not value and default is None:
            notice("Please enter a value.", "WARN")
            continue
        try:
            return parse(value)
        except (ValueError, OSError) as exc:
            notice(f"Please try again: {exc}", "WARN")


def choose(label: str, choices: list | tuple, default: str) -> str:
    heading(label)
    for index, (key, description) in enumerate(choices, 1):
        name = f"{index}. {key or 'auto'}"
        if key == default:
            name = colour(name, "1;36")
        suffix = colour(" (default)", "2") if key == default else ""
        print(f"  {name} - {description}{suffix}")

    def parse(value: str) -> str:
        for index, (key, _) in enumerate(choices, 1):
            if value == str(index) or value.casefold() == (key or "auto").casefold():
                return key
        raise ValueError("choose a listed number or name")

    return ask("Choice", default or "auto", parse)


def confirm(label: str, *, default: bool = False) -> bool:
    def parse(value: str) -> bool:
        if value.lower() in {"y", "yes", "n", "no"}:
            return value.lower() in {"y", "yes"}
        raise ValueError("enter yes or no")

    return ask(label, "yes" if default else "no", parse)


def path_value(value: str) -> Path:
    # Pasted paths may have quotes; never interpret them as shell commands.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        raise ValueError("enter a path")
    return Path(value).expanduser().resolve()


def input_path(label: str, *, directory: bool = False, default: str | None = None) -> Path:
    def parse(value: str) -> Path:
        path = path_value(value)
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"{'folder' if directory else 'file'} does not exist: {path}")
        return path

    return ask(label, default, parse)


def count_value(value: str) -> str:
    number = int(value.replace(",", "").replace("_", ""))
    if number < 1:
        raise ValueError("enter a positive whole number")
    return str(number)


def target_arguments() -> list[str]:
    mode = choose("Target input", (
        ("complex", "Protein and bound ligand in one PDB"),
        ("receptor", "Receptor PDB plus an aligned bound-ligand SDF"),
        ("prepared-target", "Existing prepared target folder"),
    ), "complex")
    path = input_path("Target folder" if mode == "prepared-target" else "PDB path", directory=mode == "prepared-target")
    arguments = [f"--{mode}", str(path)]
    if mode == "receptor":
        print("The SDF must contain the bound ligand in the receptor's coordinate frame.")
        arguments += ["--reference-ligand", str(input_path("Reference ligand SDF"))]
    elif mode == "complex":
        try:
            ligand = workflow.detect_ligand_id(path)
            notice(f"Detected bound ligand: {ligand}")
        except PipelineError as exc:
            notice(str(exc), "WARN")
            ligand = ask("Bound ligand ID (for example A:LIG:501)")
        arguments += ["--ligand-id", ligand]
    return arguments


def csv_arguments(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(65_536)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            delimiter = ","
        handle.seek(0)
        fields = next(csv.reader(handle, delimiter=delimiter), [])
    if not fields or any(not field.strip() for field in fields) or len({f.casefold() for f in fields}) != len(fields):
        raise ValueError("CSV/TSV needs a header with non-empty, distinct column names")
    default = next((f for f in fields if f.casefold() in {"smiles", "canonical_smiles", "isomeric_smiles"}), fields[0])
    smiles = choose("SMILES column", [(f, "column") for f in fields], default)
    identifier = choose("Molecule ID column", [("", "Use molecule_id/id/name when present, otherwise row numbers")]
                        + [(f, "column") for f in fields if f != smiles], "")
    return ["--input-format", "csv", "--delimiter", delimiter, "--smiles-column", smiles,
            *(["--id-column", identifier] if identifier else [])]


def source_arguments(*, default_model: str = "rl-nonisomeric") -> list[str]:
    source = choose("Molecule source", (("file", "Read a CSV, TSV or SMI library"),
                                        ("generate", "Generate molecules with iGen3")), "file")
    if source == "generate":
        count = ask("Number of molecules", parse=count_value)
        model = choose("iGen3 generation model", GENERATION_MODELS, default_model)
        return ["--generate-count", count, "--model", model]
    while True:
        path = input_path("Library path")
        suffix = path.suffix.lower()
        fmt = "csv" if suffix in {".csv", ".tsv"} else "smi"
        if suffix not in {".csv", ".tsv", ".smi", ".smiles"}:
            fmt = choose("File format", (("smi", "SMILES followed by an optional ID, no header"),
                                         ("csv", "Delimited table with a header")), fmt)
        try:
            columns = csv_arguments(path) if fmt == "csv" else ["--input-format", "smi"]
        except (ValueError, csv.Error, UnicodeError) as exc:
            notice(f"Cannot read this table: {exc}", "WARN")
            continue
        return ["--input", str(path), *columns]


def pipeline_arguments(pipeline: str) -> list[str]:
    step(1, 4 if pipeline == "rl-train" else 5, "Target")
    arguments = target_arguments()
    if pipeline == "rl-train":
        print("RL training uses the frozen protocol; its stages choose the docking presets.")
        return arguments
    step(2, 5, "Molecules and settings")
    if pipeline == "run":
        print("Fitting physically docks 300,000 UDRL molecules, plus 30,000 per AL round.")
        print("The final library is scored by the fitted model. Fit uses Uni-Dock/Vina fast.")
        def al_rounds(value: str) -> str:
            number = int(value)
            if not 0 <= number <= 5:
                raise ValueError("enter a whole number from 0 to 5")
            return str(number)

        rounds = ask("Active-learning rounds (0 disables AL; maximum 5)", "0", al_rounds)
        arguments += ["--al-rounds", rounds]
    arguments += source_arguments(default_model="rl-nonisomeric" if pipeline == "dock" else "base-isomeric")
    if pipeline == "dock":
        engine = choose("Docking engine", (("unidock", "Uni-Dock (Vina scoring)"),
                                           ("autodock-gpu", "AutoDock-GPU (AD4 scoring)")), "unidock")
        arguments += ["--engine", engine]
        arguments += ["--search-mode", choose("Search preset", (("fast", "Least search effort"),
                                                                  ("balance", "Default search effort"),
                                                                  ("detail", "Most search effort")), "balance")]
        poses = choose("Save poses", (("merged", "Combined poses.pdbqt and poses.sdf files"), ("none", "Scores only"),
                                     ("individual", "Separate PDBQT and SDF files per molecule")), "merged")
        arguments += ["--pose-output", poses]
        if poses != "none" and engine == "unidock":
            arguments += ["--num-modes", ask("Maximum poses per molecule", "1", count_value)]
        elif poses != "none":
            print("AutoDock-GPU saves one best pose per molecule.")
    else:
        def threshold(value: str) -> str:
            if value and not 0 <= float(value) <= 1:
                raise ValueError("enter a score between 0 and 1, or leave blank to save all")
            return value

        score = ask("Minimum ensemble score to save (0-1, blank saves all; this is not docking energy)", "", threshold)
        if score:
            arguments += ["--score-threshold", score]
    return arguments


def missing_assets(assets: Path, pipeline: str, al_rounds: int = 0) -> list[Path]:
    required = [assets / name for name in (
        "user-pipeline/src/igenvs_ultra/core_runtime.py",
        "iGenVS/src/igenvs/cli.py", "iGenVS/iGen3/src/igen3/cli.py",
    )]
    if pipeline == "dock":
        required.append(assets / "user-pipeline/src/igenvs_ultra/pose_export.py")
    elif pipeline == "run":
        required += workflow.required_fit_assets(assets, al_rounds)
        required += [assets / name for name in (
            "user-pipeline/src/igenvs_ultra/model_ops.py", "user-pipeline/src/igenvs_ultra/generation_worker.py",
            "gMolAI-v2.0/inference/gmolai.py", "gMolAI-v2.0/inference/models/SHA256SUMS",
        )]
    elif pipeline == "rl-train":
        required += [assets / name for name in (
            "user-pipeline/src/igenvs_ultra/rl_runtime.py", "phase-10-rl-dev/freeze.json",
            "phase-10-rl-dev/protocol.json", "phase-10-rl-dev/maintenance.json",
        )]
    return [path for path in required if not path.is_file()]


def select_assets(options: argparse.Namespace, pipeline: str, al_rounds: int) -> Path:
    try:
        assets = workflow.resolve_assets(options)
    except PipelineError:
        assets = input_path("Project/release root", directory=True)
    while True:
        missing = missing_assets(assets, pipeline, al_rounds)
        if not missing:
            return assets
        notice(f"Missing assets under {assets}:", "WARN")
        for path in missing:
            print(f"  {path.relative_to(assets)}")
        print("Use a complete project/release root with these files in their standard folders.")
        assets = input_path("Alternative project/release root (Ctrl-C to cancel)", directory=True)


def record_path(job: Path) -> Path:
    # Keep the job empty until the pipeline creates its own resume configuration.
    return job.with_name(job.name + ".launch.json")


def cli_prefix() -> list[str]:
    launcher = Path(__file__).resolve().parents[2] / "igenvs-ultra"
    return [sys.executable, str(launcher)] if launcher.is_file() else [sys.executable, "-m", "igenvs_ultra"]


def resume_command(job: Path) -> str:
    return shlex.join([*cli_prefix(), "guide", "--resume", str(job)])


def parsed_arguments(argv: list[str]) -> argparse.Namespace:
    from .cli import build_parser, validate_args

    if not argv or argv[0] not in dict(PIPELINES):
        raise PipelineError("saved command must be dock, run, or rl-train")
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return args


def resolved_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def load_record(job: Path) -> tuple[list[str], argparse.Namespace]:
    path = record_path(job)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(f"cannot read saved launcher choices: {path}; use the original CLI command to resume") from exc
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        raise PipelineError(f"unsupported launcher record: {path}")
    argv = saved.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise PipelineError(f"invalid saved command: {path}")
    args = parsed_arguments(argv)
    if args.execution != "docker" or args.dry_run or args.output_dir != job:
        raise PipelineError(f"saved command does not describe this Docker job: {path}")
    if resolved_arguments(args) != saved.get("arguments"):
        raise PipelineError(f"CLI defaults or saved choices have changed; review {path} and use the saved CLI command")
    return argv, args


def select_job(pipeline: str) -> tuple[Path, bool]:
    default = str(Path("runs") / f"{pipeline}-{datetime.now():%Y%m%d-%H%M%S}")
    while True:
        job = ask("Output folder (new or a previous guided job)", default, path_value)
        if record_path(job).exists():
            if confirm("Saved choices found. Resume this job?", default=True):
                return job, True
            continue
        if job.exists() and (not job.is_dir() or any(job.iterdir())):
            notice("This path is occupied and has no saved launcher choices. Choose a new folder.", "WARN")
            print("For an existing CLI job, rerun its original command to resume.")
            continue
        return job, False


def disk_space(job: Path) -> float:
    parent = job
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise PipelineError(f"output location is not a writable folder: {parent}")
    free = shutil.disk_usage(parent).free
    if free == 0:
        raise PipelineError(f"output filesystem has no free space: {parent}")
    return free / 2**30


def checked_capture(runtime: workflow.Runtime, tool: str, command: list[str], label: str, *, gpu: bool = True) -> str:
    result = runtime.capture(tool, command, gpu=gpu)
    if result.returncode:
        raise PipelineError(f"{label} failed:\n{(result.stderr or result.stdout).strip()[-4000:]}")
    return result.stdout


def check_runtime(args: argparse.Namespace) -> dict[str, Any]:
    print("Checking Docker, images, tools and GPU access...", flush=True)
    try:
        result = subprocess.run(["docker", "info", "--format", "{{.OSType}}"],
                                capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PipelineError("Docker is unavailable. Start Docker and check that 'docker info' works.") from exc
    if result.returncode or result.stdout.strip() != "linux":
        raise PipelineError(f"Docker must be running Linux containers; check 'docker info'.\n{result.stderr.strip()}")
    notice("Docker is ready")
    assets = args.assets_dir
    images = workflow.resolved_docker_images(args)
    for name in images if args.command == "run" else images[:1]:
        if not workflow.docker_image_exists(name):
            raise PipelineError(f"Docker image is missing: {name}. Build the project images with ./scripts/build-images.sh.")
        notice(f"Image available: {name}")
    # A preflight container must bind an existing folder, never a not-yet-created job.
    runtime = workflow.Runtime(args, assets, assets / "user-pipeline", require_gmolai=args.command == "run")
    if args.command == "rl-train":
        from .rl_workflow import _runtime_command

        raw = checked_capture(runtime, "igenvs", _runtime_command(assets, "doctor"), "RL environment check")
        report = json.loads(raw)
        if not report.get("ok"):
            raise PipelineError(f"RL environment is not ready:\n{raw}")
        igenvs = report["igenvs"]
    else:
        raw = checked_capture(runtime, "igenvs", ["igenvs", "doctor", "--json"], "iGenVS environment check")
        igenvs = json.loads(raw)
    if not igenvs.get("ok") or not igenvs.get("torch_cuda", {}).get("available"):
        raise PipelineError(f"iGenVS tools or GPU access are not ready:\n{raw}")
    notice("iGenVS tools and GPU access are ready")
    if args.command == "run":
        print("Checking gMolAI models and GPU access...", flush=True)
        checked_capture(runtime, "gmolai", runtime.model_command(["self-test"]), "Target-head check", gpu=False)
        checked_capture(runtime, "gmolai", ["python", str(assets / "gMolAI-v2.0/inference/gmolai.py"),
                                            "validate", "--device", "cuda"], "gMolAI model/GPU check")
        notice("gMolAI models and GPU access are ready")
    ids = workflow.visible_gpu_tokens(args.gpu_ids)
    if not ids:
        raise PipelineError("No GPU IDs are visible to the launcher. Check nvidia-smi and CUDA_VISIBLE_DEVICES, or use --gpu-ids.")
    return {"gpu_ids": ids, "gpu_name": igenvs["torch_cuda"].get("device_name"),
            "free_gib": disk_space(args.output_dir)}


def dry_plan(args: argparse.Namespace) -> dict[str, Any]:
    preview = argparse.Namespace(**{**vars(args), "dry_run": True})
    with contextlib.redirect_stdout(io.StringIO()):
        if args.command == "dock":
            return workflow.regular_dock(preview)
        if args.command == "run":
            return workflow.plan_run(preview)
        from .rl_workflow import train_rl

        return train_rl(preview)


def show_summary(args: argparse.Namespace, argv: list[str], plan: dict, readiness: dict) -> None:
    notice(f"Ready: {args.command}")
    field("Target", args.complex or args.receptor or args.prepared_target)
    if args.ligand_id:
        field("Bound ligand", args.ligand_id)
    if args.reference_ligand:
        field("Reference ligand", args.reference_ligand)
    if args.command != "rl-train":
        source = f"{args.generate_count:,} generated molecules ({args.model})" if args.generate_count else str(args.input)
        field("Library", source)
        if args.input and args.input_format == "csv":
            field("Columns", f"SMILES={args.smiles_column}; ID={args.id_column or 'auto'}")
    if args.command == "dock":
        field("Engine", args.engine)
        field("Preset", args.search_mode)
        field("Poses", f"{args.pose_output}; max poses: {args.num_modes}")
        print("Every admitted molecule will be physically docked.")
    elif args.command == "run":
        total = plan["fit"]["reference_docking_rows"] + plan["fit"]["al_docking_rows"]
        field("Fit", f"{args.al_rounds} AL rounds, up to {total:,} physical dockings; Uni-Dock/Vina fast.")
        field("Final library", f"neural scoring; save {'all scores' if args.score_threshold is None else 'scores >= ' + str(args.score_threshold)}.")
    else:
        field("Frozen RL protocol", plan['frozen_protocol']['id'])
        for stage in plan["frozen_protocol"]["stages"]:
            print(f"  {stage['name']}: up to {stage['maximum_updates']:,} updates ({stage['search_mode']})")
    field("GPU IDs", f"{', '.join(readiness['gpu_ids'])}; {readiness['gpu_name']}")
    field("Assets", args.assets_dir)
    field("Output", args.output_dir)
    field("Free disk", f"{readiness['free_gib']:.1f} GiB (not a job-size estimate)")
    print(f"\nCommand:\n{shlex.join([*cli_prefix(), *argv])}")


def show_results(args: argparse.Namespace) -> None:
    job = args.output_dir
    print()
    notice(f"Completed {args.command}.")
    if args.command == "dock":
        field("Docking files", job / 'docking')
        for relative in ("docking/results.csv", "docking/manifest.json", "docking/poses.pdbqt", "docking/poses.sdf"):
            if (job / relative).is_file():
                field(Path(relative).name, job / relative)
    elif args.command == "run":
        field("Scores", job / 'screens/final/results.csv')
        field("Reusable model", job / 'models/final.json')
    else:
        field("Trained generator", job / 'model')
        print("Example next step: generate 1,000 molecules with this model:")
        print(shlex.join([*cli_prefix(), "rl-generate", "--execution", "docker", "--assets-dir", str(args.assets_dir),
                          "--igenvs-docker-image", args.igenvs_docker_image, "--gpu-ids", args.gpu_ids,
                          "--model-dir", str(job / "model"), "--count", "1000", "--output", str(job / "generated.csv")]))
    field("Logs", job / 'logs')
    field("Saved choices", record_path(job))


def run_guide(options: argparse.Namespace) -> int:
    from .cli import main

    job = None
    saved = False
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
            raise PipelineError("Run this launcher in an x86-64 Linux or WSL2 terminal.")
        if options.yes and not options.resume:
            raise PipelineError("--yes is only available with --resume; new jobs need interactive choices.")
        if options.resume and (options.assets_dir or options.gpu_ids is not None):
            raise PipelineError("--resume reuses the saved assets and GPUs; omit --assets-dir and --gpu-ids.")
        if not sys.stdin.isatty() and not (options.resume and (options.yes or options.dry_run)):
            raise PipelineError("Open an interactive terminal for ./start; for scripts use the existing CLI or --resume JOB --yes.")
        heading("iGenVS-ultra guide")
        print("Enter accepts a default; Ctrl-C cancels.\n")
        if options.resume:
            job, saved = options.resume.expanduser().resolve(), True
        else:
            pipeline = options.pipeline or choose("Choose a workflow", PIPELINES, "dock")
            job, saved = select_job(pipeline)
        if saved:
            step(1, 4, "Saved choices")
            argv, args = load_record(job)
            if options.pipeline and options.pipeline != args.command:
                raise PipelineError(f"this saved job uses {args.command}, not {options.pipeline}")
            notice(f"Reusing saved {args.command} choices. Completed stages will be reused.")
        else:
            argv = [pipeline, *pipeline_arguments(pipeline), "--output-dir", str(job), "--execution", "docker"]
            initial = parsed_arguments(argv)
            assets = select_assets(options, pipeline, getattr(initial, "al_rounds", 0))
            igenvs, gmolai = workflow.resolved_docker_images(options)
            argv += ["--assets-dir", str(assets), "--igenvs-docker-image", igenvs]
            if pipeline == "run":
                argv += ["--gmolai-docker-image", gmolai]
            ids = workflow.visible_gpu_tokens(options.gpu_ids)
            if not ids:
                raise PipelineError("No GPUs are visible. Check nvidia-smi/CUDA_VISIBLE_DEVICES or provide --gpu-ids.")
            argv += ["--gpu-ids", ",".join(ids)]
            args = parsed_arguments(argv)
        missing = missing_assets(args.assets_dir, args.command, getattr(args, "al_rounds", 0))
        if missing:
            raise PipelineError("Restore the missing job assets:\n" + "\n".join(map(str, missing)))
        total_steps = 4 if saved or args.command == "rl-train" else 5
        step(total_steps - 2, total_steps, "Readiness checks")
        notice("Required assets are available")
        plan = dry_plan(args)
        readiness = check_runtime(args)
        step(total_steps - 1, total_steps, "Review")
        show_summary(args, argv, plan, readiness)
        if options.dry_run:
            notice("Preview complete. No job or launcher record was written.")
            return 0
        if not options.yes and not confirm("Start/resume this job?"):
            notice("Cancelled. No job was started.", "WARN")
            return 0
        if not saved:
            workflow.atomic_json(record_path(job), {
                "schema_version": 1, "argv": argv, "arguments": resolved_arguments(args),
                "command": shlex.join([*cli_prefix(), *argv]), "plan": plan,
            })
            saved = True
        step(total_steps, total_steps, "Run")
        print(f"To resume:\n{resume_command(job)}", flush=True)
        result = main(argv)
        if result == 0:
            show_results(args)
        else:
            notice(f"Job stopped. See the error above and logs under {job / 'logs'}.", "ERROR")
            print(f"Resume with:\n{resume_command(job)}")
        return result
    except (EOFError, KeyboardInterrupt):
        print()
        notice("Cancelled.", "WARN")
        if saved and job:
            print(f"Resume with:\n{resume_command(job)}")
        return 130
    except (PipelineError, OSError, ValueError) as exc:
        notice(f"start: {exc}", "ERROR", stream=sys.stderr)
        return 2

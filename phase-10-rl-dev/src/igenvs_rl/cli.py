"""Command-line interface for target-specific iGen3 RL."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from igen3.registry import resolve_model
from igenvs.target import load_target

from . import __version__
from .state import JobConfig, load_config, save_config
from .trainer import evaluate_job, generate_job, recover_job, train_job


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _tail_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 0.5:
        raise argparse.ArgumentTypeError("must be in (0, 0.5]")
    return parsed


def _elite_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 0.1:
        raise argparse.ArgumentTypeError("must be in (0, 0.1]")
    return parsed


def _unit_interval_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_image() -> Path:
    configured = os.environ.get("IGENVS_IMAGE")
    if configured:
        return Path(configured)
    return Path(
        "/nobackup/proj/disk/theo-storage/personal/jalil/iGenVS/containers/iGenVS.SIF"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="igenvs-rl",
        description="Target-specific iGen3 reinforcement learning using iGenVS docking.",
    )
    parser.add_argument("--version", action="version", version=f"igenvs-rl {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check iGen3, iGenVS, and visible GPUs.")
    doctor.add_argument("--target", type=Path)
    doctor.add_argument("--shards", type=_positive_int, default=1)

    initialize = subparsers.add_parser("init", help="Create a target-specific RL job.")
    initialize.add_argument("--job-dir", type=Path, required=True)
    initialize.add_argument("--target", type=Path, required=True)
    initialize.add_argument("--image", type=Path, default=_default_image())
    initialize.add_argument("--igenvs-project", type=Path, default=_default_repo_root() / "iGenVS")
    initialize.add_argument("--model-root", type=Path, default=_default_repo_root() / "iGenVS/iGen3/models")
    initialize.add_argument(
        "--initial-model-root",
        type=Path,
        help="Optional exported iGen3 model used to warm-start the policy; the prior remains --model-root.",
    )
    initialize.add_argument("--model", choices=["base-isomeric"], default="base-isomeric")
    initialize.add_argument("--oracle", choices=["igenvs", "fake"], default="igenvs")
    initialize.add_argument("--engine", choices=["unidock", "autodock-gpu"], default="unidock")
    initialize.add_argument("--scoring", choices=["auto", "vina", "vinardo", "ad4"], default="auto")
    initialize.add_argument("--search-mode", choices=["fast", "balance", "detail"], default="fast")
    initialize.add_argument("--shards", type=_positive_int, default=1)
    initialize.add_argument("--prep-workers", type=_positive_int, default=16)
    initialize.add_argument("--validation-workers", type=_positive_int, default=8)
    initialize.add_argument("--batch-size", type=_positive_int, default=256)
    initialize.add_argument("--reference-count", type=_positive_int, default=512)
    initialize.add_argument("--evaluation-count", type=_positive_int, default=256)
    initialize.add_argument("--evaluation-every", type=int, default=5)
    initialize.add_argument("--temperature", type=_positive_float, default=1.0)
    initialize.add_argument("--top-k", type=int, default=64, help="0 disables top-k filtering.")
    initialize.add_argument("--generator-seed", type=int, default=13)
    initialize.add_argument("--docking-seed", type=int, default=181129)
    initialize.add_argument("--learning-rate", type=_positive_float, default=1e-5)
    initialize.add_argument("--kl-beta", type=_positive_float, default=0.02)
    initialize.add_argument("--target-kl", type=_positive_float, default=0.05)
    initialize.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    initialize.add_argument(
        "--reward-mode",
        choices=["percentile", "hybrid", "elite", "binary-elite"],
        default="percentile",
        help="elite rewards only the frozen base model's best tail.",
    )
    initialize.add_argument("--tail-fraction", type=_tail_fraction, default=0.10)
    initialize.add_argument("--tail-weight", type=_nonnegative_float, default=1.0)
    initialize.add_argument("--elite-fraction", type=_elite_fraction, default=0.01)
    initialize.add_argument(
        "--minimum-elite-unique",
        type=_nonnegative_int,
        default=0,
        help="Reject elite-mode checkpoints with fewer distinct elite molecules in evaluation.",
    )
    initialize.add_argument(
        "--require-lipinski",
        action="store_true",
        help="Reward elite dockers only when they pass iGen3's existing Rule-of-Five check.",
    )
    initialize.add_argument(
        "--minimum-qed",
        type=_unit_interval_float,
        default=0.0,
        help="Reward molecules only when their RDKit QED is at least this value.",
    )
    initialize.add_argument(
        "--maximum-absolute-formal-charge",
        type=_nonnegative_int,
        help="Reward molecules only when their absolute RDKit formal charge is at most this value.",
    )
    initialize.add_argument(
        "--minimum-fraction-csp3",
        type=_unit_interval_float,
        default=0.0,
        help="Reward molecules only when their RDKit fraction-Csp3 is at least this value.",
    )
    initialize.add_argument(
        "--maximum-aromatic-rings",
        type=_nonnegative_int,
        help="Reward molecules only when their RDKit aromatic-ring count is at most this value.",
    )
    initialize.add_argument(
        "--reward-occurrence-cap",
        type=_positive_int,
        help="Limit each canonical molecule to this many policy-gradient occurrences per batch.",
    )
    initialize.add_argument(
        "--fresh-evaluation-docking",
        action="store_true",
        help="Re-dock every evaluation molecule instead of reading the training score cache.",
    )
    initialize.add_argument(
        "--maximum-top-molecule-fraction",
        type=_unit_interval_float,
        default=1.0,
        help="Reject checkpoints when one canonical molecule exceeds this raw evaluation fraction.",
    )
    initialize.add_argument(
        "--reward-seen-molecules",
        action="store_true",
        help="Reward every repeated occurrence while docking each unique molecule only once per frozen protocol.",
    )

    train = subparsers.add_parser("train", help="Run or resume policy updates.")
    train.add_argument("--job-dir", type=Path, required=True)
    train_count = train.add_mutually_exclusive_group(required=True)
    train_count.add_argument("--updates", type=_positive_int)
    train_count.add_argument(
        "--target-update",
        type=_positive_int,
        help="Run safely to this absolute checkpoint update (idempotent on resume).",
    )

    recover = subparsers.add_parser("recover", help="Repair checkpoint-derived state and exports without training.")
    recover.add_argument("--job-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="Freshly sample and dock a checkpoint.")
    evaluate.add_argument("--job-dir", type=Path, required=True)
    evaluate.add_argument("--count", type=_positive_int, default=256)
    evaluate.add_argument(
        "--checkpoint",
        choices=["base", "best", "latest"],
        default="best",
    )
    evaluate.add_argument("--seed", type=int, help="Explicit sampling seed for a matched raw evaluation.")

    generate = subparsers.add_parser("generate", help="Generate with the best target policy.")
    generate.add_argument("--job-dir", type=Path, required=True)
    generate.add_argument("--count", type=_positive_int, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--allow-repeats",
        action="store_true",
        help="Preserve the policy's concentrated occurrence distribution instead of forcing uniqueness.",
    )

    status = subparsers.add_parser("status", help="Show job configuration and latest metrics.")
    status.add_argument("--job-dir", type=Path, required=True)
    return parser


def _run_doctor(args: argparse.Namespace) -> int:
    igenvs = subprocess.run(
        ["igenvs", "doctor", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        igenvs_report = json.loads(igenvs.stdout)
    except json.JSONDecodeError:
        igenvs_report = {"ok": False, "stderr": igenvs.stderr.strip()}
    target_report = None
    if args.target:
        target = load_target(args.target)
        target_report = {
            "path": str(target.directory),
            "autodock_gpu": target.autodock_gpu_fld is not None,
        }
    spec = resolve_model("base-isomeric")
    report = {
        "ok": bool(igenvs_report.get("ok")) and torch.cuda.device_count() >= args.shards,
        "igenvs": igenvs_report,
        "igen3_model": spec.model_id,
        "cuda_device_count": torch.cuda.device_count(),
        "requested_shards": args.shards,
        "target": target_report,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _run_init(args: argparse.Namespace) -> None:
    job_dir = args.job_dir.expanduser().resolve()
    if job_dir.exists() and (not job_dir.is_dir() or any(job_dir.iterdir())):
        raise ValueError(f"job directory must be new or empty: {job_dir}")
    target = load_target(args.target)
    model_root = args.model_root.expanduser().resolve()
    spec = resolve_model(args.model)
    if not spec.weights_path(model_root).is_file() or not spec.vocab_path(model_root).is_file():
        raise FileNotFoundError(f"iGen3 model artifacts are missing below {model_root}")
    initial_model_root = None
    if args.initial_model_root is not None:
        initial_root = args.initial_model_root.expanduser().resolve()
        if not spec.weights_path(initial_root).is_file() or not spec.vocab_path(initial_root).is_file():
            raise FileNotFoundError(f"initial iGen3 model artifacts are missing below {initial_root}")
        if spec.vocab_path(initial_root).read_bytes() != spec.vocab_path(model_root).read_bytes():
            raise ValueError("the initial model vocabulary differs from the frozen base vocabulary")
        initial_model_root = str(initial_root)
    if args.engine == "unidock" and args.scoring == "ad4":
        raise ValueError("Uni-Dock cannot use AD4 scoring")
    if args.engine == "autodock-gpu" and args.scoring not in {"auto", "ad4"}:
        raise ValueError("AutoDock-GPU requires AD4 scoring")
    if args.engine == "autodock-gpu" and target.autodock_gpu_fld is None:
        raise ValueError("the target does not contain AutoDock-GPU maps")
    if args.top_k < 0:
        raise ValueError("top-k must be non-negative")
    if args.evaluation_every < 0:
        raise ValueError("evaluation-every must be non-negative")

    config = JobConfig(
        schema_version=1,
        target=str(target.directory),
        image=str(args.image.expanduser().resolve()),
        igenvs_project=str(args.igenvs_project.expanduser().resolve()),
        model_root=str(model_root),
        model_id=args.model,
        oracle=args.oracle,
        engine=args.engine,
        scoring=args.scoring,
        search_mode=args.search_mode,
        shards=args.shards,
        prep_workers=args.prep_workers,
        validation_workers=args.validation_workers,
        batch_size=args.batch_size,
        reference_count=args.reference_count,
        evaluation_count=args.evaluation_count,
        evaluation_every=args.evaluation_every,
        temperature=args.temperature,
        top_k=args.top_k,
        generator_seed=args.generator_seed,
        docking_seed=args.docking_seed,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        reward_mode=args.reward_mode,
        tail_fraction=args.tail_fraction,
        tail_weight=args.tail_weight,
        reward_seen_molecules=args.reward_seen_molecules,
        elite_fraction=args.elite_fraction,
        initial_model_root=initial_model_root,
        minimum_elite_unique=args.minimum_elite_unique,
        require_lipinski=args.require_lipinski,
        minimum_qed=args.minimum_qed,
        maximum_absolute_formal_charge=args.maximum_absolute_formal_charge,
        minimum_fraction_csp3=args.minimum_fraction_csp3,
        maximum_aromatic_rings=args.maximum_aromatic_rings,
        reward_occurrence_cap=args.reward_occurrence_cap,
        fresh_evaluation_docking=args.fresh_evaluation_docking,
        maximum_top_molecule_fraction=args.maximum_top_molecule_fraction,
    )
    path = save_config(job_dir, config)
    print(f"[igenvs-rl] initialized {path}")


def _run_status(job_dir: Path) -> None:
    resolved = job_dir.expanduser().resolve()
    config = load_config(resolved)
    latest: dict[str, str] | None = None
    history_path = resolved / "history.csv"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        latest = rows[-1] if rows else None
    checkpoint_summaries: dict[str, dict[str, float | int]] = {}
    for name in ("latest", "best"):
        checkpoint_path = resolved / "checkpoints" / f"{name}.pt"
        if checkpoint_path.is_file():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_summaries[name] = {
                "update": int(payload["update"]),
                "best_reward": float(payload.get("best_reward", float("nan"))),
                "kl_beta": float(payload["kl_beta"]),
            }
    report = {
        "job_dir": str(resolved),
        "target": config.target,
        "engine": config.engine,
        "search_mode": config.search_mode,
        "shards": config.shards,
        "reward_mode": config.reward_mode,
        "tail_fraction": config.tail_fraction,
        "tail_weight": config.tail_weight,
        "elite_fraction": config.elite_fraction,
        "reward_seen_molecules": config.reward_seen_molecules,
        "initial_model_root": config.initial_model_root,
        "minimum_elite_unique": config.minimum_elite_unique,
        "require_lipinski": config.require_lipinski,
        "minimum_qed": config.minimum_qed,
        "maximum_absolute_formal_charge": config.maximum_absolute_formal_charge,
        "minimum_fraction_csp3": config.minimum_fraction_csp3,
        "maximum_aromatic_rings": config.maximum_aromatic_rings,
        "reward_occurrence_cap": config.reward_occurrence_cap,
        "fresh_evaluation_docking": config.fresh_evaluation_docking,
        "maximum_top_molecule_fraction": config.maximum_top_molecule_fraction,
        "reference_ready": (resolved / "reference/scores.json").is_file(),
        "checkpoint_ready": (resolved / "checkpoints/latest.pt").is_file(),
        "best_model_ready": (resolved / "checkpoints/best.pt").is_file(),
        "checkpoints": checkpoint_summaries,
        "latest": latest,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _run_doctor(args)
        if args.command == "init":
            _run_init(args)
        elif args.command == "train":
            job_dir = args.job_dir.expanduser().resolve()
            train_job(
                job_dir,
                load_config(job_dir),
                updates=args.updates,
                target_update=args.target_update,
            )
        elif args.command == "recover":
            job_dir = args.job_dir.expanduser().resolve()
            recover_job(job_dir, load_config(job_dir))
        elif args.command == "evaluate":
            job_dir = args.job_dir.expanduser().resolve()
            metrics = evaluate_job(
                job_dir,
                load_config(job_dir),
                count=args.count,
                checkpoint_name=args.checkpoint,
                seed=args.seed,
            )
            print(json.dumps(metrics, indent=2, sort_keys=True))
        elif args.command == "generate":
            job_dir = args.job_dir.expanduser().resolve()
            generate_job(
                job_dir,
                load_config(job_dir),
                count=args.count,
                output=args.output.expanduser().resolve(),
                allow_repeats=args.allow_repeats,
            )
        elif args.command == "status":
            _run_status(args.job_dir)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"igenvs-rl: error: {exc}", file=sys.stderr)
        return 2

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Any

from .config import ConfigurationError, apply_training_plan, load_config, object_hash


def _task_value(explicit: int | None, environment_name: str) -> int:
    if explicit is not None:
        return explicit
    value = os.environ.get(environment_name)
    if value is None:
        raise ValueError(f"Pass the task argument or set {environment_name}")
    return int(value)


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2), flush=True)


def _load(args, *, allow_unconfirmed: bool = False):
    cfg = load_config(args.config, require_descriptor_confirmation=not allow_unconfirmed)
    plans = getattr(args, "plan", None) or []
    if isinstance(plans, str):
        plans = [plans]
    for plan in plans:
        apply_training_plan(cfg, plan)
    return cfg


def _apply_training_budgets(cfg: dict[str, Any], args) -> None:
    for argument, key in (
        ("node_budget", "node_budget_per_gpu"),
        ("graph_budget", "max_graphs_per_gpu"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            if value <= 0:
                raise ValueError(f"--{argument.replace('_', '-')} must be positive")
            cfg["training"][key] = int(value)


def _apply_run_directory(cfg: dict[str, Any], args) -> None:
    value = getattr(args, "run_dir", None)
    if value is None:
        return
    run_dir = os.path.abspath(
        os.path.expandvars(os.path.expanduser(value))
    )
    experiment_name = os.path.basename(os.path.normpath(run_dir))
    if not experiment_name:
        raise ValueError("--run-dir must name a run below the filesystem root")
    cfg["paths"]["run_dir"], cfg["experiment_name"] = run_dir, experiment_name


def command_check_config(args) -> None:
    cfg = _load(args, allow_unconfirmed=args.allow_unconfirmed)
    effective_plan_hash = object_hash(
        {
            "model": cfg["model"],
            "objective": cfg["objective"],
            "training": cfg["training"],
        }
    )
    _print(
        {
            "valid": True,
            "config_hash": cfg["_config_hash"],
            "descriptor_schema_hash": cfg["_descriptor_schema_hash"],
            "effective_training_plan_hash": effective_plan_hash,
            "training_plan_paths": cfg.get("_training_plan_paths", []),
            "training_plan_file_hashes": cfg.get("_training_plan_file_hashes", []),
            "model_architecture": cfg["model"].get("architecture", "vgae"),
            "max_steps": cfg["training"].get("max_steps"),
            "sources": cfg["paths"]["sources"],
        }
    )


def command_prepare(args) -> None:
    from .preprocess import prepare_tasks

    _print(prepare_tasks(_load(args)))


def command_verify_inputs(args) -> None:
    from .preprocess import verify_sources

    _print(verify_sources(_load(args)))


def command_canonicalize(args) -> None:
    from .preprocess import canonicalize_task

    _print(canonicalize_task(_load(args), _task_value(args.task, "SLURM_ARRAY_TASK_ID")))


def command_deduplicate(args) -> None:
    from .deduplicate import deduplicate_bucket

    threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    _print(deduplicate_bucket(_load(args), _task_value(args.bucket, "SLURM_ARRAY_TASK_ID"), threads))


def command_finalize_data(args) -> None:
    from .deduplicate import finalize_dataset

    _print(finalize_dataset(_load(args)))


def command_fit_scaler(args) -> None:
    from .deduplicate import fit_train_scaler

    _print(fit_train_scaler(_load(args), args.batch_size))


def command_featurize(args) -> None:
    from .graph_shards import featurize_bucket

    _print(featurize_bucket(_load(args), _task_value(args.bucket, "SLURM_ARRAY_TASK_ID")))


def command_finalize_graphs(args) -> None:
    from .graph_shards import finalize_graphs

    _print(finalize_graphs(_load(args)))


def command_train(args) -> None:
    import torch.distributed as dist

    from .train import train

    cfg = _load(args)
    _apply_run_directory(cfg, args)
    _apply_training_budgets(cfg, args)
    try:
        status = train(cfg, allow_cpu=args.allow_cpu)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    raise SystemExit(status)


def command_benchmark_training(args) -> None:
    import torch.distributed as dist

    from .train import train

    cfg = _load(args)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    _apply_run_directory(cfg, args)
    cfg["experiment_name"] = f"performance-benchmark-{args.steps}-steps"
    cfg["training"]["max_steps"] = int(args.steps)
    cfg["training"]["checkpoint_every_steps"] = int(args.steps)
    cfg["training"]["validate_every_steps"] = int(args.steps) + 1
    cfg["training"]["resume"] = "never"
    _apply_training_budgets(cfg, args)
    try:
        status = train(cfg, allow_cpu=args.allow_cpu)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    raise SystemExit(status)


def command_evaluate(args) -> None:
    import torch.distributed as dist

    from .train import evaluate_saved

    cfg = _load(args)
    _apply_run_directory(cfg, args)
    try:
        evaluate_saved(
            cfg,
            args.checkpoint,
            args.split,
            allow_cpu=args.allow_cpu,
            max_graphs=args.max_graphs,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def command_embed(args) -> None:
    import torch.distributed as dist

    from .representations import export_embeddings

    cfg = _load(args)
    _apply_run_directory(cfg, args)
    try:
        _print(
            export_embeddings(
                cfg,
                checkpoint_name=args.checkpoint,
                split=args.split,
                max_graphs=args.max_graphs,
                skip_graphs=args.skip_graphs,
                output=args.output,
                embedding_definition=args.embedding_definition,
                mean_node_weight=args.mean_node_weight,
                calibrator=args.calibrator,
                sampling_seed=args.sampling_seed,
                allow_cpu=args.allow_cpu,
            )
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def command_probe(args) -> None:
    from .probes import run_representation_probes

    cfg = _load(args)
    _print(
        run_representation_probes(
            train_embeddings=args.train_embeddings,
            validation_embeddings=args.validation_embeddings,
            work_dir=cfg["paths"]["work_dir"],
            output=args.output,
            similarity_graphs=args.similarity_graphs,
            seed=int(cfg["seed"]),
        )
    )


def command_reweight_embeddings(args) -> None:
    from .representations import reweight_hybrid_embeddings

    _print(
        reweight_hybrid_embeddings(
            args.input,
            args.output,
            mean_node_weight=args.mean_node_weight,
        )
    )


def command_fit_embedding_calibrator(args) -> None:
    from .representations import fit_embedding_calibrator

    _print(
        fit_embedding_calibrator(
            args.embeddings,
            args.output,
            minimum_graphs=args.minimum_graphs,
        )
    )


def command_benchmark_downstream(args) -> None:
    from .downstream import benchmark_moleculenet

    cfg = _load(args)
    _apply_run_directory(cfg, args)
    _print(
        benchmark_moleculenet(
            cfg,
            checkpoint_name=args.checkpoint,
            datasets_dir=args.datasets_dir,
            output=args.output,
            scaffold_splits=args.scaffold_splits,
            dataset_names=args.datasets,
            embedding_definition=args.embedding_definition,
            calibrator=args.calibrator,
            selected_only=args.selected_only,
            allow_cpu=args.allow_cpu,
        )
    )


def command_audit_downstream_overlap(args) -> None:
    from .downstream_audit import audit_pretraining_overlap

    _print(
        audit_pretraining_overlap(
            _load(args),
            datasets_dir=args.datasets_dir,
            output=args.output,
            summary_csv=args.summary_csv,
            dataset_names=args.datasets,
        )
    )


def command_benchmark_descriptor_control(args) -> None:
    from .downstream_audit import benchmark_descriptor_control

    _print(
        benchmark_descriptor_control(
            _load(args),
            datasets_dir=args.datasets_dir,
            reference_benchmark=args.reference_benchmark,
            output=args.output,
            summary_csv=args.summary_csv,
            dataset_names=args.datasets,
        )
    )


def command_audit_training_exposure(args) -> None:
    from .exposure import audit_training_exposure

    cfg = _load(args)
    _apply_run_directory(cfg, args)
    _print(
        audit_training_exposure(
            cfg,
            checkpoint_names=args.checkpoints,
            output=args.output,
            summary_csv=args.summary_csv,
        )
    )


def command_audit_downstream_exposure(args) -> None:
    from .downstream_exposure import audit_downstream_checkpoint_exposure

    cfg = _load(args)
    _apply_run_directory(cfg, args)
    _print(
        audit_downstream_checkpoint_exposure(
            cfg,
            checkpoint_names=args.checkpoints,
            datasets_dir=args.datasets_dir,
            output=args.output,
            summary_csv=args.summary_csv,
            identity_ledger_csv=args.identity_ledger_csv,
            dataset_names=args.datasets,
            workers=args.workers,
        )
    )


def command_promote_representation(args) -> None:
    from .representations import promote_representation_checkpoint

    cfg = _load(args)
    _apply_run_directory(cfg, args)
    _print(
        promote_representation_checkpoint(
            cfg,
            checkpoint_name=args.checkpoint,
            calibrator=args.calibrator,
            representation_probe=args.representation_probe,
            downstream_benchmark=args.downstream_benchmark,
            destination_name=args.destination,
            calibrator_destination_name=args.calibrator_destination,
        )
    )


def command_verify_environment(args) -> None:
    from .util import runtime_versions

    versions = runtime_versions()
    result = {
        "architecture": platform.machine(),
        "versions": versions,
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            result["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        pass
    _print(result)
    if args.require_cuda and not result["cuda_available"]:
        raise SystemExit("CUDA verification failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gmolai-retrain")
    parser.add_argument("--config", default="configs/retrain.yaml", help="Retraining YAML configuration")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-config")
    check.add_argument("--allow-unconfirmed", action="store_true")
    check.add_argument(
        "--plan",
        action="append",
        help="Repeatable training-only YAML overlay; later plans win",
    )
    check.set_defaults(handler=command_check_config)

    commands.add_parser("prepare").set_defaults(handler=command_prepare)
    commands.add_parser("verify-inputs").set_defaults(handler=command_verify_inputs)

    canonical = commands.add_parser("canonicalize")
    canonical.add_argument("--task", type=int)
    canonical.set_defaults(handler=command_canonicalize)

    deduplicate = commands.add_parser("deduplicate")
    deduplicate.add_argument("--bucket", type=int)
    deduplicate.add_argument("--threads", type=int)
    deduplicate.set_defaults(handler=command_deduplicate)

    commands.add_parser("finalize-data").set_defaults(handler=command_finalize_data)
    scaler = commands.add_parser("fit-scaler")
    scaler.add_argument("--batch-size", type=int, default=262144)
    scaler.set_defaults(handler=command_fit_scaler)

    featurize = commands.add_parser("featurize")
    featurize.add_argument("--bucket", type=int)
    featurize.set_defaults(handler=command_featurize)
    commands.add_parser("finalize-graphs").set_defaults(handler=command_finalize_graphs)

    training = commands.add_parser("train")
    training.add_argument(
        "--plan",
        action="append",
        help="Repeatable training-only YAML overlay; later plans win and graph identity is preserved",
    )
    training.add_argument("--run-dir")
    training.add_argument("--allow-cpu", action="store_true")
    training.add_argument("--node-budget", type=int)
    training.add_argument("--graph-budget", type=int)
    training.set_defaults(handler=command_train)

    benchmark = commands.add_parser("benchmark-training")
    benchmark.add_argument(
        "--plan",
        action="append",
        help="Repeatable training-only YAML overlay; later plans win and graph identity is preserved",
    )
    benchmark.add_argument("--steps", type=int, default=400)
    benchmark.add_argument("--run-dir", required=True)
    benchmark.add_argument("--allow-cpu", action="store_true")
    benchmark.add_argument("--node-budget", type=int)
    benchmark.add_argument("--graph-budget", type=int)
    benchmark.set_defaults(handler=command_benchmark_training)

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument(
        "--plan", action="append", help="Repeatable plan for runs without resolved_config.json"
    )
    evaluation.add_argument("--run-dir")
    evaluation.add_argument("--checkpoint", default="best.pt")
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--max-graphs", type=int)
    evaluation.add_argument("--allow-cpu", action="store_true")
    evaluation.set_defaults(handler=command_evaluate)

    embedding = commands.add_parser("embed")
    embedding.add_argument("--run-dir")
    embedding.add_argument(
        "--checkpoint",
        default="auto",
        help=(
            "Checkpoint path; auto requires representation-best.pt for representation "
            "runs and uses best.pt only for legacy runs"
        ),
    )
    embedding.add_argument("--split", choices=("train", "validation", "test"), default="test")
    embedding.add_argument("--max-graphs", type=int)
    embedding.add_argument(
        "--skip-graphs",
        type=int,
        default=0,
        help=(
            "Start at this offset in the deterministic stratified population; "
            "consecutive windows with the same seed are non-overlapping"
        ),
    )
    embedding.add_argument("--output")
    embedding.add_argument(
        "--embedding-definition",
        choices=(
            "auto",
            "graph_z",
            "mean_node_z",
            "projector_z",
            "hybrid",
            "raw_hybrid",
            "standardized_raw_hybrid",
        ),
        default="auto",
        help=(
            "Molecule-vector definition; auto selects the calibrated hybrid for "
            "representation checkpoints and the legacy mean-posterior vector for v4; "
            "projector_z and raw_hybrid are explicit diagnostic ablations"
        ),
    )
    embedding.add_argument(
        "--mean-node-weight",
        type=float,
        default=3.0,
        help=(
            "Relative mean_node_z block weight for hybrid and "
            "standardized_raw_hybrid embeddings"
        ),
    )
    embedding.add_argument(
        "--calibrator",
        help="Train-only calibration artifact required by standardized_raw_hybrid",
    )
    embedding.add_argument(
        "--sampling-seed",
        type=int,
        help=(
            "Independent deterministic seed for finite split sampling; defaults "
            "to the dataset seed"
        ),
    )
    embedding.add_argument("--allow-cpu", action="store_true")
    embedding.set_defaults(handler=command_embed)

    reweight = commands.add_parser("reweight-embeddings")
    reweight.add_argument("--input", required=True)
    reweight.add_argument("--output", required=True)
    reweight.add_argument("--mean-node-weight", type=float, required=True)
    reweight.set_defaults(handler=command_reweight_embeddings)

    calibrator = commands.add_parser("fit-embedding-calibrator")
    calibrator.add_argument("--embeddings", required=True)
    calibrator.add_argument("--output", required=True)
    calibrator.add_argument("--minimum-graphs", type=int, default=10000)
    calibrator.set_defaults(handler=command_fit_embedding_calibrator)

    probe = commands.add_parser("probe")
    probe.add_argument("--train-embeddings", required=True)
    probe.add_argument("--validation-embeddings", required=True)
    probe.add_argument("--output", required=True)
    probe.add_argument("--similarity-graphs", type=int, default=2000)
    probe.set_defaults(handler=command_probe)

    downstream = commands.add_parser("benchmark-downstream")
    downstream.add_argument("--run-dir", required=True)
    downstream.add_argument("--checkpoint", default="best.pt")
    downstream.add_argument("--datasets-dir", required=True)
    downstream.add_argument("--output", required=True)
    downstream.add_argument("--scaffold-splits", type=int, default=5)
    downstream.add_argument(
        "--embedding-definition",
        choices=("auto", "graph_z", "mean_node_z", "hybrid", "raw_hybrid", "standardized_raw_hybrid"),
        default="auto",
    )
    downstream.add_argument("--calibrator")
    downstream.add_argument(
        "--selected-only",
        action="store_true",
        help="Evaluate only molecule_embedding, omitting diagnostic feature baselines",
    )
    downstream.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset name to evaluate; repeat to select a subset (default: all)",
    )
    downstream.add_argument("--allow-cpu", action="store_true")
    downstream.set_defaults(handler=command_benchmark_downstream)

    overlap = commands.add_parser("audit-downstream-overlap")
    overlap.add_argument(
        "--plan",
        action="append",
        help="Repeatable recorded training-plan overlay used only to restore artifact identity",
    )
    overlap.add_argument("--datasets-dir", required=True)
    overlap.add_argument("--output", required=True)
    overlap.add_argument("--summary-csv", required=True)
    overlap.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset name to audit; repeat to select a subset (default: all six)",
    )
    overlap.set_defaults(handler=command_audit_downstream_overlap)

    descriptor_control = commands.add_parser("benchmark-descriptor-control")
    descriptor_control.add_argument(
        "--plan",
        action="append",
        help="Repeatable recorded training-plan overlay used only to restore artifact identity",
    )
    descriptor_control.add_argument("--datasets-dir", required=True)
    descriptor_control.add_argument("--reference-benchmark", required=True)
    descriptor_control.add_argument("--output", required=True)
    descriptor_control.add_argument("--summary-csv", required=True)
    descriptor_control.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset name to evaluate; repeat to select a subset (default: development panel)",
    )
    descriptor_control.set_defaults(handler=command_benchmark_descriptor_control)

    exposure = commands.add_parser("audit-training-exposure")
    exposure.add_argument(
        "--plan",
        action="append",
        help="Repeatable recorded training-plan overlay used only to restore artifact identity",
    )
    exposure.add_argument("--run-dir", required=True)
    exposure.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="Checkpoint path relative to run-dir; repeat for multiple steps",
    )
    exposure.add_argument("--output", required=True)
    exposure.add_argument("--summary-csv", required=True)
    exposure.set_defaults(handler=command_audit_training_exposure)

    downstream_exposure = commands.add_parser("audit-downstream-exposure")
    downstream_exposure.add_argument(
        "--plan",
        action="append",
        help="Repeatable recorded training-plan overlay used only to restore artifact identity",
    )
    downstream_exposure.add_argument("--run-dir", required=True)
    downstream_exposure.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="Checkpoint path relative to run-dir; repeat in any order",
    )
    downstream_exposure.add_argument("--datasets-dir", required=True)
    downstream_exposure.add_argument("--output", required=True)
    downstream_exposure.add_argument("--summary-csv", required=True)
    downstream_exposure.add_argument("--identity-ledger-csv", required=True)
    downstream_exposure.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset name to audit; repeat to select a subset (default: all six)",
    )
    downstream_exposure.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent metadata-only graph-shard readers",
    )
    downstream_exposure.set_defaults(handler=command_audit_downstream_exposure)

    promotion = commands.add_parser("promote-representation")
    promotion.add_argument("--run-dir", required=True)
    promotion.add_argument("--checkpoint", required=True)
    promotion.add_argument("--calibrator", required=True)
    promotion.add_argument("--representation-probe", required=True)
    promotion.add_argument("--downstream-benchmark", required=True)
    promotion.add_argument("--destination", default="representation-best.pt")
    promotion.add_argument(
        "--calibrator-destination", default="representation-calibrator.pt"
    )
    promotion.set_defaults(handler=command_promote_representation)

    environment = commands.add_parser("verify-environment")
    environment.add_argument("--require-cuda", action="store_true")
    environment.set_defaults(handler=command_verify_environment)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()

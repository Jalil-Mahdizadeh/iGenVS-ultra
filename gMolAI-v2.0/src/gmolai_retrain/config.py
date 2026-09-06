from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    pass


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must contain a YAML mapping")
    return _expand(value)


def load_config(path: str | Path, require_descriptor_confirmation: bool = True) -> dict[str, Any]:
    config_path = Path(path).resolve()
    cfg = load_yaml(config_path)
    if cfg.get("schema_version") != 1:
        raise ConfigurationError("Only retraining configuration schema_version=1 is supported")

    paths = cfg.get("paths", {})
    root_setting = paths.get("project_root", "..")
    project_root = (config_path.parent / root_setting).resolve()
    paths["project_root"] = str(project_root)
    for key in ("work_dir", "run_dir", "descriptor_manifest"):
        candidate = Path(paths[key])
        paths[key] = str(candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve())
    for source in paths.get("sources", []):
        candidate = Path(source["path"])
        source["path"] = str(candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve())
    cfg["paths"] = paths
    cfg["_config_path"] = str(config_path)

    descriptors = load_yaml(paths["descriptor_manifest"])
    _validate_descriptor_manifest(cfg, descriptors, require_descriptor_confirmation)
    cfg["_descriptors"] = descriptors
    cfg["_descriptor_schema_hash"] = object_hash(descriptors)

    hashable = copy.deepcopy(cfg)
    hashable.pop("_config_path", None)
    hashable.pop("_config_hash", None)
    cfg["_config_hash"] = object_hash(hashable)
    _validate_config(cfg)
    return cfg


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def apply_training_plan(cfg: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Overlay model/training choices without changing immutable data identity.

    Graph shards were built under ``_config_hash`` and are intentionally
    reusable across architecture ablations.  The overlaid sections are still
    captured by the checkpoint's independent training-plan hash.
    """
    plan_path = Path(path).resolve()
    plan = load_yaml(plan_path)
    if plan.get("schema_version") != 1:
        raise ConfigurationError("Training plans must use schema_version=1")
    allowed = {"schema_version", "model", "objective", "training"}
    unknown = sorted(set(plan) - allowed)
    if unknown:
        raise ConfigurationError(f"Unsupported training-plan keys: {', '.join(unknown)}")
    if not any(key in plan for key in ("model", "objective", "training")):
        raise ConfigurationError("Training plan does not override model, objective, or training")
    for section in ("model", "objective", "training"):
        if section in plan:
            if not isinstance(plan[section], dict):
                raise ConfigurationError(f"Training-plan {section} must be a mapping")
            _deep_update(cfg[section], plan[section])
    cfg["_training_plan_path"] = str(plan_path)
    cfg["_training_plan_file_hash"] = object_hash(plan)
    cfg.setdefault("_training_plan_paths", []).append(str(plan_path))
    cfg.setdefault("_training_plan_file_hashes", []).append(object_hash(plan))
    _validate_training_plan(cfg)
    return cfg


def _validate_descriptor_manifest(
    cfg: dict[str, Any], manifest: dict[str, Any], require_confirmation: bool
) -> None:
    if manifest.get("schema_version") != 1:
        raise ConfigurationError("Descriptor manifest must use schema_version=1")
    features = manifest.get("features")
    columns = cfg.get("data", {}).get("descriptor_columns", [])
    if not isinstance(features, list) or [str(f.get("column")) for f in features] != [str(c) for c in columns]:
        raise ConfigurationError("Descriptor manifest columns and order must exactly match data.descriptor_columns")
    names = [str(f.get("name", "")) for f in features]
    if len(set(names)) != len(names):
        raise ConfigurationError("Descriptor names must be unique")
    if require_confirmation and not manifest.get("confirmed_identical_across_sources", False):
        raise ConfigurationError(
            "Descriptor contract is unconfirmed. Complete configs/descriptors.yaml and set "
            "confirmed_identical_across_sources: true only after verifying ZINC/PubChem semantics, order, and units."
        )
    if require_confirmation:
        unresolved = []
        for index, feature in enumerate(features):
            for field in ("name", "unit", "generator"):
                value = str(feature.get(field, "")).strip()
                if not value or "UNKNOWN" in value.upper():
                    unresolved.append(f"features[{index}].{field}")
        note = str(manifest.get("confirmation_note", "")).strip()
        if not note or "REPLACE" in note.upper():
            unresolved.append("confirmation_note")
        if unresolved:
            raise ConfigurationError(
                "Descriptor manifest is marked confirmed but still has unresolved fields: "
                + ", ".join(unresolved)
            )


def _validate_config(cfg: dict[str, Any]) -> None:
    sources = cfg["paths"].get("sources", [])
    if len(sources) < 1:
        raise ConfigurationError("At least one source is required")
    names = [source["name"] for source in sources]
    if len(set(names)) != len(names):
        raise ConfigurationError("Source names must be unique")
    buckets = int(cfg["data"]["hash_buckets"])
    if buckets < 2 or buckets > 4096:
        raise ConfigurationError("data.hash_buckets must be between 2 and 4096")
    split = cfg["data"]["split"]
    total = sum(float(split[key]) for key in ("train_fraction", "validation_fraction", "test_fraction"))
    if abs(total - 1.0) > 1e-12:
        raise ConfigurationError(f"Split fractions must sum to 1.0, got {total}")
    if cfg["data"]["deduplication"].get("descriptor_conflict_policy") != "exclude":
        raise ConfigurationError("Only the safe descriptor_conflict_policy=exclude is implemented")
    _validate_training_plan(cfg)


def _validate_training_plan(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    objective = cfg["objective"]
    training = cfg["training"]
    architecture = str(model.get("architecture", "vgae"))
    if architecture not in {"vgae", "masked_graph_vicreg"}:
        raise ConfigurationError(f"Unsupported model architecture: {architecture}")
    for name in ("hidden_dim", "gine_layers"):
        if int(model[name]) <= 0:
            raise ConfigurationError(f"model.{name} must be positive")
    # Data-preparation-only configurations intentionally leave these sections
    # empty; training validates the complete overlaid plan below.
    if not objective and not training:
        return
    for name in (
        "node_mask_probability",
        "bond_feature_mask_probability",
        "bond_dropout_probability",
    ):
        value = float(objective[name])
        if not 0.0 <= value <= 1.0:
            raise ConfigurationError(f"objective.{name} must lie in [0, 1]")
    if architecture == "masked_graph_vicreg":
        for name in ("node_latent_dim", "graph_latent_dim"):
            if int(model[name]) <= 1:
                raise ConfigurationError(f"model.{name} must exceed one")
        if bool(model.get("vicreg_projector", False)) and int(
            model.get("vicreg_projector_dim", model["graph_latent_dim"])
        ) <= 1:
            raise ConfigurationError("model.vicreg_projector_dim must exceed one")
        for name in ("invariance_weight", "variance_weight", "covariance_weight"):
            if float(objective[name]) < 0:
                raise ConfigurationError(f"objective.{name} must be non-negative")
        if float(objective.get("contrastive_weight", 0.0)) < 0:
            raise ConfigurationError("objective.contrastive_weight must be non-negative")
        if float(objective.get("contrastive_temperature", 0.1)) <= 0:
            raise ConfigurationError("objective.contrastive_temperature must be positive")
        contrastive_space = str(objective.get("contrastive_space", "graph_z"))
        if contrastive_space not in {"graph_z", "projector", "mean_node_z"}:
            raise ConfigurationError(
                "objective.contrastive_space must be graph_z, projector, or mean_node_z"
            )
        if contrastive_space == "projector" and not bool(
            model.get("vicreg_projector", False)
        ):
            raise ConfigurationError(
                "objective.contrastive_space=projector requires model.vicreg_projector=true"
            )
        if contrastive_space == "mean_node_z" and bool(
            objective.get("contrastive_detach_node_gradient", False)
        ):
            raise ConfigurationError(
                "mean_node_z contrastive learning cannot detach node gradients"
            )
        invariance_space = str(objective.get("invariance_space", "graph_z"))
        if invariance_space not in {"graph_z", "projector"}:
            raise ConfigurationError(
                "objective.invariance_space must be graph_z or projector"
            )
        if invariance_space == "projector" and not bool(
            model.get("vicreg_projector", False)
        ):
            raise ConfigurationError(
                "objective.invariance_space=projector requires model.vicreg_projector=true"
            )
        if float(objective["variance_target"]) <= 0:
            raise ConfigurationError("objective.variance_target must be positive")
    if int(training["max_steps"]) <= 0:
        raise ConfigurationError("training.max_steps must be positive")
    if int(training.get("retain_every_steps", 0)) < 0:
        raise ConfigurationError("training.retain_every_steps must be non-negative")
    if "seed" in training and int(training["seed"]) < 0:
        raise ConfigurationError("training.seed must be non-negative")
    initialization_path = training.get("initialize_from_checkpoint")
    initialization_hash = training.get("initialize_from_sha256")
    if (initialization_path is None) != (initialization_hash is None):
        raise ConfigurationError(
            "training.initialize_from_checkpoint and initialize_from_sha256 must be set together"
        )
    if initialization_path is not None:
        if not Path(str(initialization_path)).is_absolute():
            raise ConfigurationError(
                "training.initialize_from_checkpoint must be an absolute path"
            )
        digest = str(initialization_hash).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ConfigurationError("training.initialize_from_sha256 must be a SHA-256 digest")
        if str(training.get("resume", "auto")) != "auto":
            raise ConfigurationError(
                "Warm-start runs require training.resume=auto for safe Slurm restart"
            )


def descriptor_names(cfg: dict[str, Any]) -> list[str]:
    return [str(item["name"]) for item in cfg["_descriptors"]["features"]]


def public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    value.pop("_descriptors", None)
    return value

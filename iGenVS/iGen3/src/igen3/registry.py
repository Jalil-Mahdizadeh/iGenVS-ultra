"""Model registry for the packaged iGen3 checkpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def default_model_root() -> Path:
    """Return the directory containing packaged model artifacts."""
    env_root = os.environ.get("IGEN3_MODEL_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd_models = Path.cwd() / "models"
    if cwd_models.exists():
        return cwd_models.resolve()

    repo_models = Path(__file__).resolve().parents[2] / "models"
    return repo_models.resolve()


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata required to load and sample from one iGen3 checkpoint."""

    model_id: str
    family: str
    stereochemistry: str
    description: str
    seq_len: int
    weights_name: str
    vocab_name: str = "vocab.pkl"
    d_model: int = 256
    n_head: int = 8
    num_layers: int = 6
    dropout: float = 0.10
    default_de_novo_temperature: float = 1.0
    default_derivative_temperature: float = 1.5
    default_top_k: int = 64

    @property
    def is_isomeric(self) -> bool:
        return self.stereochemistry == "isomeric"

    def artifact_dir(self, model_root: Path | None = None) -> Path:
        return (model_root or default_model_root()) / self.model_id.replace("-", "_")

    def weights_path(self, model_root: Path | None = None) -> Path:
        return self.artifact_dir(model_root) / self.weights_name

    def vocab_path(self, model_root: Path | None = None) -> Path:
        return self.artifact_dir(model_root) / self.vocab_name

    def default_temperature(self, mode: str) -> float:
        if mode == "derivative":
            return self.default_derivative_temperature
        return self.default_de_novo_temperature


MODEL_SPECS: dict[str, ModelSpec] = {
    "base-isomeric": ModelSpec(
        model_id="base-isomeric",
        family="base",
        stereochemistry="isomeric",
        description="Base transformer generator trained for isomeric SMILES.",
        seq_len=122,
        weights_name="iGen3_base_isomeric_256d.pth",
        default_de_novo_temperature=1.0,
        default_derivative_temperature=1.5,
    ),
    "base-nonisomeric": ModelSpec(
        model_id="base-nonisomeric",
        family="base",
        stereochemistry="nonisomeric",
        description="Base transformer generator trained for non-isomeric SMILES.",
        seq_len=112,
        weights_name="iGen3_base_nonisomeric_256d.pth",
        default_de_novo_temperature=1.0,
        default_derivative_temperature=1.5,
    ),
    "rl-isomeric": ModelSpec(
        model_id="rl-isomeric",
        family="rl",
        stereochemistry="isomeric",
        description="Reinforcement-learning tuned generator for isomeric SMILES.",
        seq_len=122,
        weights_name="iGen3_rl_qed_sa_isomeric_256d.pth",
        default_de_novo_temperature=1.2,
        default_derivative_temperature=2.0,
    ),
    "rl-nonisomeric": ModelSpec(
        model_id="rl-nonisomeric",
        family="rl",
        stereochemistry="nonisomeric",
        description="Reinforcement-learning tuned generator for non-isomeric SMILES.",
        seq_len=112,
        weights_name="iGen3_rl_qed_sa_nonisomeric_256d.pth",
        default_de_novo_temperature=1.2,
        default_derivative_temperature=2.0,
    ),
}

MODEL_ALIASES = {
    "base_iso": "base-isomeric",
    "base-iso": "base-isomeric",
    "base_isomeric": "base-isomeric",
    "base_noiso": "base-nonisomeric",
    "base-noiso": "base-nonisomeric",
    "base_nonisomeric": "base-nonisomeric",
    "rl_iso": "rl-isomeric",
    "rl-iso": "rl-isomeric",
    "rl_isomeric": "rl-isomeric",
    "rl_noiso": "rl-nonisomeric",
    "rl-noiso": "rl-nonisomeric",
    "rl_nonisomeric": "rl-nonisomeric",
}


def resolve_model(model_name: str) -> ModelSpec:
    normalized = model_name.strip().lower()
    normalized = MODEL_ALIASES.get(normalized, normalized)
    try:
        return MODEL_SPECS[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{model_name}'. Valid models: {valid}") from exc

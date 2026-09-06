"""iGen3 SMILES generation package."""

from .registry import MODEL_SPECS, ModelSpec, resolve_model

__all__ = ["MODEL_SPECS", "ModelSpec", "resolve_model"]

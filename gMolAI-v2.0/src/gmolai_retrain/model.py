from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool


class ResidualGINEEncoder(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int,
        latent: int,
        layers: int,
        dropout: float,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(node_in, hidden)
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
            )
            self.convolutions.append(GINEConv(mlp, edge_dim=edge_in, train_eps=True))
            self.normalizations.append(nn.LayerNorm(hidden))
        self.mu_head = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)
        self.dropout = float(dropout)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.input_projection(x)
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            update = convolution(hidden, edge_index, edge_attr)
            hidden = normalization(hidden + F.dropout(update, self.dropout, self.training))
            hidden = F.silu(hidden)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden).clamp(self.logvar_min, self.logvar_max)
        if sample:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar


def symmetric_pair_features(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    source, destination = edge_index
    return torch.cat((torch.abs(z[source] - z[destination]), z[source] * z[destination]), dim=-1)


class SymmetricEdgeDecoder(nn.Module):
    def __init__(self, latent: int, hidden: int, output: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        value = self.net(symmetric_pair_features(z, edge_index))
        return value.squeeze(-1) if value.shape[-1] == 1 else value


class MolecularVGAE(nn.Module):
    def __init__(self, feature_schema: dict[str, Any], descriptor_count: int, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(model_cfg["hidden_dim"])
        latent = int(model_cfg["latent_dim"])
        self.node_target_dim = int(feature_schema["node_target_dim"])
        self.edge_dim = int(feature_schema["edge_dim"])
        self.feature_schema = feature_schema
        self.encoder = ResidualGINEEncoder(
            int(feature_schema["node_input_dim"]),
            self.edge_dim,
            hidden,
            latent,
            int(model_cfg["gine_layers"]),
            float(model_cfg["dropout"]),
            float(model_cfg["logvar_min"]),
            float(model_cfg["logvar_max"]),
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, self.node_target_dim)
        )
        self.edge_existence_decoder = SymmetricEdgeDecoder(latent, hidden, 1)
        self.edge_feature_decoder = SymmetricEdgeDecoder(latent, hidden, self.edge_dim)
        self.descriptor_predictor = nn.Sequential(
            nn.Linear(3 * latent + 1, hidden * 2),
            nn.SiLU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, descriptor_count),
        )
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(x, edge_index, edge_attr, sample=sample)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        edge_existence_index: torch.Tensor | None = None,
        edge_feature_index: torch.Tensor | None = None,
        *,
        sample: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        z, mu, logvar = self.encode(x, edge_index, edge_attr, sample=sample)
        edge_existence_logits = (
            self.edge_existence_decoder(z, edge_existence_index)
            if edge_existence_index is not None
            else None
        )
        edge_feature_logits = (
            self.edge_feature_decoder(z, edge_feature_index)
            if edge_feature_index is not None
            else None
        )
        return (
            z,
            mu,
            logvar,
            self.node_decoder(z),
            # Descriptor supervision defines the deterministic molecular
            # representation, so train and evaluate this head from the
            # posterior mean. Feeding sampled ``z`` here lets the nonlinear
            # max pool infer properties from posterior noise/variance; at
            # evaluation time ``sample=False`` then presents a completely
            # different input distribution and discards that learned signal.
            self.predict_descriptors(mu, batch),
            edge_existence_logits,
            edge_feature_logits,
        )

    def predict_descriptors(self, z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean = global_mean_pool(z, batch)
        total = global_add_pool(z, batch)
        maximum = global_max_pool(z, batch)
        count = torch.bincount(batch, minlength=mean.shape[0]).to(z.dtype).log1p().unsqueeze(-1)
        return self.descriptor_predictor(torch.cat((mean, total, maximum, count), dim=-1))


class DeterministicGINEEncoder(nn.Module):
    """A deterministic atom encoder for transferable molecular representations."""

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int,
        latent: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(node_in, hidden)
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
            )
            self.convolutions.append(GINEConv(mlp, edge_dim=edge_in, train_eps=True))
            self.normalizations.append(nn.LayerNorm(hidden))
        self.output_projection = nn.Linear(hidden, latent)
        self.output_normalization = nn.LayerNorm(latent)
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(x)
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            update = convolution(hidden, edge_index, edge_attr)
            hidden = normalization(hidden + F.dropout(update, self.dropout, self.training))
            hidden = F.silu(hidden)
        return self.output_normalization(self.output_projection(hidden))


class GraphConditionedEdgeDecoder(nn.Module):
    """Decode an unordered atom pair while forcing use of its graph embedding."""

    def __init__(self, node_latent: int, graph_latent: int, hidden: int, output: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_latent + graph_latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output),
        )

    def forward(
        self,
        node_z: torch.Tensor,
        graph_z: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        graph_for_pair = graph_z[batch[edge_index[0]]]
        value = self.net(torch.cat((symmetric_pair_features(node_z, edge_index), graph_for_pair), dim=-1))
        return value.squeeze(-1) if value.shape[-1] == 1 else value


class MolecularRepresentationModel(nn.Module):
    """Masked graph autoencoder with an explicit deterministic molecule vector.

    Every auxiliary decoder is conditioned on ``graph_z``. This makes the
    exported vector part of the reconstruction path rather than an incidental
    pooling of atom latents used only by the descriptor head.
    """

    def __init__(self, feature_schema: dict[str, Any], descriptor_count: int, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(model_cfg["hidden_dim"])
        node_latent = int(model_cfg.get("node_latent_dim", model_cfg.get("latent_dim", hidden)))
        graph_latent = int(model_cfg.get("graph_latent_dim", model_cfg.get("latent_dim", hidden)))
        dropout = float(model_cfg["dropout"])
        self.node_target_dim = int(feature_schema["node_target_dim"])
        self.edge_dim = int(feature_schema["edge_dim"])
        self.node_latent_dim = node_latent
        self.graph_latent_dim = graph_latent
        self.latent_dim = graph_latent
        self.feature_schema = feature_schema
        self.encoder = DeterministicGINEEncoder(
            int(feature_schema["node_input_dim"]),
            self.edge_dim,
            hidden,
            node_latent,
            int(model_cfg["gine_layers"]),
            dropout,
        )
        self.graph_readout = nn.Sequential(
            nn.Linear(3 * node_latent + 1, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, graph_latent),
            nn.LayerNorm(graph_latent),
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(node_latent + graph_latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.node_target_dim),
        )
        self.edge_existence_decoder = GraphConditionedEdgeDecoder(
            node_latent, graph_latent, hidden, 1
        )
        self.edge_feature_decoder = GraphConditionedEdgeDecoder(
            node_latent, graph_latent, hidden, self.edge_dim
        )
        self.descriptor_predictor = nn.Sequential(
            nn.Linear(graph_latent, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, descriptor_count),
        )
        self.vicreg_projector = None
        if bool(model_cfg.get("vicreg_projector", False)):
            projector_dim = int(model_cfg.get("vicreg_projector_dim", graph_latent))
            self.vicreg_projector = nn.Sequential(
                nn.Linear(graph_latent, hidden * 2),
                nn.LayerNorm(hidden * 2),
                nn.SiLU(),
                nn.Linear(hidden * 2, projector_dim),
            )

    def _pool(self, node_z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean = global_mean_pool(node_z, batch)
        total = global_add_pool(node_z, batch)
        maximum = global_max_pool(node_z, batch)
        count = torch.bincount(batch, minlength=mean.shape[0]).to(node_z.dtype).clamp_min(1)
        normalized_total = total / count.sqrt().unsqueeze(-1)
        size = count.log1p().unsqueeze(-1)
        return self.graph_readout(torch.cat((mean, normalized_total, maximum, size), dim=-1))

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic atom and molecule embeddings."""
        node_z = self.encoder(x, edge_index, edge_attr)
        return node_z, self._pool(node_z, batch)

    def combine_molecule_embedding(
        self,
        node_z: torch.Tensor,
        graph_z: torch.Tensor,
        batch: torch.Tensor,
        *,
        mean_node_weight: float = 3.0,
    ) -> torch.Tensor:
        """Combine already encoded graph and atom blocks."""
        if mean_node_weight <= 0:
            raise ValueError("mean_node_weight must be positive")
        mean_node_z = global_mean_pool(node_z, batch)
        return torch.cat(
            (
                F.normalize(graph_z.float(), dim=-1),
                float(mean_node_weight) * F.normalize(mean_node_z.float(), dim=-1),
            ),
            dim=-1,
        )

    def combine_raw_molecule_embedding(
        self,
        node_z: torch.Tensor,
        graph_z: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate raw graph and mean-atom blocks before train calibration."""
        return torch.cat(
            (graph_z.float(), global_mean_pool(node_z, batch).float()), dim=-1
        )

    @staticmethod
    def apply_molecule_calibration(
        raw_embedding: torch.Tensor,
        coordinate_mean: torch.Tensor,
        coordinate_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Apply immutable train-split coordinate statistics to raw vectors."""
        if (
            coordinate_mean.ndim != 1
            or coordinate_scale.ndim != 1
            or coordinate_mean.shape != coordinate_scale.shape
            or coordinate_mean.shape[0] != raw_embedding.shape[-1]
            or not torch.isfinite(coordinate_mean).all()
            or not torch.isfinite(coordinate_scale).all()
            or bool(torch.any(coordinate_scale <= 0))
        ):
            raise ValueError("Invalid molecule calibration tensors")
        return (raw_embedding.float() - coordinate_mean) / coordinate_scale

    def molecule_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        *,
        mean_node_weight: float = 3.0,
    ) -> torch.Tensor:
        """Return the legacy unit-block hybrid molecule vector.

        The unit-normalized graph block carries globally supervised and
        reconstruction-conditioned information.  The weighted mean-atom block
        preserves smooth local chemical geometry for clustering/retrieval.
        """
        node_z, graph_z = self.encode(x, edge_index, edge_attr, batch)
        return self.combine_molecule_embedding(
            node_z,
            graph_z,
            batch,
            mean_node_weight=mean_node_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        edge_existence_index: torch.Tensor | None = None,
        edge_feature_index: torch.Tensor | None = None,
        *,
        view2_x: torch.Tensor | None = None,
        view2_edge_index: torch.Tensor | None = None,
        view2_edge_attr: torch.Tensor | None = None,
        contrastive_detach_node: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        node_z, graph_z = self.encode(x, edge_index, edge_attr, batch)
        mean_node_z = global_mean_pool(node_z, batch)
        graph_for_node = graph_z[batch]
        graph_z2 = None
        node_z2 = None
        if view2_x is not None:
            if view2_edge_index is None or view2_edge_attr is None:
                raise ValueError("The second corrupted view requires x, edge_index, and edge_attr")
            node_z2, graph_z2 = self.encode(
                view2_x, view2_edge_index, view2_edge_attr, batch
            )
        mean_node_z2 = (
            global_mean_pool(node_z2, batch) if node_z2 is not None else None
        )
        regularization_z = (
            self.vicreg_projector(graph_z)
            if self.vicreg_projector is not None
            else graph_z
        )
        regularization_z2 = (
            self.vicreg_projector(graph_z2)
            if self.vicreg_projector is not None and graph_z2 is not None
            else graph_z2
        )
        contrastive_z = regularization_z
        contrastive_z2 = regularization_z2
        if contrastive_detach_node:
            if node_z2 is None:
                raise ValueError("Detached-node contrastive output requires a second view")
            detached_graph_z = self._pool(node_z.detach(), batch)
            detached_graph_z2 = self._pool(node_z2.detach(), batch)
            contrastive_z = (
                self.vicreg_projector(detached_graph_z)
                if self.vicreg_projector is not None
                else detached_graph_z
            )
            contrastive_z2 = (
                self.vicreg_projector(detached_graph_z2)
                if self.vicreg_projector is not None
                else detached_graph_z2
            )
        return {
            "node_z": node_z,
            "mean_node_z": mean_node_z,
            "mean_node_z2": mean_node_z2,
            "graph_z": graph_z,
            "graph_z2": graph_z2,
            "regularization_z": regularization_z,
            "regularization_z2": regularization_z2,
            "contrastive_z": contrastive_z,
            "contrastive_z2": contrastive_z2,
            "node_logits": self.node_decoder(torch.cat((node_z, graph_for_node), dim=-1)),
            "descriptor_prediction": self.descriptor_predictor(graph_z),
            "descriptor_prediction2": (
                self.descriptor_predictor(graph_z2) if graph_z2 is not None else None
            ),
            "existence_logits": (
                self.edge_existence_decoder(node_z, graph_z, batch, edge_existence_index)
                if edge_existence_index is not None
                else None
            ),
            "edge_logits": (
                self.edge_feature_decoder(node_z, graph_z, batch, edge_feature_index)
                if edge_feature_index is not None
                else None
            ),
        }


def vicreg_terms(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    variance_target: float = 1.0,
    epsilon: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return invariance, variance-floor, and covariance-redundancy losses."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("VICReg inputs must be equally shaped [graphs, features] tensors")
    invariance = F.mse_loss(first, second)
    variance_penalties = []
    covariance_penalties = []
    for value in (first, second):
        centered = value - value.mean(dim=0)
        standard_deviation = torch.sqrt(centered.square().mean(dim=0) + epsilon)
        variance_penalties.append(F.relu(float(variance_target) - standard_deviation).mean())
        denominator = max(1, value.shape[0] - 1)
        covariance = centered.T @ centered / denominator
        off_diagonal = (
            covariance.flatten()[:-1]
            .view(covariance.shape[0] - 1, covariance.shape[0] + 1)[:, 1:]
            .flatten()
        )
        covariance_penalties.append(off_diagonal.square().sum() / covariance.shape[0])
    variance = torch.stack(variance_penalties).mean()
    covariance = torch.stack(covariance_penalties).mean()
    return invariance, variance, covariance


def nt_xent_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric cross-view InfoNCE loss for graph embeddings."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("NT-Xent inputs must be equally shaped [graphs, features] tensors")
    if temperature <= 0:
        raise ValueError("NT-Xent temperature must be positive")
    if first.shape[0] == 0:
        raise ValueError("NT-Xent requires at least one graph")
    first_normalized = F.normalize(first.float(), dim=-1)
    second_normalized = F.normalize(second.float(), dim=-1)
    logits = first_normalized @ second_normalized.T / float(temperature)
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


@dataclass
class CorruptedGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_mask: torch.Tensor
    unique_positive_edge_index: torch.Tensor
    unique_positive_edge_attr: torch.Tensor
    edge_target_mask: torch.Tensor
    edge_drop_mask: torch.Tensor


def corrupt_graph_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    node_probability: float,
    edge_feature_probability: float,
    edge_dropout_probability: float,
    generator: torch.Generator,
) -> CorruptedGraph:
    node_mask = torch.rand(x.shape[0], device=x.device, generator=generator) < node_probability
    if node_probability > 0 and x.shape[0]:
        # Preserve the original fallback semantics without synchronizing CUDA
        # merely to convert ``node_mask.any()`` into a Python boolean.
        node_mask[0] |= ~node_mask.any()
    corrupted_x = x.clone()
    corrupted_x[node_mask] = 0

    source, destination = edge_index
    pair_key = torch.minimum(source, destination) * x.shape[0] + torch.maximum(source, destination)
    unique_keys, inverse = torch.unique(pair_key, sorted=True, return_inverse=True)
    unique_count = unique_keys.numel()
    unique_drop = torch.rand(unique_count, device=x.device, generator=generator) < edge_dropout_probability
    unique_feature_mask = (
        torch.rand(unique_count, device=x.device, generator=generator) < edge_feature_probability
    )
    if unique_count and edge_dropout_probability > 0:
        unique_drop[0] |= ~unique_drop.any()
    if unique_count and edge_feature_probability > 0:
        unique_feature_mask[0] |= ~unique_feature_mask.any()
    directed_drop = unique_drop[inverse]
    directed_feature_mask = unique_feature_mask[inverse]
    keep = ~directed_drop
    corrupted_edge_index = edge_index[:, keep]
    corrupted_edge_attr = edge_attr[keep].clone()
    corrupted_edge_attr[directed_feature_mask[keep]] = 0

    canonical_directed = source < destination
    positive_edge_index = edge_index[:, canonical_directed]
    positive_edge_attr = edge_attr[canonical_directed]
    canonical_inverse = inverse[canonical_directed]
    edge_target_mask = unique_drop[canonical_inverse] | unique_feature_mask[canonical_inverse]
    edge_drop_mask = unique_drop[canonical_inverse]
    return CorruptedGraph(
        x=corrupted_x,
        edge_index=corrupted_edge_index,
        edge_attr=corrupted_edge_attr,
        node_mask=node_mask,
        unique_positive_edge_index=positive_edge_index,
        unique_positive_edge_attr=positive_edge_attr,
        edge_target_mask=edge_target_mask,
        edge_drop_mask=edge_drop_mask,
    )


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1)).mean()


def grouped_feature_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    groups: list[dict[str, Any]],
) -> torch.Tensor:
    """Use categorical CE for one-hot groups and BCE for binary features."""
    losses = []
    offset = 0
    for group in groups:
        if group["kind"] == "binary":
            losses.append(
                F.binary_cross_entropy_with_logits(logits[:, offset], target[:, offset])
            )
            offset += 1
        elif group["kind"] == "one_hot":
            width = len(group["values"]) + int(group.get("other", False))
            losses.append(
                F.cross_entropy(logits[:, offset : offset + width], target[:, offset : offset + width].argmax(dim=-1))
            )
            offset += width
        else:
            raise ValueError(f"Unsupported feature group kind: {group['kind']}")
    if offset != logits.shape[1] or target.shape[1] != logits.shape[1]:
        raise ValueError("Feature group widths do not match decoder tensors")
    return torch.stack(losses).mean()

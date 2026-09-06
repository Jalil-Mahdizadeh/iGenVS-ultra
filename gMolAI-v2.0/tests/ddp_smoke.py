"""Run directly with: torchrun --standalone --nproc_per_node=2 tests/ddp_smoke.py"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Batch, Data

from gmolai_retrain.model import MolecularRepresentationModel, MolecularVGAE
from gmolai_retrain.schema import feature_schema
from gmolai_retrain.train import _losses_for_batch, _sample_negative_candidates


def valid_features(groups, rows):
    width = sum(
        1 if group["kind"] == "binary" else len(group["values"]) + int(group["other"])
        for group in groups
    )
    value = torch.zeros((rows, width))
    offset = 0
    for group in groups:
        if group["kind"] == "binary":
            offset += 1
        else:
            value[:, offset] = 1
            offset += len(group["values"]) + int(group["other"])
    return value


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(7)
    schema = feature_schema(True, 0)
    model = MolecularVGAE(
        schema,
        13,
        {
            "hidden_dim": 16,
            "latent_dim": 8,
            "gine_layers": 2,
            "dropout": 0.0,
            "logvar_min": -10.0,
            "logvar_max": 6.0,
        },
    )
    model.to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = Batch.from_data_list(
        [
            Data(
                x=valid_features(schema["node_groups"], 6),
                edge_index=torch.tensor(
                    [
                        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
                    ]
                ),
                edge_attr=valid_features(schema["edge_groups"], 10),
                y=torch.zeros((1, 13)),
                graph_id=torch.tensor([100 + rank]),
            )
        ]
    )
    cfg = {
        "seed": 42,
        "objective": {
            "node_mask_probability": 0.15,
            "bond_feature_mask_probability": 0.15,
            "bond_dropout_probability": 0.15,
            "easy_negative_ratio": 1.0,
            "hard_negative_ratio": 1.0,
            "hard_pool_ratio": 5.0,
            "node_weight": 1.0,
            "edge_existence_weight": 1.0,
            "edge_feature_weight": 1.0,
            "descriptor_weight": 1.0,
            "kl_beta_max": 0.125,
            "kl_warmup_steps": 2,
        },
    }
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prepared_batch = batch.clone()
        candidates = _sample_negative_candidates(
            prepared_batch,
            cfg,
            step=step,
            rank=rank,
            training=True,
        )
        if device.type == "cuda":
            prepared_batch = prepared_batch.pin_memory()
            candidates = candidates.pin_memory()
        losses = _losses_for_batch(
            model,
            prepared_batch,
            device,
            cfg,
            step=step,
            rank=rank,
            training=True,
            candidates=candidates,
        )
        losses["total"].backward()
        missing = [name for name, parameter in model.module.named_parameters() if parameter.grad is None]
        if missing:
            raise RuntimeError(f"DDP parameters without gradients: {missing}")
        optimizer.step()
    checksum = torch.stack([parameter.detach().double().sum() for parameter in model.parameters()]).sum()
    checksums = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(checksums, checksum)
    if not all(torch.allclose(checksums[0], value, atol=1e-12, rtol=0) for value in checksums[1:]):
        raise RuntimeError(f"DDP parameters diverged: {checksums}")

    representation_cfg = {
        "seed": 42,
        "model": {
            "architecture": "masked_graph_vicreg",
            "hidden_dim": 16,
            "node_latent_dim": 8,
            "graph_latent_dim": 8,
            "vicreg_projector": True,
            "vicreg_projector_dim": 8,
            "gine_layers": 2,
            "dropout": 0.0,
        },
        "objective": {
            "node_mask_probability": 0.30,
            "bond_feature_mask_probability": 0.30,
            "bond_dropout_probability": 0.15,
            "easy_negative_ratio": 1.0,
            "hard_negative_ratio": 1.0,
            "hard_pool_ratio": 5.0,
            "node_weight": 1.0,
            "edge_existence_weight": 0.5,
            "edge_feature_weight": 1.0,
            "descriptor_weight": 0.25,
            "invariance_weight": 1.0,
            "variance_weight": 1.0,
            "covariance_weight": 0.04,
            "variance_target": 1.0,
            "contrastive_space": "projector",
            "contrastive_detach_node_gradient": True,
            "contrastive_weight": 0.1,
            "contrastive_temperature": 0.1,
        },
    }
    representation = MolecularRepresentationModel(
        schema, 13, representation_cfg["model"]
    ).to(device)
    representation = DistributedDataParallel(
        representation,
        device_ids=[local_rank] if device.type == "cuda" else None,
    )
    representation_optimizer = torch.optim.AdamW(representation.parameters(), lr=1e-3)
    for step in range(2):
        representation_optimizer.zero_grad(set_to_none=True)
        prepared_batch = batch.clone()
        candidates = _sample_negative_candidates(
            prepared_batch,
            representation_cfg,
            step=step,
            rank=rank,
            training=True,
        )
        if device.type == "cuda":
            prepared_batch = prepared_batch.pin_memory()
            candidates = candidates.pin_memory()
        losses = _losses_for_batch(
            representation,
            prepared_batch,
            device,
            representation_cfg,
            step=step,
            rank=rank,
            training=True,
            candidates=candidates,
        )
        losses["total"].backward()
        missing = [
            name
            for name, parameter in representation.module.named_parameters()
            if parameter.grad is None
        ]
        if missing:
            raise RuntimeError(f"Representation DDP parameters without gradients: {missing}")
        representation_optimizer.step()
    checksum = torch.stack(
        [parameter.detach().double().sum() for parameter in representation.parameters()]
    ).sum()
    checksums = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(checksums, checksum)
    if not all(torch.allclose(checksums[0], value, atol=1e-12, rtol=0) for value in checksums[1:]):
        raise RuntimeError(f"Representation DDP parameters diverged: {checksums}")
    if rank == 0:
        print(
            f"{dist.get_world_size()}-rank legacy and representation DDP smoke tests ({device.type}): OK",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

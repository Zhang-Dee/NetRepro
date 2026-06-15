from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import negative_sampling

from .model import ModelConfig, NetRepro


@dataclass
class ChemicalFineTuneConfig:
    fingerprint_dim: int = 2048
    fingerprint_hidden_dim: int = 128
    projected_node_dim: int = 32
    lambda_perturbation: float = 0.1
    negative_ratio: float = 1.0


class NetReproChemicalExtension(nn.Module):
    """
    Zero-shot extension from chemical features to shared perturbation embeddings.

    This module mirrors the logic in the original Chem_FT prototype while reusing
    the cleaned NetRepro backbone. The backbone is typically loaded from a trained
    NetRepro checkpoint, then frozen during chemical fine-tuning.
    """

    def __init__(
        self,
        backbone_config: ModelConfig,
        chemical_config: ChemicalFineTuneConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = NetRepro(backbone_config)
        self.chemical_config = chemical_config or ChemicalFineTuneConfig()

        self.fingerprint_proj = nn.Sequential(
            nn.Linear(
                self.chemical_config.fingerprint_dim,
                self.chemical_config.fingerprint_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.chemical_config.fingerprint_hidden_dim,
                backbone_config.specific_dim,
            ),
        )
        self.perturbation_head = nn.Sequential(
            nn.Linear(
                backbone_config.specific_dim + backbone_config.shared_dim,
                64,
            ),
            nn.ReLU(),
            nn.Linear(64, backbone_config.shared_dim),
        )

        node_input_dim = (
            backbone_config.specific_dim
            + backbone_config.shared_dim
            + backbone_config.shared_dim
            + backbone_config.specific_dim
        )
        self.embedding_proj = nn.Sequential(
            nn.Linear(node_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(backbone_config.dropout),
            nn.Linear(64, self.chemical_config.projected_node_dim),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    @staticmethod
    def decode_edges(node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return (node_embeddings[src] * node_embeddings[dst]).sum(dim=-1)

    def predict_node_perturbation(
        self,
        fingerprints: torch.Tensor,
        dmso_shared_node_centroid: torch.Tensor,
    ) -> torch.Tensor:
        chemical = self.fingerprint_proj(fingerprints)
        num_genes = dmso_shared_node_centroid.size(0)

        chemical = chemical.unsqueeze(1).expand(-1, num_genes, -1)
        dmso_context = dmso_shared_node_centroid.unsqueeze(0).expand(
            fingerprints.size(0),
            -1,
            -1,
        )
        return self.perturbation_head(torch.cat([chemical, dmso_context], dim=-1))

    def build_projected_node_embeddings(
        self,
        dmso_specific_node_centroid: torch.Tensor,
        dmso_shared_node_centroid: torch.Tensor,
        node_perturbation: torch.Tensor,
        fingerprints: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_genes, _ = node_perturbation.shape
        chemical = self.fingerprint_proj(fingerprints).unsqueeze(1).expand(-1, num_genes, -1)
        base = torch.cat(
            [
                dmso_specific_node_centroid.unsqueeze(0).expand(batch_size, -1, -1),
                dmso_shared_node_centroid.unsqueeze(0).expand(batch_size, -1, -1),
            ],
            dim=-1,
        )
        decoder_input = torch.cat([base, node_perturbation, chemical], dim=-1)
        return self.embedding_proj(decoder_input.reshape(batch_size * num_genes, -1))


@torch.no_grad()
def compute_dmso_node_centroid(
    backbone: NetRepro,
    dmso_graphs: Batch | Data,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = dmso_graphs.to(device)
    if not hasattr(batch, "batch") or batch.batch is None:
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
    encoded = backbone.encode_branch(batch, domain="cell")

    shared_node = encoded["shared_node"]
    specific_node = encoded["specific_node"]
    num_graphs = int(batch.batch.max().item()) + 1
    num_genes = shared_node.size(0) // num_graphs

    shared_centroid = shared_node.view(num_graphs, num_genes, -1).mean(dim=0)
    specific_centroid = specific_node.view(num_graphs, num_genes, -1).mean(dim=0)
    return {
        "shared_node_centroid": shared_centroid,
        "specific_node_centroid": specific_centroid,
    }


def zero_shot_graph_reconstruction_loss(
    model: NetReproChemicalExtension,
    projected_node_embeddings: torch.Tensor,
    graph_batch: Batch | Data,
    negative_ratio: float = 1.0,
) -> torch.Tensor:
    positive_edge_index = graph_batch.edge_index
    num_pos = positive_edge_index.size(1)
    if num_pos == 0:
        return projected_node_embeddings.new_tensor(0.0)

    negative_edge_index = negative_sampling(
        edge_index=positive_edge_index,
        num_nodes=graph_batch.num_nodes,
        num_neg_samples=max(1, int(num_pos * negative_ratio)),
        method="sparse",
        force_undirected=True,
    )

    pos_scores = model.decode_edges(projected_node_embeddings, positive_edge_index)
    neg_scores = model.decode_edges(projected_node_embeddings, negative_edge_index)
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)

    return F.binary_cross_entropy_with_logits(
        torch.cat([pos_scores, neg_scores]),
        torch.cat([pos_labels, neg_labels]),
    )


def zero_shot_finetune_step(
    model: NetReproChemicalExtension,
    treated_graph_batch: Batch | Data,
    fingerprints: torch.Tensor,
    dmso_specific_node_centroid: torch.Tensor,
    dmso_shared_node_centroid: torch.Tensor,
    device: torch.device,
    lambda_perturbation: float = 0.1,
    negative_ratio: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    treated_graph_batch = treated_graph_batch.to(device)
    fingerprints = fingerprints.to(device)
    dmso_specific_node_centroid = dmso_specific_node_centroid.to(device)
    dmso_shared_node_centroid = dmso_shared_node_centroid.to(device)

    batch_size = int(treated_graph_batch.batch.max().item()) + 1
    num_genes = dmso_shared_node_centroid.size(0)

    treated_specific = model.backbone.cell_specific_encoder(treated_graph_batch)
    treated_shared = model.backbone.shared_encoder(treated_graph_batch)
    real_perturbation = treated_shared.view(batch_size, num_genes, -1) - dmso_shared_node_centroid.unsqueeze(0)

    predicted_perturbation = model.predict_node_perturbation(
        fingerprints,
        dmso_shared_node_centroid,
    )

    projected_node_embeddings = model.build_projected_node_embeddings(
        dmso_specific_node_centroid,
        dmso_shared_node_centroid,
        real_perturbation,
        fingerprints,
    )
    reconstruction_loss = zero_shot_graph_reconstruction_loss(
        model,
        projected_node_embeddings,
        treated_graph_batch,
        negative_ratio=negative_ratio,
    )

    perturbation_loss = F.mse_loss(
        predicted_perturbation.mean(dim=1),
        real_perturbation.mean(dim=1),
    )
    total_loss = reconstruction_loss + lambda_perturbation * perturbation_loss

    return total_loss, {
        "reconstruction_loss": reconstruction_loss.detach(),
        "perturbation_loss": perturbation_loss.detach(),
        "treated_specific": treated_specific.detach(),
        "real_perturbation": real_perturbation.detach(),
        "predicted_perturbation": predicted_perturbation.detach(),
    }


@torch.no_grad()
def validate_zero_shot_extension(
    model: NetReproChemicalExtension,
    treated_graph_batch: Batch | Data,
    fingerprints: torch.Tensor,
    dmso_specific_node_centroid: torch.Tensor,
    dmso_shared_node_centroid: torch.Tensor,
    device: torch.device,
    negative_ratio: float = 1.0,
) -> dict[str, float]:
    treated_graph_batch = treated_graph_batch.to(device)
    fingerprints = fingerprints.to(device)
    dmso_specific_node_centroid = dmso_specific_node_centroid.to(device)
    dmso_shared_node_centroid = dmso_shared_node_centroid.to(device)

    batch_size = int(treated_graph_batch.batch.max().item()) + 1
    num_genes = dmso_shared_node_centroid.size(0)
    treated_shared = model.backbone.shared_encoder(treated_graph_batch)
    real_perturbation = treated_shared.view(batch_size, num_genes, -1) - dmso_shared_node_centroid.unsqueeze(0)
    predicted_perturbation = model.predict_node_perturbation(
        fingerprints,
        dmso_shared_node_centroid,
    )

    real_projected = model.build_projected_node_embeddings(
        dmso_specific_node_centroid,
        dmso_shared_node_centroid,
        real_perturbation,
        fingerprints,
    )
    predicted_projected = model.build_projected_node_embeddings(
        dmso_specific_node_centroid,
        dmso_shared_node_centroid,
        predicted_perturbation,
        fingerprints,
    )

    real_reconstruction = zero_shot_graph_reconstruction_loss(
        model,
        real_projected,
        treated_graph_batch,
        negative_ratio=negative_ratio,
    )
    predicted_reconstruction = zero_shot_graph_reconstruction_loss(
        model,
        predicted_projected,
        treated_graph_batch,
        negative_ratio=negative_ratio,
    )

    cosine = F.cosine_similarity(
        real_perturbation.mean(dim=1),
        predicted_perturbation.mean(dim=1),
        dim=1,
    ).mean()

    return {
        "real_reconstruction_loss": float(real_reconstruction.item()),
        "predicted_reconstruction_loss": float(predicted_reconstruction.item()),
        "perturbation_cosine": float(cosine.item()),
    }

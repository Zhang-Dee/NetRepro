from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


@dataclass
class ReconstructionLossConfig:
    num_negative_ratio: float = 1.0


class GraphReconstructionLoss(nn.Module):
    """
    Graph reconstruction loss using positive and negative sampled edges.
    """

    def __init__(self, num_negative_ratio: float = 1.0) -> None:
        super().__init__()
        self.num_negative_ratio = num_negative_ratio

    def forward(
        self,
        decoder: nn.Module,
        node_embeddings: torch.Tensor,
        positive_edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        num_pos = positive_edge_index.size(1)
        num_neg = max(1, int(num_pos * self.num_negative_ratio))

        negative_edge_index = negative_sampling(
            edge_index=positive_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg,
            method="sparse",
        )

        pos_logits = decoder(node_embeddings, positive_edge_index)
        neg_logits = decoder(node_embeddings, negative_edge_index)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
        return pos_loss + neg_loss


class GraphTripletLoss(nn.Module):
    """
    Triplet loss on graph-level shared embeddings.
    Anchor: cancer tissue
    Positive: untreated cell line
    Negative: healthy tissue
    """

    def __init__(self, margin: float = 0.2, hard: bool = True) -> None:
        super().__init__()
        self.margin = margin
        self.hard = hard

    def forward(
        self,
        anchor_graph_emb: torch.Tensor,
        positive_graph_emb: torch.Tensor,
        negative_graph_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_dist = torch.cdist(anchor_graph_emb, positive_graph_emb, p=2)
        neg_dist = torch.cdist(anchor_graph_emb, negative_graph_emb, p=2)

        if self.hard:
            pos_dist = pos_dist.max(dim=1).values
            neg_dist = neg_dist.min(dim=1).values
        else:
            pos_dist = pos_dist.mean(dim=1)
            neg_dist = neg_dist.mean(dim=1)

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0).mean()
        satisfied = (pos_dist + self.margin) < neg_dist
        return loss, satisfied

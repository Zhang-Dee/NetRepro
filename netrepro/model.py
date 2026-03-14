from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch.autograd import Function
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


@dataclass
class ModelConfig:
    input_dim: int = 20
    shared_dim: int = 16
    specific_dim: int = 32
    shared_layers: int = 2
    specific_layers: int = 2
    gat_heads: int = 2
    dropout: float = 0.2
    discriminator_hidden_dim: int = 16


class GNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gnn_type: str,
        num_layers: int,
        dropout: float,
        gat_heads: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.gnn_type = gnn_type
        self.num_layers = num_layers

        hidden_dims = [input_dim]
        if num_layers == 1:
            hidden_dims.append(output_dim)
        elif num_layers == 2:
            hidden_dims.extend([64, output_dim])
        else:
            hidden_dims.extend([64, 48, output_dim])

        for i in range(num_layers):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]

            if gnn_type == "gat":
                is_last = i == num_layers - 1
                heads = 1 if is_last else gat_heads
                conv = GATConv(in_dim, out_dim, heads=heads, concat=True)
                norm_dim = out_dim * heads
            else:
                conv = GCNConv(in_dim, out_dim)
                norm_dim = out_dim

            self.layers.append(conv)
            self.norms.append(nn.BatchNorm1d(norm_dim))

        self.final_proj = None
        if gnn_type == "gat" and num_layers >= 1:
            final_out_dim = hidden_dims[-1]
            self.final_proj = nn.Linear(final_out_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x, edge_index)
            x = norm(x)
            if i < len(self.layers) - 1:
                x = torch.nn.functional.elu(x) if self.gnn_type == "gat" else torch.relu(x)
                x = self.dropout(x)

        if self.final_proj is not None and x.size(-1) != self.final_proj.out_features:
            x = self.final_proj(x)
        return x


class EdgeDecoder(nn.Module):
    """
    Graph reconstruction decoder.
    Reconstructs edge existence from concatenated shared + specific node embeddings.
    """

    def __init__(self, node_emb_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_features = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)
        logits = self.edge_mlp(edge_features).squeeze(-1)
        return logits


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NetRepro(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.shared_encoder = GNNEncoder(
            input_dim=config.input_dim,
            output_dim=config.shared_dim,
            gnn_type="gat",
            num_layers=config.shared_layers,
            dropout=config.dropout,
            gat_heads=config.gat_heads,
        )

        self.tissue_specific_encoder = GNNEncoder(
            input_dim=config.input_dim,
            output_dim=config.specific_dim,
            gnn_type="gcn",
            num_layers=config.specific_layers,
            dropout=config.dropout,
        )

        self.cell_specific_encoder = GNNEncoder(
            input_dim=config.input_dim,
            output_dim=config.specific_dim,
            gnn_type="gcn",
            num_layers=config.specific_layers,
            dropout=config.dropout,
        )

        combined_dim = config.shared_dim + config.specific_dim
        self.tissue_decoder = EdgeDecoder(combined_dim, hidden_dim=64, dropout=config.dropout)
        self.cell_decoder = EdgeDecoder(combined_dim, hidden_dim=64, dropout=config.dropout)

        self.discriminator = DomainDiscriminator(
            input_dim=config.shared_dim,
            hidden_dim=config.discriminator_hidden_dim,
            dropout=config.dropout,
        )

    @staticmethod
    def _graph_embedding(node_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return global_mean_pool(node_emb, batch)

    def encode_branch(self, data: Data, domain: str) -> Dict[str, torch.Tensor]:
        shared_node = self.shared_encoder(data)
        if domain == "tissue":
            specific_node = self.tissue_specific_encoder(data)
        elif domain == "cell":
            specific_node = self.cell_specific_encoder(data)
        else:
            raise ValueError(f"Unknown domain: {domain}")

        graph_shared = self._graph_embedding(shared_node, data.batch)
        combined_node = torch.cat([shared_node, specific_node], dim=-1)

        return {
            "shared_node": shared_node,
            "specific_node": specific_node,
            "shared_graph": graph_shared,
            "combined_node": combined_node,
        }

    def forward(
        self,
        tissue_normal: Data,
        tissue_cancer: Data,
        cell_treated: Data,
        cell_dmso: Data,
        alpha: float,
    ) -> Dict[str, torch.Tensor]:
        tissue_normal_out = self.encode_branch(tissue_normal, domain="tissue")
        tissue_cancer_out = self.encode_branch(tissue_cancer, domain="tissue")
        cell_dmso_out = self.encode_branch(cell_dmso, domain="cell")
        cell_treated_out = self.encode_branch(cell_treated, domain="cell")

        shared_graphs = torch.cat(
            [
                tissue_cancer_out["shared_graph"],
                tissue_normal_out["shared_graph"],
                cell_dmso_out["shared_graph"],
                cell_treated_out["shared_graph"],
            ],
            dim=0,
        )

        reversed_shared = GradientReversal.apply(shared_graphs, alpha)
        domain_logits = self.discriminator(reversed_shared)

        return {
            "tissue_normal": tissue_normal_out,
            "tissue_cancer": tissue_cancer_out,
            "cell_dmso": cell_dmso_out,
            "cell_treated": cell_treated_out,
            "domain_logits": domain_logits,
        }

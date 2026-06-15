from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data


@torch.no_grad()
def _shared_graph_embeddings(model, graphs: Batch | Data, device: torch.device) -> torch.Tensor:
    batch = graphs.to(device)
    if not hasattr(batch, "batch") or batch.batch is None:
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
    shared_node = model.shared_encoder(batch)
    return model._graph_embedding(shared_node, batch.batch)


@torch.no_grad()
def summarize_state_embeddings(model, graphs: Batch | Data, device: torch.device) -> torch.Tensor:
    """
    Aggregate one biological state into a single shared embedding.

    The manuscript defines perturbations between state-level shared embeddings,
    so we average graph-level embeddings from repeated sampled networks here.
    """

    graph_embeddings = _shared_graph_embeddings(model, graphs, device)
    return graph_embeddings.mean(dim=0)


@torch.no_grad()
def compute_reversion_score(
    model,
    tissue_normal_graphs: Batch | Data,
    tissue_cancer_graphs: Batch | Data,
    cell_dmso_graphs: Batch | Data,
    cell_treated_graphs: Batch | Data,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Compute the manuscript's network perturbation reversion score.

    A higher score should mean stronger reversal, so we negate the cosine
    similarity between disease and drug perturbation vectors.
    """

    tissue_normal = summarize_state_embeddings(model, tissue_normal_graphs, device)
    tissue_cancer = summarize_state_embeddings(model, tissue_cancer_graphs, device)
    cell_dmso = summarize_state_embeddings(model, cell_dmso_graphs, device)
    cell_treated = summarize_state_embeddings(model, cell_treated_graphs, device)

    disease_delta = tissue_cancer - tissue_normal
    drug_delta = cell_treated - cell_dmso

    raw_cosine = F.cosine_similarity(
        disease_delta.unsqueeze(0),
        drug_delta.unsqueeze(0),
        dim=1,
    ).squeeze(0)
    reversion_score = -raw_cosine

    return {
        "reversion_score": reversion_score,
        "raw_cosine_similarity": raw_cosine,
        "tissue_normal_embedding": tissue_normal,
        "tissue_cancer_embedding": tissue_cancer,
        "cell_dmso_embedding": cell_dmso,
        "cell_treated_embedding": cell_treated,
        "disease_delta": disease_delta,
        "drug_delta": drug_delta,
    }

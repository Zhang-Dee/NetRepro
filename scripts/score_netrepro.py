from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netrepro.data import GraphBuildConfig, PairedGraphDataset
from netrepro.io import align_gene_space, load_expression_matrix
from netrepro.model import ModelConfig, NetRepro
from netrepro.scoring import compute_reversion_score


def _build_dataset(
    normal_or_dmso: pd.DataFrame,
    cancer_or_treated: pd.DataFrame,
    domain_label: int,
    num_graphs: int,
    max_repeats: int,
    seed: int,
    graph_config: GraphBuildConfig,
) -> PairedGraphDataset:
    return PairedGraphDataset(
        df_a=normal_or_dmso,
        df_b=cancer_or_treated,
        domain_label=domain_label,
        num_graphs=num_graphs,
        group_size=20,
        max_repeats=(max_repeats, max_repeats),
        seed=seed,
        config=graph_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the NetRepro reversion score.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--tissue-normal", required=True, type=str)
    parser.add_argument("--tissue-cancer", required=True, type=str)
    parser.add_argument("--cell-dmso", required=True, type=str)
    parser.add_argument("--cell-treated", required=True, type=str)
    parser.add_argument(
        "--matrix-layout",
        choices=["auto", "genes_by_samples", "samples_by_genes"],
        default="auto",
    )
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--edge-threshold", type=float, default=0.8)
    parser.add_argument("--graph-method", type=str, choices=["correlation", "wgcna"], default="correlation")
    parser.add_argument("--wgcna-type", type=str, choices=["unsigned", "signed"], default="unsigned")
    parser.add_argument("--wgcna-power", type=int, default=6)
    parser.add_argument("--num-graphs", type=int, default=100)
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shared-dim", type=int, default=16)
    parser.add_argument("--specific-dim", type=int, default=32)
    parser.add_argument("--shared-layers", type=int, default=2)
    parser.add_argument("--specific-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--disc-hidden", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    load_results = [
        load_expression_matrix(args.tissue_normal, layout=args.matrix_layout, sep=args.sep),
        load_expression_matrix(args.tissue_cancer, layout=args.matrix_layout, sep=args.sep),
        load_expression_matrix(args.cell_dmso, layout=args.matrix_layout, sep=args.sep),
        load_expression_matrix(args.cell_treated, layout=args.matrix_layout, sep=args.sep),
    ]
    aligned = align_gene_space(*(result.matrix for result in load_results))
    tissue_normal, tissue_cancer, cell_dmso, cell_treated = aligned

    graph_cfg = GraphBuildConfig(
        edge_threshold=args.edge_threshold,
        graph_method=args.graph_method,
        wgcna_type=args.wgcna_type,
        wgcna_power=args.wgcna_power,
    )
    tissue_dataset = _build_dataset(
        tissue_normal,
        tissue_cancer,
        domain_label=0,
        num_graphs=args.num_graphs,
        max_repeats=args.max_repeats,
        seed=args.seed,
        graph_config=graph_cfg,
    )
    cell_dataset = _build_dataset(
        cell_dmso,
        cell_treated,
        domain_label=1,
        num_graphs=args.num_graphs,
        max_repeats=args.max_repeats,
        seed=args.seed,
        graph_config=graph_cfg,
    )

    tissue_normal_batch, tissue_cancer_batch = next(
        iter(DataLoader(tissue_dataset, batch_size=len(tissue_dataset), shuffle=False))
    )
    cell_dmso_batch, cell_treated_batch = next(
        iter(DataLoader(cell_dataset, batch_size=len(cell_dataset), shuffle=False))
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = NetRepro(
        ModelConfig(
            input_dim=20,
            shared_dim=args.shared_dim,
            specific_dim=args.specific_dim,
            shared_layers=args.shared_layers,
            specific_layers=args.specific_layers,
            gat_heads=args.gat_heads,
            dropout=args.dropout,
            discriminator_hidden_dim=args.disc_hidden,
        )
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    result = compute_reversion_score(
        model,
        tissue_normal_batch,
        tissue_cancer_batch,
        cell_dmso_batch,
        cell_treated_batch,
        device=device,
    )
    payload = {
        "reversion_score": float(result["reversion_score"].item()),
        "raw_cosine_similarity": float(result["raw_cosine_similarity"].item()),
        "num_genes": int(tissue_normal.shape[0]),
        "num_graphs": int(args.num_graphs),
    }
    print(payload)

    if args.output:
        pd.DataFrame([payload]).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

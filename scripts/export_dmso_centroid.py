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

from netrepro.chem import compute_dmso_node_centroid
from netrepro.data import GraphBuildConfig, PairedGraphDataset
from netrepro.io import load_expression_matrix
from netrepro.model import ModelConfig, NetRepro


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DMSO node centroids for the zero-shot extension.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--cell-dmso", required=True, type=str)
    parser.add_argument("--matrix-layout", choices=["auto", "genes_by_samples", "samples_by_genes"], default="auto")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--edge-threshold", type=float, default=0.8)
    parser.add_argument("--graph-method", type=str, choices=["correlation", "wgcna"], default="correlation")
    parser.add_argument("--wgcna-type", type=str, choices=["unsigned", "signed"], default="unsigned")
    parser.add_argument("--wgcna-power", type=int, default=6)
    parser.add_argument("--num-graphs", type=int, default=100)
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shared-dim", type=int, default=16)
    parser.add_argument("--specific-dim", type=int, default=32)
    parser.add_argument("--shared-layers", type=int, default=2)
    parser.add_argument("--specific-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--disc-hidden", type=int, default=16)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    dmso_result = load_expression_matrix(args.cell_dmso, layout=args.matrix_layout, sep=args.sep)
    graph_cfg = GraphBuildConfig(
        edge_threshold=args.edge_threshold,
        graph_method=args.graph_method,
        wgcna_type=args.wgcna_type,
        wgcna_power=args.wgcna_power,
    )
    dataset = PairedGraphDataset(
        df_a=dmso_result.matrix,
        df_b=dmso_result.matrix,
        domain_label=1,
        num_graphs=args.num_graphs,
        group_size=20,
        max_repeats=(args.max_repeats, args.max_repeats),
        seed=args.seed,
        config=graph_cfg,
    )
    dmso_batch, _ = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))

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

    centroids = compute_dmso_node_centroid(model, dmso_batch, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genes = dmso_result.matrix.index

    pd.DataFrame(
        centroids["shared_node_centroid"].cpu().numpy(),
        index=genes,
    ).to_csv(output_dir / "dmso_shared_node_centroid.csv")
    pd.DataFrame(
        centroids["specific_node_centroid"].cpu().numpy(),
        index=genes,
    ).to_csv(output_dir / "dmso_specific_node_centroid.csv")
    print(f"Saved centroids to {output_dir}")


if __name__ == "__main__":
    main()

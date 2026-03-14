from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from netrepro.data import GraphBuildConfig, PairedGraphDataset
from netrepro.losses import GraphReconstructionLoss, GraphTripletLoss
from netrepro.models import ModelConfig, NetRepro
from netrepro.train import TrainingConfig, fit


def load_expression_table(path: str) -> pd.DataFrame:
    """
    Expected format:
    rows = genes
    columns = samples
    index_col = 0
    """
    return pd.read_csv(path, index_col=0)


def build_loaders(args):
    graph_cfg = GraphBuildConfig(
        edge_threshold=args.edge_threshold,
        graph_method=args.graph_method,
        wgcna_type=args.wgcna_type,
        wgcna_power=args.wgcna_power,
        scale_each_matrix=args.scale_each_matrix,
    )

    tissue_train = PairedGraphDataset(
        df_a=load_expression_table(args.tissue_normal_train),
        df_b=load_expression_table(args.tissue_cancer_train),
        domain_label=0,
        num_graphs=args.num_graphs_train,
        group_size=20,
        max_repeats=(args.max_repeats, args.max_repeats),
        seed=args.seed,
        config=graph_cfg,
    )

    cell_train = PairedGraphDataset(
        df_a=load_expression_table(args.cell_dmso_train),
        df_b=load_expression_table(args.cell_treated_train),
        domain_label=1,
        num_graphs=args.num_graphs_train,
        group_size=20,
        max_repeats=(args.max_repeats, args.max_repeats),
        seed=args.seed,
        config=graph_cfg,
    )

    tissue_val = PairedGraphDataset(
        df_a=load_expression_table(args.tissue_normal_val),
        df_b=load_expression_table(args.tissue_cancer_val),
        domain_label=0,
        num_graphs=args.num_graphs_val,
        group_size=20,
        max_repeats=(args.max_repeats, args.max_repeats),
        seed=args.seed + 1,
        config=graph_cfg,
    )

    cell_val = PairedGraphDataset(
        df_a=load_expression_table(args.cell_dmso_val),
        df_b=load_expression_table(args.cell_treated_val),
        domain_label=1,
        num_graphs=args.num_graphs_val,
        group_size=20,
        max_repeats=(args.max_repeats, args.max_repeats),
        seed=args.seed + 1,
        config=graph_cfg,
    )

    train_tissue_loader = DataLoader(tissue_train, batch_size=args.batch_size, shuffle=True)
    train_cell_loader = DataLoader(cell_train, batch_size=args.batch_size, shuffle=True)
    val_tissue_loader = DataLoader(tissue_val, batch_size=args.batch_size, shuffle=False)
    val_cell_loader = DataLoader(cell_val, batch_size=args.batch_size, shuffle=False)

    return train_cell_loader, train_tissue_loader, val_cell_loader, val_tissue_loader


def main():
    parser = argparse.ArgumentParser(description="Train NetRepro.")
    parser.add_argument("--tissue-normal-train", type=str, required=True)
    parser.add_argument("--tissue-cancer-train", type=str, required=True)
    parser.add_argument("--cell-dmso-train", type=str, required=True)
    parser.add_argument("--cell-treated-train", type=str, required=True)

    parser.add_argument("--tissue-normal-val", type=str, required=True)
    parser.add_argument("--tissue-cancer-val", type=str, required=True)
    parser.add_argument("--cell-dmso-val", type=str, required=True)
    parser.add_argument("--cell-treated-val", type=str, required=True)

    parser.add_argument("--save-path", type=str, default="checkpoints/netrepro_best.pt")
    parser.add_argument("--graph-method", type=str, choices=["correlation", "wgcna"], default="correlation")
    parser.add_argument("--wgcna-type", type=str, choices=["unsigned", "signed"], default="unsigned")
    parser.add_argument("--wgcna-power", type=int, default=6)
    parser.add_argument("--edge-threshold", type=float, default=0.8)
    parser.add_argument("--scale-each-matrix", action="store_true")

    parser.add_argument("--num-graphs-train", type=int, default=1000)
    parser.add_argument("--num-graphs-val", type=int, default=200)
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--shared-dim", type=int, default=16)
    parser.add_argument("--specific-dim", type=int, default=32)
    parser.add_argument("--shared-layers", type=int, default=2)
    parser.add_argument("--specific-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--disc-hidden", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-recon", type=float, default=0.01)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=3)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cell_loader, train_tissue_loader, val_cell_loader, val_tissue_loader = build_loaders(args)

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

    criterion_recon = GraphReconstructionLoss(num_negative_ratio=1.0)
    criterion_adv = torch.nn.CrossEntropyLoss()
    criterion_trip = GraphTripletLoss(margin=args.triplet_margin, hard=True)

    fit(
        model=model,
        train_cell_loader=train_cell_loader,
        train_tissue_loader=train_tissue_loader,
        val_cell_loader=val_cell_loader,
        val_tissue_loader=val_tissue_loader,
        criterion_recon=criterion_recon,
        criterion_adv=criterion_adv,
        criterion_trip=criterion_trip,
        config=TrainingConfig(
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_recon=args.lambda_recon,
            warmup_epochs=args.warmup_epochs,
        ),
        device=device,
        save_path=Path(args.save_path),
    )


if __name__ == "__main__":
    main()

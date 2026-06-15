from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netrepro.chem import (
    ChemicalFineTuneConfig,
    NetReproChemicalExtension,
    validate_zero_shot_extension,
    zero_shot_finetune_step,
)
from netrepro.datasets import TreatedCompoundDataset, TreatedCompoundDatasetConfig
from netrepro.data import GraphBuildConfig
from netrepro.io import load_fingerprint_table
from netrepro.model import ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the chemistry zero-shot extension.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--treated-dir", required=True, type=str)
    parser.add_argument("--fingerprints", required=True, type=str)
    parser.add_argument("--dmso-shared-centroid", required=True, type=str)
    parser.add_argument("--dmso-specific-centroid", required=True, type=str)
    parser.add_argument("--matrix-layout", choices=["auto", "genes_by_samples", "samples_by_genes"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-perturbation", type=float, default=0.1)
    parser.add_argument("--fingerprint-dim", type=int, default=2048)
    parser.add_argument("--projected-node-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-path", required=True, type=str)
    parser.add_argument("--shared-dim", type=int, default=16)
    parser.add_argument("--specific-dim", type=int, default=32)
    parser.add_argument("--shared-layers", type=int, default=2)
    parser.add_argument("--specific-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--disc-hidden", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    fingerprint_table = load_fingerprint_table(args.fingerprints)
    dataset = TreatedCompoundDataset(
        treated_dir=args.treated_dir,
        fingerprint_table=fingerprint_table,
        config=TreatedCompoundDatasetConfig(
            matrix_layout=args.matrix_layout,
            graph_config=GraphBuildConfig(),
            seed=args.seed,
        ),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = NetReproChemicalExtension(
        ModelConfig(
            input_dim=20,
            shared_dim=args.shared_dim,
            specific_dim=args.specific_dim,
            shared_layers=args.shared_layers,
            specific_layers=args.specific_layers,
            gat_heads=args.gat_heads,
            dropout=args.dropout,
            discriminator_hidden_dim=args.disc_hidden,
        ),
        ChemicalFineTuneConfig(
            fingerprint_dim=args.fingerprint_dim,
            projected_node_dim=args.projected_node_dim,
            lambda_perturbation=args.lambda_perturbation,
        ),
    )
    backbone_state = torch.load(args.checkpoint, map_location=device)
    model.backbone.load_state_dict(backbone_state)
    model.freeze_backbone()
    model.to(device)

    dmso_shared = torch.tensor(__import__("pandas").read_csv(args.dmso_shared_centroid, index_col=0).values, dtype=torch.float32)
    dmso_specific = torch.tensor(__import__("pandas").read_csv(args.dmso_specific_centroid, index_col=0).values, dtype=torch.float32)

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = float("-inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for treated_graphs, fingerprints, _compound_ids in loader:
            optimizer.zero_grad()
            loss, _metrics = zero_shot_finetune_step(
                model,
                treated_graphs,
                fingerprints,
                dmso_specific,
                dmso_shared,
                device=device,
                lambda_perturbation=args.lambda_perturbation,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        validation_batch = next(iter(loader))
        val_metrics = validate_zero_shot_extension(
            model,
            validation_batch[0],
            validation_batch[1],
            dmso_specific,
            dmso_shared,
            device=device,
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={sum(losses) / max(1, len(losses)):.4f} | "
            f"val_real_recon={val_metrics['real_reconstruction_loss']:.4f} | "
            f"val_pred_recon={val_metrics['predicted_reconstruction_loss']:.4f} | "
            f"val_pert_cos={val_metrics['perturbation_cosine']:.4f}"
        )
        if val_metrics["perturbation_cosine"] > best_metric:
            best_metric = val_metrics["perturbation_cosine"]
            torch.save(model.state_dict(), save_path)

    print(f"Best zero-shot model saved to {save_path}")


if __name__ == "__main__":
    main()

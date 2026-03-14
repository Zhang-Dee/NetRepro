from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


@dataclass
class TrainingConfig:
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_recon: float = 0.01
    warmup_epochs: int = 3
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-3
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.8


class EarlyStopping:
    def __init__(self, patience: int, delta: float, save_path: str | Path) -> None:
        self.patience = patience
        self.delta = delta
        self.save_path = Path(save_path)
        self.best_loss = None
        self.counter = 0

    def __call__(self, value: float, model: nn.Module) -> bool:
        if self.best_loss is None or value < self.best_loss - self.delta:
            self.best_loss = value
            self.counter = 0
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            return False

        self.counter += 1
        return self.counter >= self.patience


def _make_alpha(epoch: int, num_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    p = float(epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-5.0 * p))) - 1.0)


def _domain_labels_from_batches(
    tissue_cancer,
    tissue_normal,
    cell_dmso,
    cell_treated,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            tissue_cancer.domain,
            tissue_normal.domain,
            cell_dmso.domain,
            cell_treated.domain,
        ],
        dim=0,
    ).to(device)


def _step(
    model: nn.Module,
    batch_cell,
    batch_tissue,
    criterion_recon,
    criterion_adv,
    criterion_trip,
    alpha: float,
    device: torch.device,
    lambda_recon: float,
    optimizer=None,
) -> Dict[str, float]:
    cell_dmso, cell_treated = batch_cell
    tissue_normal, tissue_cancer = batch_tissue

    cell_dmso = cell_dmso.to(device)
    cell_treated = cell_treated.to(device)
    tissue_normal = tissue_normal.to(device)
    tissue_cancer = tissue_cancer.to(device)

    domain_labels = _domain_labels_from_batches(
        tissue_cancer, tissue_normal, cell_dmso, cell_treated, device
    )

    outputs = model(
        tissue_normal=tissue_normal,
        tissue_cancer=tissue_cancer,
        cell_treated=cell_treated,
        cell_dmso=cell_dmso,
        alpha=alpha,
    )

    loss_recon = (
        criterion_recon(
            model.tissue_decoder,
            outputs["tissue_normal"]["combined_node"],
            tissue_normal.edge_index,
            tissue_normal.num_nodes,
        )
        + criterion_recon(
            model.tissue_decoder,
            outputs["tissue_cancer"]["combined_node"],
            tissue_cancer.edge_index,
            tissue_cancer.num_nodes,
        )
        + criterion_recon(
            model.cell_decoder,
            outputs["cell_dmso"]["combined_node"],
            cell_dmso.edge_index,
            cell_dmso.num_nodes,
        )
        + criterion_recon(
            model.cell_decoder,
            outputs["cell_treated"]["combined_node"],
            cell_treated.edge_index,
            cell_treated.num_nodes,
        )
    )

    loss_adv = criterion_adv(outputs["domain_logits"], domain_labels)

    loss_trip, triplet_ok = criterion_trip(
        outputs["tissue_cancer"]["shared_graph"],
        outputs["cell_dmso"]["shared_graph"],
        outputs["tissue_normal"]["shared_graph"],
    )

    total_loss = lambda_recon * loss_recon + loss_adv + loss_trip

    if optimizer is not None:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    preds = torch.argmax(outputs["domain_logits"], dim=1).detach().cpu()
    labels = domain_labels.detach().cpu()

    return {
        "loss": float(total_loss.item()),
        "loss_recon": float(loss_recon.item()),
        "loss_adv": float(loss_adv.item()),
        "loss_trip": float(loss_trip.item()),
        "preds": preds,
        "labels": labels,
        "triplet_acc": float(triplet_ok.float().mean().item()),
    }


def train_epoch(
    model: nn.Module,
    dataloader_cell,
    dataloader_tissue,
    optimizer,
    criterion_recon,
    criterion_adv,
    criterion_trip,
    alpha: float,
    device: torch.device,
    lambda_recon: float,
) -> Dict[str, float]:
    model.train()

    losses = []
    triplet_accs = []
    all_preds = []
    all_labels = []

    for batch_cell, batch_tissue in zip(dataloader_cell, dataloader_tissue):
        out = _step(
            model=model,
            batch_cell=batch_cell,
            batch_tissue=batch_tissue,
            criterion_recon=criterion_recon,
            criterion_adv=criterion_adv,
            criterion_trip=criterion_trip,
            alpha=alpha,
            device=device,
            lambda_recon=lambda_recon,
            optimizer=optimizer,
        )
        losses.append(out["loss"])
        triplet_accs.append(out["triplet_acc"])
        all_preds.append(out["preds"])
        all_labels.append(out["labels"])

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    return {
        "loss": float(sum(losses) / len(losses)),
        "domain_acc": float(accuracy_score(labels, preds)),
        "triplet_acc": float(sum(triplet_accs) / len(triplet_accs)),
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader_cell,
    dataloader_tissue,
    criterion_recon,
    criterion_adv,
    criterion_trip,
    alpha: float,
    device: torch.device,
    lambda_recon: float,
) -> Dict[str, float]:
    model.eval()

    losses = []
    triplet_accs = []
    all_preds = []
    all_labels = []

    for batch_cell, batch_tissue in zip(dataloader_cell, dataloader_tissue):
        out = _step(
            model=model,
            batch_cell=batch_cell,
            batch_tissue=batch_tissue,
            criterion_recon=criterion_recon,
            criterion_adv=criterion_adv,
            criterion_trip=criterion_trip,
            alpha=alpha,
            device=device,
            lambda_recon=lambda_recon,
            optimizer=None,
        )
        losses.append(out["loss"])
        triplet_accs.append(out["triplet_acc"])
        all_preds.append(out["preds"])
        all_labels.append(out["labels"])

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    return {
        "loss": float(sum(losses) / len(losses)),
        "domain_acc": float(accuracy_score(labels, preds)),
        "triplet_acc": float(sum(triplet_accs) / len(triplet_accs)),
    }


def fit(
    model: nn.Module,
    train_cell_loader,
    train_tissue_loader,
    val_cell_loader,
    val_tissue_loader,
    criterion_recon,
    criterion_adv,
    criterion_trip,
    config: TrainingConfig,
    device: torch.device,
    save_path: str | Path,
) -> None:
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        delta=config.early_stopping_delta,
        save_path=save_path,
    )

    for epoch in range(config.num_epochs):
        alpha = _make_alpha(epoch, config.num_epochs, config.warmup_epochs)

        train_metrics = train_epoch(
            model=model,
            dataloader_cell=train_cell_loader,
            dataloader_tissue=train_tissue_loader,
            optimizer=optimizer,
            criterion_recon=criterion_recon,
            criterion_adv=criterion_adv,
            criterion_trip=criterion_trip,
            alpha=alpha,
            device=device,
            lambda_recon=config.lambda_recon,
        )

        val_metrics = validate_epoch(
            model=model,
            dataloader_cell=val_cell_loader,
            dataloader_tissue=val_tissue_loader,
            criterion_recon=criterion_recon,
            criterion_adv=criterion_adv,
            criterion_trip=criterion_trip,
            alpha=alpha,
            device=device,
            lambda_recon=config.lambda_recon,
        )

        print(
            f"Epoch {epoch + 1:03d}/{config.num_epochs} | "
            f"alpha={alpha:.3f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"train_domain_acc={train_metrics['domain_acc']:.4f} | "
            f"val_domain_acc={val_metrics['domain_acc']:.4f} | "
            f"train_triplet_acc={train_metrics['triplet_acc']:.4f} | "
            f"val_triplet_acc={val_metrics['triplet_acc']:.4f}"
        )

        scheduler.step()

        if early_stopping(val_metrics["loss"], model):
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

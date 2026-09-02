"""
src/training/trainer.py
──────────────────────────────────────────────────────────────────────────────
MLflow-instrumented GNN training loop.

Key engineering decisions:
  • NeighborLoader    — mini-batch training on large graphs. Full-graph GNN
                        training on 6M+ edges would OOM on an 8GB GPU.
  • Focal Loss        — handles extreme class imbalance (fraud ≈ 0.13%)
  • Time-based split  — nodes are assigned to train/val/test based on the
                        maximum step (hour) of their transactions, not randomly
  • Gradient clipping — prevents exploding gradients in deep GNNs
  • LR scheduler      — ReduceLROnPlateau on validation PR-AUC
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.models.focal_loss import FocalLoss
from src.training.evaluate import evaluate_model


# ── Split masks ───────────────────────────────────────────────────────────────

def create_temporal_masks(
    data: Data,
    node_steps: torch.Tensor,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/val/test boolean masks based on time.

    node_steps: (N,) tensor with the max transaction step for each node.
    Earlier nodes → train, later → val/test.
    """
    n = data.num_nodes
    sorted_steps = node_steps.sort().values
    val_cutoff = float(sorted_steps[int(n * (1 - val_fraction - test_fraction))])
    test_cutoff = float(sorted_steps[int(n * (1 - test_fraction))])

    train_mask = node_steps <= val_cutoff
    val_mask = (node_steps > val_cutoff) & (node_steps <= test_cutoff)
    test_mask = node_steps > test_cutoff

    logger.info(
        f"Temporal split → train: {train_mask.sum():,} | "
        f"val: {val_mask.sum():,} | test: {test_mask.sum():,}"
    )
    return train_mask, val_mask, test_mask


# ── GNN Trainer ───────────────────────────────────────────────────────────────

class GNNTrainer:
    """
    Generic trainer for GCN / GraphSAGE / GAT models.

    Parameters
    ----------
    model          : The GNN model instance
    data           : PyG Data object (full graph)
    train_mask     : Boolean mask for training nodes
    val_mask       : Boolean mask for validation nodes
    device         : torch device ('cuda' or 'cpu')
    lr             : Learning rate
    weight_decay   : L2 regularisation
    focal_alpha    : Focal loss alpha (minority class weight)
    focal_gamma    : Focal loss gamma (focusing parameter)
    num_epochs     : Maximum training epochs
    patience       : Early stopping patience (on val PR-AUC)
    batch_size     : Nodes per NeighborLoader batch
    num_neighbors  : Neighbours to sample per layer [layer1, layer2, layer3]
    output_dir     : Where to save the best model checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        num_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        num_neighbors: Optional[list] = None,
        output_dir: Path = Path("outputs/checkpoints"),
    ) -> None:
        self.model = model.to(device)
        self.data = data
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if num_neighbors is None:
            num_neighbors = [25, 10, 5]  # 3 layers
        self.num_neighbors = num_neighbors

        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5,
            patience=7
        )

        # NeighborLoader for mini-batch training
        train_node_ids = train_mask.nonzero(as_tuple=True)[0]
        self.train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_node_ids,
            shuffle=True,
            num_workers=0,  # Windows compatibility (no fork)
        )

    def train_epoch(self) -> float:
        """Run one training epoch, return mean loss."""
        self.model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(batch.x, batch.edge_index)
            # Only compute loss on the "seed" nodes (first batch_size nodes)
            out = logits[:batch.batch_size]
            labels = batch.y[:batch.batch_size].float()

            loss = self.criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch.batch_size
            total_nodes += batch.batch_size

        return total_loss / total_nodes

    @torch.no_grad()
    def evaluate(self, mask: torch.Tensor) -> Tuple[float, np.ndarray]:
        """Mini-batch evaluation using NeighborLoader to prevent CUDA Out of Memory."""
        self.model.eval()
        mask_ids = mask.nonzero(as_tuple=True)[0]
        eval_loader = NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=1024,
            input_nodes=mask_ids,
            shuffle=False,
            num_workers=0,
        )
        probas_list = []
        y_true_list = []
        for batch in eval_loader:
            batch = batch.to(self.device)
            logits = self.model(batch.x, batch.edge_index)
            out = logits[:batch.batch_size]
            p = torch.sigmoid(out).cpu().numpy()
            probas_list.append(p)
            y_true_list.append(batch.y[:batch.batch_size].cpu().numpy())

        proba = np.concatenate(probas_list)
        y_true = np.concatenate(y_true_list)
        pr_auc = float(average_precision_score(y_true, proba))
        return pr_auc, proba

    def fit(self, experiment_name: str, model_name: str) -> Dict:
        """
        Full training loop with MLflow tracking and early stopping.

        Returns dict with best metrics and path to saved model.
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=model_name):
            # Log hyperparams
            mlflow.log_params({
                "model": model_name,
                "lr": self.optimizer.param_groups[0]["lr"],
                "num_epochs": self.num_epochs,
                "patience": self.patience,
            })

            best_val_pr_auc = 0.0
            epochs_no_improve = 0
            best_checkpoint = self.output_dir / f"{model_name}_best.pt"

            for epoch in range(1, self.num_epochs + 1):
                t0 = time.perf_counter()
                train_loss = self.train_epoch()
                val_pr_auc, _ = self.evaluate(self.val_mask)
                epoch_time = time.perf_counter() - t0

                self.scheduler.step(val_pr_auc)

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_pr_auc": val_pr_auc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }, step=epoch)

                logger.info(
                    f"Epoch {epoch:03d}/{self.num_epochs} | "
                    f"loss={train_loss:.4f} | val_PR-AUC={val_pr_auc:.4f} | "
                    f"time={epoch_time:.1f}s"
                )

                if val_pr_auc > best_val_pr_auc:
                    best_val_pr_auc = val_pr_auc
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), best_checkpoint)
                    logger.success(f"New best val PR-AUC: {best_val_pr_auc:.4f} → saved")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.warning(f"Early stopping at epoch {epoch}")
                        break

            # Load best and evaluate on test
            self.model.load_state_dict(torch.load(best_checkpoint, weights_only=True))
            _, test_proba = self.evaluate(torch.ones(self.data.num_nodes, dtype=torch.bool))

            # Compute final metrics
            test_metrics = evaluate_model(
                self.data.y.cpu().numpy(),
                test_proba,
                model_name=model_name,
            )
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_artifact(str(best_checkpoint))

            logger.success(f"[{model_name}] Test PR-AUC: {test_metrics['pr_auc']:.4f}")

        return {
            "model_name": model_name,
            "best_val_pr_auc": best_val_pr_auc,
            "test_metrics": test_metrics,
            "checkpoint": str(best_checkpoint),
            "test_proba": test_proba,
        }

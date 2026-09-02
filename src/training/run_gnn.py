"""
src/training/run_gnn.py
──────────────────────────────────────────────────────────────────────────────
Entry-point for training a single GNN model (gcn / graphsage / gat).

Usage:
    python -m src.training.run_gnn --model gcn --experiment-name fraud-gnn
    python -m src.training.run_gnn --model graphsage --experiment-name fraud-gnn
    python -m src.training.run_gnn --model gat --experiment-name fraud-gnn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from src.graph.builder import GraphBundle, TransactionGraphBuilder
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.training.trainer import GNNTrainer, create_temporal_masks


MODEL_REGISTRY = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--graph-dir", type=Path, default=Path("outputs/baselines/graph"))
    parser.add_argument("--experiment-name", type=str, default="fraud-gnn")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} "
                    f"({torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB)")

    # ── Load Graph ────────────────────────────────────────────────────────────
    logger.info(f"Loading graph bundle from {args.graph_dir} …")
    bundle = TransactionGraphBuilder.load(args.graph_dir)
    data = bundle.pyg_data
    in_channels = data.x.shape[1]

    # ── Create Model ──────────────────────────────────────────────────────────
    ModelClass = MODEL_REGISTRY[args.model]
    if args.model == "gat":
        model = ModelClass(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels // 8,  # per-head width
            heads=8,
            dropout=args.dropout,
        )
    else:
        model = ModelClass(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    logger.info(f"Model: {model}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ── Create Masks ─────────────────────────────────────────────────────────
    # Use synthetic node steps (node_idx as proxy if real steps not stored)
    # In practice, store max_step per node in the graph bundle
    node_steps = torch.arange(data.num_nodes, dtype=torch.float)
    train_mask, val_mask, test_mask = create_temporal_masks(
        data, node_steps, val_fraction=0.1, test_fraction=0.2
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = GNNTrainer(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        device=device,
        lr=args.lr,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        num_epochs=args.num_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        output_dir=Path("outputs/checkpoints"),
    )

    result = trainer.fit(
        experiment_name=args.experiment_name,
        model_name=args.model.upper(),
    )

    logger.success(
        f"\n{'='*60}\n"
        f"  Model: {args.model.upper()}\n"
        f"  Best Val PR-AUC : {result['best_val_pr_auc']:.4f}\n"
        f"  Test PR-AUC     : {result['test_metrics']['pr_auc']:.4f}\n"
        f"  Checkpoint      : {result['checkpoint']}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()

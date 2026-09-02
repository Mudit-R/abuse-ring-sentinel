"""
scripts/run_pipeline.py
──────────────────────────────────────────────────────────────────────────────
End-to-end pipeline runner: data download → graph build → feature engineering
→ baseline training → GNN training (GCN, GraphSAGE, GAT) → evaluation →
comparison table → results export.

Usage:
    .venv\Scripts\python.exe scripts/run_pipeline.py

All results logged to MLflow (mlruns/) and saved to outputs/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import numpy as np
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


def check_gpu():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        logger.success(f"GPU: {name} ({mem} GB VRAM)")
        return True
    else:
        logger.warning("No GPU found — running on CPU (training will be slow)")
        return False


def step_build_graph(csv_path: Path, graph_dir: Path) -> "GraphBundle":
    from src.graph.builder import TransactionGraphBuilder, load_paysim

    if (graph_dir / "pyg_data.pt").exists():
        logger.info("Graph already built — loading from cache")
        return TransactionGraphBuilder.load(graph_dir)

    df = load_paysim(csv_path)
    builder = TransactionGraphBuilder(fraud_types_only=True)
    bundle = builder.build(df)
    builder.save(bundle, graph_dir)
    return bundle, df


def step_features(bundle, df, graph_dir: Path):
    import pandas as pd
    from src.graph.features import (
        StructuralFeatureComputer,
        compute_temporal_features,
        build_full_feature_matrix,
    )

    struct_path = graph_dir / "structural_features.parquet"
    temporal_path = graph_dir / "temporal_features.parquet"

    if struct_path.exists():
        logger.info("Features already computed — loading from cache")
        struct = pd.read_parquet(struct_path)
        temporal = pd.read_parquet(temporal_path)
    else:
        logger.info("Computing structural features (NetworkX CPU) …")
        computer = StructuralFeatureComputer(use_gpu=False)
        struct = computer.compute(bundle.nx_graph)
        struct.to_parquet(struct_path, index=False)

        logger.info("Computing temporal features …")
        temporal = compute_temporal_features(df, bundle.account_to_idx)
        temporal.to_parquet(temporal_path, index=False)

    full_features = build_full_feature_matrix(struct, temporal, bundle.pyg_data.x)

    # Update pyg_data.x with the full feature matrix (22 features instead of 11)
    bundle.pyg_data.x = full_features.float()
    logger.success(f"Feature matrix: {full_features.shape}")

    return struct, temporal, full_features


def step_baselines(full_features, bundle, df, output_dir: Path) -> dict:
    import numpy as np
    from src.models.baselines import (
        temporal_train_test_split,
        train_logistic_regression,
        train_xgboost,
        train_lightgbm,
    )
    from src.training.evaluate import print_comparison_table

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "baseline_results.json"

    if results_path.exists():
        logger.info("Baselines already trained — loading results")
        with open(results_path) as f:
            return json.load(f)

    X = full_features.numpy()
    y = bundle.node_labels.numpy()

    # Build node steps for temporal split
    account_max_step = df.groupby("nameOrig")["step"].max()
    node_steps = np.zeros(len(bundle.account_to_idx), dtype=np.float32)
    for acc, idx in bundle.account_to_idx.items():
        node_steps[idx] = float(account_max_step.get(acc, 0))

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X, y, node_steps, test_fraction=0.2
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}, "
                f"Train fraud: {y_train.mean():.4%}, Test fraud: {y_test.mean():.4%}")

    db_path = str(ROOT / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("fraud-baselines")

    results = {}
    with mlflow.start_run(run_name="Baselines_All"):
        results["LogisticRegression"] = train_logistic_regression(X_train, y_train, X_test, y_test)
        xgb_out = train_xgboost(X_train, y_train, X_test, y_test)
        results["XGBoost"] = xgb_out["metrics"]
        lgbm_out = train_lightgbm(X_train, y_train, X_test, y_test)
        results["LightGBM"] = lgbm_out["metrics"]

    print_comparison_table(results)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save XGBoost model for API serving
    import pickle
    with open(output_dir / "xgboost_best.pkl", "wb") as f:
        pickle.dump(xgb_out["model"], f)

    return results


def step_train_gnn(model_name: str, bundle, output_dir: Path, device: torch.device) -> dict:
    import numpy as np
    from src.models.gcn import GCN
    from src.models.graphsage import GraphSAGE
    from src.models.gat import GAT
    from src.training.trainer import GNNTrainer, create_temporal_masks

    results_path = output_dir / f"gnn_{model_name}_results.json"
    if results_path.exists():
        logger.info(f"{model_name.upper()} already trained — loading results")
        with open(results_path) as f:
            return json.load(f)

    in_channels = bundle.pyg_data.x.shape[1]
    MODEL_MAP = {
        "gcn": lambda: GCN(in_channels=in_channels, hidden_channels=128, num_layers=3, dropout=0.3),
        "graphsage": lambda: GraphSAGE(in_channels=in_channels, hidden_channels=256, num_layers=3, dropout=0.3),
        "gat": lambda: GAT(in_channels=in_channels, hidden_channels=32, heads=4, dropout=0.3),
    }

    model = MODEL_MAP[model_name]()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{model_name.upper()}: {n_params:,} parameters")

    # Temporal masks based on node index (proxy for time ordering)
    node_steps = torch.arange(bundle.pyg_data.num_nodes, dtype=torch.float)
    train_mask, val_mask, test_mask = create_temporal_masks(
        bundle.pyg_data, node_steps, val_fraction=0.1, test_fraction=0.2
    )

    db_path = str(ROOT / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    batch_sz = 1024 if model_name == "gat" else 2048
    trainer = GNNTrainer(
        model=model,
        data=bundle.pyg_data,
        train_mask=train_mask,
        val_mask=val_mask,
        device=device,
        lr=1e-3,
        focal_alpha=0.5,
        focal_gamma=2.0,
        num_epochs=60,        # Reduced for full pipeline run
        patience=12,
        batch_size=batch_sz,
        num_neighbors=[20, 10, 5],
        output_dir=output_dir,
    )

    result = trainer.fit(
        experiment_name="fraud-gnn",
        model_name=model_name.upper(),
    )

    metrics_to_save = {
        "model": model_name,
        "best_val_pr_auc": result["best_val_pr_auc"],
        **{f"test_{k}": v for k, v in result["test_metrics"].items()},
    }
    with open(results_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    return metrics_to_save


def print_final_table(baseline_results: dict, gnn_results: dict):
    """Print the final comparison table with all models."""
    table = Table(title="[bold]Final Model Comparison — Fraud Detection (PaySim 6.3M transactions)[/bold]")
    table.add_column("Model", style="bold cyan", min_width=18)
    table.add_column("PR-AUC", style="bold green", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("P@500", justify="right")
    table.add_column("R@500", justify="right")

    all_results = {}
    for name, m in baseline_results.items():
        all_results[name] = m

    for name, m in gnn_results.items():
        display_name = f"GNN-{name.upper()}"
        row_metrics = {
            "pr_auc": m.get("test_pr_auc", m.get("pr_auc", 0)),
            "roc_auc": m.get("test_roc_auc", m.get("roc_auc", 0)),
            "f1": m.get("test_f1", m.get("f1", 0)),
            "precision": m.get("test_precision", m.get("precision", 0)),
            "recall": m.get("test_recall", m.get("recall", 0)),
            "precision_at_500": m.get("test_precision_at_500", m.get("precision_at_500", 0)),
            "recall_at_500": m.get("test_recall_at_500", m.get("recall_at_500", 0)),
        }
        all_results[display_name] = row_metrics

    for model_name, metrics in all_results.items():
        table.add_row(
            model_name,
            f"{metrics.get('pr_auc', 0):.4f}",
            f"{metrics.get('roc_auc', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('precision_at_500', 0):.4f}",
            f"{metrics.get('recall_at_500', 0):.4f}",
        )

    console.print("\n")
    console.print(table)
    return all_results


def main():
    console.rule("[bold green]Graph-Based Fraud Detection — Full Pipeline[/bold green]")
    t_start = time.time()

    has_gpu = check_gpu()
    device = torch.device("cuda" if has_gpu else "cpu")

    CSV_PATH = ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
    GRAPH_DIR = ROOT / "outputs" / "baselines" / "graph"
    BASELINE_DIR = ROOT / "outputs" / "baselines"
    GNN_DIR = ROOT / "outputs" / "checkpoints"

    # ── Step 1: Check data ─────────────────────────────────────────────────────
    if not CSV_PATH.exists():
        logger.error(f"Dataset not found: {CSV_PATH}")
        logger.info("Run: .venv\\Scripts\\python.exe scripts/download_data.py")
        sys.exit(1)

    console.rule("Step 1/5: Building Graph")
    result = step_build_graph(CSV_PATH, GRAPH_DIR)
    if isinstance(result, tuple):
        bundle, df = result
    else:
        bundle = result
        from src.graph.builder import load_paysim
        df = load_paysim(CSV_PATH)

    # ── Step 2: Features ───────────────────────────────────────────────────────
    console.rule("Step 2/5: Feature Engineering (22 features per account)")
    struct, temporal, full_features = step_features(bundle, df, GRAPH_DIR)

    # ── Step 3: Baselines ──────────────────────────────────────────────────────
    console.rule("Step 3/5: Baseline Models (LogReg + XGBoost + LightGBM)")
    baseline_results = step_baselines(full_features, bundle, df, BASELINE_DIR)

    # ── Step 4: GNN Training ───────────────────────────────────────────────────
    gnn_results = {}
    for model_name in ["gcn", "graphsage", "gat"]:
        console.rule(f"Step 4/5: Training {model_name.upper()}")
        gnn_results[model_name] = step_train_gnn(model_name, bundle, GNN_DIR, device)

    # ── Step 5: Final comparison table ────────────────────────────────────────
    console.rule("Step 5/5: Results")
    all_results = print_final_table(baseline_results, gnn_results)

    # Save combined results
    with open(ROOT / "outputs" / "final_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.success(f"Results saved to outputs/final_results.json")

    elapsed = time.time() - t_start
    console.print(f"\n[bold green]✅ Pipeline complete in {elapsed/60:.1f} minutes[/bold green]")
    console.print("[dim]Next: .venv\\Scripts\\python.exe scripts/update_readme.py[/dim]")
    console.print("[dim]Then: .venv\\Scripts\\uvicorn.exe src.api.main:app --host 0.0.0.0 --port 8000[/dim]")


if __name__ == "__main__":
    main()

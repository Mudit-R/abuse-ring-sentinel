"""
src/training/run_baselines.py
──────────────────────────────────────────────────────────────────────────────
Entry-point script for training all baseline models.

Usage:
    python -m src.training.run_baselines \
        --data-path data/raw/PS_20174392719_1491204439457_log.csv \
        --output-dir outputs/baselines \
        --experiment-name fraud-baselines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from loguru import logger

from src.graph.builder import TransactionGraphBuilder, load_paysim
from src.graph.features import (
    StructuralFeatureComputer,
    compute_temporal_features,
    build_full_feature_matrix,
)
from src.models.baselines import (
    temporal_train_test_split,
    train_logistic_regression,
    train_xgboost,
    train_lightgbm,
)
from src.training.evaluate import print_comparison_table

FEATURE_NAMES = [
    # Tabular (11)
    "total_sent_log", "total_received_log", "tx_count_out", "tx_count_in",
    "unique_dest_count", "unique_src_count", "avg_sent_log", "avg_received_log",
    "balance_drain_ratio", "night_tx_fraction", "fraud_type_fraction",
    # Structural (6)
    "in_degree", "out_degree", "degree_ratio", "pagerank",
    "k_core_number", "local_clustering_coefficient",
    # Temporal (5)
    "tx_velocity_24h", "tx_velocity_7d", "amount_velocity_24h",
    "amount_velocity_7d", "amount_spike_ratio",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--experiment-name", type=str, default="fraud-baselines")
    parser.add_argument("--use-gpu-features", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("mlruns/")
    mlflow.set_experiment(args.experiment_name)

    # ── Load Data ────────────────────────────────────────────────────────────
    logger.info("Step 1/4: Loading and preprocessing data …")
    df = load_paysim(args.data_path)

    # ── Build Graph & Features ────────────────────────────────────────────────
    logger.info("Step 2/4: Building graph and computing features …")
    builder = TransactionGraphBuilder(fraud_types_only=True)
    bundle = builder.build(df)
    builder.save(bundle, args.output_dir / "graph")

    computer = StructuralFeatureComputer(use_gpu=args.use_gpu_features)
    structural_features = computer.compute(bundle.nx_graph)
    temporal_features = compute_temporal_features(df, bundle.account_to_idx)

    X_full = build_full_feature_matrix(
        structural_features, temporal_features, bundle.pyg_data.x
    ).numpy()
    y = bundle.node_labels.numpy()

    # Get the max step per account for temporal split
    account_max_step = df.groupby("nameOrig")["step"].max()
    node_steps = np.zeros(len(bundle.account_to_idx), dtype=np.float32)
    for acc, idx in bundle.account_to_idx.items():
        node_steps[idx] = float(account_max_step.get(acc, 0))

    logger.info(f"Feature matrix: {X_full.shape}, Fraud rate: {y.mean():.4%}")

    # ── Time-Based Split ─────────────────────────────────────────────────────
    logger.info("Step 3/4: Applying time-based train/test split …")
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X_full, y, node_steps, test_fraction=0.2
    )

    # ── Train Models ─────────────────────────────────────────────────────────
    logger.info("Step 4/4: Training baseline models …")
    results = {}

    with mlflow.start_run(run_name="Baselines"):
        lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
        results["LogisticRegression"] = lr_metrics

        xgb_result = train_xgboost(X_train, y_train, X_test, y_test, FEATURE_NAMES)
        results["XGBoost"] = xgb_result["metrics"]

        lgbm_result = train_lightgbm(X_train, y_train, X_test, y_test, FEATURE_NAMES)
        results["LightGBM"] = lgbm_result["metrics"]

    # ── Report ────────────────────────────────────────────────────────────────
    print_comparison_table(results)
    with open(args.output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Results saved to {args.output_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()

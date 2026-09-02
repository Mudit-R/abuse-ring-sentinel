"""
src/training/evaluate.py
──────────────────────────────────────────────────────────────────────────────
Evaluation utilities for all models.

Metric choices (explain these in interviews):
──────────────────────────────────────────────
• PR-AUC (Average Precision)
    With 0.13% fraud rate, ROC-AUC is misleadingly optimistic — a model that
    predicts 0.14% of transactions as fraud still gets a high ROC-AUC.
    PR-AUC directly measures precision vs. recall trade-off on the minority
    class, making it the correct primary metric for imbalanced fraud detection.

• Precision@K / Recall@K
    Fraud analysts can only review K cases per day. Precision@K answers:
    "of the K accounts we flag, what fraction are actually fraud?"
    Recall@K answers: "of all fraud accounts, how many do we catch in top-K?"
    This is the operational metric that maps directly to business cost.

• F1 Score
    Harmonic mean of precision and recall — useful for comparing models at a
    single operating point (threshold = 0.5 by default).

References:
    Davis & Goadrich, "The Relationship Between Precision-Recall and ROC
    Curves", ICML 2006.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from rich.console import Console
from rich.table import Table


console = Console()


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    threshold: Optional[float] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute the full evaluation suite for a binary classifier.

    Parameters
    ----------
    y_true     : Ground-truth binary labels
    y_proba    : Predicted probabilities (positive class)
    model_name : Name for logging
    threshold  : Decision threshold (if None, optimal F1 threshold is automatically computed)
    k_values   : Values of K for Precision@K / Recall@K

    Returns
    -------
    dict of metric_name → float (logged to MLflow)
    """
    if k_values is None:
        k_values = [100, 500, 1000]

    # For highly imbalanced 130:1 fraud datasets, calculate the optimal operating threshold
    if threshold is None:
        cand_thresholds = np.linspace(0.01, 0.80, 80)
        best_f1 = -1.0
        best_t = 0.5
        for t in cand_thresholds:
            p = (y_proba >= t).astype(int)
            f = f1_score(y_true, p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        threshold = best_t

    y_pred = (y_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    # Precision@K and Recall@K
    n_fraud = y_true.sum()
    sorted_indices = np.argsort(y_proba)[::-1]  # highest scores first
    for k in k_values:
        if k > len(y_true):
            continue
        top_k_indices = sorted_indices[:k]
        p_at_k = y_true[top_k_indices].sum() / k
        r_at_k = y_true[top_k_indices].sum() / (n_fraud + 1e-6)
        metrics[f"precision_at_{k}"] = float(p_at_k)
        metrics[f"recall_at_{k}"] = float(r_at_k)

    logger.info(
        f"[{model_name}] PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | "
        f"F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f}"
    )
    for k in k_values:
        if f"precision_at_{k}" in metrics:
            logger.info(
                f"[{model_name}] Precision@{k}={metrics[f'precision_at_{k}']:.4f} | "
                f"Recall@{k}={metrics[f'recall_at_{k}']:.4f}"
            )

    return metrics


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Pretty-print a comparison table of all models.

    Results format:
        {"ModelName": {"pr_auc": 0.85, "roc_auc": 0.91, ...}, ...}
    """
    table = Table(title="Model Comparison — Fraud Detection")
    table.add_column("Model", style="bold cyan")
    table.add_column("PR-AUC", style="bold green")
    table.add_column("ROC-AUC")
    table.add_column("F1")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("P@500")
    table.add_column("R@500")

    for model_name, metrics in results.items():
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

    console.print(table)

"""
src/models/baselines.py
──────────────────────────────────────────────────────────────────────────────
Tabular baseline models: Logistic Regression, XGBoost, LightGBM.

IMPORTANT (say this in interviews):
    Most real production fraud systems ARE tabular models. XGBoost/LightGBM on
    well-engineered features (including graph-structural features computed
    offline) can match or beat GNNs on latency-sensitive endpoints.

    The baseline exists to show:
      1. You understand the production landscape (not just deep learning hype)
      2. The GNNs must beat a strong baseline to justify the complexity
      3. You know how to quantify improvement rigorously
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.training.evaluate import evaluate_model, print_comparison_table


# ── Time-Based Split ──────────────────────────────────────────────────────────

def temporal_train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    steps: np.ndarray,
    test_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by time (step) rather than randomly.

    WHY: Random splits leak future transaction behaviour into training.
    In a real payment system, you can only train on historical data and
    evaluate on future data. Random splits silently inflate all metrics.

    This is one of the most common ML mistakes in fraud systems.
    Explicitly calling it out demonstrates production ML maturity.
    """
    split_step = np.quantile(steps, 1 - test_fraction)
    train_mask = steps <= split_step
    test_mask = steps > split_step

    logger.info(
        f"Time-based split at step {split_step:.0f}: "
        f"train={train_mask.sum():,}, test={test_mask.sum():,}"
    )
    return (
        features[train_mask], features[test_mask],
        labels[train_mask], labels[test_mask],
    )


# ── Logistic Regression ───────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "fraud-baselines",
) -> Dict[str, float]:
    """Train Logistic Regression with class-weight balancing."""
    with mlflow.start_run(run_name="LogisticRegression", nested=True):
        params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs"}
        mlflow.log_params(params)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ])
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_proba, model_name="LogisticRegression")
        mlflow.log_metrics(metrics)

        # Save model
        mlflow.sklearn.log_model(pipe, "logistic_regression")
    return metrics


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
) -> Dict[str, Any]:
    """Train XGBoost with scale_pos_weight for class imbalance."""
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logger.info(f"XGBoost scale_pos_weight: {scale_pos_weight:.1f} (neg/pos ratio)")

    with mlflow.start_run(run_name="XGBoost", nested=True):
        params = {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_pos_weight,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "eval_metric": "aucpr",
            # use_label_encoder removed in XGBoost 2.0
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100,
        )
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_proba, model_name="XGBoost")
        mlflow.log_metrics(metrics)

        if feature_names:
            fi = dict(zip(feature_names, model.feature_importances_))
            top_features = sorted(fi.items(), key=lambda x: -x[1])[:20]
            logger.info(f"XGBoost top features: {top_features}")
            mlflow.log_dict(fi, "feature_importances.json")

        mlflow.xgboost.log_model(model, "xgboost")

    return {"metrics": metrics, "model": model}


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
) -> Dict[str, Any]:
    """Train LightGBM with is_unbalance=True."""
    with mlflow.start_run(run_name="LightGBM", nested=True):
        params = {
            "n_estimators": 600,
            "max_depth": 7,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "is_unbalance": True,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "metric": "average_precision",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        mlflow.log_params(params)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
        )
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_proba, model_name="LightGBM")
        mlflow.log_metrics(metrics)

        if feature_names:
            fi = dict(zip(
                feature_names if feature_names else [f"f{i}" for i in range(X_train.shape[1])],
                model.feature_importances_
            ))
            mlflow.log_dict(fi, "feature_importances.json")

        mlflow.lightgbm.log_model(model, "lightgbm")

    return {"metrics": metrics, "model": model}

"""
src/federated/flower_sim.py
──────────────────────────────────────────────────────────────────────────────
Federated Learning simulation using Flower (flwr).

Simulates 3 "banks" training a shared fraud detection model without sharing
raw transaction data. Only model gradients/weights are communicated.

Why this matters:
    Real AML systems face a fundamental privacy dilemma — the best fraud
    detection would require sharing data across banks (a fraudster caught at
    Bank A is likely active at Banks B and C). But raw data sharing violates
    GDPR, data localisation laws, and competitive confidentiality.

    Federated Learning solves this: banks collaboratively train a shared model
    while keeping their transaction data on-premises. Only model updates
    (not data) leave the institution.

    This directly maps to NPCI's stated interest in "federated and
    privacy-preserving AI" — and almost no other candidate will have built
    a working demo of it.

Architecture:
    • 3 simulated clients (Bank A, B, C) each with a partition of PaySim
    • FedAvg aggregation strategy (Federated Averaging)
    • XGBoost classifier as the local model (practical for tabular fraud data)
    • Flower simulation mode (no actual network — runs in-process for demo)

References:
    McMahan et al., "Communication-Efficient Learning of Deep Networks from
    Decentralized Data", AISTATS 2017. (FedAvg paper)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler


# ── Data Partitioning ─────────────────────────────────────────────────────────

def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 3,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition features and labels into n_clients non-overlapping shards.

    In a real federated setup, each shard would live on a different
    institution's servers. Here we simulate this in-memory.

    Uses a stratified split to ensure each client sees some fraud examples
    (important given the very low fraud rate).
    """
    rng = np.random.RandomState(seed)

    # Separate fraud and non-fraud indices
    fraud_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    rng.shuffle(fraud_idx)
    rng.shuffle(normal_idx)

    partitions = []
    fraud_splits = np.array_split(fraud_idx, n_clients)
    normal_splits = np.array_split(normal_idx, n_clients)

    for i in range(n_clients):
        idx = np.concatenate([fraud_splits[i], normal_splits[i]])
        rng.shuffle(idx)
        partitions.append((X[idx], y[idx]))
        logger.info(
            f"  Client {i} (Bank {chr(65+i)}): {len(idx):,} samples, "
            f"{fraud_splits[i].shape[0]} fraud ({fraud_splits[i].shape[0]/len(idx):.4%})"
        )

    return partitions


# ── Flower Client ─────────────────────────────────────────────────────────────

class FraudDetectionClient(fl.client.NumPyClient):
    """
    Flower client representing one bank's fraud detection node.

    Uses SGDClassifier (logistic regression with SGD) because it supports
    warm_start + partial_fit, which is the standard way to implement
    federated training with sklearn-style models.

    In production, this would be wrapped in a secure enclave (TEE) or
    differentially private (DP-SGD) to prevent gradient inversion attacks.
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        self.model = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=1,
            warm_start=True,
            random_state=42,
        )
        # Initialize with one pass to set coef_ shape
        self.model.fit(self.X_train_scaled[:10], self.y_train[:10])

    def get_parameters(self, config) -> List[np.ndarray]:
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Local training step (one round of gradient descent)."""
        self.set_parameters(parameters)
        self.model.partial_fit(
            self.X_train_scaled, self.y_train, classes=[0, 1]
        )
        return self.get_parameters({}), len(self.X_train), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """Local evaluation — returns loss and PR-AUC."""
        self.set_parameters(parameters)
        y_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        pr_auc = average_precision_score(self.y_test, y_proba)
        loss = 1.0 - pr_auc  # Flower expects a loss (lower = better)
        logger.info(f"  Bank {chr(65 + self.client_id)} — PR-AUC: {pr_auc:.4f}")
        return float(loss), len(self.X_test), {"pr_auc": float(pr_auc)}


# ── Federated Simulation ───────────────────────────────────────────────────────

def run_federated_simulation(
    X: np.ndarray,
    y: np.ndarray,
    X_test_global: np.ndarray,
    y_test_global: np.ndarray,
    n_clients: int = 3,
    n_rounds: int = 10,
) -> Dict:
    """
    Run a full federated training simulation using Flower's in-process mode.

    Returns the global model metrics after n_rounds of FedAvg.
    """
    logger.info(f"Starting federated simulation: {n_clients} banks, {n_rounds} rounds")

    partitions = partition_data(X, y, n_clients=n_clients)

    # Create clients
    clients = [
        FraudDetectionClient(
            client_id=i,
            X_train=partitions[i][0],
            y_train=partitions[i][1],
            X_test=X_test_global,
            y_test=y_test_global,
        )
        for i in range(n_clients)
    ]

    # FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
    )

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)],
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # Extract final metrics
    final_metrics = {}
    if history.metrics_distributed:
        for key, values in history.metrics_distributed.items():
            final_metrics[key] = values[-1][1] if values else None

    logger.success(f"Federated training complete. Final metrics: {final_metrics}")
    return {
        "history": history,
        "final_metrics": final_metrics,
        "n_rounds": n_rounds,
        "n_clients": n_clients,
    }

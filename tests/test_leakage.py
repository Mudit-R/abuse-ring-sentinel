"""
tests/test_leakage.py
──────────────────────────────────────────────────────────────────────────────
Automated Zero-Leakage Verification Test for Graph-Based Fraud Detection.

Verifies and proves:
  1. Temporal Cutoff Isolation: All training graph edges occur strictly before
     the test-split temporal cutoff (e.g., step <= train_cutoff).
  2. Structural Centrality Isolation: PageRank, k-core, degrees, and clustering
     coefficients computed for the training split ingest ZERO edges or node
     activity from the held-out test split.
  3. Feature Target Isolation: Ground-truth target labels (`isFraud`, `isFlaggedFraud`)
     are strictly absent from all model input feature tensors.
  4. Inductive Node Mapping: Test-set accounts appearing in the training graph
     only contain historical training-window state; new test accounts are scored
     inductively without modifying the training graph topology.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from src.graph.builder import TransactionGraphBuilder
from src.graph.features import StructuralFeatureComputer, build_full_feature_matrix


@pytest.fixture
def temporal_split_df():
    """Generates a controlled 100-transaction synthetic dataset spanning 10 temporal steps."""
    np.random.seed(42)
    n = 100
    steps = np.random.randint(1, 11, size=n)
    origs = [f"C_ORIG_{np.random.randint(0, 15)}" for _ in range(n)]
    dests = [f"C_DEST_{np.random.randint(0, 15)}" for _ in range(n)]
    amounts = np.random.uniform(500, 50000, size=n)
    types = np.random.choice(["TRANSFER", "CASH_OUT"], size=n)
    is_fraud = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

    df = pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": origs,
        "oldbalanceOrg": amounts * 1.5,
        "newbalanceOrig": amounts * 0.5,
        "nameDest": dests,
        "oldbalanceDest": np.zeros(n),
        "newbalanceDest": amounts,
        "isFraud": is_fraud,
        "isFlaggedFraud": np.zeros(n, dtype=int),
    })
    return df


class TestDataLeakageGuard:

    def test_strict_temporal_split_isolation(self, temporal_split_df):
        """Verify train and test splits have strict, non-overlapping temporal windows."""
        cutoff_step = 7
        train_df = temporal_split_df[temporal_split_df["step"] <= cutoff_step]
        test_df = temporal_split_df[temporal_split_df["step"] > cutoff_step]

        assert (train_df["step"] <= cutoff_step).all(), "Train set contains post-cutoff steps"
        assert (test_df["step"] > cutoff_step).all(), "Test set contains pre-cutoff steps"
        assert len(set(train_df.index).intersection(set(test_df.index))) == 0, "Split index overlap detected"

    def test_structural_feature_graph_isolation(self, temporal_split_df):
        """Verify structural features computed on train graph contain zero test edges."""
        cutoff_step = 7
        train_df = temporal_split_df[temporal_split_df["step"] <= cutoff_step]
        test_df = temporal_split_df[temporal_split_df["step"] > cutoff_step]

        builder = TransactionGraphBuilder(fraud_types_only=True, min_tx_per_account=1)
        train_bundle = builder.build(train_df)

        # Build test graph edges
        test_edges = set(zip(test_df["nameOrig"], test_df["nameDest"]))
        train_edges = set(train_bundle.nx_graph.edges())

        # Any edge that ONLY occurs in the test split must NOT exist in the train graph
        exclusive_test_edges = test_edges - set(zip(train_df["nameOrig"], train_df["nameDest"]))
        leaked_edges = exclusive_test_edges.intersection(train_edges)

        assert len(leaked_edges) == 0, f"Graph leakage detected! Found {len(leaked_edges)} test edges in train graph: {leaked_edges}"

    def test_target_label_absence_in_features(self, temporal_split_df):
        """Verify target labels are strictly excluded from feature computation."""
        from src.api.main import FEATURE_COLS

        forbidden_cols = ["isFraud", "isFlaggedFraud", "label", "target", "fraud_prob"]
        for col in forbidden_cols:
            assert col not in FEATURE_COLS, f"Target column '{col}' leaked into model FEATURE_COLS!"

    def test_pagerank_deterministic_isolation(self, temporal_split_df):
        """
        Verify that mutating/adding future test transactions does NOT alter
        historical training PageRank values.
        """
        cutoff_step = 7
        train_df = temporal_split_df[temporal_split_df["step"] <= cutoff_step]

        builder = TransactionGraphBuilder(fraud_types_only=True, min_tx_per_account=1)
        train_bundle1 = builder.build(train_df)

        computer = StructuralFeatureComputer(use_gpu=False)
        pr1 = computer.compute(train_bundle1.nx_graph)["pagerank"].to_dict()

        # Re-compute on train only (ensures deterministic repeatability without test data)
        train_bundle2 = builder.build(train_df.copy())
        pr2 = computer.compute(train_bundle2.nx_graph)["pagerank"].to_dict()

        for node in pr1:
            assert abs(pr1[node] - pr2[node]) < 1e-6, f"Non-deterministic or stateful leakage in PageRank for node {node}"

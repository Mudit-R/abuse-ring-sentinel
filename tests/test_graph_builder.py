"""
tests/test_graph_builder.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the graph construction pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.graph.builder import TransactionGraphBuilder, FRAUD_TX_TYPES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal PaySim-shaped DataFrame for testing."""
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5, 6],
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [1000.0, 2000.0, 500.0, 3000.0, 1500.0, 800.0],
        "nameOrig": ["C001", "C001", "C002", "C003", "C004", "C005"],
        "oldbalanceOrg": [5000.0, 4000.0, 1000.0, 6000.0, 3000.0, 2000.0],
        "newbalanceOrig": [4000.0, 2000.0, 500.0, 3000.0, 1500.0, 1200.0],
        "nameDest": ["C002", "C003", "C004", "C005", "C001", "C002"],
        "oldbalanceDest": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "newbalanceDest": [1000.0, 2000.0, 500.0, 3000.0, 1500.0, 800.0],
        "isFraud": [1, 0, 0, 1, 0, 0],
        "isFlaggedFraud": [0, 0, 0, 0, 0, 0],
    })


@pytest.fixture
def bundle(sample_df):
    builder = TransactionGraphBuilder(fraud_types_only=True, min_tx_per_account=1)
    return builder.build(sample_df)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGraphBuilder:

    def test_no_self_loops(self, bundle):
        ei = bundle.pyg_data.edge_index
        assert (ei[0] != ei[1]).all(), "Graph should have no self-loops"

    def test_directed_graph(self, bundle):
        """Edges should be directed (not necessarily symmetric)."""
        G = bundle.nx_graph
        assert G.is_directed(), "Graph must be directed"

    def test_node_count(self, sample_df, bundle):
        """All unique accounts appear as nodes."""
        all_accs = set(sample_df["nameOrig"]) | set(sample_df["nameDest"])
        # Filter to only accounts that appear in TRANSFER/CASH_OUT edges
        fraud_df = sample_df[sample_df["type"].isin(FRAUD_TX_TYPES)]
        expected_accs = set(fraud_df["nameOrig"]) | set(fraud_df["nameDest"])
        assert len(bundle.account_to_idx) == len(expected_accs)

    def test_fraud_labels_nonzero(self, bundle):
        """At least one node should be labelled as fraud."""
        assert bundle.node_labels.sum() > 0

    def test_pyg_data_dimensions(self, bundle):
        n = bundle.pyg_data.num_nodes
        f = bundle.pyg_data.x.shape[1]
        assert bundle.pyg_data.x.shape == (n, f), "Node feature matrix shape mismatch"
        assert bundle.pyg_data.y.shape == (n,), "Labels shape mismatch"
        assert bundle.pyg_data.edge_index.shape[0] == 2, "edge_index must have 2 rows"

    def test_edge_attr_dimensions(self, bundle):
        e = bundle.pyg_data.edge_index.shape[1]
        assert bundle.pyg_data.edge_attr.shape == (e, 3), "Edge attrs: (E, 3) expected"

    def test_node_features_no_nan(self, bundle):
        assert not torch.isnan(bundle.pyg_data.x).any(), "NaNs in node features"

    def test_label_consistency(self, bundle, sample_df):
        """Fraud origin accounts should be labelled 1."""
        fraud_origins = set(sample_df.loc[sample_df["isFraud"] == 1, "nameOrig"])
        for acc in fraud_origins:
            if acc in bundle.account_to_idx:
                idx = bundle.account_to_idx[acc]
                assert bundle.node_labels[idx] == 1, f"{acc} should be labelled fraud"

    def test_idx_to_account_inverse(self, bundle):
        """account_to_idx and idx_to_account should be inverses."""
        for acc, idx in bundle.account_to_idx.items():
            assert bundle.idx_to_account[idx] == acc

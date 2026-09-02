"""
tests/test_features.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for feature engineering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.graph.builder import TransactionGraphBuilder
from src.graph.features import (
    StructuralFeatureComputer,
    compute_temporal_features,
    build_full_feature_matrix,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "step": list(range(1, 11)),
        "type": ["TRANSFER", "CASH_OUT"] * 5,
        "amount": [float(i * 1000) for i in range(1, 11)],
        "nameOrig": ["C00" + str(i % 3) for i in range(10)],
        "oldbalanceOrg": [5000.0] * 10,
        "newbalanceOrig": [4000.0] * 10,
        "nameDest": ["C00" + str((i + 1) % 3) for i in range(10)],
        "oldbalanceDest": [0.0] * 10,
        "newbalanceDest": [1000.0] * 10,
        "isFraud": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "isFlaggedFraud": [0] * 10,
    })


@pytest.fixture
def bundle(sample_df):
    builder = TransactionGraphBuilder(fraud_types_only=True, min_tx_per_account=1)
    return builder.build(sample_df)


class TestStructuralFeatures:

    def test_cpu_features_shape(self, bundle):
        computer = StructuralFeatureComputer(use_gpu=False)
        feat_df = computer.compute(bundle.nx_graph)
        n = bundle.nx_graph.number_of_nodes()
        assert feat_df.shape[0] == n, "Feature rows should match node count"
        assert feat_df.shape[1] == 6, "Expected 6 structural features"

    def test_no_nan_in_features(self, bundle):
        computer = StructuralFeatureComputer(use_gpu=False)
        feat_df = computer.compute(bundle.nx_graph)
        assert not feat_df.isnull().any().any(), "No NaNs allowed in structural features"

    def test_degree_nonnegative(self, bundle):
        computer = StructuralFeatureComputer(use_gpu=False)
        feat_df = computer.compute(bundle.nx_graph)
        assert (feat_df["in_degree"] >= 0).all()
        assert (feat_df["out_degree"] >= 0).all()

    def test_pagerank_sums_to_one(self, bundle):
        computer = StructuralFeatureComputer(use_gpu=False)
        feat_df = computer.compute(bundle.nx_graph)
        pr_sum = feat_df["pagerank"].sum()
        assert abs(pr_sum - 1.0) < 0.01, f"PageRank should sum to ~1, got {pr_sum:.4f}"

    def test_clustering_coefficient_range(self, bundle):
        computer = StructuralFeatureComputer(use_gpu=False)
        feat_df = computer.compute(bundle.nx_graph)
        cc = feat_df["local_clustering_coefficient"]
        assert (cc >= 0).all() and (cc <= 1).all(), "Clustering coefficient must be in [0, 1]"


class TestTemporalFeatures:

    def test_temporal_features_shape(self, sample_df, bundle):
        feat_df = compute_temporal_features(sample_df, bundle.account_to_idx)
        assert feat_df.shape[0] == len(bundle.account_to_idx)
        assert feat_df.shape[1] == 5

    def test_no_nan_in_temporal(self, sample_df, bundle):
        feat_df = compute_temporal_features(sample_df, bundle.account_to_idx)
        assert not feat_df.isnull().any().any()

    def test_velocity_nonnegative(self, sample_df, bundle):
        feat_df = compute_temporal_features(sample_df, bundle.account_to_idx)
        assert (feat_df["tx_velocity_24h"] >= 0).all()
        assert (feat_df["tx_velocity_7d"] >= 0).all()

    def test_7d_ge_24h(self, sample_df, bundle):
        """7-day velocity should always be >= 24h velocity."""
        feat_df = compute_temporal_features(sample_df, bundle.account_to_idx)
        assert (feat_df["tx_velocity_7d"] >= feat_df["tx_velocity_24h"]).all()


class TestFullFeatureMatrix:

    def test_matrix_shape(self, sample_df, bundle):
        computer = StructuralFeatureComputer(use_gpu=False)
        struct = computer.compute(bundle.nx_graph)
        temporal = compute_temporal_features(sample_df, bundle.account_to_idx)
        full = build_full_feature_matrix(struct, temporal, bundle.pyg_data.x)
        n = bundle.pyg_data.num_nodes
        assert full.shape == (n, 11 + 6 + 5), f"Expected (N, 22), got {full.shape}"

    def test_no_nan_in_full_matrix(self, sample_df, bundle):
        computer = StructuralFeatureComputer(use_gpu=False)
        struct = computer.compute(bundle.nx_graph)
        temporal = compute_temporal_features(sample_df, bundle.account_to_idx)
        full = build_full_feature_matrix(struct, temporal, bundle.pyg_data.x)
        assert not torch.isnan(full).any()

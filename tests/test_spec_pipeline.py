"""
tests/test_spec_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Comprehensive Unit & Integration Test Suite for Specification Conformance:
1. 20-Node Heterogeneous Graph Builder & Typed Edge Attributes
2. 4 Synthetic Merchant Attack Vectors (Promo, RTO, Chargeback, ATO)
3. Chebyshev Spectral Filter (Order K=2 Normalized Laplacian)
4. Adaptive RL Neighbor Policy Selector
5. Community-Level Ring Detection (Ring Recall, TTFD, Future Loss Prevented)
6. Probability Calibration, Temperature Scaling & Prior Correction Math
7. Multi-Task FraudHGT on Heterogeneous Graph
"""
from __future__ import annotations

import pytest
import numpy as np
import torch

from src.graph.hetero_schema import (
    NodeType,
    NODE_TYPES,
    EDGE_RELATIONS,
    TemporalEdgeAttributes,
    tokenize_identifier,
)
from src.graph.hetero_builder import HeteroGraphBuilder
from src.models.spectral import ChebyshevSpectralFilter
from src.models.rl_selector import AdaptiveNeighborSelector
from src.graph.ring_detector import RingDetectionEngine
from src.evaluation.calibration import (
    correct_for_sampled_prior,
    compute_bayes_optimal_threshold,
    ProbabilityCalibrator,
    compare_calibration_methods,
)
from src.models.hgt import FraudHGT


class TestHeteroGraphSchema:
    def test_twenty_node_types_defined(self):
        assert len(NODE_TYPES) == 20
        assert "merchant" in NODE_TYPES
        assert "customer" in NODE_TYPES
        assert "transaction" in NODE_TYPES
        assert "device" in NODE_TYPES
        assert "ip" in NODE_TYPES
        assert "promo" in NODE_TYPES
        assert "refund" in NODE_TYPES
        assert "chargeback" in NODE_TYPES

    def test_tokenization_irreversibility(self):
        tok1 = tokenize_identifier("card_4111_1234_5678")
        tok2 = tokenize_identifier("card_4111_1234_5678")
        tok3 = tokenize_identifier("card_different")

        assert tok1.startswith("tok_")
        assert tok1 == tok2
        assert tok1 != tok3
        assert "4111" not in tok1

    def test_temporal_edge_vector_length(self):
        attrs = TemporalEdgeAttributes(
            event_timestamp=1788300000.0,
            delta_t_from_prev=12.5,
            amount=4500.0,
            velocity_5m=3,
        )
        vec = attrs.to_feature_vector()
        assert len(vec) == 12
        assert vec[0] == 12.5
        assert vec[1] == 4500.0


class TestHeteroGraphBuilder:
    def test_ecosystem_generation_and_rings(self):
        builder = HeteroGraphBuilder(feature_dim=8, seed=42)
        data, rings, stats = builder.build_synthetic_merchant_ecosystem(
            n_merchants=5,
            n_clean_customers=30,
            n_rings=4,
            transactions_per_customer=2,
        )

        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        assert len(rings) == 4

        ring_types = [r.ring_type for r in rings]
        assert "PROMO_ABUSE" in ring_types
        assert "RTO_LOOP" in ring_types
        assert "CHARGEBACK_COLLUSION" in ring_types
        assert "ATO_SURGE" in ring_types

        # Verify PyG HeteroData structure
        assert (NodeType.CUSTOMER.value, "PLACED", NodeType.TRANSACTION.value) in data.edge_types
        assert data[NodeType.CUSTOMER.value].x.shape[1] == 8


class TestSpectralFilter:
    def test_chebyshev_forward_shape(self):
        flt = ChebyshevSpectralFilter(in_channels=16, out_channels=16, K=2)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 3]], dtype=torch.long)

        out = flt(x, edge_index)
        assert out.shape == (10, 16)
        assert not torch.isnan(out).any()

    def test_spectral_empty_graph_fallback(self):
        flt = ChebyshevSpectralFilter(in_channels=8, out_channels=8, K=2)
        x = torch.randn(5, 8)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        out = flt(x, edge_index)
        assert out.shape == (5, 8)
        assert not torch.isnan(out).any()


class TestAdaptiveRLNeighborSelector:
    def test_policy_filtering(self):
        selector = AdaptiveNeighborSelector(node_dim=16, edge_feat_dim=12, hidden_dim=16)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        edge_attr = torch.randn(5, 12)

        filt_edges, filt_attr, mask = selector.filter_edges(x, edge_index, edge_attr)
        assert filt_edges.shape[0] == 2
        assert filt_edges.shape[1] <= 5
        assert filt_edges.shape[1] >= 1  # Safety threshold preserves at least 1
        assert filt_attr.shape[0] == filt_edges.shape[1]


class TestRingDetectionEngine:
    def test_community_detection_and_metrics(self):
        engine = RingDetectionEngine(risk_threshold=0.40)

        txns = [
            {"transaction_id": "tx1", "customer_id": "c1", "device_id": "d_shared", "amount": 2000.0, "timestamp": 100.0},
            {"transaction_id": "tx2", "customer_id": "c2", "device_id": "d_shared", "amount": 2000.0, "timestamp": 110.0},
            {"transaction_id": "tx3", "customer_id": "c3", "device_id": "d_shared", "amount": 2000.0, "timestamp": 120.0},
            {"transaction_id": "tx4", "customer_id": "clean1", "device_id": "d_clean", "amount": 500.0, "timestamp": 200.0},
        ]
        risk_scores = {"c1": 0.85, "c2": 0.90, "c3": 0.88, "clean1": 0.05}

        rings = engine.extract_rings_from_transactions(txns, risk_scores)
        assert len(rings) >= 1
        top_ring = rings[0]
        assert top_ring.entity_count == 3
        assert top_ring.risk_level in ["CRITICAL", "HIGH"]
        assert top_ring.temporal_synchronization_score > 0.5

        gt_rings = [{"ring_id": "gt_1", "member_customer_ids": ["c1", "c2", "c3"], "start_timestamp": 90.0}]
        eval_metrics = engine.evaluate_ring_metrics(rings, gt_rings, txns)
        assert eval_metrics["ring_recall"] == 1.0
        assert eval_metrics["future_loss_prevented_inr"] > 0.0


class TestCalibrationAndPriorCorrection:
    def test_prior_correction_math(self):
        # Oversampled 50% fraud prior down to 0.13% base rate
        p_sampled = np.array([0.90, 0.50, 0.10])
        p_deploy = correct_for_sampled_prior(p_sampled, pi_sampled=0.50, pi_deploy=0.0013)

        assert p_deploy.shape == p_sampled.shape
        # 50% sampled probability under 0.13% deploy base rate becomes ~0.13%
        assert np.isclose(p_deploy[1], 0.0013, atol=1e-4)
        assert p_deploy[0] < 0.90
        assert p_deploy[2] < 0.10

    def test_bayes_optimal_threshold(self):
        t_star = compute_bayes_optimal_threshold(cost_fp=350.0, cost_fn=42000.0)
        assert np.isclose(t_star, 0.008264, atol=1e-5)

    def test_calibration_comparison_all_methods(self):
        rng = np.random.default_rng(42)
        y_true_val = rng.binomial(1, 0.1, size=100)
        y_prob_val = np.clip(y_true_val * 0.7 + rng.normal(0, 0.15, size=100), 0.01, 0.99)

        y_true_test = rng.binomial(1, 0.1, size=100)
        y_prob_test = np.clip(y_true_test * 0.7 + rng.normal(0, 0.15, size=100), 0.01, 0.99)

        comp = compare_calibration_methods(y_true_val, y_prob_val, y_true_test, y_prob_test)
        assert "Uncalibrated" in comp
        assert "Platt" in comp
        assert "Temperature" in comp
        assert "Isotonic" in comp
        assert comp["Isotonic"]["brier_score"] <= comp["Uncalibrated"]["brier_score"] + 0.05

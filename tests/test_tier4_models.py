"""
tests/test_tier4_models.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for Tier 4 research models: PC-GNN (WWW 2021) & CARE-GNN (CIKM 2020).
"""
import pytest
import torch

from src.models.pc_gnn import PCGNN, PCGNNSampler
from src.models.care_gnn import CAREGNN, SimilarityNeighborFilter


def test_pc_gnn_sampler():
    sampler = PCGNNSampler(fraud_oversample_ratio=2.0, clean_undersample_ratio=0.5)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    sampled = sampler.sample_subgraph(edge_index, labels, num_nodes=4)
    assert sampled.shape[0] == 2
    assert sampled.shape[1] > 0


def test_pc_gnn_forward():
    model = PCGNN(in_channels=10, hidden_channels=16, heads=2)
    x = torch.randn(8, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
    labels = torch.tensor([0, 1, 0, 1, 0, 0, 0, 1], dtype=torch.long)

    model.train()
    logits_train = model(x, edge_index, labels=labels)
    assert logits_train.shape == (8,)

    model.eval()
    logits_eval = model(x, edge_index)
    assert logits_eval.shape == (8,)


def test_care_gnn_similarity_filter():
    filt = SimilarityNeighborFilter(in_channels=8, similarity_threshold=0.3)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    filtered = filt.filter_edges(x, edge_index)
    assert filtered.shape[0] == 2
    assert filtered.shape[1] >= 1


def test_care_gnn_forward():
    model = CAREGNN(in_channels=10, hidden_channels=16, similarity_threshold=0.3)
    x = torch.randn(6, 10)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    logits = model(x, edge_index)
    assert logits.shape == (6,)


def test_fraud_hgt_forward():
    from src.models.hgt import FraudHGT

    node_types = ["customer", "merchant", "device"]
    edge_types = [
        ("customer", "TRANSACTED_AT", "merchant"),
        ("customer", "USED_DEVICE", "device"),
    ]
    metadata = (node_types, edge_types)
    in_dims = {"customer": 12, "merchant": 8, "device": 4}

    model = FraudHGT(metadata=metadata, in_dims=in_dims, hidden_channels=16, out_channels=16, num_heads=2)

    x_dict = {
        "customer": torch.randn(10, 12),
        "merchant": torch.randn(4, 8),
        "device": torch.randn(6, 4),
    }
    edge_index_dict = {
        ("customer", "TRANSACTED_AT", "merchant"): torch.tensor([[0, 1, 2, 3], [0, 1, 0, 2]], dtype=torch.long),
        ("customer", "USED_DEVICE", "device"): torch.tensor([[0, 1, 2, 4], [0, 1, 2, 3]], dtype=torch.long),
    }

    out = model(x_dict, edge_index_dict)
    assert "p_global" in out
    assert "p_promo" in out
    assert "p_return" in out
    assert "p_chargeback" in out
    assert "p_ato" in out
    assert out["p_global"].shape[0] == 10


def test_graph_info_nce_loss():
    from src.models.contrastive import GraphInfoNCELoss

    loss_fn = GraphInfoNCELoss(temperature=0.2)
    anchors = torch.randn(8, 16)
    positives = torch.randn(8, 16)
    negatives = torch.randn(8, 5, 16)

    loss = loss_fn(anchors, positives, negatives)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

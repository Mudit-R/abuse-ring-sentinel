"""
src/models/care_gnn.py
──────────────────────────────────────────────────────────────────────────────
CARE-GNN: Camouflage-Resistant Graph Neural Network for Fraud Detection

Reference:
    Dou et al., "Enhancing Graph Neural Network-based Fraud Detectors against
    Camouflaged Fraudsters", In Proceedings of ACM CIKM 2020.
    https://arxiv.org/abs/2008.08692

Theoretical Motivation:
    Fraud rings deliberately connect to clean legitimate accounts (relation camouflage)
    and synthesize legitimate feature signatures (feature camouflage) to dilute their
    graph connectivity.

    CARE-GNN addresses this via:
    1. Label-aware similarity scoring between node representations.
    2. Adaptive threshold / top-p neighbor selection that filters out dissimilar,
       camouflaging connections during message aggregation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityNeighborFilter(nn.Module):
    """
    Computes pairwise feature similarity between connected nodes and filters
    out camouflaged edges whose similarity falls below an adaptive threshold.
    """

    def __init__(self, in_channels: int, similarity_threshold: float = 0.45) -> None:
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.proj = nn.Linear(in_channels, in_channels, bias=False)

    def filter_edges(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        threshold: float | None = None,
    ) -> torch.Tensor:
        th = threshold if threshold is not None else self.similarity_threshold
        src, dst = edge_index[0], edge_index[1]

        # Project features for similarity measurement
        h_src = F.normalize(self.proj(x[src]), p=2, dim=-1)
        h_dst = F.normalize(self.proj(x[dst]), p=2, dim=-1)

        # Cosine similarity across edges
        sim = (h_src * h_dst).sum(dim=-1)

        # Filter out edges where similarity is below threshold (relation camouflage)
        keep_mask = sim >= th
        if keep_mask.sum() == 0:
            # Fallback: keep top 20% if all pruned
            k = max(1, int(0.2 * len(sim)))
            _, top_idx = torch.topk(sim, k)
            keep_mask = torch.zeros_like(sim, dtype=torch.bool)
            keep_mask[top_idx] = True

        return edge_index[:, keep_mask]


class CAREGNN(nn.Module):
    """
    CARE-GNN Architecture: Similarity-guided adaptive neighbor filtering
    coupled with multi-layer graph aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        similarity_threshold: float = 0.45,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.neighbor_filter = SimilarityNeighborFilter(in_channels, similarity_threshold)
        self.fc1 = nn.Linear(in_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Step 1: Filter camouflaged edges using similarity metric
        filtered_edge_index = self.neighbor_filter.filter_edges(x, edge_index)

        # Step 2: Mean aggregation over filtered neighbors
        src, dst = filtered_edge_index[0], filtered_edge_index[1]
        agg = torch.zeros_like(x)
        ones = torch.zeros((x.size(0), 1), device=x.device)

        agg.index_add_(0, dst, x[src])
        ones.index_add_(0, dst, torch.ones((dst.size(0), 1), device=x.device))
        ones = torch.clamp(ones, min=1.0)
        neighbor_repr = agg / ones

        # Step 3: Self + Neighbor concatenation & MLP classification
        combined = torch.cat([x, neighbor_repr], dim=-1)
        h = F.relu(self.fc1(combined))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.fc2(h))

        logits = self.classifier(h).squeeze(-1)
        return logits

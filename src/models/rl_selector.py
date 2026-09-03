"""
src/models/rl_selector.py
──────────────────────────────────────────────────────────────────────────────
RL-Inspired Adaptive Neighbor Selection Policy for Camouflage Defense.

References:
    - DiG-In-GNN: Discriminative Guidance-Informed Graph Neural Network, AAAI 2024.
    - Section 6.3 of the Production Merchant Fraud GNN Specification.

Mechanism:
    Evaluates candidate edges between target nodes and their neighbors under
    a policy network trained to drop camouflaged / decoy connections while
    penalizing computational latency and degree explosion.
"""
from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveNeighborSelector(nn.Module):
    """
    Policy network for adaptive edge filtering.
    Decides whether to retain or prune candidate edges.
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_feat_dim: int = 12,
        hidden_dim: int = 32,
        keep_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.keep_threshold = keep_threshold

        # Input state: [h_target (D), h_neighbor (D), cos_sim (1), delta_t (1), edge_attr (E)]
        state_dim = node_dim * 2 + 2 + edge_feat_dim

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def compute_edge_probabilities(
        self,
        h_target: torch.Tensor,
        h_neighbor: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the keep probability for each candidate edge.
        """
        # Cosine feature similarity
        norm_t = F.normalize(h_target, dim=-1)
        norm_n = F.normalize(h_neighbor, dim=-1)
        cos_sim = (norm_t * norm_n).sum(dim=-1, keepdim=True)

        dt = delta_t if delta_t is not None else torch.zeros((h_target.size(0), 1), device=h_target.device)
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        dt_norm = torch.log1p(torch.clamp(dt, min=0.0))

        if edge_attr is None:
            edge_attr = torch.zeros((h_target.size(0), 12), device=h_target.device)

        state = torch.cat([h_target, h_neighbor, cos_sim, dt_norm, edge_attr], dim=-1)
        keep_probs = self.policy_net(state).squeeze(-1)
        return keep_probs

    def filter_edges(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Filters edge_index based on policy threshold.
        Returns: (filtered_edge_index, filtered_edge_attr, keep_mask)
        """
        if edge_index.numel() == 0:
            return edge_index, edge_attr, torch.empty(0, dtype=torch.bool, device=edge_index.device)

        src, dst = edge_index[0], edge_index[1]
        probs = self.compute_edge_probabilities(
            h_target=x[dst],
            h_neighbor=x[src],
            edge_attr=edge_attr,
        )

        keep_mask = probs >= self.keep_threshold

        # Safety: preserve at least top 15% if threshold drops everything
        if keep_mask.sum() == 0:
            k = max(1, int(0.15 * len(probs)))
            _, top_idx = torch.topk(probs, k)
            keep_mask = torch.zeros_like(probs, dtype=torch.bool)
            keep_mask[top_idx] = True

        filtered_edge_index = edge_index[:, keep_mask]
        filtered_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None

        return filtered_edge_index, filtered_edge_attr, keep_mask

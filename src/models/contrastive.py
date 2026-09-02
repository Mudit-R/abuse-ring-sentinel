"""
src/models/contrastive.py
──────────────────────────────────────────────────────────────────────────────
Contrastive Neighbor Selection & InfoNCE Loss for Fraud Graph Representation

References:
    - Zhang et al., "DiG-In-GNN: Discriminative Guidance-Informed Graph Neural Network
      for Fraud Detection", AAAI 2024.
    - CACO-GNN: Contrastive Graph Neural Network-based Camouflaged Fraud Detector,
      Information Sciences 2022.
    - Wang et al., "TH-GCL: Temporal Heterogeneous Graph Contrastive Learning",
      IEEE Access 2025.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphInfoNCELoss(nn.Module):
    """
    Supervised/Self-Supervised InfoNCE loss for graph node representations.
    
    Pulls embeddings of genuine colluding ring members together while pushing
    decoy/camouflaged connections apart.
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        anchor_embeddings   : (N, D) Anchor node representations
        positive_embeddings : (N, D) Same-ring or true relational neighbor representations
        negative_embeddings : (N, K, D) Decoy / negative neighbor representations

        Returns
        -------
        loss : Scalar InfoNCE loss
        """
        anchor = F.normalize(anchor_embeddings, dim=-1)
        positive = F.normalize(positive_embeddings, dim=-1)
        negatives = F.normalize(negative_embeddings, dim=-1)

        # Positive pair similarity (N, 1)
        pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / self.temperature

        # Negative pairs similarity (N, K)
        neg_sim = torch.einsum("nd,nkd->nk", anchor, negatives) / self.temperature

        # Logits: (N, 1 + K) where index 0 is the positive class
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)

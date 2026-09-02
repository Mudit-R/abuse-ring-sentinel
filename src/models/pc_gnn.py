"""
src/models/pc_gnn.py
──────────────────────────────────────────────────────────────────────────────
PC-GNN: Pick-and-Choose Neighbor Sampler for Imbalanced Graph Fraud Detection

Reference:
    Liu et al., "Pick and Choose: A GNN-based Imbalanced Learning Approach
    for Fraud Detection", In Proceedings of The Web Conference (WWW 2021).
    https://dl.acm.org/doi/10.1145/3442381.3449989

Theoretical Motivation:
    In transaction graphs with extreme class imbalance (e.g. 0.13% fraud),
    standard isotropic or attention-based message passing suffers from
    "neighborhood dilution": because >99% of adjacent nodes belong to the
    legitimate majority class, standard aggregation dilutes the minority fraud
    signal with majority noise.

    PC-GNN addresses this via:
    1. Label-balanced anchor sampling: ensuring minority class nodes are evenly
       represented in training batches.
    2. Over-sampling minority neighbors & under-sampling majority neighbors
       during GNN message aggregation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm


class PCGNNSampler:
    """
    Imbalanced neighborhood sampler inspired by PC-GNN (Liu et al., WWW 2021).
    
    Over-samples edges connecting to minority (fraud) nodes and under-samples
    edges connecting to majority (clean) nodes to prevent neighborhood dilution.
    """

    def __init__(
        self,
        fraud_oversample_ratio: float = 4.0,
        clean_undersample_ratio: float = 0.3,
        random_seed: int = 42,
    ) -> None:
        self.fraud_ratio = fraud_oversample_ratio
        self.clean_ratio = clean_undersample_ratio
        self.seed = random_seed

    def sample_subgraph(
        self,
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Samples a balanced edge_index given node labels.

        Parameters
        ----------
        edge_index : (2, E) graph edges
        labels     : (N,) binary labels (1=fraud, 0=clean)
        num_nodes  : Total number of nodes

        Returns
        -------
        sampled_edge_index : (2, E_sampled) balanced edges
        """
        src, dst = edge_index[0], edge_index[1]
        is_dst_fraud = labels[dst] == 1

        fraud_edge_indices = torch.nonzero(is_dst_fraud).squeeze(-1)
        clean_edge_indices = torch.nonzero(~is_dst_fraud).squeeze(-1)

        # Under-sample clean edges
        num_clean_keep = int(len(clean_edge_indices) * self.clean_ratio)
        perm = torch.randperm(len(clean_edge_indices))
        selected_clean = clean_edge_indices[perm[:num_clean_keep]]

        # Over-sample fraud edges (replicate with replacement if needed)
        repeat_count = int(self.fraud_ratio)
        selected_fraud = fraud_edge_indices.repeat(repeat_count)

        combined_indices = torch.cat([selected_clean, selected_fraud])
        return edge_index[:, combined_indices]


class PCGNN(nn.Module):
    """
    PC-GNN Model: Integrates Pick-and-Choose Neighbor Sampling with GATv2 layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.sampler = PCGNNSampler()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.bn1 = BatchNorm(hidden_channels * heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False)
        self.bn2 = BatchNorm(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # If in training mode and labels provided, apply PC-GNN balanced sampling
        if self.training and labels is not None:
            edge_index = self.sampler.sample_subgraph(edge_index, labels, x.size(0))

        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)

        logits = self.classifier(h).squeeze(-1)
        return logits

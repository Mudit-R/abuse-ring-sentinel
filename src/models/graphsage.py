"""
src/models/graphsage.py
──────────────────────────────────────────────────────────────────────────────
GraphSAGE (SAmple and aggreGatE) for node-level fraud classification.

Architecture:
    3× SAGEConv layers with BatchNorm + ReLU + Dropout
    → Linear classifier

Reference:
    Hamilton et al., "Inductive Representation Learning on Large Graphs",
    NeurIPS 2017. https://arxiv.org/abs/1706.02216

Why GraphSAGE over GCN for production?
    GraphSAGE is INDUCTIVE — it learns an aggregation function that can be
    applied to nodes not seen during training. In a fraud system, new accounts
    appear constantly. GCN requires retraining on the full graph; GraphSAGE
    can score new nodes by sampling their neighbourhoods at inference time.

    It also supports mini-batch training via NeighborLoader, making it the
    preferred architecture when the graph has millions of nodes (like PaySim).

Aggregator:
    Using 'mean' aggregator (default). 'max' is faster but loses distribution
    information. 'lstm' is richer but slower and harder to explain.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv


class GraphSAGE(nn.Module):
    """
    3-layer GraphSAGE model.

    Parameters
    ----------
    in_channels      : Input feature dimension
    hidden_channels  : Hidden layer width
    out_channels     : Output classes (1 for binary)
    num_layers       : Number of SAGE conv layers
    dropout          : Dropout rate
    aggr             : Aggregation function — 'mean', 'max', 'lstm'
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(BatchNorm(hidden_channels))

        # Last conv
        self.convs.append(SAGEConv(hidden_channels, hidden_channels // 2, aggr=aggr))
        self.bns.append(BatchNorm(hidden_channels // 2))

        self.classifier = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)  # (N,)

    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return x

    def __repr__(self) -> str:
        return (
            f"GraphSAGE(in={self.convs[0].in_channels}, "
            f"hidden={self.convs[0].out_channels}, "
            f"layers={self.num_layers}, "
            f"dropout={self.dropout})"
        )

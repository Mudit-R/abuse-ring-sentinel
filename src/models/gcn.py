"""
src/models/gcn.py
──────────────────────────────────────────────────────────────────────────────
Graph Convolutional Network (GCN) for node-level fraud classification.

Architecture:
    3× GCNConv layers with BatchNorm + ReLU + Dropout
    → Linear classifier → sigmoid output

Reference:
    Kipf & Welling, "Semi-supervised Classification with Graph Convolutional
    Networks", ICLR 2017. https://arxiv.org/abs/1609.02907

Why GCN for fraud?
    GCN aggregates features from a node's 1-hop neighbourhood. A fraudulent
    account's neighbours (accounts it transacted with) often have correlated
    behaviour — high velocity, balance drains, etc. GCN captures this
    "guilt by association" signal that tabular models miss.

Limitation (talk about this in interviews):
    GCN is transductive — it cannot generalise to unseen nodes without
    retraining. For production, GraphSAGE (inductive) is preferred.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv


class GCN(nn.Module):
    """
    3-layer Graph Convolutional Network.

    Parameters
    ----------
    in_channels   : Input feature dimension (from feature matrix)
    hidden_channels : Width of hidden layers
    out_channels  : Number of output classes (1 for binary classification)
    num_layers    : Number of GCN conv layers (default: 3)
    dropout       : Dropout rate applied after each hidden layer
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=True))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=True))
            self.bns.append(BatchNorm(hidden_channels))

        # Output conv layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels // 2, normalize=True))
        self.bns.append(BatchNorm(hidden_channels // 2))

        # Final classifier head
        self.classifier = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : Node feature matrix (N, F)
        edge_index : Graph connectivity (2, E)

        Returns
        -------
        logits : (N, 1) — raw scores (apply sigmoid for probabilities)
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:  # no dropout before last conv
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)  # (N,)

    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return pre-classifier embeddings for explainability / visualisation."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return x

    def __repr__(self) -> str:
        return (
            f"GCN(in={self.convs[0].in_channels}, "
            f"hidden={self.convs[0].out_channels}, "
            f"layers={self.num_layers}, "
            f"dropout={self.dropout})"
        )

"""
src/models/gat.py
──────────────────────────────────────────────────────────────────────────────
Graph Attention Network (GAT) for node-level fraud classification.

Architecture:
    3× GATv2Conv layers (multi-head) with BatchNorm + ELU + Dropout
    → Linear classifier

Reference:
    Brody et al., "How Attentive are Graph Attention Networks?",
    ICLR 2022. https://arxiv.org/abs/2105.14491
    (GATv2 fixes the static attention limitation of original GAT)

Why GAT for fraud?
    Not all neighbours are equally suspicious. A mule account transacting with
    5 known-fraud accounts and 100 normal accounts should weight the fraud
    connections more heavily. GAT learns per-edge attention weights, allowing
    the model to focus on the highest-risk connections.

    The learned attention weights are also interpretable — you can visualise
    which edges drove the fraud prediction, which is exactly what AML
    compliance teams need.

Hyperparameter notes:
    - heads: 8 multi-head attention is standard. Produces richer representations
      but multiplies memory by `heads`.
    - concat: True during hidden layers (concatenate heads), False at output
      (average heads) to keep the final representation compact.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv


class GAT(nn.Module):
    """
    3-layer Graph Attention Network (GATv2).

    Parameters
    ----------
    in_channels     : Input feature dimension
    hidden_channels : Width per attention head
    out_channels    : Output classes (1 for binary)
    num_layers      : Number of GAT conv layers
    heads           : Number of attention heads in hidden layers
    dropout         : Dropout on attention coefficients and node features
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer — multi-head, concatenate outputs
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads,
                      dropout=dropout, concat=True)
        )
        self.bns.append(BatchNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels,
                          heads=heads, dropout=dropout, concat=True)
            )
            self.bns.append(BatchNorm(hidden_channels * heads))

        # Output layer — average heads (concat=False)
        self.convs.append(
            GATv2Conv(hidden_channels * heads, hidden_channels,
                      heads=1, dropout=dropout, concat=False)
        )
        self.bns.append(BatchNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        """
        Parameters
        ----------
        x                       : (N, F) node features
        edge_index              : (2, E) graph connectivity
        return_attention_weights: If True, also returns the last layer's
                                  attention weights for visualisation.

        Returns
        -------
        logits : (N,)
        (optionally) (edge_index, alpha) for the final attention layer
        """
        attn_out = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            is_last = i == self.num_layers - 1
            if is_last and return_attention_weights:
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                attn_out = (ei, alpha)
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            if not is_last:
                x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x).squeeze(-1)

        if return_attention_weights and attn_out is not None:
            return logits, attn_out
        return logits

    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
        return x

    def __repr__(self) -> str:
        return (
            f"GAT(in={self.convs[0].in_channels}, "
            f"hidden={self.convs[0].out_channels}, "
            f"layers={self.num_layers})"
        )

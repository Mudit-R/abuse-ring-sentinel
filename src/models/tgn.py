"""
src/models/tgn.py
──────────────────────────────────────────────────────────────────────────────
Temporal Graph Network (TGN) — stretch goal model.

TGNs are the state-of-the-art for fraud detection on dynamic graphs where
the TIMING of transactions matters (not just the static structure).

Why temporal graphs for fraud?
    A mule account might look normal if you only look at the static graph —
    it has reasonable in/out ratios and is not directly connected to known
    fraudsters. But the TIMING pattern reveals it:
      • Receives a large transfer → immediately (within 1-2 hours) → cashes out
    TGN captures this velocity signal at the graph level, not just in tabular
    rolling windows.

Architecture:
    TGN uses a Memory module to track each node's state over time, plus
    a Graph Attention network to aggregate neighbourhood information at each
    event. This implementation uses the PyTorch Geometric Temporal library's
    TGN components.

Reference:
    Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs",
    ICML 2020 Workshop. https://arxiv.org/abs/2006.10637

Note:
    Full TGN training is computationally intensive. For the project, we train
    on a 10% time-ordered subsample and compare PR-AUC to the static GAT.
    Expected improvement: ~3-5 PR-AUC points on temporal test set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.nn import TGNMemory, TransformerConv

try:
    from torch_geometric_temporal.nn.recurrent import EvolveGCNH
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    logger.warning(
        "torch_geometric_temporal not installed. "
        "EvolveGCN unavailable. Install with: pip install torch-geometric-temporal"
    )


# ── TGN Memory Module ─────────────────────────────────────────────────────────

class TemporalFraudGNN(nn.Module):
    """
    Simplified TGN-style model using PyG's TGNMemory + Graph Attention.

    This is not the full TGN paper implementation, but captures the key ideas:
      1. Memory: each node maintains a state vector that evolves with each event
      2. Message passing: aggregate from neighbours using their current memory

    For the full implementation, use the official TGN code:
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int = 100,
        time_dim: int = 100,
        embedding_dim: int = 100,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=nn.Linear(2 * memory_dim + raw_msg_dim + time_dim, memory_dim),
            aggregator_module=nn.GRUCell(memory_dim, memory_dim),
        )

        self.gnn_conv = TransformerConv(
            in_channels=memory_dim,
            out_channels=embedding_dim // 4,
            heads=4,
            dropout=dropout,
            edge_dim=raw_msg_dim,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.dropout = dropout

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a batch of temporal events.

        src, dst : (E,) source and destination node IDs of events
        t        : (E,) event timestamps
        msg      : (E, D) raw event messages (edge features)
        """
        z, last_update = self.memory(src, dst, t, msg)
        embeddings = self.gnn_conv(z, edge_index, edge_attr)
        return self.classifier(embeddings).squeeze(-1)

    def detach_memory(self) -> None:
        """Detach memory state from computational graph (call between batches)."""
        self.memory.detach_()


# ── EvolveGCN (Alternative Temporal Architecture) ────────────────────────────

class EvolveGCNFraud(nn.Module):
    """
    EvolveGCN-H: evolves GCN weights over time using a GRU.

    Simpler than TGN but very effective for periodic snapshot graphs.
    Use this if you want to model the graph as monthly snapshots rather
    than individual events.

    Reference:
        Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks for
        Dynamic Graphs", AAAI 2020. https://arxiv.org/abs/1902.10191
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if not TEMPORAL_AVAILABLE:
            raise ImportError(
                "torch_geometric_temporal required. "
                "pip install torch-geometric-temporal"
            )

        self.convs = nn.ModuleList([
            EvolveGCNH(in_channels=in_channels, num_of_nodes=1)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = torch.relu(x)
        return self.classifier(x).squeeze(-1)

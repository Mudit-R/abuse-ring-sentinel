"""
src/models/hgt.py
──────────────────────────────────────────────────────────────────────────────
FraudHGT: Heterogeneous Graph Transformer for E-Commerce & Merchant Abuse Rings

References:
    - Hu et al., "Heterogeneous Graph Transformer", In Proceedings of The Web Conference (WWW 2020).
    - Zhang et al., "eFraudCom: An E-commerce Fraud Detection System via Competitive GNNs", ACM TOIS (2022).
    - Adyen Global Risk Engine (2024–2025): Entity-relation graph projections for merchant risk.

Architecture:
    Heterogeneous Event/Entity Graph:
    Node Types: ['customer', 'merchant', 'device', 'address', 'vpa', 'promo']
    Edge Types:
      - ('customer', 'TRANSACTED_AT', 'merchant')
      - ('customer', 'USED_DEVICE', 'device')
      - ('customer', 'DELIVERED_TO', 'address')
      - ('customer', 'PAID_WITH', 'vpa')
      - ('customer', 'APPLIED', 'promo')
      
    Outputs multi-task risk heads:
      - p_promo (Promo-Abuse Ring Risk)
      - p_return (RTO / Return Fraud Risk)
      - p_chargeback (Chargeback Collusion Risk)
      - p_ato (Account-Takeover Surge Risk)
"""
from __future__ import annotations

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroRelationAttention(nn.Module):
    """
    Relation-aware multi-head attention convolution for heterogeneous graphs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.num_heads = num_heads
        self.d_k = out_channels // num_heads

        node_types, edge_types = metadata

        # Type-specific Q, K, V projections
        self.q_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, out_channels) for nt in node_types
        })
        self.k_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, out_channels) for nt in node_types
        })
        self.v_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, out_channels) for nt in node_types
        })

        # Relation-specific weighting matrices
        self.rel_weights = nn.ParameterDict({
            f"{src}__{rel}__{dst}": nn.Parameter(torch.ones(num_heads))
            for (src, rel, dst) in edge_types
        })

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {k: torch.zeros((x.size(0), self.out_channels), device=x.device) for k, x in x_dict.items()}
        counts: Dict[str, torch.Tensor] = {k: torch.zeros((x.size(0), 1), device=x.device) for k, x in x_dict.items()}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel, dst_type = edge_type
            rel_key = f"{src_type}__{rel}__{dst_type}"

            if src_type not in x_dict or dst_type not in x_dict:
                continue

            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            if len(src_nodes) == 0:
                continue

            # Compute Q on target/dst and K, V on source
            Q = self.q_proj[dst_type](x_dict[dst_type][dst_nodes]).view(-1, self.num_heads, self.d_k)
            K = self.k_proj[src_type](x_dict[src_type][src_nodes]).view(-1, self.num_heads, self.d_k)
            V = self.v_proj[src_type](x_dict[src_type][src_nodes]).view(-1, self.num_heads, self.d_k)

            # Scaled Dot-Product Attention
            scores = (Q * K).sum(dim=-1) / (self.d_k ** 0.5)
            if rel_key in self.rel_weights:
                scores = scores * self.rel_weights[rel_key]

            alpha = torch.sigmoid(scores).unsqueeze(-1) # (E, Heads, 1)
            msg = (alpha * V).view(-1, self.out_channels)

            out_dict[dst_type].index_add_(0, dst_nodes, msg)
            counts[dst_type].index_add_(0, dst_nodes, torch.ones((len(dst_nodes), 1), device=x_dict[dst_type].device))

        # Average aggregation and residual connection
        res_dict = {}
        for k, v in x_dict.items():
            norm_factor = torch.clamp(counts[k], min=1.0)
            res_dict[k] = (out_dict[k] / norm_factor) + (self.q_proj[k](v) if self.in_channels == self.out_channels else v)
        return res_dict


class FraudHGT(nn.Module):
    """
    Heterogeneous Graph Transformer for multi-entity merchant payment graphs.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_dims: Dict[str, int],
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.dropout = dropout

        # Feature projection per node type
        self.projections = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_channels)
            for node_type, dim in in_dims.items()
        })

        # Heterogeneous Attention Layers
        self.convs = nn.ModuleList([
            HeteroRelationAttention(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.node_emb = nn.Linear(hidden_channels, out_channels)

        # Multi-task Specialized Fraud Heads
        self.promo_head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.return_head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.chargeback_head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ato_head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Global Merchant Consensus Combiner
        self.merchant_combiner = nn.Linear(4 + out_channels, 1)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Step 1: Linear feature projection
        h_dict = {
            k: F.relu(self.projections[k](x))
            for k, x in x_dict.items()
            if k in self.projections
        }

        # Step 2: Message Passing
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in h_dict.items()}

        # Step 3: Embeddings
        z_dict = {k: F.normalize(self.node_emb(v), dim=-1) for k, v in h_dict.items()}

        # Step 4: Multi-task heads on customer/merchant nodes
        target_repr = z_dict.get("customer", z_dict.get("merchant"))
        if target_repr is None:
            first_key = list(z_dict.keys())[0]
            target_repr = z_dict[first_key]

        p_promo = torch.sigmoid(self.promo_head(target_repr)).squeeze(-1)
        p_return = torch.sigmoid(self.return_head(target_repr)).squeeze(-1)
        p_chargeback = torch.sigmoid(self.chargeback_head(target_repr)).squeeze(-1)
        p_ato = torch.sigmoid(self.ato_head(target_repr)).squeeze(-1)

        # Combined merchant risk
        stacked_heads = torch.stack([p_promo, p_return, p_chargeback, p_ato], dim=-1)
        combined_input = torch.cat([stacked_heads, target_repr], dim=-1)
        p_global = torch.sigmoid(self.merchant_combiner(combined_input)).squeeze(-1)

        return {
            "p_global": p_global,
            "p_promo": p_promo,
            "p_return": p_return,
            "p_chargeback": p_chargeback,
            "p_ato": p_ato,
            "embeddings": z_dict,
        }

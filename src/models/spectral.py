"""
src/models/spectral.py
──────────────────────────────────────────────────────────────────────────────
Lightweight Chebyshev Spectral Graph Filter Branch (Camouflage Defense).

References:
    - SplitGNN: Spectral/Heterophily-Aware Fraud Detection, ACM CIKM 2023.
    - Hammond et al., "Wavelets on Graphs via Spectral Graph Theory", ACHA 2011.
    - Section 6.2 of the Production Merchant Fraud GNN Specification:
      Normalized Laplacian L = I - D^(-1/2) A D^(-1/2) with order K=2 Chebyshev
      polynomial spectral filtering approximation.

Theoretical Motivation:
    Fraud rings deliberately create camouflaged edges to clean nodes (heterophily).
    Standard low-pass GNN spatial aggregators over-smooth and mix fraudulent and
    clean signals. The spectral filter separates high-frequency boundary discrepancies
    from low-frequency community cluster signals.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebyshevSpectralFilter(nn.Module):
    """
    Lightweight Chebyshev Spectral Filter (Order K=2).
    Operates on node feature representations using the normalized graph Laplacian.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 2) -> None:
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learnable Chebyshev polynomial coefficients theta_k
        self.weight = nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    @staticmethod
    def compute_scaled_laplacian(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the symmetric normalized Laplacian L = I - D^(-1/2) A D^(-1/2)
        scaled to [-1, 1] for stable Chebyshev recurrence.
        """
        if edge_index.numel() == 0 or num_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), torch.empty(0, device=edge_index.device)

        src, dst = edge_index[0], edge_index[1]

        # Degree computation
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        # Normalized adjacency edge values: A_norm = D^(-1/2) A D^(-1/2)
        norm_val = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        # L_scaled = (2 / lambda_max) * L - I, where lambda_max ~ 2.0
        # Simplifies to: - D^(-1/2) A D^(-1/2)
        return edge_index, -norm_val

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Applies K-order Chebyshev recurrence:
            T_0(L) = x
            T_1(L) = L_scaled @ x
            T_k(L) = 2 * L_scaled @ T_{k-1}(L) - T_{k-2}(L)
        """
        num_nodes = x.size(0)
        if num_nodes == 0:
            return torch.empty((0, self.out_channels), device=x.device)

        edge_idx, edge_weight = self.compute_scaled_laplacian(edge_index, num_nodes)

        def sparse_matmul(src_x: torch.Tensor) -> torch.Tensor:
            if edge_idx.numel() == 0:
                return torch.zeros_like(src_x)
            out = torch.zeros_like(src_x)
            weighted_x = src_x[edge_idx[0]] * edge_weight.unsqueeze(-1)
            out.index_add_(0, edge_idx[1], weighted_x)
            return out

        # T_0 = x
        T0 = x
        out = torch.matmul(T0, self.weight[0])

        if self.K >= 1:
            # T_1 = L_scaled @ x
            T1 = sparse_matmul(x)
            out = out + torch.matmul(T1, self.weight[1])

        if self.K >= 2:
            # T_2 = 2 * L_scaled @ T1 - T0
            T2 = 2.0 * sparse_matmul(T1) - T0
            out = out + torch.matmul(T2, self.weight[2])

        out = out + self.bias
        return out

"""
src/models/focal_loss.py
──────────────────────────────────────────────────────────────────────────────
Focal Loss for extreme class imbalance.

PaySim fraud rate ≈ 0.13% — standard BCE will achieve 99.87% accuracy by
predicting all-benign. Focal Loss down-weights easy negatives and focuses
training on hard, misclassified examples.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    https://arxiv.org/abs/1708.02002

Parameters:
    alpha : float
        Weighting factor for the minority (fraud) class. Typical range: 0.25–0.75.
        Higher alpha → more emphasis on fraud recall.
    gamma : float
        Focusing parameter. gamma=0 → standard BCE.
        gamma=2 (default) is the value from the original paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss for node-level fraud classification."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N,) raw output from the final linear layer (pre-sigmoid)
        targets : (N,) binary labels {0, 1}
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)

        # p_t = p for positive class, (1-p) for negative class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # alpha_t = alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction!r}"

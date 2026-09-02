"""
src/drift/psi.py
──────────────────────────────────────────────────────────────────────────────
Population Stability Index (PSI) for model drift detection.

PSI measures how much the distribution of a variable has shifted between
a reference (training) distribution and a current (production) distribution.

Thresholds (industry standard):
    PSI < 0.10  : No significant change, model is stable
    PSI < 0.20  : Moderate change, monitor closely
    PSI >= 0.20 : Major shift, trigger model retraining

Why this matters for fraud:
    Fraud patterns evolve continuously — attackers adapt to known detection
    patterns. A model trained on January data may have PSI > 0.25 by March.
    PSI checks are a lightweight leading indicator that catch distribution
    shift BEFORE performance metrics degrade (which requires ground truth
    labels that arrive with delay in fraud systems).

This is the kind of production ML maturity that FAANG and fintech companies
look for — it shows you think beyond training accuracy to real deployment
lifecycle management.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ── Core PSI Implementation ───────────────────────────────────────────────────

def compute_psi_single(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Compute PSI for a single feature or score distribution.

    PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)

    Parameters
    ----------
    reference : Distribution from training (reference period)
    current   : Distribution from production (current period)
    n_bins    : Number of buckets (10 is standard for PSI)
    epsilon   : Smoothing term to avoid log(0)

    Returns
    -------
    psi : float — higher = more drift
    """
    # Use reference distribution to define bin edges
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / len(reference) + epsilon
    cur_pct = cur_counts / len(current) + epsilon

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_psi_features(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute PSI for every feature column.

    Returns dict: feature_name → psi_value
    """
    n_features = reference_features.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    psi_dict = {}
    for i, name in enumerate(feature_names):
        psi_dict[name] = compute_psi_single(
            reference_features[:, i], current_features[:, i], n_bins=n_bins
        )

    return psi_dict


# ── Drift Monitor ─────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Monitors model score distribution and feature distributions for drift.

    Usage:
        monitor = DriftMonitor(reference_scores, reference_features, feature_names)
        alert = monitor.check(current_scores, current_features)
        if alert.has_drift:
            trigger_retraining()
    """

    PSI_WARN = 0.10
    PSI_ALERT = 0.20

    def __init__(
        self,
        reference_scores: np.ndarray,
        reference_features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_bins: int = 10,
    ) -> None:
        self.reference_scores = reference_scores
        self.reference_features = reference_features
        self.feature_names = feature_names
        self.n_bins = n_bins

    def check(
        self,
        current_scores: np.ndarray,
        current_features: np.ndarray,
    ) -> "DriftReport":
        """
        Run PSI on model scores + all features.
        """
        score_psi = compute_psi_single(
            self.reference_scores, current_scores, n_bins=self.n_bins
        )
        feature_psi = compute_psi_features(
            self.reference_features, current_features,
            feature_names=self.feature_names, n_bins=self.n_bins,
        )

        report = DriftReport(
            score_psi=score_psi,
            feature_psi=feature_psi,
        )
        report.log()
        return report

    def save_reference(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            scores=self.reference_scores,
            features=self.reference_features,
        )
        logger.info(f"Reference distribution saved to {path}")

    @classmethod
    def load(cls, path: Path, feature_names: Optional[List[str]] = None) -> "DriftMonitor":
        data = np.load(path)
        return cls(
            reference_scores=data["scores"],
            reference_features=data["features"],
            feature_names=feature_names,
        )


class DriftReport:
    """Result of a drift check."""

    PSI_WARN = 0.10
    PSI_ALERT = 0.20

    def __init__(self, score_psi: float, feature_psi: Dict[str, float]) -> None:
        self.score_psi = score_psi
        self.feature_psi = feature_psi

        self.drifted_features = {
            k: v for k, v in feature_psi.items() if v >= self.PSI_ALERT
        }
        self.warned_features = {
            k: v for k, v in feature_psi.items()
            if self.PSI_WARN <= v < self.PSI_ALERT
        }
        self.has_drift = (score_psi >= self.PSI_ALERT) or bool(self.drifted_features)

    def log(self) -> None:
        if self.score_psi >= self.PSI_ALERT:
            logger.warning(
                f"⚠️  MODEL SCORE DRIFT DETECTED: PSI={self.score_psi:.4f} ≥ {self.PSI_ALERT}. "
                "Trigger model retraining."
            )
        elif self.score_psi >= self.PSI_WARN:
            logger.warning(f"Model score PSI={self.score_psi:.4f} — monitor closely.")
        else:
            logger.success(f"Model score PSI={self.score_psi:.4f} — stable.")

        if self.drifted_features:
            logger.warning(f"Drifted features (PSI ≥ {self.PSI_ALERT}): {self.drifted_features}")
        if self.warned_features:
            logger.warning(f"Warned features (PSI ≥ {self.PSI_WARN}): {self.warned_features}")

    def to_dict(self) -> Dict:
        return {
            "score_psi": self.score_psi,
            "has_drift": self.has_drift,
            "drifted_features": self.drifted_features,
            "warned_features": self.warned_features,
            "all_feature_psi": self.feature_psi,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Drift report saved: {path}")

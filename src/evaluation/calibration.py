"""
src/evaluation/calibration.py
──────────────────────────────────────────────────────────────────────────────
Probability Calibration & Reliability Assessment for Fraud Risk Models.

Under extreme class imbalance (0.13% fraud base rate), tree and neural network
raw predicted probabilities are often uncalibrated. This module evaluates and
corrects probability calibration using Isotonic Regression and Platt Scaling.

Key Metrics:
  - Brier Score (mean squared error of probability forecasts)
  - Expected Calibration Error (ECE across 10 probability bins)
  - Reliability Diagram coordinate generation
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Any
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute standard Brier Score: (1/N) * sum((prob - y)^2)."""
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


class ProbabilityCalibrator:
    """Calibrates model probability outputs using Isotonic Regression or Platt Scaling."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        elif method == "platt":
            self.calibrator = LogisticRegression(solver="lbfgs")
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        self.is_fitted = False

    def fit(self, y_prob_val: np.ndarray, y_true_val: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "isotonic":
            self.calibrator.fit(y_prob_val, y_true_val)
        else:
            self.calibrator.fit(y_prob_val.reshape(-1, 1), y_true_val)
        self.is_fitted = True
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return y_prob
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(y_prob), 0.0, 1.0)
        else:
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    def evaluate_calibration_lift(
        self,
        y_true_test: np.ndarray,
        y_prob_uncalibrated: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculates before and after calibration metrics and reliability diagram data.
        """
        y_prob_calibrated = self.calibrate(y_prob_uncalibrated)

        uncal_brier = compute_brier_score(y_true_test, y_prob_uncalibrated)
        cal_brier = compute_brier_score(y_true_test, y_prob_calibrated)

        uncal_ece = compute_ece(y_true_test, y_prob_uncalibrated)
        cal_ece = compute_ece(y_true_test, y_prob_calibrated)

        # Compute reliability curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_true_test, y_prob_uncalibrated, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_true_test, y_prob_calibrated, n_bins=10)

        return {
            "uncalibrated": {
                "brier_score": round(uncal_brier, 5),
                "ece": round(uncal_ece, 5),
                "prob_true": prob_true_uncal.tolist(),
                "prob_pred": prob_pred_uncal.tolist(),
            },
            "calibrated": {
                "brier_score": round(cal_brier, 5),
                "ece": round(cal_ece, 5),
                "prob_true": prob_true_cal.tolist(),
                "prob_pred": prob_pred_cal.tolist(),
            },
            "brier_reduction_pct": round((1.0 - cal_brier / max(uncal_brier, 1e-6)) * 100, 2),
            "ece_reduction_pct": round((1.0 - cal_ece / max(uncal_ece, 1e-6)) * 100, 2),
        }

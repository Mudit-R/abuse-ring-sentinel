"""
src/evaluation/calibration.py
──────────────────────────────────────────────────────────────────────────────
Probability Calibration, Prior Correction, and Bayes-Optimal Decisioning.

Conforms to Sections 0, 13, 14, and 15 of the Merchant Fraud GNN Specification:
1. Prior Correction for Oversampled Training Priors (Section 14):
   odds_deploy = odds_sampled * [pi_d / (1 - pi_d)] / [pi_s / (1 - pi_s)]
2. Calibration Comparison (Section 13):
   - Platt Scaling (Sigmoid)
   - Temperature Scaling (Logit Temperature T)
   - Isotonic Regression (Monotonic Non-Decreasing Step Function)
3. Mathematical Cost-Aware Thresholding (Section 0 & 15):
   T*_Bayes = C_FP / (C_FP + C_FN) = 350 / (350 + 42000) ~= 0.008264 (0.83%)
   Clarified against Operational Cascade Threshold T* = 0.42 (analyst budget-bounded).
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Any, Optional
import numpy as np
from scipy.optimize import minimize
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


def correct_for_sampled_prior(
    y_prob_sampled: np.ndarray,
    pi_sampled: float,
    pi_deploy: float = 0.0013,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Applies odds transformation to correct for training-time positive oversampling (Section 14).
        odds_deploy = odds_sampled * [pi_d / (1 - pi_d)] / [pi_s / (1 - pi_s)]
        p_deploy = odds_deploy / (1 + odds_deploy)
    """
    p_clipped = np.clip(y_prob_sampled, epsilon, 1.0 - epsilon)
    odds_sampled = p_clipped / (1.0 - p_clipped)

    deploy_factor = (pi_deploy / (1.0 - pi_deploy)) / (pi_sampled / (1.0 - pi_sampled))
    odds_deploy = odds_sampled * deploy_factor
    p_deploy = odds_deploy / (1.0 + odds_deploy)

    return np.clip(p_deploy, 0.0, 1.0)


def compute_bayes_optimal_threshold(
    cost_fp: float = 350.0,
    cost_fn: float = 42000.0,
) -> float:
    """
    Mathematically exact Bayes-optimal threshold for calibrated probabilities:
        T* = C_FP / (C_FP + C_FN)
    For INR 350 FP and INR 42,000 FN, yields ~0.008264 (0.83%).
    """
    return cost_fp / (cost_fp + cost_fn)


class TemperatureScaler:
    """Optimizes temperature T > 0 on validation set to calibrate neural logits."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        def nll(t_val):
            t = max(t_val[0], 1e-2)
            scaled = np.clip(logits / t, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-scaled))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

        res = minimize(nll, [1.0], method="Nelder-Mead")
        self.temperature = float(max(res.x[0], 0.05))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.clip(logits / self.temperature, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-scaled))
        return np.clip(p, 0.0, 1.0)


class ProbabilityCalibrator:
    """Calibrates model probability outputs using Isotonic Regression, Platt Scaling, or Temperature Scaling."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        elif method == "platt":
            self.calibrator = LogisticRegression(solver="lbfgs")
        elif method == "temperature":
            self.calibrator = TemperatureScaler()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        self.is_fitted = False

    def fit(self, y_prob_val: np.ndarray, y_true_val: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "isotonic":
            self.calibrator.fit(y_prob_val, y_true_val)
        elif self.method == "temperature":
            # Invert sigmoid to approximate logits
            p_clip = np.clip(y_prob_val, 1e-6, 1.0 - 1e-6)
            logits = np.log(p_clip / (1.0 - p_clip))
            self.calibrator.fit(logits, y_true_val)
        else:
            self.calibrator.fit(y_prob_val.reshape(-1, 1), y_true_val)
        self.is_fitted = True
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return y_prob
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(y_prob), 0.0, 1.0)
        elif self.method == "temperature":
            p_clip = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
            logits = np.log(p_clip / (1.0 - p_clip))
            return self.calibrator.predict_proba(logits)
        else:
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    def evaluate_calibration_lift(
        self,
        y_true_test: np.ndarray,
        y_prob_uncalibrated: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates before and after calibration metrics and reliability diagram data."""
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
            "bayes_optimal_threshold": round(compute_bayes_optimal_threshold(), 6),
            "operational_threshold": 0.42,
        }


def compare_calibration_methods(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    y_true_test: np.ndarray,
    y_prob_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compares Platt Scaling, Temperature Scaling, and Isotonic Regression side-by-side (Section 13).
    """
    methods = ["platt", "temperature", "isotonic"]
    results = {}

    uncal_brier = compute_brier_score(y_true_test, y_prob_test)
    uncal_ece = compute_ece(y_true_test, y_prob_test)
    results["Uncalibrated"] = {"brier_score": round(uncal_brier, 5), "ece": round(uncal_ece, 5)}

    for m in methods:
        cal = ProbabilityCalibrator(method=m).fit(y_prob_val, y_true_val)
        p_cal = cal.calibrate(y_prob_test)
        results[m.capitalize()] = {
            "brier_score": round(compute_brier_score(y_true_test, p_cal), 5),
            "ece": round(compute_ece(y_true_test, p_cal), 5),
        }

    return results

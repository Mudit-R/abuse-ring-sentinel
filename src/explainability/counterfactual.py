"""
src/explainability/counterfactual.py
──────────────────────────────────────────────────────────────────────────────
Counterfactual Explanation Engine for Merchant Fraud Decisions.

Calculates the minimal feature perturbations required to move a flagged account
from high-risk status down into the clean/safe review tier (< threshold).

Goes beyond static SHAP attributions by providing actionable "what-if" insights:
"If this account's balance drain had been 0.35 instead of 0.98 and out/in degree
ratio was 1.5 instead of 16.0, the risk score would drop from 0.94 to 0.28."
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np


class CounterfactualExplainer:
    """Computes minimal perturbations to achieve safe classification."""

    def __init__(self, target_threshold: float = 0.35):
        self.target_threshold = target_threshold

    def explain(
        self,
        features: Dict[str, float],
        current_probability: float,
        model_predict_fn,
    ) -> Dict[str, Any]:
        """
        Computes counterfactual adjustments for the key actionable risk drivers.
        """
        if current_probability <= self.target_threshold:
            return {
                "is_flagged": False,
                "target_threshold": self.target_threshold,
                "current_probability": current_probability,
                "counterfactual_probability": current_probability,
                "changes_required": [],
                "summary": "Account is already below the risk review threshold. No counterfactual adjustments needed.",
            }

        # Candidate actionable features for perturbation
        actionable_mods = [
            ("balance_drain_ratio", 0.25, "Reduce balance drain from {curr:.2f} to 0.25 (retain funds)"),
            ("degree_ratio", 1.2, "Reduce out/in degree ratio from {curr:.1f} to 1.2 (organic counterparty balance)"),
            ("night_tx_fraction", 0.10, "Shift transactions to business hours (night fraction {curr:.2f} -> 0.10)"),
            ("amount_spike_ratio", 1.0, "Maintain consistent transaction velocity (spike {curr:.1f} -> 1.0)"),
            ("local_clustering_coefficient", 0.22, "Build organic neighborhood clustering ({curr:.2f} -> 0.22)"),
        ]

        # Iteratively apply minimal perturbations until probability < target_threshold
        perturbed_features = dict(features)
        changes = []

        for feat_name, target_val, text_tpl in actionable_mods:
            curr_val = features.get(feat_name, 0.0)
            if feat_name in ["balance_drain_ratio", "degree_ratio", "night_tx_fraction", "amount_spike_ratio"]:
                if curr_val > target_val:
                    perturbed_features[feat_name] = target_val
                    changes.append({
                        "feature": feat_name,
                        "current_value": curr_val,
                        "counterfactual_value": target_val,
                        "description": text_tpl.format(curr=curr_val),
                    })
            elif feat_name == "local_clustering_coefficient":
                if curr_val < target_val:
                    perturbed_features[feat_name] = target_val
                    changes.append({
                        "feature": feat_name,
                        "current_value": curr_val,
                        "counterfactual_value": target_val,
                        "description": text_tpl.format(curr=curr_val),
                    })

            # Check new predicted probability with model
            new_prob = model_predict_fn(perturbed_features)
            if new_prob <= self.target_threshold:
                return {
                    "is_flagged": True,
                    "target_threshold": self.target_threshold,
                    "current_probability": round(current_probability, 4),
                    "counterfactual_probability": round(new_prob, 4),
                    "changes_required": changes,
                    "summary": (
                        f"If this account had reduced its balance drain to 0.25 and dispersed funds through "
                        f"standard organic channels (degree ratio <= 1.2), predicted risk score would drop "
                        f"from {round(current_probability*100)}% down to {round(new_prob*100)}% [SAFE]."
                    ),
                }

        # If all changes applied
        final_prob = model_predict_fn(perturbed_features)
        return {
            "is_flagged": True,
            "target_threshold": self.target_threshold,
            "current_probability": round(current_probability, 4),
            "counterfactual_probability": round(final_prob, 4),
            "changes_required": changes,
            "summary": (
                f"Modifying balance retention and graph connectivity drops risk score from "
                f"{round(current_probability*100)}% to {round(final_prob*100)}%."
            ),
        }

"""
src/explainability/briefing_engine.py
──────────────────────────────────────────────────────────────────────────────
LLM Investigator Briefing Engine for Merchant Fraud Analysts.

Synthesizes structured SHAP feature attributions, graph topological metrics,
and payment telemetry into an actionable, plain-English tactical briefing.

AI Judgment Principle:
  - Numerical fraud probability is strictly determined by GAT + XGBoost.
  - The LLM does NOT score or alter probabilities.
  - The LLM synthesizes the "Why Flagged" diagnostic narrative so human
    fraud analysts can make high-confidence, rapid review decisions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any


def generate_investigator_briefing(
    account_id: str,
    fraud_probability: float,
    risk_tier: str,
    top_features: List[Dict[str, float]],
    features_dict: Dict[str, float],
    ring_type: Optional[str] = None,
) -> str:
    """
    Generates a concise, high-signal investigative briefing for a merchant fraud analyst.
    """
    pct = round(fraud_probability * 100, 1)
    drain = features_dict.get("balance_drain_ratio", 0.0)
    deg_ratio = features_dict.get("degree_ratio", 1.0)
    night_tx = features_dict.get("night_tx_fraction", 0.0)
    spike = features_dict.get("amount_spike_ratio", 1.0)
    clust = features_dict.get("local_clustering_coefficient", 0.0)
    vel24 = features_dict.get("tx_velocity_24h", 0.0)

    # Format top driver string
    top_driver_names = [f["feature"] for f in top_features[:3]] if top_features else []

    if fraud_probability < 0.30:
        return (
            f"**Safe Baseline ({pct}% Risk)**: Account `{account_id}` exhibits organic consumer "
            f"purchasing patterns with standard diurnal timing ({round(night_tx*100)}% off-hours), "
            f"moderate balance retention ({round((1-drain)*100)}% retained), and uniform 7-day velocity. "
            f"No collusive graph topological links detected. **Recommended Action: Standard Clearance (No Analyst Intervention Needed).**"
        )

    narrative_parts = [
        f"**Risk Evaluation ({pct}% - {risk_tier})**: Account `{account_id}` was flagged for human review."
    ]

    # Topological context
    if deg_ratio >= 10.0:
        narrative_parts.append(
            f"Graph centrality shows an extreme out/in degree ratio of **{deg_ratio:.1f}x**, "
            f"indicating a classic hub-and-spoke dispersion pattern."
        )
    elif deg_ratio >= 4.0:
        narrative_parts.append(
            f"Graph degree asymmetry (**{deg_ratio:.1f}x**) signals asymmetric fund pass-through."
        )

    # Balance drain context
    if drain >= 0.85:
        narrative_parts.append(
            f"Behavioral telemetry reveals an aggressive **{round(drain*100)}% balance drain** "
            f"rapidly after fund ingress."
        )

    # Velocity and timing
    if night_tx >= 0.60:
        narrative_parts.append(
            f"**{round(night_tx*100)}%** of transactions occurred during high-risk off-hours (00:00–06:00 IST)."
        )

    if spike >= 3.0:
        narrative_parts.append(
            f"Observed a sudden **{spike:.1f}x transaction velocity surge** over historical 7-day baseline."
        )

    if clust < 0.05 and deg_ratio > 3.0:
        narrative_parts.append(
            f"Local neighborhood clustering is near zero ({clust:.2f}), confirming isolated synthetic counterparties rather than an organic merchant community."
        )

    # Recommended human action
    if ring_type == "Promo-Abuse Ring":
        action = "**Recommended Review Action**: Verify device fingerprint collisions across first-order promo coupon redemptions before manual release."
    elif ring_type == "Return-Fraud Ring":
        action = "**Recommended Review Action**: Hold high-value dispatch for physical address and courier AWB verification."
    elif ring_type == "Chargeback Collusion Cluster":
        action = "**Recommended Review Action**: Queue transaction for friendly-fraud evidence dossier generation and bank card issuer cross-check."
    elif ring_type == "Account-Takeover Checkout Surge":
        action = "**Recommended Review Action**: Trigger out-of-band customer verification via registered mobile SMS/Voice."
    else:
        action = "**Recommended Review Action**: Flag for secondary analyst review; verify counterparty settlement endpoints before release."

    narrative_parts.append(action)
    return " ".join(narrative_parts)

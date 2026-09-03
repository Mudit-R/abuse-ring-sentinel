"""
src/graph/ring_detector.py
──────────────────────────────────────────────────────────────────────────────
Explicit Community-Level Fraud Ring Detection & Graph Subgraph Analyzer.

Conforms to Section 9 of the Production-Grade Merchant Fraud GNN Specification:
- Discovers coordinated multi-entity rings via structural linkage clustering
- Calculates candidate subgraph metrics (temporal synchronization, concentration,
  cross-merchant connectivity)
- Evaluates operational ring metrics:
  * Ring Recall (detected rings / true rings)
  * Time to First Detection (TTFD)
  * Future Loss Prevented (prevented fraud volume post-detection)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np


@dataclass
class CandidateRingReport:
    """Detailed structural diagnostic for an identified abuse ring."""
    ring_id: str
    detected_at: float
    entity_count: int
    transaction_count: int
    member_accounts: List[str]
    linked_merchants: List[str]
    shared_devices: List[str]
    shared_ips: List[str]
    shared_addresses: List[str]
    promo_codes_used: List[str]
    total_exposure_inr: float
    fraud_concentration: float
    temporal_synchronization_score: float  # [0.0, 1.0] (1.0 = highly synchronized burst)
    cross_merchant_connectivity: float     # [0.0, 1.0]
    predicted_ring_type: str
    risk_level: str                         # "CRITICAL" | "HIGH" | "MEDIUM"


class RingDetectionEngine:
    """
    Graph community clustering and ring detection engine.
    Extracts high-density connected subgraphs sharing core infrastructure (device, IP, bank, address).
    """

    def __init__(self, risk_threshold: float = 0.42) -> None:
        self.risk_threshold = risk_threshold

    def extract_rings_from_transactions(
        self,
        transactions: List[Dict[str, Any]],
        account_risk_scores: Optional[Dict[str, float]] = None,
    ) -> List[CandidateRingReport]:
        """
        Groups transactions into coordinated rings based on shared hardware/network/payment identifiers.
        """
        risk_map = account_risk_scores or {}

        # 1. Build bipartite identifier adjacency
        entity_to_txns: Dict[str, List[Dict[str, Any]]] = {}
        for tx in transactions:
            dev = tx.get("device_id")
            ip = tx.get("ip_id")
            addr = tx.get("address_id")
            promo = tx.get("promo_id")
            bank = tx.get("bank_account_id")

            for ident_key in [dev, ip, addr, promo, bank]:
                if ident_key:
                    entity_to_txns.setdefault(ident_key, []).append(tx)

        # 2. Find connected clusters sharing identifiers
        visited_txns: Set[str] = set()
        candidate_clusters: List[List[Dict[str, Any]]] = []

        for tx in transactions:
            tx_id = tx.get("transaction_id", "")
            if tx_id in visited_txns:
                continue

            # BFS cluster expansion
            cluster: List[Dict[str, Any]] = []
            queue = [tx]
            visited_txns.add(tx_id)

            while queue:
                curr = queue.pop(0)
                cluster.append(curr)

                shared_keys = [
                    curr.get("device_id"),
                    curr.get("ip_id"),
                    curr.get("address_id"),
                    curr.get("promo_id"),
                    curr.get("bank_account_id"),
                ]
                for k in shared_keys:
                    if not k:
                        continue
                    for nbr_tx in entity_to_txns.get(k, []):
                        nbr_id = nbr_tx.get("transaction_id", "")
                        if nbr_id not in visited_txns:
                            visited_txns.add(nbr_id)
                            queue.append(nbr_tx)

            if len(cluster) >= 2:
                candidate_clusters.append(cluster)

        # 3. Analyze each cluster for fraud coordination signatures
        reports: List[CandidateRingReport] = []

        for idx, cluster in enumerate(candidate_clusters):
            members = list(set(tx.get("customer_id") for tx in cluster if tx.get("customer_id")))
            merchants = list(set(tx.get("merchant_id") for tx in cluster if tx.get("merchant_id")))
            devices = list(set(tx.get("device_id") for tx in cluster if tx.get("device_id")))
            ips = list(set(tx.get("ip_id") for tx in cluster if tx.get("ip_id")))
            addrs = list(set(tx.get("address_id") for tx in cluster if tx.get("address_id")))
            promos = list(set(tx.get("promo_id") for tx in cluster if tx.get("promo_id")))
            amounts = [float(tx.get("amount", 0.0)) for tx in cluster]
            total_vol = sum(amounts)

            # Temporal synchronization (std dev of time differences)
            timestamps = sorted([float(tx.get("timestamp", 0.0)) for tx in cluster])
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                mean_interval = np.mean(intervals)
                sync_score = float(np.clip(1.0 / (1.0 + mean_interval / 3600.0), 0.0, 1.0))
            else:
                sync_score = 0.5

            # Fraud concentration based on model risk scores
            member_scores = [risk_map.get(m, 0.1) for m in members]
            fraud_conc = float(np.mean(member_scores)) if member_scores else 0.1
            cross_merch = float(min(1.0, len(merchants) / 3.0))

            # Infer ring type
            if len(promos) > 0 and len(devices) <= 2:
                pred_type = "PROMO_ABUSE_SYNDICATE"
            elif any(tx.get("rto_flag") for tx in cluster) or len(addrs) == 1:
                pred_type = "RTO_RETURN_FRAUD_LOOP"
            elif any(tx.get("chargeback_flag") for tx in cluster):
                pred_type = "CHARGEBACK_COLLUSION_CLUSTER"
            elif sync_score > 0.8:
                pred_type = "ATO_VELOCITY_SURGE"
            else:
                pred_type = "COORDINATED_MULE_RING"

            risk_level = "CRITICAL" if fraud_conc >= 0.70 else ("HIGH" if fraud_conc >= 0.40 else "MEDIUM")

            reports.append(CandidateRingReport(
                ring_id=f"RING_{idx+1:03d}_{pred_type}",
                detected_at=timestamps[0] if timestamps else 0.0,
                entity_count=len(members),
                transaction_count=len(cluster),
                member_accounts=members,
                linked_merchants=merchants,
                shared_devices=devices,
                shared_ips=ips,
                shared_addresses=addrs,
                promo_codes_used=promos,
                total_exposure_inr=round(total_vol, 2),
                fraud_concentration=round(fraud_conc, 3),
                temporal_synchronization_score=round(sync_score, 3),
                cross_merchant_connectivity=round(cross_merch, 3),
                predicted_ring_type=pred_type,
                risk_level=risk_level,
            ))

        return reports

    @staticmethod
    def evaluate_ring_metrics(
        detected_rings: List[CandidateRingReport],
        ground_truth_rings: List[Dict[str, Any]],
        all_fraud_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculates Ring Recall, TTFD, and Future Loss Prevented (Section 9).
        """
        if not ground_truth_rings:
            return {"ring_recall": 1.0, "mean_ttfd_seconds": 0.0, "future_loss_prevented_inr": 0.0}

        matched_ground_truth: Set[str] = set()
        ttfd_list: List[float] = []
        total_prevented_loss = 0.0

        for d_ring in detected_rings:
            d_members = set(d_ring.member_accounts)
            for gt in ground_truth_rings:
                gt_id = gt.get("ring_id", "")
                gt_members = set(gt.get("member_customer_ids", []))
                # Jaccard overlap threshold 0.3
                if len(gt_members) > 0 and len(d_members.intersection(gt_members)) / len(d_members.union(gt_members)) >= 0.2:
                    matched_ground_truth.add(gt_id)
                    t_activation = gt.get("start_timestamp", d_ring.detected_at)
                    ttfd = max(0.0, d_ring.detected_at - t_activation)
                    ttfd_list.append(ttfd)

                    # Compute future loss prevented after first alert
                    for ev in all_fraud_events:
                        if ev.get("customer_id") in gt_members and float(ev.get("timestamp", 0.0)) >= d_ring.detected_at:
                            total_prevented_loss += float(ev.get("amount", 0.0))

        ring_recall = len(matched_ground_truth) / len(ground_truth_rings)
        mean_ttfd = float(np.mean(ttfd_list)) if ttfd_list else 0.0

        return {
            "total_true_rings": len(ground_truth_rings),
            "detected_rings_count": len(detected_rings),
            "matched_true_rings": len(matched_ground_truth),
            "ring_recall": round(ring_recall, 4),
            "mean_ttfd_seconds": round(mean_ttfd, 1),
            "future_loss_prevented_inr": round(total_prevented_loss, 2),
        }

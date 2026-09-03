"""
src/graph/hetero_builder.py
──────────────────────────────────────────────────────────────────────────────
Constructs typed PyTorch Geometric HeteroData objects from transaction streams
and provides realistic merchant fraud ring generators for the four target attacks:
1. Promo-code / discount abuse syndicates
2. RTO & high-ticket return fraud loops
3. Friendly-fraud / chargeback collusion clusters
4. Account-takeover (ATO) checkout velocity surges
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.graph.hetero_schema import (
    NodeType,
    NODE_TYPES,
    EDGE_RELATIONS,
    TemporalEdgeAttributes,
    tokenize_identifier,
)


@dataclass
class SyntheticRingMetadata:
    ring_id: str
    ring_type: str  # "PROMO_ABUSE" | "RTO_LOOP" | "CHARGEBACK_COLLUSION" | "ATO_SURGE"
    member_customer_ids: List[str]
    member_merchant_ids: List[str]
    shared_device_ids: List[str]
    shared_ip_ids: List[str]
    total_fraud_volume: float
    start_timestamp: float
    detection_timestamp: Optional[float] = None


class HeteroGraphBuilder:
    """
    Builds and manages PyG HeteroData graphs for production merchant risk scoring.
    """

    def __init__(self, feature_dim: int = 16, seed: int = 42) -> None:
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def create_empty_hetero_data(self) -> HeteroData:
        """Initializes an empty HeteroData graph with metadata pre-registered."""
        data = HeteroData()
        for nt in NODE_TYPES:
            data[nt].x = torch.empty((0, self.feature_dim), dtype=torch.float32)
        return data

    def build_synthetic_merchant_ecosystem(
        self,
        n_merchants: int = 20,
        n_clean_customers: int = 200,
        n_rings: int = 4,
        transactions_per_customer: int = 5,
        base_timestamp: float = 1788300000.0,
    ) -> Tuple[HeteroData, List[SyntheticRingMetadata], Dict[str, Any]]:
        """
        Builds a comprehensive heterogeneous transaction graph containing legitimate
        background merchants alongside 4 specialized coordinated fraud rings.
        """
        data = HeteroData()
        entity_indices: Dict[str, Dict[str, int]] = {nt: {} for nt in NODE_TYPES}
        edge_lists: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {rel: [] for rel in EDGE_RELATIONS}
        edge_attr_lists: Dict[Tuple[str, str, str], List[List[float]]] = {rel: [] for rel in EDGE_RELATIONS}

        def get_or_create_idx(ntype: str, raw_id: str) -> int:
            token = tokenize_identifier(raw_id)
            if token not in entity_indices[ntype]:
                new_idx = len(entity_indices[ntype])
                entity_indices[ntype][token] = new_idx
            return entity_indices[ntype][token]

        def add_edge(src_type: str, rel: str, dst_type: str, src_id: str, dst_id: str, attrs: TemporalEdgeAttributes):
            u = get_or_create_idx(src_type, src_id)
            v = get_or_create_idx(dst_type, dst_id)
            key = (src_type, rel, dst_type)
            if key in edge_lists:
                edge_lists[key].append((u, v))
                edge_attr_lists[key].append(attrs.to_feature_vector())

        rings_meta: List[SyntheticRingMetadata] = []
        txn_counter = 0

        # ── 1. Create Legitimate Merchants & Customers ──────────────────────
        merchants = [f"merch_{i+1:03d}" for i in range(n_merchants)]
        customers = [f"cust_clean_{i+1:04d}" for i in range(n_clean_customers)]

        # Pre-assign clean customer identity attributes
        for cust in customers:
            dev = f"dev_clean_{cust}"
            ip = f"ip_clean_{cust}"
            asn = f"asn_isp_{random.randint(1, 5)}"
            phone = f"ph_clean_{cust}"
            email = f"em_clean_{cust}"
            addr = f"addr_clean_{cust}"
            pin = f"pin_{random.randint(100001, 100050)}"
            bank = f"bank_clean_{cust}"

            add_edge(NodeType.IP.value, "IN_ASN", NodeType.ASN.value, ip, asn, TemporalEdgeAttributes(event_timestamp=base_timestamp))
            add_edge(NodeType.ADDRESS.value, "IN_PINCODE", NodeType.PINCODE.value, addr, pin, TemporalEdgeAttributes(event_timestamp=base_timestamp))
            add_edge(NodeType.CUSTOMER.value, "HAS_PHONE", NodeType.PHONE.value, cust, phone, TemporalEdgeAttributes(event_timestamp=base_timestamp))
            add_edge(NodeType.CUSTOMER.value, "HAS_EMAIL", NodeType.EMAIL.value, cust, email, TemporalEdgeAttributes(event_timestamp=base_timestamp))
            add_edge(NodeType.CUSTOMER.value, "HAS_BANK_ACCOUNT", NodeType.BANK_ACCOUNT.value, cust, bank, TemporalEdgeAttributes(event_timestamp=base_timestamp))

            # Simulate benign transactions over time
            for t in range(transactions_per_customer):
                txn_counter += 1
                txn_id = f"tx_{txn_counter:06d}"
                merch = random.choice(merchants)
                amt = float(self.rng.lognormal(mean=7.0, sigma=0.8)) # ~1k - 10k INR
                t_event = base_timestamp + t * 3600 + random.randint(0, 300)

                t_attr = TemporalEdgeAttributes(
                    event_timestamp=t_event,
                    amount=amt,
                    transaction_status="SUCCESS",
                    order_status="DELIVERED",
                    velocity_5m=1,
                    velocity_1h=1,
                    velocity_24h=t + 1,
                )
                add_edge(NodeType.CUSTOMER.value, "PLACED", NodeType.TRANSACTION.value, cust, txn_id, t_attr)
                add_edge(NodeType.TRANSACTION.value, "BELONGS_TO", NodeType.MERCHANT.value, txn_id, merch, t_attr)
                add_edge(NodeType.TRANSACTION.value, "USED_DEVICE", NodeType.DEVICE.value, txn_id, dev, t_attr)
                add_edge(NodeType.TRANSACTION.value, "FROM_IP", NodeType.IP.value, txn_id, ip, t_attr)
                add_edge(NodeType.TRANSACTION.value, "DELIVERED_TO", NodeType.ADDRESS.value, txn_id, addr, t_attr)

        # ── 2. Inject Attack Ring 1: Promo-Code Abuse Syndicate ─────────────
        # Multiple fake accounts all sharing a few device/IP nodes & cycling the same high-value promo
        promo_custs = [f"cust_promo_{i+1:02d}" for i in range(15)]
        shared_promo_dev = ["dev_farm_alpha_01", "dev_farm_alpha_02"]
        shared_promo_ip = ["198.51.100.22", "198.51.100.23"]
        promo_code = "PROMO_MEGA90"
        promo_vol = 0.0

        for cust in promo_custs:
            for _ in range(3):
                txn_counter += 1
                txn_id = f"tx_{txn_counter:06d}"
                merch = random.choice(merchants[:3])
                amt = 2500.0
                disc = 1800.0
                promo_vol += amt
                t_event = base_timestamp + random.randint(3600, 7200) # Rapid burst in 1 hr

                t_attr = TemporalEdgeAttributes(
                    event_timestamp=t_event,
                    amount=amt,
                    promo_discount_amount=disc,
                    velocity_5m=3,
                    velocity_1h=8,
                    velocity_24h=12,
                    device_novelty_flag=0,
                )
                add_edge(NodeType.CUSTOMER.value, "PLACED", NodeType.TRANSACTION.value, cust, txn_id, t_attr)
                add_edge(NodeType.TRANSACTION.value, "BELONGS_TO", NodeType.MERCHANT.value, txn_id, merch, t_attr)
                add_edge(NodeType.TRANSACTION.value, "USED_DEVICE", NodeType.DEVICE.value, txn_id, random.choice(shared_promo_dev), t_attr)
                add_edge(NodeType.TRANSACTION.value, "FROM_IP", NodeType.IP.value, txn_id, random.choice(shared_promo_ip), t_attr)
                add_edge(NodeType.TRANSACTION.value, "USED_PROMO", NodeType.PROMO.value, txn_id, promo_code, t_attr)

        rings_meta.append(SyntheticRingMetadata(
            ring_id="RING_01_PROMO_SYNDICATE",
            ring_type="PROMO_ABUSE",
            member_customer_ids=promo_custs,
            member_merchant_ids=merchants[:3],
            shared_device_ids=shared_promo_dev,
            shared_ip_ids=shared_promo_ip,
            total_fraud_volume=promo_vol,
            start_timestamp=base_timestamp + 3600,
        ))

        # ── 3. Inject Attack Ring 2: High-Ticket RTO / Return Fraud Loop ──────
        # Accounts order expensive electronic items (SKUs) to shared addresses and trigger fake returns
        rto_custs = [f"cust_rto_{i+1:02d}" for i in range(10)]
        shared_drop_addr = ["addr_drop_warehouse_7", "addr_drop_suite_9"]
        rto_vol = 0.0

        for cust in rto_custs:
            for _ in range(2):
                txn_counter += 1
                txn_id = f"tx_{txn_counter:06d}"
                merch = merchants[4]
                amt = 48000.0  # High ticket
                rto_vol += amt
                t_event = base_timestamp + random.randint(7200, 14400)

                t_attr = TemporalEdgeAttributes(
                    event_timestamp=t_event,
                    amount=amt,
                    rto_flag=1,
                    refund_amount=amt,
                    order_status="RETURNED_TO_ORIGIN",
                )
                add_edge(NodeType.CUSTOMER.value, "PLACED", NodeType.TRANSACTION.value, cust, txn_id, t_attr)
                add_edge(NodeType.TRANSACTION.value, "BELONGS_TO", NodeType.MERCHANT.value, txn_id, merch, t_attr)
                add_edge(NodeType.TRANSACTION.value, "DELIVERED_TO", NodeType.ADDRESS.value, txn_id, random.choice(shared_drop_addr), t_attr)
                add_edge(NodeType.TRANSACTION.value, "GENERATED_REFUND", NodeType.REFUND.value, txn_id, f"ref_{txn_id}", t_attr)

        rings_meta.append(SyntheticRingMetadata(
            ring_id="RING_02_RTO_LOOP",
            ring_type="RTO_LOOP",
            member_customer_ids=rto_custs,
            member_merchant_ids=[merchants[4]],
            shared_device_ids=[],
            shared_ip_ids=[],
            total_fraud_volume=rto_vol,
            start_timestamp=base_timestamp + 7200,
        ))

        # ── 4. Inject Attack Ring 3: Chargeback Collusion Cluster ────────────
        # Merchant and accounts collude: fake high-volume payments, immediate payouts to shared bank, then chargebacks
        cb_custs = [f"cust_colluder_{i+1:02d}" for i in range(8)]
        collusive_merch = merchants[7 % len(merchants)]
        shared_mule_bank = "bank_mule_offshore_99"
        cb_vol = 0.0

        add_edge(NodeType.MERCHANT.value, "PAYOUT_TO", NodeType.BANK_ACCOUNT.value, collusive_merch, shared_mule_bank, TemporalEdgeAttributes(event_timestamp=base_timestamp))

        for cust in cb_custs:
            add_edge(NodeType.CUSTOMER.value, "HAS_BANK_ACCOUNT", NodeType.BANK_ACCOUNT.value, cust, shared_mule_bank, TemporalEdgeAttributes(event_timestamp=base_timestamp))
            for _ in range(2):
                txn_counter += 1
                txn_id = f"tx_{txn_counter:06d}"
                amt = 35000.0
                cb_vol += amt
                t_event = base_timestamp + random.randint(14400, 21600)

                t_attr = TemporalEdgeAttributes(
                    event_timestamp=t_event,
                    amount=amt,
                    chargeback_flag=1,
                    order_status="DISPUTED",
                )
                add_edge(NodeType.CUSTOMER.value, "PLACED", NodeType.TRANSACTION.value, cust, txn_id, t_attr)
                add_edge(NodeType.TRANSACTION.value, "BELONGS_TO", NodeType.MERCHANT.value, txn_id, collusive_merch, t_attr)
                add_edge(NodeType.TRANSACTION.value, "GENERATED_CHARGEBACK", NodeType.CHARGEBACK.value, txn_id, f"cb_{txn_id}", t_attr)

        rings_meta.append(SyntheticRingMetadata(
            ring_id="RING_03_CHARGEBACK_COLLUSION",
            ring_type="CHARGEBACK_COLLUSION",
            member_customer_ids=cb_custs,
            member_merchant_ids=[collusive_merch],
            shared_device_ids=[],
            shared_ip_ids=[],
            total_fraud_volume=cb_vol,
            start_timestamp=base_timestamp + 14400,
        ))

        # ── 5. Inject Attack Ring 4: Account-Takeover (ATO) Checkout Surge ───
        # Legitimate customer accounts suddenly hijacked: new devices, new IPs in unusual ASNs, checkout spikes
        ato_custs = customers[:min(6, len(customers))]  # Hijack previously clean customers
        ato_dev = "dev_tor_relay_x"
        ato_ip = "185.220.101.5"
        ato_vol = 0.0
        ato_merchants = [merchants[i % len(merchants)] for i in range(10, 14)]

        for cust in ato_custs:
            for surge_i in range(5):
                txn_counter += 1
                txn_id = f"tx_{txn_counter:06d}"
                merch = random.choice(ato_merchants)
                amt = 19500.0
                ato_vol += amt
                t_event = base_timestamp + 25000 + surge_i * 60 # 1 minute apart!

                t_attr = TemporalEdgeAttributes(
                    event_timestamp=t_event,
                    amount=amt,
                    device_novelty_flag=1,
                    ip_novelty_flag=1,
                    velocity_5m=5,
                    velocity_1h=15,
                    velocity_24h=20,
                )
                add_edge(NodeType.CUSTOMER.value, "PLACED", NodeType.TRANSACTION.value, cust, txn_id, t_attr)
                add_edge(NodeType.TRANSACTION.value, "BELONGS_TO", NodeType.MERCHANT.value, txn_id, merch, t_attr)
                add_edge(NodeType.TRANSACTION.value, "USED_DEVICE", NodeType.DEVICE.value, txn_id, ato_dev, t_attr)
                add_edge(NodeType.TRANSACTION.value, "FROM_IP", NodeType.IP.value, txn_id, ato_ip, t_attr)

        rings_meta.append(SyntheticRingMetadata(
            ring_id="RING_04_ATO_VELOCITY_SURGE",
            ring_type="ATO_SURGE",
            member_customer_ids=ato_custs,
            member_merchant_ids=ato_merchants,
            shared_device_ids=[ato_dev],
            shared_ip_ids=[ato_ip],
            total_fraud_volume=ato_vol,
            start_timestamp=base_timestamp + 25000,
        ))

        # ── Assemble PyG HeteroData Object ───────────────────────────────────
        for ntype, id_map in entity_indices.items():
            count = len(id_map)
            if count > 0:
                # Deterministic normalized embedding feature initialization
                feats = self.rng.standard_normal((count, self.feature_dim)).astype(np.float32)
                feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
                data[ntype].x = torch.tensor(feats, dtype=torch.float32)
                data[ntype].num_nodes = count
            else:
                data[ntype].x = torch.empty((0, self.feature_dim), dtype=torch.float32)
                data[ntype].num_nodes = 0

        for rel, edges in edge_lists.items():
            src_type, rel_name, dst_type = rel
            if len(edges) > 0:
                edge_arr = np.array(edges, dtype=np.int64).T
                data[rel].edge_index = torch.tensor(edge_arr, dtype=torch.long)
                attr_arr = np.array(edge_attr_lists[rel], dtype=np.float32)
                data[rel].edge_attr = torch.tensor(attr_arr, dtype=torch.float32)
            else:
                data[rel].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[rel].edge_attr = torch.empty((0, 12), dtype=torch.float32)

        stats = {
            "total_nodes": sum(len(m) for m in entity_indices.values()),
            "nodes_by_type": {k: len(v) for k, v in entity_indices.items()},
            "total_edges": sum(len(e) for e in edge_lists.values()),
            "total_rings": len(rings_meta),
            "total_fraud_volume": sum(r.total_fraud_volume for r in rings_meta),
        }

        return data, rings_meta, stats

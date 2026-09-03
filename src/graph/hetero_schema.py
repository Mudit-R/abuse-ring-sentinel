"""
src/graph/hetero_schema.py
──────────────────────────────────────────────────────────────────────────────
Heterogeneous Graph Ontology Definition for Merchant Fraud & Abuse Ring Detection.

Conforms strictly to Section 2 of the Production-Grade Merchant Fraud GNN Specification:
- 20 Required Node Types
- Explicit Typed Edges (no simplistic clique projection)
- Temporal Edge Attributes
- PII Tokenization Safeguards
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any
import hashlib


# ── 20 Required Node Types (Section 2.1) ──────────────────────────────────────

class NodeType(str, Enum):
    MERCHANT = "merchant"
    CUSTOMER = "customer"
    TRANSACTION = "transaction"
    PAYMENT_INSTRUMENT = "payment_instrument"
    UPI_VPA = "upi_vpa"
    BANK_ACCOUNT = "bank_account"
    DEVICE = "device"
    IP = "ip"
    ASN = "asn"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    PINCODE = "pincode_or_geohash"
    PROMO = "promo"
    ORDER = "order"
    SHIPMENT = "shipment"
    REFUND = "refund"
    CHARGEBACK = "chargeback"
    PRODUCT_SKU = "product_sku"
    SESSION = "session"


NODE_TYPES: List[str] = [nt.value for nt in NodeType]


# ── Explicit Directed Edge Types (Section 2.2) ────────────────────────────────

EDGE_RELATIONS: List[Tuple[str, str, str]] = [
    (NodeType.CUSTOMER.value, "PLACED", NodeType.TRANSACTION.value),
    (NodeType.TRANSACTION.value, "BELONGS_TO", NodeType.MERCHANT.value),
    (NodeType.TRANSACTION.value, "USED_DEVICE", NodeType.DEVICE.value),
    (NodeType.TRANSACTION.value, "FROM_IP", NodeType.IP.value),
    (NodeType.IP.value, "IN_ASN", NodeType.ASN.value),
    (NodeType.TRANSACTION.value, "USED_PAYMENT", NodeType.PAYMENT_INSTRUMENT.value),
    (NodeType.TRANSACTION.value, "USED_VPA", NodeType.UPI_VPA.value),
    (NodeType.CUSTOMER.value, "HAS_BANK_ACCOUNT", NodeType.BANK_ACCOUNT.value),
    (NodeType.MERCHANT.value, "PAYOUT_TO", NodeType.BANK_ACCOUNT.value),
    (NodeType.TRANSACTION.value, "DELIVERED_TO", NodeType.ADDRESS.value),
    (NodeType.ADDRESS.value, "IN_PINCODE", NodeType.PINCODE.value),
    (NodeType.CUSTOMER.value, "HAS_PHONE", NodeType.PHONE.value),
    (NodeType.CUSTOMER.value, "HAS_EMAIL", NodeType.EMAIL.value),
    (NodeType.TRANSACTION.value, "USED_PROMO", NodeType.PROMO.value),
    (NodeType.TRANSACTION.value, "CONTAINS", NodeType.PRODUCT_SKU.value),
    (NodeType.TRANSACTION.value, "HAS_ORDER", NodeType.ORDER.value),
    (NodeType.ORDER.value, "HAS_SHIPMENT", NodeType.SHIPMENT.value),
    (NodeType.TRANSACTION.value, "GENERATED_REFUND", NodeType.REFUND.value),
    (NodeType.TRANSACTION.value, "GENERATED_CHARGEBACK", NodeType.CHARGEBACK.value),
    (NodeType.CUSTOMER.value, "HAS_SESSION", NodeType.SESSION.value),
    (NodeType.SESSION.value, "USED_DEVICE", NodeType.DEVICE.value),
    (NodeType.SESSION.value, "FROM_IP", NodeType.IP.value),
]


# Default Top-K neighbor caps per relation to prevent degree explosion (Section 4.3)
RELATION_TOP_K_DEFAULTS: Dict[str, int] = {
    "USED_DEVICE": 8,
    "DELIVERED_TO": 8,
    "USED_PAYMENT": 8,
    "USED_VPA": 8,
    "USED_PROMO": 16,
    "FROM_IP": 4,
    "BELONGS_TO": 16,
    "PLACED": 16,
    "HAS_BANK_ACCOUNT": 8,
    "PAYOUT_TO": 8,
    "DEFAULT": 12,
}


# ── Edge Attribute Schema (Section 2.3) ───────────────────────────────────────

@dataclass
class TemporalEdgeAttributes:
    """
    Standard temporal edge feature payload.
    Never leaks future information into historical records.
    """
    event_timestamp: float
    delta_t_from_prev: float = 0.0
    amount: float = 0.0
    currency: str = "INR"
    payment_method: str = "UPI"
    transaction_status: str = "SUCCESS"
    order_status: str = "DELIVERED"
    refund_amount: float = 0.0
    chargeback_flag: int = 0
    rto_flag: int = 0
    promo_discount_amount: float = 0.0
    merchant_category: str = "RETAIL"
    device_novelty_flag: int = 0
    ip_novelty_flag: int = 0
    address_novelty_flag: int = 0
    velocity_5m: int = 1
    velocity_1h: int = 1
    velocity_24h: int = 1

    def to_feature_vector(self) -> List[float]:
        return [
            float(self.delta_t_from_prev),
            float(self.amount),
            float(self.refund_amount),
            float(self.chargeback_flag),
            float(self.rto_flag),
            float(self.promo_discount_amount),
            float(self.device_novelty_flag),
            float(self.ip_novelty_flag),
            float(self.address_novelty_flag),
            float(self.velocity_5m),
            float(self.velocity_1h),
            float(self.velocity_24h),
        ]


# ── Tokenization / PII Protection Safeguard (Section 26) ─────────────────────

def tokenize_identifier(raw_identifier: str, salt: str = "sentinel_v2_salt") -> str:
    """
    Irreversibly hashes sensitive entity identifiers (card PAN, bank account, phone).
    Raw credentials never enter the graph feature store.
    """
    if not raw_identifier:
        return "tok_empty"
    hashed = hashlib.sha256(f"{salt}::{raw_identifier}".encode("utf-8")).hexdigest()
    return f"tok_{hashed[:16]}"

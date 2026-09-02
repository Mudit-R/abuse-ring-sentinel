"""
src/api/razorpay_simulator.py
──────────────────────────────────────────────────────────────────────────────
Razorpay Merchant Checkout Simulation & Planted Fraud-Ring Generator.

Generates synthetic transactions styled exactly like real Razorpay India
merchant checkout telemetry:
  - Standard Razorpay entity IDs: `order_Rzp...`, `pay_...`, `cust_...`
  - Indian Payment Rails: UPI (PhonePe, Google Pay, Paytm, CRED, BHIM),
    Cards (Visa, Mastercard, RuPay), NetBanking (HDFC, ICICI, SBI, Axis)
  - Merchant Categories (MCC):
      5732 (Consumer Electronics)
      5651 (Fashion & Apparel)
      5812 (Quick-Commerce & Food Delivery)
      5999 (Digital Goods, Gaming & Gift Cards)
  - Realistic INR ticket sizes (₹199 to ₹85,000)
  - Planted Merchant Fraud Rings:
      1. Promo-Abuse Ring (Syndicate exploiting first-order discounts via multi-account fan-in)
      2. Return-Fraud Ring (High-value electronics orders routed to drop addresses followed by empty-box return claims)
      3. Account-Takeover Checkout Surge (Compromised merchant session draining stored balances at 03:00 AM)
      4. Chargeback Collusion Cluster (Coordinated card chargebacks across friendly fraud networks)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np


UPI_HANDLES = ["@okhdfcbank", "@okaxis", "@paytm", "@ybl", "@ibl", "@ptaxis", "@apl"]
UPI_APPS = ["Google Pay", "PhonePe", "Paytm UPI", "CRED UPI", "BHIM UPI", "WhatsApp Pay"]
BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Punjab National Bank"]
CARD_NETWORKS = ["RuPay", "Visa", "Mastercard"]

MERCHANT_CATEGORIES = [
    {"mcc": "5732", "name": "Electronics & Gadgets", "avg_ticket": 14500.0, "risk_base": 0.04},
    {"mcc": "5651", "name": "Fashion & Lifestyle", "avg_ticket": 2400.0, "risk_base": 0.02},
    {"mcc": "5812", "name": "Quick-Commerce & Delivery", "avg_ticket": 480.0, "risk_base": 0.01},
    {"mcc": "5999", "name": "Digital Vouchers & Gaming", "avg_ticket": 3200.0, "risk_base": 0.07},
]


def generate_razorpay_id(prefix: str = "pay") -> str:
    """Generate a realistic Razorpay alphanumeric entity ID."""
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    random_str = "".join(random.choices(charset, k=14))
    return f"{prefix}_{random_str}"


@dataclass
class RazorpayTransaction:
    payment_id: str
    order_id: str
    account_id: str
    merchant_name: str
    mcc_code: str
    amount_inr: float
    method: str  # "upi" | "card" | "netbanking"
    method_detail: str  # e.g. "Google Pay (user@okhdfcbank)"
    timestamp: str
    ip_hash: str
    device_fingerprint: str
    risk_score: float
    risk_tier: str  # "LOW" | "ELEVATED" | "HIGH" | "CRITICAL"
    is_planted_ring: bool
    ring_type: Optional[str] = None
    recommended_action: str = "Allow Transaction (Standard Clearance)"
    human_confirmation_required: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


class RazorpayStreamSimulator:
    """Simulates real-time Razorpay merchant payment events and planted abuse rings."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.tx_counter = 0

    def generate_clean_transaction(self) -> RazorpayTransaction:
        """Generate a normal, legitimate merchant transaction."""
        self.tx_counter += 1
        mcc_info = random.choice(MERCHANT_CATEGORIES)
        method_type = random.choices(["upi", "card", "netbanking"], weights=[0.68, 0.24, 0.08])[0]

        if method_type == "upi":
            app = random.choice(UPI_APPS)
            vpa = f"user_{random.randint(1000, 9999)}{random.choice(UPI_HANDLES)}"
            detail = f"{app} ({vpa})"
        elif method_type == "card":
            network = random.choice(CARD_NETWORKS)
            last4 = f"{random.randint(1000, 9999)}"
            detail = f"{network} •••• {last4}"
        else:
            bank = random.choice(BANKS)
            detail = f"NetBanking ({bank})"

        # Realistic log-normal amount
        amount = max(149.0, round(float(np.random.lognormal(np.log(mcc_info["avg_ticket"]), 0.5)), 2))
        account_id = f"acc_rzp_{random.randint(100000, 999999)}"

        risk_score = round(float(np.random.beta(1.5, 25.0)), 4)  # Mostly < 0.15
        risk_tier = "LOW" if risk_score < 0.30 else "ELEVATED"

        return RazorpayTransaction(
            payment_id=generate_razorpay_id("pay"),
            order_id=generate_razorpay_id("order"),
            account_id=account_id,
            merchant_name=f"Merchant_{mcc_info['mcc']}",
            mcc_code=mcc_info["mcc"],
            amount_inr=amount,
            method=method_type,
            method_detail=detail,
            timestamp=time.strftime("%H:%M:%S"),
            ip_hash=f"ip_{random.randint(100, 999)}.{random.randint(10, 99)}",
            device_fingerprint=f"dev_{random.randint(10000, 99999)}",
            risk_score=risk_score,
            risk_tier=risk_tier,
            is_planted_ring=False,
            ring_type=None,
            recommended_action="Allow Transaction (Standard Clearance)",
            human_confirmation_required=False,
        )

    def generate_planted_ring_wave(self, ring_type: str = "Promo-Abuse Ring") -> List[RazorpayTransaction]:
        """
        Injects a coordinated batch of 4-8 transactions belonging to a planted merchant fraud ring.
        """
        txs = []
        shared_device = f"dev_shared_cluster_{random.randint(100, 999)}"
        shared_ip = f"ip_vpn_exit_{random.randint(10, 99)}"

        if ring_type == "Promo-Abuse Ring":
            # 6 colluding accounts redeeming a ₹500 first-order promo coupon within 90 seconds
            shared_dest = "vpa_promo_reseller@okhdfcbank"
            for i in range(6):
                acc = f"acc_syndicate_{i+1:03d}"
                tx = RazorpayTransaction(
                    payment_id=generate_razorpay_id("pay"),
                    order_id=generate_razorpay_id("order"),
                    account_id=acc,
                    merchant_name="QuickCommerce_5812",
                    mcc_code="5812",
                    amount_inr=599.00,  # Just above minimum discount threshold
                    method="upi",
                    method_detail=f"PhonePe ({acc}@paytm) -> {shared_dest}",
                    timestamp=time.strftime("%H:%M:%S"),
                    ip_hash=shared_ip,
                    device_fingerprint=shared_device,
                    risk_score=0.912 + (i * 0.012),
                    risk_tier="CRITICAL",
                    is_planted_ring=True,
                    ring_type="Promo-Abuse Ring",
                    recommended_action="Recommend Manual Review: Coordinated Promo Abuse Syndicate Detected",
                    human_confirmation_required=True,
                )
                txs.append(tx)

        elif ring_type == "Return-Fraud Ring":
            # High-ticket electronics orders with rapid balance drain and synthetic identity
            for i in range(4):
                acc = f"acc_return_ring_{i+1:03d}"
                tx = RazorpayTransaction(
                    payment_id=generate_razorpay_id("pay"),
                    order_id=generate_razorpay_id("order"),
                    account_id=acc,
                    merchant_name="Electronics_5732",
                    mcc_code="5732",
                    amount_inr=68490.00,
                    method="card",
                    method_detail=f"Mastercard •••• {8800 + i}",
                    timestamp=time.strftime("%H:%M:%S"),
                    ip_hash=shared_ip,
                    device_fingerprint=shared_device,
                    risk_score=0.885 + (i * 0.02),
                    risk_tier="HIGH",
                    is_planted_ring=True,
                    ring_type="Return-Fraud Ring",
                    recommended_action="Recommend Manual Review: High Return-Risk Cluster (Flag for Physical Delivery Verification)",
                    human_confirmation_required=True,
                )
                txs.append(tx)

        elif ring_type == "Chargeback Collusion Cluster":
            # Card collusion cluster across multiple merchant categories
            for i in range(5):
                acc = f"acc_cb_cluster_{i+1:03d}"
                tx = RazorpayTransaction(
                    payment_id=generate_razorpay_id("pay"),
                    order_id=generate_razorpay_id("order"),
                    account_id=acc,
                    merchant_name="DigitalGaming_5999",
                    mcc_code="5999",
                    amount_inr=8500.00,
                    method="card",
                    method_detail=f"Visa •••• {4120 + i}",
                    timestamp=time.strftime("%H:%M:%S"),
                    ip_hash=shared_ip,
                    device_fingerprint=shared_device,
                    risk_score=0.945 + (i * 0.008),
                    risk_tier="CRITICAL",
                    is_planted_ring=True,
                    ring_type="Chargeback Collusion Cluster",
                    recommended_action="Recommend Manual Review: Coordinated Chargeback Collusion Syndicate",
                    human_confirmation_required=True,
                )
                txs.append(tx)

        else:  # Account-Takeover Checkout Surge
            for i in range(5):
                acc = f"acc_ato_victim_{i+1:03d}"
                tx = RazorpayTransaction(
                    payment_id=generate_razorpay_id("pay"),
                    order_id=generate_razorpay_id("order"),
                    account_id=acc,
                    merchant_name="DigitalVouchers_5999",
                    mcc_code="5999",
                    amount_inr=15000.00,
                    method="netbanking",
                    method_detail=f"NetBanking (HDFC) -> Instant Voucher Drain",
                    timestamp=time.strftime("%H:%M:%S"),
                    ip_hash=shared_ip,
                    device_fingerprint=shared_device,
                    risk_score=0.965,
                    risk_tier="CRITICAL",
                    is_planted_ring=True,
                    ring_type="Account-Takeover Checkout Surge",
                    recommended_action="Recommend Manual Review: Urgent ATO Checkout Drain Burst (Step-Up Out-of-Band Auth)",
                    human_confirmation_required=True,
                )
                txs.append(tx)

        return txs

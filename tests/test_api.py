"""
tests/test_api.py
──────────────────────────────────────────────────────────────────────────────
Integration tests for FastAPI fraud detection, Redis feature store, audit trails,
and Razorpay stream endpoints.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, COST_OPTIMAL_THRESHOLD


@pytest.fixture(scope="module")
def client():
    """Create a test client with the app's lifespan managed."""
    with TestClient(app) as c:
        yield c


EXAMPLE_ACCOUNT = {
    "account_id": "C_TEST_001",
    "total_sent_log": 12.5,
    "total_received_log": 10.2,
    "tx_count_out": 45.0,
    "tx_count_in": 3.0,
    "unique_dest_count": 40.0,
    "unique_src_count": 3.0,
    "avg_sent_log": 8.3,
    "avg_received_log": 9.1,
    "balance_drain_ratio": 0.95,
    "night_tx_fraction": 0.8,
    "fraud_type_fraction": 1.0,
    "in_degree": 3.0,
    "out_degree": 45.0,
    "degree_ratio": 15.0,
    "pagerank": 0.0023,
    "k_core_number": 5.0,
    "local_clustering_coefficient": 0.02,
    "tx_velocity_24h": 12.0,
    "tx_velocity_7d": 45.0,
    "amount_velocity_24h": 10.5,
    "amount_velocity_7d": 12.5,
    "amount_spike_ratio": 2.3,
}


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "gpu_available" in data
        assert "redis_connected" in data
        assert "uptime_seconds" in data


class TestPredictEndpoint:

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=EXAMPLE_ACCOUNT)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json=EXAMPLE_ACCOUNT).json()
        assert "account_id" in data
        assert "fraud_probability" in data
        assert "is_flagged" in data
        assert "risk_tier" in data
        assert "top_contributing_features" in data
        assert "model_version" in data
        assert "cache_hit" in data
        assert "scoring_latency_ms" in data
        assert "recommended_action" in data
        assert data["human_confirmation_required"] is True
        assert "llm_investigator_briefing" in data
        assert "counterfactual_explanation" in data
        assert "shap_values" in data

    def test_predict_probability_range(self, client):
        data = client.post("/predict", json=EXAMPLE_ACCOUNT).json()
        prob = data["fraud_probability"]
        assert 0.0 <= prob <= 1.0, f"fraud_probability out of range: {prob}"

    def test_predict_risk_tier_valid(self, client):
        data = client.post("/predict", json=EXAMPLE_ACCOUNT).json()
        assert data["risk_tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_predict_human_guardrail_enforced(self, client):
        data = client.post("/predict", json=EXAMPLE_ACCOUNT).json()
        assert data["human_confirmation_required"] is True
        assert any(keyword in data["recommended_action"] for keyword in ["Flag", "Allow", "Review", "Verification"])


class TestAuditTrailEndpoints:

    def test_confirm_audit_action(self, client):
        req_payload = {
            "account_id": "acc_test_audit_01",
            "decision": "CONFIRMED_FLAG",
            "analyst_id": "analyst_sarah_01",
            "risk_score": 0.94,
            "notes": "Verified high-velocity balance drain across synthetic endpoints.",
            "action_taken": "Hold for Merchant Verification",
        }
        resp = client.post("/audit/confirm-action", json=req_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "audit_id" in data
        assert data["account_id"] == "acc_test_audit_01"
        assert data["status"] == "RECORDED"

    def test_list_audit_trail(self, client):
        resp = client.get("/audit/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_records" in data
        assert data["total_records"] >= 1

    def test_export_audit_trail(self, client):
        resp_json = client.get("/audit/export?format=json")
        assert resp_json.status_code == 200
        assert "audit_trail" in resp_json.json()

        resp_csv = client.get("/audit/export?format=csv")
        assert resp_csv.status_code == 200
        assert "audit_id" in resp_csv.text


class TestRazorpaySimulatorEndpoints:

    def test_get_stream_event(self, client):
        resp = client.get("/razorpay/stream-event")
        assert resp.status_code == 200
        data = resp.json()
        assert "payment_id" in data
        assert data["payment_id"].startswith("pay_")
        assert "amount_inr" in data
        assert "method" in data

    def test_inject_ring_wave(self, client):
        resp = client.post("/razorpay/inject-ring?ring_type=Promo-Abuse%20Ring")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ring_type"] == "Promo-Abuse Ring"
        assert len(data["transactions"]) == 6
        assert data["transactions"][0]["is_planted_ring"] is True


class TestEvaluationEndpoints:

    def test_get_benchmark(self, client):
        resp = client.get("/evaluation/benchmark")
        assert resp.status_code == 200

    def test_get_ablation(self, client):
        resp = client.get("/evaluation/ablation")
        assert resp.status_code == 200

    def test_get_adversarial(self, client):
        resp = client.get("/evaluation/adversarial")
        assert resp.status_code == 200

    def test_get_spec_ablation_matrix(self, client):
        resp = client.get("/metrics/ablation-matrix")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 8
        assert data[0]["model"] == "XGBoost Standalone"

    def test_get_camouflage_stress_test(self, client):
        resp = client.get("/metrics/camouflage-stress")
        assert resp.status_code == 200
        data = resp.json()
        assert "k_injected_decoy_neighbors" in data
        assert "curves" in data

    def test_evaluate_hetero_ring(self, client):
        payload = {
            "transactions": [
                {"transaction_id": "t1", "customer_id": "c1", "device_id": "d1", "amount": 1000.0, "timestamp": 100.0},
                {"transaction_id": "t2", "customer_id": "c2", "device_id": "d1", "amount": 1200.0, "timestamp": 105.0},
            ],
            "risk_scores": {"c1": 0.8, "c2": 0.85},
        }
        resp = client.post("/evaluate/hetero-ring", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_rings" in data
        assert data["total_rings"] >= 1

"""
Unit and integration tests for FastAPI application endpoints.
Uses FastAPI TestClient to test health, overview, metrics, statistics, predictions, and simulator.
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "disclaimer" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_ready"] is True
    assert data["model_name"] == "Random Forest"


def test_overview_endpoint():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_observations"] == 10000
    assert "class_distribution" in data
    assert data["class_distribution"]["failure_count"] == 339
    assert data["selected_model"]["model_name"] == "Random Forest"
    assert data["selected_model"]["decision_threshold"] == 0.31


def test_model_metrics_endpoint():
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "out_of_fold_validation_results" in data
    assert "held_out_test_results" in data
    assert "feature_importance_rf" in data

    rf_test = data["held_out_test_results"]["random_forest"]["default_threshold_0_5"]
    assert rf_test["average_precision_pr_auc"] > 0.80


def test_statistics_endpoint():
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.json()
    assert "numeric_tests" in data
    assert "categorical_test" in data
    assert "interpretations" in data
    assert len(data["numeric_tests"]) == 7


def test_predict_endpoint_valid():
    payload = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 30,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "estimated_anomaly_score" in data
    assert 0.0 <= data["estimated_anomaly_score"] <= 1.0
    assert data["predicted_label"] in [0, 1]
    assert data["status"] in ["NORMAL_OPERATION", "ATTENTION_ADVISED", "MAINTENANCE_REQUIRED"]


def test_predict_endpoint_invalid_type():
    payload = {
        "Type": "Z",  # Invalid type
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 30,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity from Pydantic


def test_simulator_initialize_and_step():
    # 1. Initialize
    init_res = client.post("/api/simulator/initialize", json={"scenario": "normal", "machine_type": "L", "seed": 42})
    assert init_res.status_code == 200
    init_data = init_res.json()
    assert "state" in init_data
    state = init_data["state"]
    assert state["tick"] == 0

    # 2. Step
    step_res = client.post("/api/simulator/step", json={"state": state})
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert "next_state" in step_data
    assert step_data["next_state"]["tick"] == 1
    assert "reading" in step_data
    assert "prediction" in step_data
    assert step_data["metadata"]["is_simulated"] is True


def test_fastapi_root_entrypoint_compatibility():
    """Ensures root app.py exposes the exact FastAPI app auto-detected by Vercel."""
    from app import app as root_app
    from fastapi.testclient import TestClient

    test_client = TestClient(root_app)
    res = test_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


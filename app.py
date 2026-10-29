"""
FastAPI application for Predictive Maintenance Intelligence System.
Root entrypoint designed for local execution and serverless Vercel deployment.
Provides stateless condition monitoring, live simulation stepping, and analytical reports.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.config import (
    ALL_PREDICTOR_COLUMNS,
    FEATURE_RANGES,
    OUTPUTS_DIR,
    SAFETY_DISCLAIMER,
)
from src.models.predictor import ProductionPredictor, get_predictor
from src.simulator.stream import (
    VALID_MACHINE_TYPES,
    VALID_SCENARIOS,
    init_simulator_state,
    step_simulation,
)

app = FastAPI(
    title="Predictive Maintenance Intelligence API",
    description="Statistical Condition-Based Monitoring and Anomaly Estimation API (UCI AI4I 2020 Benchmark)",
    version="1.0.0",
)

# CORS Configuration
# Configurable via environment variable ALLOWED_ORIGINS (comma-separated)
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
is_wildcard = "*" in allowed_origins or not allowed_origins
if is_wildcard:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=not is_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Pydantic Request / Response Schemas
# -------------------------------------------------------------
class SensorInputSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    machine_type: Literal["L", "M", "H"] = Field(..., alias="Type")
    air_temp: float = Field(..., alias="Air temperature [K]", ge=250.0, le=350.0)
    process_temp: float = Field(..., alias="Process temperature [K]", ge=250.0, le=350.0)
    rotational_speed: float = Field(..., alias="Rotational speed [rpm]", ge=500.0, le=4000.0)
    torque: float = Field(..., alias="Torque [Nm]", ge=0.0, le=150.0)
    tool_wear: float = Field(..., alias="Tool wear [min]", ge=0.0, le=500.0)

    def to_reading_dict(self) -> Dict[str, Any]:
        return {
            "Type": self.machine_type,
            "Air temperature [K]": self.air_temp,
            "Process temperature [K]": self.process_temp,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear,
        }


class SimulatorInitRequest(BaseModel):
    scenario: Literal["normal", "elevated_load", "thermal_stress", "high_tool_wear"] = "normal"
    machine_type: Literal["L", "M", "H"] = "L"
    seed: int = 42


class SimulatorStepRequest(BaseModel):
    state: Dict[str, Any]


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Predictive Maintenance Intelligence API",
        "status": "online",
        "documentation": "/docs",
        "version": "1.0.0",
        "disclaimer": SAFETY_DISCLAIMER,
    }


@app.get("/health")
def health():
    """Health check endpoint without leaking internal filesystem paths."""
    try:
        predictor = get_predictor()
        model_ready = predictor.pipeline is not None
        model_name = predictor.model_name
    except Exception:
        model_ready = False
        model_name = "Unavailable"

    return {
        "status": "healthy" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_name": model_name,
        "version": "1.0.0",
    }


@app.get("/api/overview")
def get_overview():
    """Returns dataset KPIs, class distribution, sensor limits, and model configuration."""
    metrics_path = OUTPUTS_DIR / "model_evaluation_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation metrics not found. Run model training first.",
        )

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    predictor = get_predictor()

    return {
        "dataset_name": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "doi": "https://doi.org/10.24432/C5HS5C",
        "license": "CC BY 4.0",
        "total_observations": 10000,
        "class_distribution": {
            "normal_count": 9661,
            "failure_count": 339,
            "failure_rate_pct": 3.39,
        },
        "quality_variants": {
            "L (Low)": {"count": 6000, "pct": 60.0},
            "M (Medium)": {"count": 2997, "pct": 29.97},
            "H (High)": {"count": 1003, "pct": 10.03},
        },
        "sensor_specifications": FEATURE_RANGES,
        "selected_model": {
            "model_name": predictor.model_name,
            "decision_threshold": predictor.threshold,
            "selection_criterion": "Out-of-fold F2-score maximization on training split",
            "feature_count": len(ALL_PREDICTOR_COLUMNS),
            "features": ALL_PREDICTOR_COLUMNS,
        },
        "synthetic_benchmark_notice": SAFETY_DISCLAIMER,
    }


@app.get("/api/model/metrics")
def get_model_metrics():
    """Returns genuine saved out-of-fold and held-out test evaluation metrics."""
    metrics_path = OUTPUTS_DIR / "model_evaluation_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metrics report not found.",
        )

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    return {
        "dataset_split": metrics_data["dataset_split"],
        "metadata": metrics_data["metadata"],
        "out_of_fold_validation_results": metrics_data["out_of_fold_validation_results"],
        "held_out_test_results": metrics_data["held_out_test_results"],
        "feature_importance_rf": metrics_data.get("feature_importance_rf", {}),
        "feature_coefficients_lr": metrics_data.get("feature_coefficients_lr", {}),
    }


@app.get("/api/statistics")
def get_statistics():
    """Returns genuine SciPy non-parametric tests, FDR corrections, and Chi-Square results."""
    stats_path = OUTPUTS_DIR / "statistical_analysis_report.json"
    if not stats_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Statistical analysis report not found.",
        )

    with open(stats_path, "r") as f:
        stats_data = json.load(f)

    return stats_data


@app.post("/api/predict")
def predict_telemetry(payload: SensorInputSchema):
    """
    Validates sensor inputs and generates model-estimated anomaly scores.
    """
    try:
        predictor = get_predictor()
        reading_dict = payload.to_reading_dict()
        prediction = predictor.predict(reading_dict)[0]
        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/api/simulator/initialize")
def initialize_simulator(req: SimulatorInitRequest):
    """
    Initializes a new simulator state for the requested scenario and machine type.
    Stateless on the server; the state is returned to the client to manage.
    """
    try:
        state = init_simulator_state(
            scenario=req.scenario,
            machine_type=req.machine_type,
            seed=req.seed,
        )
        return {
            "state": state,
            "scenario": req.scenario,
            "machine_type": req.machine_type,
            "is_simulated": True,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulator initialization error: {str(e)}",
        )


@app.post("/api/simulator/step")
def step_simulator(req: SimulatorStepRequest):
    """
    Takes client-provided simulator state, steps forward by one tick,
    computes model prediction on the resulting reading, and returns updated state.
    """
    try:
        if not req.state or not isinstance(req.state, dict):
            raise ValueError("Invalid simulator state payload.")

        next_state, sim_output = step_simulation(req.state)
        features = sim_output["features"]
        metadata = sim_output["metadata"]

        predictor = get_predictor()
        prediction = predictor.predict(features)[0]

        return {
            "next_state": next_state,
            "reading": features,
            "metadata": metadata,
            "prediction": prediction,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulator step error: {str(e)}",
        )

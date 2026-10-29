"""
Unit tests for the Production Prediction Interface.
Tests artifact loading, validation, score bounds, and error handling without fabricating metrics.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.config import ALL_PREDICTOR_COLUMNS, METADATA_PATH
from src.models.predictor import ProductionPredictor, predict_reading


@pytest.fixture
def sample_reading() -> dict:
    return {
        "Type": "L",
        "Air temperature [K]": 300.2,
        "Process temperature [K]": 310.1,
        "Rotational speed [rpm]": 1520,
        "Torque [Nm]": 41.5,
        "Tool wear [min]": 65,
    }


def test_predictor_loads_production_artifacts():
    """Ensures production model and metadata load cleanly."""
    if not METADATA_PATH.exists():
        pytest.skip("Production model metadata not present on disk.")

    predictor = ProductionPredictor()
    assert predictor.model_name == "Random Forest"
    assert 0.0 < predictor.threshold < 1.0
    assert predictor.pipeline is not None


def test_predictor_single_and_batch_inference(sample_reading):
    """Verifies that predictor handles single dict and batch DataFrame accurately."""
    if not METADATA_PATH.exists():
        pytest.skip("Production model metadata not present on disk.")

    predictor = ProductionPredictor()

    # Single dictionary
    single_res = predictor.predict(sample_reading)
    assert len(single_res) == 1
    r0 = single_res[0]

    assert 0.0 <= r0["estimated_anomaly_score"] <= 1.0
    assert r0["predicted_label"] in [0, 1]
    assert r0["status"] in ["NORMAL_OPERATION", "ATTENTION_ADVISED", "MAINTENANCE_REQUIRED"]
    assert "synthetic" in r0["score_interpretation"].lower()

    # Batch DataFrame
    df = pd.DataFrame([sample_reading, sample_reading])
    batch_res = predictor.predict(df)
    assert len(batch_res) == 2


def test_predictor_rejection_of_missing_columns(sample_reading):
    """Verifies that missing predictor columns trigger a clear ValueError."""
    predictor = ProductionPredictor()
    bad_reading = sample_reading.copy()
    del bad_reading["Torque [Nm]"]

    with pytest.raises(ValueError, match="missing required predictor columns"):
        predictor.predict(bad_reading)


def test_predictor_rejection_of_nulls(sample_reading):
    """Verifies that null / NaN values are rejected."""
    predictor = ProductionPredictor()
    bad_reading = sample_reading.copy()
    bad_reading["Rotational speed [rpm]"] = None

    with pytest.raises(ValueError, match="null/NaN values"):
        predictor.predict(bad_reading)


def test_predictor_rejection_of_invalid_type(sample_reading):
    """Verifies that invalid Type categories raise ValueError."""
    predictor = ProductionPredictor()
    bad_reading = sample_reading.copy()
    bad_reading["Type"] = "InvalidType"

    with pytest.raises(ValueError, match="Invalid 'Type' categories"):
        predictor.predict(bad_reading)


def test_predictor_missing_metadata_error(tmp_path):
    """Verifies that missing metadata raises a RuntimeError rather than arbitrary fallback."""
    fake_path = tmp_path / "non_existent_metadata.json"
    with pytest.raises(RuntimeError, match="not found"):
        ProductionPredictor(metadata_path=fake_path)


def test_predict_reading_convenience_function(sample_reading):
    """Verifies the module-level convenience function."""
    if not METADATA_PATH.exists():
        pytest.skip("Production model metadata not present on disk.")

    res = predict_reading(sample_reading)
    assert "estimated_anomaly_score" in res
    assert "status" in res

"""
Unit tests for data validation, schema conformity, and data loader functions.
Designed to run strictly offline without requiring internet connectivity.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    ID_COLUMNS,
    LEAKAGE_FAILURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from src.data.loader import (
    validate_dataset,
    load_dataset,
    get_dataset_summary,
    EXPECTED_COLUMNS,
)


@pytest.fixture
def valid_mock_df() -> pd.DataFrame:
    """Creates a valid in-memory mock dataframe matching the UCI schema."""
    data = {
        "UDI": [1, 2, 3],
        "Product ID": ["L47181", "M14860", "H39480"],
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [298.1, 298.2, 298.3],
        "Process temperature [K]": [308.6, 308.7, 308.8],
        "Rotational speed [rpm]": [1551, 1408, 1498],
        "Torque [Nm]": [42.8, 46.3, 49.4],
        "Tool wear [min]": [0, 3, 5],
        "Machine failure": [0, 0, 1],
        "TWF": [0, 0, 0],
        "HDF": [0, 0, 0],
        "PWF": [0, 0, 0],
        "OSF": [0, 0, 1],
        "RNF": [0, 0, 0],
    }
    return pd.DataFrame(data)


def test_validate_dataset_valid(valid_mock_df):
    """Verifies that a well-formed DataFrame passes validation."""
    report = validate_dataset(valid_mock_df, strict_row_count=False)
    assert report["status"] == "VALID"
    assert report["row_count"] == 3
    assert report["missing_values"] == 0
    assert report["valid_types"] == ["H", "L", "M"]


def test_validate_dataset_missing_column(valid_mock_df):
    """Verifies that dropping a required column raises a ValueError."""
    bad_df = valid_mock_df.drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(bad_df)


def test_validate_dataset_null_values(valid_mock_df):
    """Verifies that null / NaN values in features raise a ValueError."""
    bad_df = valid_mock_df.copy()
    bad_df.loc[0, "Air temperature [K]"] = None
    with pytest.raises(ValueError, match="null values"):
        validate_dataset(bad_df)


def test_validate_dataset_invalid_binary_label(valid_mock_df):
    """Verifies that non-binary target or failure labels raise a ValueError."""
    bad_df = valid_mock_df.copy()
    bad_df.loc[0, TARGET_COLUMN] = 2  # Non-binary
    with pytest.raises(ValueError, match="must only contain binary values"):
        validate_dataset(bad_df)


def test_validate_dataset_invalid_type_category(valid_mock_df):
    """Verifies that unsupported Type categories raise a ValueError."""
    bad_df = valid_mock_df.copy()
    bad_df.loc[0, "Type"] = "X"
    with pytest.raises(ValueError, match="invalid values"):
        validate_dataset(bad_df)


def test_validate_dataset_negative_sensor_values(valid_mock_df):
    """Verifies that non-positive physical sensor values raise a ValueError."""
    bad_df = valid_mock_df.copy()
    bad_df.loc[0, "Rotational speed [rpm]"] = -100
    with pytest.raises(ValueError, match="Rotational speed contains non-positive values"):
        validate_dataset(bad_df)


def test_get_dataset_summary(valid_mock_df):
    """Verifies that get_dataset_summary produces accurate metrics."""
    summary = get_dataset_summary(valid_mock_df)
    assert summary["total_records"] == 3
    assert summary["target_distribution"]["Normal (0)"] == 2
    assert summary["target_distribution"]["Failure (1)"] == 1
    assert summary["failure_mode_counts"]["OSF"] == 1
    assert summary["type_distribution"]["L"] == 1
    assert "Torque [Nm]" in summary["sensor_statistics"]


def test_load_dataset_from_csv(tmp_path, valid_mock_df):
    """Verifies that load_dataset reads and validates a CSV from disk."""
    csv_file = tmp_path / "mock_ai4i.csv"
    valid_mock_df.to_csv(csv_file, index=False)

    loaded_df = load_dataset(csv_file, validate=True, strict_row_count=False)
    assert len(loaded_df) == 3
    assert list(loaded_df.columns) == EXPECTED_COLUMNS


def test_leakage_columns_are_defined_in_config():
    """Ensures leakage prevention list contains all 5 failure modes and 2 identifiers."""
    assert "UDI" in ID_COLUMNS
    assert "Product ID" in ID_COLUMNS
    for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert col in LEAKAGE_FAILURE_COLUMNS
    assert TARGET_COLUMN not in CATEGORICAL_FEATURES
    assert TARGET_COLUMN not in NUMERIC_FEATURES


def test_real_dataset_integrity():
    """Validates the local copy of ai4i2020.csv if present on disk."""
    if not RAW_DATA_PATH.exists():
        pytest.skip("ai4i2020.csv not present locally; skipping real dataset check.")

    df = load_dataset(RAW_DATA_PATH, validate=True, strict_row_count=True)
    assert len(df) == 10000
    assert len(df.columns) == 14
    summary = get_dataset_summary(df)
    assert summary["total_records"] == 10000
    assert summary["target_distribution"]["Failure (1)"] == 339
    assert summary["target_distribution"]["Normal (0)"] == 9661
    assert summary["duplicates"]["all_columns"] == 0

